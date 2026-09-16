"""Compact SAINT-style model for tabular churn classification.

SAINT (Somepalli et al., 2021) is not shipped in PyTorch Tabular, so this
module implements the two attention mechanisms from the paper:

- Inter-feature attention: each feature token attends to the other features
  in the same row (same idea as FT-Transformer).
- Inter-sample attention: each feature token attends to the same feature
  across other rows in the mini-batch (the SAINT-specific row mixer).

Numerical features are projected to tokens (value * weight + bias). Categorical
features use learned embeddings. A CLS token is used for classification.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader, Dataset


@dataclass
class SaintConfig:
    dim: int = 32
    depth: int = 3
    num_heads: int = 4
    dropout: float = 0.1
    ffn_mult: int = 4
    lr: float = 1e-3
    weight_decay: float = 1e-4
    batch_size: int = 256
    max_epochs: int = 20
    patience: int = 5
    seed: int = 42
    use_intersample: bool = True


class TelcoTensorDataset(Dataset):
    def __init__(self, x_cat: np.ndarray, x_num: np.ndarray, y: np.ndarray):
        self.x_cat = torch.as_tensor(x_cat, dtype=torch.long)
        self.x_num = torch.as_tensor(x_num, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long)

    def __len__(self) -> int:
        return int(self.y.shape[0])

    def __getitem__(self, idx: int):
        return self.x_cat[idx], self.x_num[idx], self.y[idx]


class _ResidualAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, dropout: float, ffn_mult: int):
        super().__init__()
        self.norm_attn = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.drop = nn.Dropout(dropout)
        self.norm_ff = nn.LayerNorm(dim)
        hidden = dim * ffn_mult
        self.ff = nn.Sequential(
            nn.Linear(dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm_attn(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop(attn_out)
        x = x + self.ff(self.norm_ff(x))
        return x


class SaintLayer(nn.Module):
    def __init__(
        self,
        dim: int,
        num_heads: int,
        dropout: float,
        ffn_mult: int,
        use_intersample: bool,
    ):
        super().__init__()
        self.feature_block = _ResidualAttention(dim, num_heads, dropout, ffn_mult)
        self.use_intersample = use_intersample
        if use_intersample:
            self.sample_block = _ResidualAttention(dim, num_heads, dropout, ffn_mult)

    def forward(self, x: torch.Tensor, apply_intersample: bool) -> torch.Tensor:
        # x: (batch, n_tokens, dim) — attention across features
        x = self.feature_block(x)
        if self.use_intersample and apply_intersample and x.size(0) > 1:
            # (n_tokens, batch, dim) — attention across rows for each feature
            x_t = x.transpose(0, 1)
            x_t = self.sample_block(x_t)
            x = x_t.transpose(0, 1)
        return x


class SaintClassifier(nn.Module):
    def __init__(
        self,
        cardinalities: Sequence[int],
        n_num: int,
        config: SaintConfig,
    ):
        super().__init__()
        self.config = config
        dim = config.dim
        self.n_cat = len(cardinalities)
        self.n_num = n_num
        # +1 slot per categorical column for unseen values at inference
        self.cat_embeds = nn.ModuleList(
            [nn.Embedding(int(card) + 1, dim) for card in cardinalities]
        )
        if n_num > 0:
            self.num_weight = nn.Parameter(torch.randn(n_num, dim) * 0.02)
            self.num_bias = nn.Parameter(torch.zeros(n_num, dim))
        n_tokens = 1 + self.n_cat + n_num
        self.cls = nn.Parameter(torch.zeros(1, 1, dim))
        self.col_embed = nn.Parameter(torch.randn(1, n_tokens, dim) * 0.02)
        self.layers = nn.ModuleList(
            [
                SaintLayer(
                    dim=dim,
                    num_heads=config.num_heads,
                    dropout=config.dropout,
                    ffn_mult=config.ffn_mult,
                    use_intersample=config.use_intersample,
                )
                for _ in range(config.depth)
            ]
        )
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Sequential(
            nn.Linear(dim, dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(dim, 2),
        )
        self._reset_parameters()

    def _reset_parameters(self) -> None:
        nn.init.trunc_normal_(self.cls, std=0.02)
        for emb in self.cat_embeds:
            nn.init.trunc_normal_(emb.weight, std=0.02)

    def tokenize(self, x_cat: torch.Tensor, x_num: torch.Tensor) -> torch.Tensor:
        batch = x_cat.size(0)
        tokens = [self.cls.expand(batch, -1, -1)]
        if self.n_cat:
            cat_tokens = torch.stack(
                [emb(x_cat[:, i]) for i, emb in enumerate(self.cat_embeds)],
                dim=1,
            )
            tokens.append(cat_tokens)
        if self.n_num:
            num_tokens = x_num.unsqueeze(-1) * self.num_weight + self.num_bias
            tokens.append(num_tokens)
        x = torch.cat(tokens, dim=1)
        return x + self.col_embed

    def forward(
        self,
        x_cat: torch.Tensor,
        x_num: torch.Tensor,
        apply_intersample: Optional[bool] = None,
    ) -> torch.Tensor:
        if apply_intersample is None:
            apply_intersample = bool(self.training)
        x = self.tokenize(x_cat, x_num)
        for layer in self.layers:
            x = layer(x, apply_intersample=apply_intersample)
        cls = self.norm(x[:, 0])
        return self.head(cls)


class SaintPreprocessor:
    """Train-only fit of label encoders + numeric scaler."""

    def __init__(self, categorical_cols: Sequence[str], numerical_cols: Sequence[str]):
        self.categorical_cols = list(categorical_cols)
        self.numerical_cols = list(numerical_cols)
        self.encoders: Dict[str, LabelEncoder] = {}
        self.cardinalities: List[int] = []
        self.scaler = StandardScaler()

    def fit(self, frame) -> "SaintPreprocessor":
        self.cardinalities = []
        for col in self.categorical_cols:
            enc = LabelEncoder()
            values = frame[col].astype(str).fillna("__missing__").to_numpy()
            enc.fit(values)
            self.encoders[col] = enc
            self.cardinalities.append(len(enc.classes_))
        if self.numerical_cols:
            self.scaler.fit(frame[self.numerical_cols].to_numpy(dtype=np.float32))
        return self

    def transform(self, frame) -> Tuple[np.ndarray, np.ndarray]:
        cat_cols = []
        for col in self.categorical_cols:
            enc = self.encoders[col]
            raw = frame[col].astype(str).fillna("__missing__").to_numpy()
            mapping = {cls: i for i, cls in enumerate(enc.classes_)}
            unknown = len(enc.classes_)
            cat_cols.append(np.array([mapping.get(v, unknown) for v in raw], dtype=np.int64))
        x_cat = (
            np.stack(cat_cols, axis=1)
            if cat_cols
            else np.zeros((len(frame), 0), dtype=np.int64)
        )
        if self.numerical_cols:
            x_num = self.scaler.transform(
                frame[self.numerical_cols].to_numpy(dtype=np.float32)
            )
        else:
            x_num = np.zeros((len(frame), 0), dtype=np.float32)
        return x_cat, x_num


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_device() -> torch.device:
    if torch.backends.mps.is_available():
        return torch.device("mps")
    if torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


@torch.no_grad()
def predict_proba(
    model: SaintClassifier,
    x_cat: np.ndarray,
    x_num: np.ndarray,
    device: torch.device,
    batch_size: int = 512,
    apply_intersample: bool = False,
) -> np.ndarray:
    """Return P(churn=Yes). Intersample is off by default so one-row scoring is valid."""
    model.eval()
    loader = DataLoader(
        TelcoTensorDataset(x_cat, x_num, np.zeros(len(x_cat), dtype=np.int64)),
        batch_size=batch_size,
        shuffle=False,
    )
    probs = []
    for cat_b, num_b, _ in loader:
        logits = model(
            cat_b.to(device),
            num_b.to(device),
            apply_intersample=apply_intersample,
        )
        probs.append(torch.softmax(logits, dim=-1)[:, 1].detach().cpu().numpy())
    return np.concatenate(probs, axis=0)


def train_saint(
    train_frame,
    val_frame,
    categorical_cols: Sequence[str],
    numerical_cols: Sequence[str],
    target_col: str = "Churn",
    config: Optional[SaintConfig] = None,
    device: Optional[torch.device] = None,
    checkpoint_path: Optional[Path] = None,
) -> Tuple[SaintClassifier, SaintPreprocessor, Dict[str, float]]:
    config = config or SaintConfig()
    device = device or resolve_device()
    _seed_everything(config.seed)

    y_train = (train_frame[target_col].astype(str) == "Yes").to_numpy(dtype=np.int64)
    y_val = (val_frame[target_col].astype(str) == "Yes").to_numpy(dtype=np.int64)

    preprocessor = SaintPreprocessor(categorical_cols, numerical_cols).fit(train_frame)
    x_cat_tr, x_num_tr = preprocessor.transform(train_frame)
    x_cat_va, x_num_va = preprocessor.transform(val_frame)

    model = SaintClassifier(preprocessor.cardinalities, x_num_tr.shape[1], config).to(device)
    n_no = max(int((y_train == 0).sum()), 1)
    n_yes = max(int((y_train == 1).sum()), 1)
    class_weight = torch.tensor([1.0, n_no / n_yes], dtype=torch.float32, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.lr, weight_decay=config.weight_decay
    )

    train_loader = DataLoader(
        TelcoTensorDataset(x_cat_tr, x_num_tr, y_train),
        batch_size=config.batch_size,
        shuffle=True,
        drop_last=False,
    )

    best_auc = -1.0
    stale = 0
    best_state = None
    history: Dict[str, float] = {}

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        running = 0.0
        n_seen = 0
        for cat_b, num_b, y_b in train_loader:
            cat_b = cat_b.to(device)
            num_b = num_b.to(device)
            y_b = y_b.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(cat_b, num_b, apply_intersample=True)
            loss = criterion(logits, y_b)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running += float(loss.item()) * y_b.size(0)
            n_seen += y_b.size(0)

        val_prob = predict_proba(
            model, x_cat_va, x_num_va, device, batch_size=config.batch_size, apply_intersample=False
        )
        val_auc = _roc_auc(y_val, val_prob)
        train_loss = running / max(n_seen, 1)
        print(
            f"  SAINT epoch {epoch:02d}/{config.max_epochs}  "
            f"train_loss={train_loss:.4f}  val_auc={val_auc:.4f}"
        )
        if val_auc > best_auc + 1e-4:
            best_auc = val_auc
            stale = 0
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= config.patience:
                print("  SAINT early stopping")
                break

    if best_state is not None:
        model.load_state_dict(best_state)
    model.to(device)
    history["best_val_auc"] = float(best_auc)

    if checkpoint_path is not None:
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "state_dict": model.state_dict(),
                "config": asdict(config),
                "cardinalities": preprocessor.cardinalities,
                "categorical_cols": preprocessor.categorical_cols,
                "numerical_cols": preprocessor.numerical_cols,
                "encoders": preprocessor.encoders,
                "scaler": preprocessor.scaler,
            },
            checkpoint_path,
        )
    return model, preprocessor, history


def load_saint(checkpoint_path: Path, device: Optional[torch.device] = None):
    device = device or resolve_device()
    blob = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = SaintConfig(**blob["config"])
    model = SaintClassifier(blob["cardinalities"], len(blob["numerical_cols"]), config)
    model.load_state_dict(blob["state_dict"])
    model.to(device)
    preprocessor = SaintPreprocessor(blob["categorical_cols"], blob["numerical_cols"])
    preprocessor.encoders = blob["encoders"]
    preprocessor.cardinalities = blob["cardinalities"]
    preprocessor.scaler = blob["scaler"]
    return model, preprocessor


def _roc_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score

    if len(np.unique(y_true)) < 2:
        return 0.5
    return float(roc_auc_score(y_true, y_prob))

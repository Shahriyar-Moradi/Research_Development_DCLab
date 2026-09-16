"""Binary TabTransformer / FTTransformer trainers.

Architecture is lucidrains/tab-transformer-pytorch. Training and preprocessing
follow the original papers plus the practical recipes in:

- Gorishniy et al. 2021 / Suzuki Kaggle FT-Transformer notebook
  https://www.kaggle.com/code/masatakasuzuki/ft-transformer-transformer-for-tabular-data
  QuantileTransformer(normal) on numerics, AdamW 1e-4, wd 1e-5 (no decay on
  embeddings / LayerNorm / bias), patience 16, no LR schedule.
- Arash Khoeini, FTTransformer explainer
  https://arashk.medium.com/fttransformer-transformer-architecture-for-tabular-datasets-d4bfe591d6fb
  One token per feature (cat embedding + numeric tokenizer), then encoder layers.
- Aravind Kolli, TabTransformer guide
  https://aravindkolli.medium.com/mastering-tabular-data-with-tabtransformer-a-comprehensive-guide-119f6dbf5a79
  StandardScaler on numerics, dropout, Adam-family lr ~1e-3, short LR warmup.
  We do **not** copy that article's Wine-Quality model (a linear map to sequence
  length 1). Huang et al. attend over categorical column embeddings.
"""

from __future__ import annotations

import copy
import time
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, QuantileTransformer, StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from tab_transformer_pytorch import FTTransformer, TabTransformer

from train_tabpfn import metrics, pick_device


@dataclass
class TabTransformerConfig:
    kind: str = "tab"  # "tab" | "ft"
    dim: int = 32
    depth: int = 6
    heads: int = 8
    attn_dropout: float = 0.1
    ff_dropout: float = 0.1
    epochs: int = 200
    batch_size: int = 256
    lr: float = 1e-3
    weight_decay: float = 1e-4
    patience: int = 16
    warmup_epochs: int = 0
    val_size: float = 0.15
    seed: int = 42
    num_residual_streams: int = 1
    numeric_transform: str = "standard"  # "standard" | "quantile"


def default_config(kind: str) -> TabTransformerConfig:
    if kind == "ft":
        # Gorishniy Table 12 defaults, sized for these tables (d_token 192, 3 blocks).
        return TabTransformerConfig(
            kind="ft",
            dim=192,
            depth=3,
            heads=8,
            attn_dropout=0.2,
            ff_dropout=0.1,
            epochs=200,
            batch_size=256,
            lr=1e-4,
            weight_decay=1e-5,
            patience=16,
            warmup_epochs=0,
            numeric_transform="quantile",
        )
    # Huang TabTransformer + Kolli training tips (warmup, dropout, lr 1e-3).
    return TabTransformerConfig(
        kind="tab",
        dim=32,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        epochs=200,
        batch_size=256,
        lr=1e-3,
        weight_decay=1e-4,
        patience=16,
        warmup_epochs=5,
        numeric_transform="standard",
    )


def cat_num_cols(X: pd.DataFrame) -> tuple[list[str], list[str]]:
    cat_cols = list(X.select_dtypes(include=["category", "object", "bool"]).columns)
    num_cols = [c for c in X.columns if c not in cat_cols]
    return cat_cols, num_cols


def _seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _resolve_device(requested: str | None) -> torch.device:
    name = requested or pick_device()
    if name == "mps" and not torch.backends.mps.is_available():
        name = "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        name = "cpu"
    return torch.device(name)


def _numeric_pipeline(transform: str, n_samples: int, seed: int) -> Pipeline:
    steps: list[tuple[str, object]] = [("imp", SimpleImputer(strategy="median"))]
    if transform == "quantile":
        n_quantiles = max(2, min(1000, n_samples))
        steps.append(
            (
                "qt",
                QuantileTransformer(
                    n_quantiles=n_quantiles,
                    output_distribution="normal",
                    subsample=min(100_000, max(n_samples, 10)),
                    random_state=seed,
                ),
            )
        )
    else:
        steps.append(("sc", StandardScaler()))
    return Pipeline(steps)


class _FeatureEncoder:
    """Integer category IDs in [0, n); unknown/missing as -1 (library specials)."""

    def __init__(self, numeric_transform: str = "standard", seed: int = 42) -> None:
        self.numeric_transform = numeric_transform
        self.seed = seed
        self.cat_cols: list[str] = []
        self.num_cols: list[str] = []
        self.categories: tuple[int, ...] = ()
        self.pre: ColumnTransformer | None = None

    def fit(self, X: pd.DataFrame) -> _FeatureEncoder:
        self.cat_cols, self.num_cols = cat_num_cols(X)
        transformers = []
        if self.cat_cols:
            transformers.append(
                (
                    "cat",
                    OrdinalEncoder(
                        handle_unknown="use_encoded_value",
                        unknown_value=-1,
                        encoded_missing_value=-1,
                    ),
                    self.cat_cols,
                )
            )
        if self.num_cols:
            transformers.append(
                (
                    "num",
                    _numeric_pipeline(self.numeric_transform, len(X), self.seed),
                    self.num_cols,
                )
            )
        self.pre = ColumnTransformer(transformers, remainder="drop")
        encoded = self.pre.fit_transform(X)
        n_cat = len(self.cat_cols)
        cat = np.asarray(encoded[:, :n_cat], dtype=np.float64) if n_cat else np.zeros((len(X), 0))
        card = []
        for i in range(n_cat):
            known = cat[:, i][cat[:, i] >= 0]
            card.append(int(known.max()) + 1 if known.size else 1)
        self.categories = tuple(card)
        return self

    def transform(self, X: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        if self.pre is None:
            raise RuntimeError("encoder is not fitted")
        encoded = np.asarray(self.pre.transform(X), dtype=np.float64)
        n_cat = len(self.cat_cols)
        cat = encoded[:, :n_cat].astype(np.int64) if n_cat else np.zeros((len(X), 0), dtype=np.int64)
        num = (
            encoded[:, n_cat:].astype(np.float32)
            if self.num_cols
            else np.zeros((len(X), 0), dtype=np.float32)
        )
        return cat, num


def _build_model(cfg: TabTransformerConfig, categories: tuple[int, ...], n_cont: int) -> nn.Module:
    cats = categories if categories else (1,)
    common = dict(
        categories=cats,
        num_continuous=n_cont,
        dim=cfg.dim,
        depth=cfg.depth,
        heads=cfg.heads,
        dim_out=1,
        attn_dropout=cfg.attn_dropout,
        ff_dropout=cfg.ff_dropout,
        num_residual_streams=cfg.num_residual_streams,
    )
    if cfg.kind == "ft":
        return FTTransformer(**common)
    return TabTransformer(**common, mlp_act=nn.ReLU())


def _maybe_dummy_categories(cat: np.ndarray, categories: tuple[int, ...]) -> np.ndarray:
    if categories:
        return cat
    return np.zeros((len(cat), 1), dtype=np.int64)


def _tensors(
    cat: np.ndarray, num: np.ndarray, y: np.ndarray | None, categories: tuple[int, ...]
) -> tuple[torch.Tensor, ...]:
    x_categ = torch.as_tensor(_maybe_dummy_categories(cat, categories), dtype=torch.long)
    x_cont = torch.as_tensor(num, dtype=torch.float32)
    if y is None:
        return x_categ, x_cont
    return x_categ, x_cont, torch.as_tensor(y, dtype=torch.float32)


def _adamw(model: nn.Module, lr: float, weight_decay: float) -> torch.optim.AdamW:
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        lower = name.lower()
        if param.ndim <= 1 or lower.endswith("bias") or "norm" in lower or "embed" in lower:
            no_decay.append(param)
        else:
            decay.append(param)
    return torch.optim.AdamW(
        [
            {"params": decay, "weight_decay": weight_decay},
            {"params": no_decay, "weight_decay": 0.0},
        ],
        lr=lr,
    )


@torch.no_grad()
def _predict_logits(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    model.eval()
    chunks: list[np.ndarray] = []
    for batch in loader:
        x_categ, x_cont = batch[0].to(device), batch[1].to(device)
        logits = model(x_categ, x_cont).squeeze(-1)
        chunks.append(logits.detach().cpu().numpy())
    return np.concatenate(chunks, axis=0)


def _predict_proba(model: nn.Module, loader: DataLoader, device: torch.device) -> np.ndarray:
    logits = _predict_logits(model, loader, device)
    return 1.0 / (1.0 + np.exp(-np.clip(logits, -30, 30)))


def fit_tab_transformer(
    X_train: pd.DataFrame,
    y_train: pd.Series | np.ndarray,
    X_test: pd.DataFrame,
    y_test: pd.Series | np.ndarray,
    *,
    cfg: TabTransformerConfig | None = None,
    device: str | None = None,
) -> dict:
    cfg = cfg or default_config("tab")
    _seed_everything(cfg.seed)
    torch_device = _resolve_device(device)
    y_train_arr = np.asarray(y_train).astype(np.int64)
    y_test_arr = np.asarray(y_test).astype(np.int64)

    if len(X_train) >= 40 and y_train_arr.min() != y_train_arr.max():
        X_fit, X_val, y_fit, y_val = train_test_split(
            X_train,
            y_train_arr,
            test_size=cfg.val_size,
            random_state=cfg.seed,
            stratify=y_train_arr,
        )
    else:
        X_fit, y_fit = X_train, y_train_arr
        X_val, y_val = X_train, y_train_arr

    encoder = _FeatureEncoder(cfg.numeric_transform, cfg.seed).fit(X_fit)
    cat_fit, num_fit = encoder.transform(X_fit)
    cat_val, num_val = encoder.transform(X_val)
    cat_te, num_te = encoder.transform(X_test)
    categories = encoder.categories if encoder.categories else (1,)

    model = _build_model(cfg, categories, num_fit.shape[1]).to(torch_device)
    n_pos = max(int((y_fit == 1).sum()), 1)
    n_neg = max(int((y_fit == 0).sum()), 1)
    pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32, device=torch_device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    opt = _adamw(model, cfg.lr, cfg.weight_decay)
    warmup = max(int(cfg.warmup_epochs), 0)

    def _lr_mult(epoch: int) -> float:
        if warmup <= 0:
            return 1.0
        if epoch < warmup:
            return float(epoch + 1) / float(warmup)
        return 1.0

    scheduler = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda=_lr_mult)

    train_ds = TensorDataset(*_tensors(cat_fit, num_fit, y_fit, categories))
    val_ds = TensorDataset(*_tensors(cat_val, num_val, None, categories))
    test_ds = TensorDataset(*_tensors(cat_te, num_te, None, categories))
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False)

    best_state = copy.deepcopy(model.state_dict())
    best_auc = -1.0
    bad = 0
    t0 = time.perf_counter()
    try:
        for epoch in range(cfg.epochs):
            model.train()
            for x_categ, x_cont, yb in train_loader:
                x_categ = x_categ.to(torch_device)
                x_cont = x_cont.to(torch_device)
                yb = yb.to(torch_device)
                opt.zero_grad(set_to_none=True)
                logits = model(x_categ, x_cont).squeeze(-1)
                loss = criterion(logits, yb)
                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
            scheduler.step()
            val_proba = _predict_proba(model, val_loader, torch_device)
            try:
                auc = float(roc_auc_score(y_val, val_proba))
            except ValueError:
                auc = 0.5
            if auc > best_auc + 1e-4:
                best_auc = auc
                best_state = copy.deepcopy(model.state_dict())
                bad = 0
            else:
                bad += 1
                if bad >= cfg.patience:
                    break
    except RuntimeError:
        if torch_device.type != "cpu":
            return fit_tab_transformer(
                X_train,
                y_train,
                X_test,
                y_test,
                cfg=cfg,
                device="cpu",
            )
        raise
    fit_s = time.perf_counter() - t0

    model.load_state_dict(best_state)
    t1 = time.perf_counter()
    proba = _predict_proba(model, test_loader, torch_device)
    pred_s = time.perf_counter() - t1
    pred = (proba >= 0.5).astype(int)
    name = "TabTransformer" if cfg.kind == "tab" else "FTTransformer"
    out = metrics(y_test_arr, pred, proba)
    out.update(
        {
            "model": name,
            "fit_seconds": round(fit_s, 3),
            "predict_seconds": round(pred_s, 3),
            "notes": (
                f"lucidrains/{name} {cfg.numeric_transform} dim={cfg.dim} depth={cfg.depth} "
                f"heads={cfg.heads} lr={cfg.lr} wd={cfg.weight_decay} warmup={warmup} "
                f"patience={cfg.patience} valAUC={best_auc:.3f} device={torch_device}"
            ),
            "val_auc": best_auc,
            "device": str(torch_device),
        }
    )
    return out


def _merge_cfg(kind: str, **kwargs) -> TabTransformerConfig:
    cfg = default_config(kind)
    if not kwargs:
        return cfg
    return replace(cfg, **kwargs)


def run_tab_transformer(X_train, y_train, X_test, y_test, *, device: str | None = None, **kwargs) -> dict:
    return fit_tab_transformer(
        X_train,
        y_train,
        X_test,
        y_test,
        cfg=_merge_cfg("tab", **kwargs),
        device=device,
    )


def run_ft_transformer(X_train, y_train, X_test, y_test, *, device: str | None = None, **kwargs) -> dict:
    return fit_tab_transformer(
        X_train,
        y_train,
        X_test,
        y_test,
        cfg=_merge_cfg("ft", **kwargs),
        device=device,
    )

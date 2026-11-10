"""Holdout crash-safety and split governance.

Prevents holdout replay, silent re-scoring after crashes, and split drift across project stages.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Optional, Sequence


def split_fingerprint(train_indices: Sequence[Any], holdout_indices: Sequence[Any]) -> str:
    """Canonical SHA-256 hex digest of train and holdout row indices.

    Guarantees split reproducibility and detects any partition drift between data locking and final scoring.
    """
    try:
        train_sorted = sorted(int(i) for i in train_indices)
        holdout_sorted = sorted(int(i) for i in holdout_indices)
    except (TypeError, ValueError):
        train_sorted = sorted(str(i) for i in train_indices)
        holdout_sorted = sorted(str(i) for i in holdout_indices)

    payload = json.dumps({"train": train_sorted, "holdout": holdout_sorted}, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass
class HoldoutReceipt:
    receipt_id: str
    project_id: str
    stage: str
    split_fingerprint: str
    intended_at: str
    consumed: bool = False
    consumed_at: Optional[str] = None
    holdout_uses: int = 1
    reuse_reason: Optional[str] = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "HoldoutReceipt":
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


def record_intent(
    receipt_path: Path | str,
    project_id: str,
    split_fingerprint: str,
    holdout_uses: int = 1,
    stage: str = "final",
    reuse_reason: Optional[str] = None,
) -> HoldoutReceipt:
    """Atomically record intent to evaluate on the holdout.

    Must be written BEFORE any holdout data is read or evaluated.
    """
    path = Path(receipt_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    receipt_id = hashlib.sha256(f"{project_id}-{stage}-{time.time()}".encode("utf-8")).hexdigest()[:16]
    receipt = HoldoutReceipt(
        receipt_id=receipt_id,
        project_id=project_id,
        stage=stage,
        split_fingerprint=split_fingerprint,
        intended_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        consumed=False,
        consumed_at=None,
        holdout_uses=holdout_uses,
        reuse_reason=reuse_reason,
    )
    _atomic_save(path, receipt.to_dict())
    return receipt


def mark_consumed(receipt_path: Path | str, receipt: HoldoutReceipt) -> HoldoutReceipt:
    """Atomically mark the holdout receipt as consumed after scoring completes."""
    path = Path(receipt_path)
    receipt.consumed = True
    receipt.consumed_at = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    _atomic_save(path, receipt.to_dict())
    return receipt


def read_receipt(receipt_path: Path | str) -> Optional[HoldoutReceipt]:
    """Read existing holdout receipt if present, returning None if missing or corrupted."""
    path = Path(receipt_path)
    if not path.is_file():
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return HoldoutReceipt.from_dict(data)
    except Exception:
        return None


def check_receipt(
    receipt_path: Path | str,
    current_fingerprint: str,
    reuse_reason: Optional[str] = None,
) -> tuple[bool, Optional[str]]:
    """Check whether a holdout evaluation can proceed.

    Returns (True, None) if permitted, or (False, reason) if prohibited.
    """
    receipt = read_receipt(receipt_path)
    if receipt is None:
        return True, None

    # Detect split drift
    if receipt.split_fingerprint and current_fingerprint and receipt.split_fingerprint != current_fingerprint:
        return False, (
            f"Holdout split drift detected: current split fingerprint ({current_fingerprint[:12]}...) "
            f"differs from receipt ({receipt.split_fingerprint[:12]}...). "
            "Data split was modified after initial lock."
        )

    # Detect replay without reuse reason
    if not reuse_reason:
        if receipt.consumed:
            return False, (
                f"Holdout has already been consumed (receipt {receipt.receipt_id} at {receipt.consumed_at}). "
                "Holdout re-evaluation requires an explicit reuse_reason to guard against test set leakage."
            )
        else:
            return False, (
                f"Previous holdout intent recorded (receipt {receipt.receipt_id} at {receipt.intended_at}) "
                "without successful completion (possible crash during evaluation). "
                "Holdout re-evaluation after crash requires an explicit reuse_reason."
            )

    return True, None


def verify_split(train_indices: Sequence[Any], holdout_indices: Sequence[Any], expected_fingerprint: str) -> None:
    """Verify that train and holdout partitions match the expected split fingerprint."""
    fp = split_fingerprint(train_indices, holdout_indices)
    if fp != expected_fingerprint:
        raise ValueError(
            f"Split fingerprint mismatch: expected {expected_fingerprint[:12]}..., got {fp[:12]}..."
        )


def _atomic_save(path: Path, data: dict[str, Any]) -> None:
    """Write data to path atomically using a temporary file in the same directory."""
    dir_path = path.parent
    dir_path.mkdir(parents=True, exist_ok=True)
    temp_file = None
    try:
        with tempfile.NamedTemporaryFile("w", dir=dir_path, delete=False, encoding="utf-8") as tf:
            temp_file = Path(tf.name)
            json.dump(data, tf, indent=2, ensure_ascii=False)
            tf.flush()
            os.fsync(tf.fileno())
        os.replace(temp_file, path)
    except Exception:
        if temp_file and temp_file.exists():
            try:
                temp_file.unlink()
            except OSError:
                pass
        raise

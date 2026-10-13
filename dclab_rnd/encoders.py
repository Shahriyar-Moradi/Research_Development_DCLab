"""Pretrained encoders for dclab_core.embeddings (package 16.12), on the studio's side.

- Text: no pretrained text encoder is wired. The gateway's ``embedding`` purpose may see only the evidence records, never
  a workspace's rows or texts (its data limit), so encoding user texts needs a purpose of its own with its own limit:
  the owner's decision.
- :class:`TorchvisionImage`: a ResNet-18 from torchvision with weights read from a local file the owner provides. Nothing
  is downloaded: without the file the encoder refuses and says where to put it.

The local encoders that need no model (``HashingText``, ``RandomFeatures``) live in dclab_core and are baselines.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Sequence

import numpy as np


class TorchvisionImage:
    pretrained = True

    def __init__(self, weights: str | Path, size: int = 224):
        path = Path(weights)
        if not path.is_file():
            raise ValueError(f"no image encoder weights at {path}: place a torchvision ResNet-18 state dict there (DCLab never downloads one)")
        import torch
        import torchvision

        model = torchvision.models.resnet18(weights=None)
        model.load_state_dict(torch.load(path, map_location="cpu", weights_only=True))
        model.fc = torch.nn.Identity()
        import hashlib

        self.model, self.size, self.torch = model.eval(), size, torch
        self.name = f"resnet18-{hashlib.sha256(path.read_bytes()).hexdigest()[:12]}-{size}"  # the weights and the size, not a file name

    def encode(self, items: Sequence[Any]) -> np.ndarray:
        torch = self.torch
        batch = []
        for image in items:
            raw = np.asarray(image)
            a = raw.astype(np.float32) / (255.0 if np.issubdtype(raw.dtype, np.integer) else 1.0)  # photos in 0-255 to 0-1
            a = np.repeat(a[..., None], 3, axis=2) if a.ndim == 2 else a[..., :3]
            t = torch.from_numpy(a).permute(2, 0, 1)[None]
            t = torch.nn.functional.interpolate(t, size=(self.size, self.size), mode="bilinear", align_corners=False)
            mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            batch.append((t - mean) / std)
        with torch.no_grad():
            return self.model(torch.cat(batch)).numpy().astype(np.float32)

"""Camera-anchored window dataset for SLat (shape) flow fine-tuning.

Same layout, indexing and conditioning as ``ChunkCellSSLatent``, but each
sample is the window's sparse structured latent from ``slat_1024.npz``
(o-voxel resolution 1024 -> latent grid 64^3):

* ``coords``: [N, 3] int voxel positions on the 64^3 grid;
* ``feats``: [N, 32] latent features, standardized with ``normalization``.

Empty windows are excluded (there is no geometry to encode). ``collate_fn``
batches into one ``SparseTensor`` (batch index prepended to coords); with
``split_size`` it forms token-balanced groups.
"""
import os
from typing import Optional

import numpy as np
import torch

from ..modules.sparse.basic import SparseTensor
from ..utils.data_utils import load_balanced_group_indices
from .chunk_cell_ss_latent import ChunkCellSSLatent


class ChunkCellSLat(ChunkCellSSLatent):
    """
    Args:
        normalization: {"mean": [32], "std": [32]} feats standardization.
        pretrained_slat_dec / resolution: shape decoder for validation renders.
        Remaining arguments: see ``ChunkCellSSLatent``.
    """

    LATENT_FILENAME = "slat_1024.npz"
    INCLUDE_EMPTY = False

    def __init__(
        self,
        roots: str,
        *,
        normalization: Optional[dict] = None,
        pretrained_slat_dec: str = "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16",
        resolution: int = 1024,
        **kwargs,
    ):
        self.mean = self.std = None
        if normalization is not None:
            self.mean = torch.tensor(normalization["mean"], dtype=torch.float32).reshape(1, -1)
            self.std = torch.tensor(normalization["std"], dtype=torch.float32).reshape(1, -1)
        self.pretrained_slat_dec = pretrained_slat_dec
        self.resolution = int(resolution)
        super().__init__(roots, **kwargs)

    def __getitem__(self, index: int) -> dict:
        chunk_dir, image_path, depth_m, lateral_m, _ = self.instances[index]
        data = np.load(os.path.join(chunk_dir, self.LATENT_FILENAME))
        feats = torch.from_numpy(data["feats"]).float()
        if self.mean is not None:
            feats = (feats - self.mean) / self.std
        return {"coords": torch.from_numpy(data["coords"].astype(np.int32)), "feats": feats,
                "depth_m": depth_m, "lateral_m": lateral_m, **self._load_view(image_path)}

    @classmethod
    def _collate_group(cls, sub_batch):
        coords, feats, layout, start = [], [], [], 0
        for i, b in enumerate(sub_batch):
            n = b["coords"].shape[0]
            coords.append(torch.cat([torch.full((n, 1), i, dtype=torch.int32), b["coords"]], dim=-1))
            feats.append(b["feats"])
            layout.append(slice(start, start + n))
            start += n
        x_0 = SparseTensor(coords=torch.cat(coords), feats=torch.cat(feats))
        x_0._shape = torch.Size([len(sub_batch), *sub_batch[0]["feats"].shape[1:]])
        x_0.register_spatial_cache("layout", layout)
        return {"x_0": x_0, **cls._collate_common(sub_batch)}

    @classmethod
    def collate_fn(cls, batch, split_size=None):
        if split_size is None:
            return cls._collate_group(batch)
        group_idx = load_balanced_group_indices([b["coords"].shape[0] for b in batch], split_size)
        return [cls._collate_group([batch[i] for i in group]) for group in group_idx if group]

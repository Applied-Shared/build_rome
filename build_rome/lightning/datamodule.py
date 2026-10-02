"""LightningDataModule: builds the train/val window datasets from the config
(``data.train`` / ``data.val`` roots, ``dataset.args`` kwargs) and their loaders."""

from __future__ import annotations

import json
from typing import Optional

import lightning.pytorch as pl
from easydict import EasyDict as edict
from torch.utils.data import DataLoader, WeightedRandomSampler

from build_rome import datasets
from build_rome.utils.data_utils import TokenBalancedDistributedBatchSampler


class SSFlowDataModule(pl.LightningDataModule):
    def __init__(self, cfg: edict):
        super().__init__()
        self.cfg = cfg
        self.dataset_name = cfg.dataset.name
        self.dataset_args = dict(cfg.dataset.args)
        self.batch_size = int(cfg.trainer.args.batch_size_per_gpu)
        self.num_workers = int(cfg.lightning.get("num_workers", 0))
        # Token-balanced batching (sparse SLat/PBR datasets): equalize
        # per-rank token loads each step so DDP ranks don't straggle.
        # Requires Trainer(use_distributed_sampler=False) — train_lightning.py
        # reads the same flag.
        self.token_balanced = bool(cfg.lightning.get("token_balanced_batches", False))
        # Data roots as plain dicts -> JSON strings (dataset parses JSON).
        self.train_roots = json.dumps(dict(cfg.data.train))
        self.val_roots = json.dumps(dict(cfg.data.get("val", cfg.data.train)))
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage: Optional[str] = None) -> None:
        dataset_cls = getattr(datasets, self.dataset_name)
        if self.train_ds is None:
            self.train_ds = dataset_cls(self.train_roots, **self.dataset_args)
        if self.val_ds is None:
            self.val_ds = dataset_cls(self.val_roots, **self.dataset_args)

    def train_dataloader(self) -> DataLoader:
        if self.token_balanced:
            loads = getattr(self.train_ds, "latent_loads", None)
            assert loads is not None, (
                "token_balanced_batches requires a dataset exposing latent_loads "
                f"({type(self.train_ds).__name__} does not)"
            )
            assert getattr(self.train_ds, "sample_weights", None) is None, (
                "token_balanced_batches is mutually exclusive with weighted sampling"
            )
            return DataLoader(
                self.train_ds,
                batch_sampler=TokenBalancedDistributedBatchSampler(loads, self.batch_size),
                num_workers=self.num_workers,
                collate_fn=self.train_ds.collate_fn,
                persistent_workers=self.num_workers > 0,
            )
        # balance_roots: weighted sampling with replacement (small roots repeat)
        sampler = None
        if getattr(self.train_ds, "sample_weights", None) is not None:
            sampler = WeightedRandomSampler(
                self.train_ds.sample_weights, num_samples=len(self.train_ds), replacement=True
            )
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            shuffle=sampler is None,
            sampler=sampler,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.train_ds.collate_fn,  # split_size defaults to None -> single dict
            persistent_workers=self.num_workers > 0,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_ds,
            batch_size=self.batch_size,
            shuffle=False,
            drop_last=False,
            num_workers=self.num_workers,
            collate_fn=self.val_ds.collate_fn,
            persistent_workers=self.num_workers > 0,
        )

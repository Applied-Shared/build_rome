"""Lightning module for SLat (shape) flow fine-tuning.

Subclass of ``SSFlowLitModule`` with the sparse flow-matching engine
(``x_0`` is a batched ``SparseTensor``: coords fixed, feats denoised).
Validation renders decode GT and predicted latents with the frozen shape
decoder and spiral-render normal maps (input | GT | prediction); coords are
ground truth at validation, so the renders show surface quality.
"""
from __future__ import annotations

import os

import torch
from easydict import EasyDict as edict

from build_rome import models
from build_rome.lightning.render import (
    compose_side_by_side,
    image_panel,
    merge_chunk_meshes,
    render_mesh_spiral,
    write_video_file,
)
from build_rome.lightning.ss_flow_module import SSFlowLitModule
from build_rome.trainers.flow_matching.sparse_flow_matching import CameraConditionedSparseFlowMatchingCFGTrainer


class SLatFlowLitModule(SSFlowLitModule):
    ENGINE_CLS = CameraConditionedSparseFlowMatchingCFGTrainer

    def __init__(self, cfg: edict, init_from_pretrained: bool = True):
        super().__init__(cfg, init_from_pretrained)
        dargs = cfg.dataset.args
        self.pretrained_slat_dec = dargs.get(
            "pretrained_slat_dec", "microsoft/TRELLIS.2-4B/ckpts/shape_dec_next_dc_f16c32_fp16")
        self.slat_resolution = int(dargs.get("resolution", 1024))
        norm = dargs.get("normalization", None)
        self.slat_mean = torch.tensor(norm["mean"]).reshape(1, -1) if norm else None
        self.slat_std = torch.tensor(norm["std"]).reshape(1, -1) if norm else None
        self.slat_dec = None

    @staticmethod
    def _make_noise(x_0):
        return x_0.replace(torch.randn_like(x_0.feats))

    def _loading_slat_dec(self):
        if self.slat_dec is None:
            with torch.inference_mode(False):  # same story as _loading_ss_dec
                dec = models.from_pretrained(self.pretrained_slat_dec)
                dec.set_resolution(self.slat_resolution)
                dec = dec.to(self.device).eval()
            object.__setattr__(self, "slat_dec", dec)

    def _decode_slat_batch(self, z):
        """Denormalize a batched sparse latent and decode to per-sample meshes."""
        if self.slat_mean is not None:
            z = z * self.slat_std.to(z.device) + self.slat_mean.to(z.device)
        return [self.slat_dec(z[i : i + 1])[0] for i in range(z.shape[0])]  # one at a time: meshes are large

    @torch.no_grad()
    def _render_scene_shard(self) -> None:
        ds, shard = self._render_picks()
        if not shard:
            return
        self._loading_slat_dec()
        extrinsics, intrinsics = self._spiral_cameras()
        os.makedirs(self._render_dir(), exist_ok=True)
        group_size = int(self.targs.batch_size_per_gpu)
        for i_scene, (_, indices) in shard:
            samples = [ds[i] for i in indices]
            transforms = [ds.chunk_to_world(i) for i in indices]
            gt_meshes, pred_meshes = [], []
            for g0 in range(0, len(samples), group_size):
                batch = ds.collate_fn(samples[g0 : g0 + group_size])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred_latent = self._sample(batch)
                with torch.autocast(device_type="cuda", enabled=False):
                    gt_meshes += self._decode_slat_batch(batch["x_0"].float())
                    pred_meshes += self._decode_slat_batch(pred_latent.float())
            with torch.autocast(device_type="cuda", enabled=False):
                frames = [image_panel(samples[0]["cond"][0], self.render_num_frames, self.render_resolution)]
                for meshes in (gt_meshes, pred_meshes):
                    scene = merge_chunk_meshes(meshes, transforms, self.render_cutaway_z_m)
                    frames.append(render_mesh_spiral(scene, extrinsics, intrinsics, self.render_resolution,
                                                     near_clip_frac=self.render_near_clip_frac))
                path = os.path.join(self._render_dir(), f"val_{i_scene:03d}_step{self.global_step:06d}."
                                                        f"{self.render_video_format}")
                write_video_file(compose_side_by_side(*frames), path, fps=self.render_fps)
            del gt_meshes, pred_meshes
            torch.cuda.empty_cache()

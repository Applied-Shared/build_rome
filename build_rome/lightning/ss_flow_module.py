"""PyTorch Lightning module for SS flow fine-tuning.

The flow-matching objective, DINO conditioning and classifier-free guidance
come from ``CameraConditionedFlowMatchingCFGTrainer``; this module only builds
the denoiser (LoRA + conditioning branches), delegates the loss to a bare
instance of that trainer (its heavy ``BasicTrainer.__init__`` is bypassed),
and adds optimizer groups and per-scene validation renders.
"""
from __future__ import annotations

import glob
import os
from typing import Any, Dict

import lightning.pytorch as pl
import torch
from easydict import EasyDict as edict
from lightning.pytorch.utilities import rank_zero_only

from build_rome import models
from build_rome.lightning.render import (
    build_spiral_cameras,
    compose_side_by_side,
    decode_latent_to_voxels,
    image_panel,
    merge_chunk_voxels,
    render_voxel_spiral,
    write_video_file,
)
from build_rome.trainers.basic import BasicTrainer
from build_rome.trainers.flow_matching.flow_matching import CameraConditionedFlowMatchingCFGTrainer
from build_rome.trainers.utils import LinearWarmupLRScheduler, build_optimizer_param_groups
from build_rome.utils.model_wrapper_utils import build_single_model

# Conditioning-branch parameters: absent from the pretrained base, trained from
# their zero init, and kept out of weight decay (decay would pull the zero-init
# injection linears back toward zero).
_BRANCH_PREFIXES = ("depth_lift_proj_linears.", "depth_geom_")


class SSFlowLitModule(pl.LightningModule):
    """SS flow fine-tuning.

    Args:
        cfg: full experiment config (edict) with ``models`` / ``trainer`` /
            ``dataset`` / ``lightning`` blocks.
        init_from_pretrained: load ``trainer.args.finetune_ckpt.denoiser``
            before training.
    """

    # Loss engine class; the sparse (SLat) subclass swaps this.
    ENGINE_CLS = CameraConditionedFlowMatchingCFGTrainer

    def __init__(self, cfg: edict, init_from_pretrained: bool = True):
        super().__init__()
        self.cfg = cfg
        self.targs = cfg.trainer.args

        # LoRA freezes the base; the conditioning branches are new, trainable
        # parameters created in the constructor, so re-enable them.
        self.denoiser = build_single_model(edict(cfg.models.denoiser))
        for name, p in self.denoiser.named_parameters():
            if name.startswith(_BRANCH_PREFIXES):
                p.requires_grad_(True)
        if init_from_pretrained:
            self._load_pretrained()
        self.engine = self._build_loss_engine()

        vr = cfg.lightning.get("validation_render", edict({})) if "lightning" in cfg else edict({})
        self.render_enable = bool(vr.get("enable", True))
        self.render_num_frames = int(vr.get("num_frames", 60))
        self.render_resolution = int(vr.get("resolution", 384))
        self.render_r = float(vr.get("r", 2.0))
        self.render_fov = float(vr.get("fov", 40.0))
        self.render_base_elev_deg = float(vr.get("base_elevation_deg", 20.0))
        self.render_elev_amp_deg = float(vr.get("elevation_amp_deg", 30.0))
        self.render_elev_cycles = float(vr.get("elevation_cycles", 1.0))
        self.render_steps = int(vr.get("sample_steps", 50))
        self.render_guidance = float(vr.get("guidance_strength", 3.0))
        self.render_max_scenes = vr.get("max_scenes", None)
        if self.render_max_scenes is not None:
            self.render_max_scenes = int(self.render_max_scenes)
        # drop merged-scene geometry above this height (m) so the orbit sees in
        self.render_cutaway_z_m = float(vr.get("cutaway_z_m", 2.2))
        # per-frame camera-facing clip (unit-cube fraction; null disables)
        self.render_near_clip_frac = vr.get("near_clip_frac", 0.15)
        if self.render_near_clip_frac is not None:
            self.render_near_clip_frac = float(self.render_near_clip_frac)
        self.render_video_format = str(vr.get("video_format", "gif"))
        self.render_fps = int(vr.get("fps", 15))
        self.pretrained_ss_dec = cfg.dataset.args.get(
            "pretrained_ss_dec", "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16")
        self.ss_dec = None
        self._scene_map_printed = False

        self.save_hyperparameters({"cfg": dict(cfg), "init_from_pretrained": init_from_pretrained})

    # ------------------------------------------------------------------ setup
    def _load_pretrained(self) -> None:
        finetune_ckpt = self.targs.get("finetune_ckpt", None)
        if not finetune_ckpt or "denoiser" not in finetune_ckpt:
            return
        path = str(finetune_ckpt["denoiser"])
        if path.endswith(".ckpt"):
            # Lightning checkpoint from a previous run: strip the module prefix
            full = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
            state_dict = {k[len("denoiser."):]: v for k, v in full.items() if k.startswith("denoiser.")}
        else:
            # local .pt or HF safetensors, via BasicTrainer's loader (bound to a
            # shim carrying the two attributes it reads)
            loader_ctx = edict({"is_master": rank_zero_only.rank == 0, "device": "cpu"})
            state_dict = BasicTrainer.load_finetune_model_ckpt(loader_ctx, path, "denoiser")

        model_state = self.denoiser.state_dict()
        filtered = {k: v for k, v in state_dict.items() if k in model_state and v.shape == model_state[k].shape}
        self.denoiser.load_state_dict(filtered, strict=False)
        # strict=False may only tolerate the parameters we added (LoRA, branches)
        unrestored = [n for n, _ in self.denoiser.named_parameters()
                      if n not in filtered and "lora_" not in n and not n.startswith(_BRANCH_PREFIXES)]
        assert not unrestored, f"pretrained load left base parameters unrestored: {unrestored[:10]}"
        rank_zero_only(print)(f"[{type(self).__name__}] loaded pretrained denoiser: {len(filtered)} tensors "
                              f"restored, {len(state_dict) - len(filtered)} skipped")

    def _build_loss_engine(self):
        engine = object.__new__(type(self).ENGINE_CLS)
        engine.training_models = {"denoiser": self.denoiser}
        engine.models = {"denoiser": self.denoiser}
        engine.t_schedule = dict(self.targs.t_schedule)
        engine.sigma_min = float(self.targs.sigma_min)
        engine.p_uncond = float(self.targs.get("p_uncond", 0.0))  # CFG dropout
        engine.image_cond_model = None  # DINO, instantiated lazily on first use
        engine.image_cond_model_config = dict(self.targs.image_cond_model)
        engine.is_master = True
        engine.world_size = 1
        return engine

    # -------------------------------------------------------------- fwd/steps
    def _losses(self, batch: Dict[str, Any], training: bool):
        fn = self.engine.training_losses if training else self.engine.validation_losses
        terms, _ = fn(**batch)
        return terms

    def training_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        terms = self._losses(batch, training=True)
        bs = batch["x_0"].shape[0]
        self.log("train/loss", terms["loss"], prog_bar=True, on_step=True, on_epoch=False, batch_size=bs)
        self.log("train/mse", terms["mse"], on_step=True, on_epoch=False, batch_size=bs)
        return terms["loss"]

    def validation_step(self, batch: Dict[str, Any], batch_idx: int) -> torch.Tensor:
        prev_p = self.engine.p_uncond
        self.engine.p_uncond = 0.0  # no CFG dropout while measuring
        try:
            terms = self._losses(batch, training=False)
        finally:
            self.engine.p_uncond = prev_p
        bs = batch["x_0"].shape[0]
        self.log("val/loss", terms["loss"], prog_bar=True, on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        self.log("val/mse", terms["mse"], on_step=False, on_epoch=True, batch_size=bs, sync_dist=True)
        return terms["loss"]

    def on_validation_epoch_end(self) -> None:
        # One video per val scene (input | GT | prediction). Under DDP each rank
        # renders a round-robin shard; the wandb run only exists on rank 0.
        if not (self.render_enable and self._is_wandb_logger()):
            return
        try:
            self._render_scene_shard()
        except Exception as e:  # rendering must never crash training
            print(f"[{type(self).__name__}] render skipped on rank {self.trainer.global_rank}: "
                  f"{type(e).__name__}: {e}")
        self.trainer.strategy.barrier()  # every rank must reach it, even after a failure
        if self.trainer.is_global_zero:
            try:
                self._upload_scene_videos()
            except Exception as e:
                print(f"[{type(self).__name__}] video upload skipped: {type(e).__name__}: {e}")

    # ----------------------------------------------------------- val rendering
    def _render_dir(self) -> str:
        return os.path.join(self.trainer.default_root_dir or ".", "spiral_renders")

    def _render_picks(self):
        ds = self.trainer.datamodule.val_ds
        picks = ds.first_view_chunks_per_scene()  # [(scene, [chunk indices]), ...]
        if self.render_max_scenes is not None:
            picks = picks[: self.render_max_scenes]
        if self.trainer.is_global_zero and not self._scene_map_printed:
            for i, (scene, idxs) in enumerate(picks):
                print(f"[{type(self).__name__}] val render {i:03d} -> {scene} ({len(idxs)} chunks)")
            self._scene_map_printed = True
        return ds, list(enumerate(picks))[self.trainer.global_rank :: self.trainer.world_size]

    def _spiral_cameras(self):
        return build_spiral_cameras(
            num_frames=self.render_num_frames, r=self.render_r, fov=self.render_fov,
            base_elevation_deg=self.render_base_elev_deg, elevation_amp_deg=self.render_elev_amp_deg,
            elevation_cycles=self.render_elev_cycles)

    @torch.no_grad()
    def _render_scene_shard(self) -> None:
        """Per scene: predict every chunk of the first view, place them with
        each sidecar's chunk_to_world, and spiral-render input | GT | prediction."""
        ds, shard = self._render_picks()
        if not shard:
            return
        self._loading_ss_dec()
        extrinsics, intrinsics = self._spiral_cameras()
        os.makedirs(self._render_dir(), exist_ok=True)
        group_size = int(self.targs.batch_size_per_gpu)
        for i_scene, (_, indices) in shard:
            samples = [ds[i] for i in indices]
            transforms = [ds.chunk_to_world(i) for i in indices]
            gt_voxels, pred_voxels = [], []
            for g0 in range(0, len(samples), group_size):
                batch = ds.collate_fn(samples[g0 : g0 + group_size])
                batch = {k: v.to(self.device) for k, v in batch.items()}
                pred_latent = self._sample(batch)
                with torch.autocast(device_type="cuda", enabled=False):  # decoder/renderer need float32
                    gt_voxels += decode_latent_to_voxels(self.ss_dec, batch["x_0"].float())
                    pred_voxels += decode_latent_to_voxels(self.ss_dec, pred_latent.float())
            with torch.autocast(device_type="cuda", enabled=False):
                frames = [image_panel(samples[0]["cond"][0], self.render_num_frames, self.render_resolution)]
                for voxels in (gt_voxels, pred_voxels):
                    scene = merge_chunk_voxels(voxels, transforms, self.render_cutaway_z_m)
                    frames.append(render_voxel_spiral(scene, extrinsics, intrinsics, self.render_resolution,
                                                      near_clip_frac=self.render_near_clip_frac))
                path = os.path.join(self._render_dir(), f"val_{i_scene:03d}_step{self.global_step:06d}."
                                                        f"{self.render_video_format}")
                write_video_file(compose_side_by_side(*frames), path, fps=self.render_fps)

    def _upload_scene_videos(self) -> None:
        """Rank 0: log every video all ranks wrote for this step."""
        pattern = os.path.join(self._render_dir(), f"val_*_step{self.global_step:06d}.{self.render_video_format}")
        for path in sorted(glob.glob(pattern)):
            i_scene = os.path.basename(path).split("_")[1]
            self.logger.log_video(key=f"val/{i_scene}", videos=[path], step=self.global_step)

    def _is_wandb_logger(self) -> bool:
        from lightning.pytorch.loggers import WandbLogger

        return isinstance(self.logger, WandbLogger)

    def _loading_ss_dec(self):
        if self.ss_dec is None:
            # loaded outside Lightning's inference_mode and kept out of the
            # module tree (frozen: DDP must not broadcast it, checkpoints must
            # not embed it)
            with torch.inference_mode(False):
                dec = models.from_pretrained(self.pretrained_ss_dec).to(self.device).eval()
            object.__setattr__(self, "ss_dec", dec)

    @staticmethod
    def _make_noise(x_0):
        """Noise like x_0 (the sparse subclass overrides for SparseTensor)."""
        return torch.randn_like(x_0)

    @torch.no_grad()
    def _sample(self, batch: Dict[str, Any]):
        """Sample latents for a batch's conditioning (bf16, like training)."""
        data = {k: v for k, v in batch.items() if k != "x_0"}
        data.setdefault("extrinsics", None)
        data.setdefault("intrinsics", None)
        denoiser = self.engine.training_models["denoiser"]
        if hasattr(denoiser, "module"):
            denoiser = denoiser.module
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            args = self.engine.get_inference_cond(denoiser, x_0=batch["x_0"], **data)
            return self.engine.get_sampler().sample(
                denoiser, noise=self._make_noise(batch["x_0"]), **args, steps=self.render_steps,
                guidance_strength=self.render_guidance, verbose=False).samples

    # ------------------------------------------------------------- optimizer
    def configure_optimizers(self):
        opt_cfg = self.targs.optimizer
        named = list(self.denoiser.named_parameters())
        branch_params = [p for n, p in named if n.startswith(_BRANCH_PREFIXES) and p.requires_grad]
        rest = [(n, p) for n, p in named if not n.startswith(_BRANCH_PREFIXES)]
        # LoRA / new parameters fall into the lora / new groups (config LRs)
        param_groups, stats = build_optimizer_param_groups(
            rest, lr=opt_cfg.args.lr, weight_decay=opt_cfg.args.weight_decay, pretrained_param_names=None)
        if branch_params:
            lr_cfg = opt_cfg.args.lr
            embed_lr = float(lr_cfg.get("embed", lr_cfg["new"])) if isinstance(lr_cfg, dict) else float(lr_cfg)
            param_groups.append({"name": "branches", "params": branch_params, "lr": embed_lr, "weight_decay": 0.0})
            stats["branches"] = sum(p.numel() for p in branch_params)
            stats["total"] += stats["branches"]
        rank_zero_only(print)(f"[{type(self).__name__}] optimizer param-group sizes: {stats}")

        extra = {}
        if "betas" in opt_cfg.args:
            extra["betas"] = tuple(opt_cfg.args.betas)
        if "eps" in opt_cfg.args:
            extra["eps"] = float(opt_cfg.args.eps)
        optimizer = getattr(torch.optim, opt_cfg.get("name", "AdamW"))(param_groups, **extra)
        sched_cfg = self.targs.get("lr_scheduler", None)
        if sched_cfg is None:
            return optimizer
        scheduler = LinearWarmupLRScheduler(optimizer, warmup_steps=int(sched_cfg.args.warmup_steps))
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "interval": "step", "frequency": 1}}

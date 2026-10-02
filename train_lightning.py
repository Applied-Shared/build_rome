"""Fine-tune the SS or SLat flow model (PyTorch Lightning).

Usage:
    # one node, 8 GPUs
    torchrun --nproc_per_node 8 train_lightning.py --config configs/ss.yaml data_root=/path/to/data

    # selected GPUs, one process per GPU spawned by Lightning
    python train_lightning.py --config configs/slat.yaml --gpu 0,2 data_root=/path/to/data

Extra ``key.path=value`` arguments are OmegaConf dotlist overrides applied on
top of the YAML, e.g. ``lightning.wandb.enable=true trainer.args.batch_size_per_gpu=2``.
Validation cadence is ``lightning.val_check_interval`` (optimizer steps).
"""

from __future__ import annotations

import argparse
import os

from easydict import EasyDict as edict
from omegaconf import OmegaConf


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Fine-tune the SS / SLat flow model.",
        epilog="Extra key.path=value arguments are OmegaConf dotlist overrides for the YAML config.",
    )
    p.add_argument("--config", required=True, help="Path to YAML config (configs/ss.yaml or configs/slat.yaml).")
    p.add_argument(
        "--gpu",
        default="0",
        help="GPU(s) to use, e.g. '0' or '0,2' (sets CUDA_VISIBLE_DEVICES; >1 GPU launches DDP). "
        "Ignored if CUDA_VISIBLE_DEVICES is already set (e.g. when launching via torchrun).",
    )
    p.add_argument("--output_dir", default=None, help="Override lightning.output_dir.")
    p.add_argument("--max_steps", type=int, default=None, help="Override lightning.max_steps.")
    p.add_argument("--val_check_interval", type=int, default=None, help="Override lightning.val_check_interval.")
    p.add_argument("--precision", default=None, help="Override lightning.precision (e.g. 32, bf16-mixed).")
    p.add_argument("--no_pretrained", action="store_true", help="Skip loading the finetune checkpoint.")
    p.add_argument(
        "--eval_before_train",
        action="store_true",
        help="Run a single validation pass (incl. spiral render) before training, then exit.",
    )
    wandb_group = p.add_mutually_exclusive_group()
    wandb_group.add_argument("--wandb", dest="wandb", action="store_true", help="Force-enable wandb.")
    wandb_group.add_argument("--no_wandb", dest="wandb", action="store_false", help="Disable wandb.")
    p.set_defaults(wandb=None)
    args, overrides = p.parse_known_args()
    bad = [o for o in overrides if "=" not in o or o.startswith("-")]
    if bad:
        p.error(f"unrecognized arguments: {' '.join(bad)} (config overrides must be key.path=value)")
    args.overrides = overrides
    return args


def main() -> None:
    args = parse_args()

    # Pin GPUs before importing torch/lightning. If CUDA_VISIBLE_DEVICES is
    # already set (user export or torchrun), it wins over --gpu. Lightning then
    # uses all visible GPUs (devices=-1) and launches DDP when there are >1.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    if not os.environ.get("CUDA_VISIBLE_DEVICES"):
        os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    n_gpus = len(os.environ["CUDA_VISIBLE_DEVICES"].split(","))

    import lightning.pytorch as pl
    from lightning.pytorch.callbacks import TQDMProgressBar
    from lightning.pytorch.callbacks.progress.tqdm_progress import convert_inf

    # Non-zero DDP ranks: mute the transformers "Loading weights" progress bar
    # (DINOv3 load) so startup output isn't repeated once per GPU.
    if int(os.environ.get("RANK", os.environ.get("LOCAL_RANK", "0"))) != 0:
        import transformers

        transformers.utils.logging.disable_progress_bar()

    from build_rome.lightning import (
        SLatFlowLitModule,
        SSFlowDataModule,
        SSFlowLitModule,
    )

    class StepProgressBar(TQDMProgressBar):
        """Show a single continuous ``global_step / max_steps`` training bar.

        With a tiny dataset each epoch is only a step or two, so the
        default epoch-based bar shows a confusing ``Epoch N`` with ``0/1``.
        This drives the bar by the global optimizer step instead.
        """

        def _step_total(self, trainer):
            if trainer.max_steps and trainer.max_steps > 0:
                return trainer.max_steps
            return convert_inf(self.total_train_batches)

        def on_train_epoch_start(self, trainer, *_):
            # Do NOT reset per epoch: keep one continuous step-based bar.
            if self.train_progress_bar is None:
                self.train_progress_bar = self.init_train_tqdm()
            self.train_progress_bar.total = self._step_total(trainer)
            self.train_progress_bar.set_description("Training")

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            n = trainer.global_step
            bar = self.train_progress_bar
            if bar is not None and self._should_update(n, bar.total):
                bar.n = n
                bar.refresh()
                bar.set_postfix(self.get_metrics(trainer, pl_module))

    conf = OmegaConf.load(args.config)
    if args.overrides:
        conf = OmegaConf.merge(conf, OmegaConf.from_dotlist(args.overrides))
        print(f"[train_lightning] config overrides: {args.overrides}")
    # The rest of the stack (module/datamodule) expects edict/plain containers.
    cfg = edict(OmegaConf.to_container(conf, resolve=True))

    lcfg = cfg.lightning
    if args.output_dir is not None:
        lcfg.output_dir = args.output_dir
    if args.max_steps is not None:
        lcfg.max_steps = args.max_steps
    if args.val_check_interval is not None:
        lcfg.val_check_interval = args.val_check_interval
    if args.precision is not None:
        lcfg.precision = args.precision
    if args.wandb is not None:
        lcfg.wandb.enable = args.wandb

    os.makedirs(lcfg.output_dir, exist_ok=True)

    init_from_pretrained = bool(lcfg.get("init_from_pretrained", True)) and not args.no_pretrained

    # Build data + model. Sparse trainers (SLat: x_0 is a SparseTensor) get the
    # sparse Lightning module; dense (SS) trainers keep the original one.
    dm = SSFlowDataModule(cfg)
    if "Sparse" in cfg.trainer.name:
        module_cls = SLatFlowLitModule
    else:
        module_cls = SSFlowLitModule
    module = module_cls(cfg, init_from_pretrained=init_from_pretrained)

    # Logger (wandb optional).
    logger = True
    wandb_cfg = lcfg.get("wandb", edict({"enable": False}))
    if wandb_cfg.get("enable", False):
        from lightning.pytorch.loggers import WandbLogger

        logger = WandbLogger(
            project=wandb_cfg.get("project", "image-to-scene"),
            name=wandb_cfg.get("name", None),
            save_dir=lcfg.output_dir,
            mode=wandb_cfg.get("mode", "online"),
            # Continue an EXISTING wandb run (same curves) across relaunches:
            # lightning.wandb.id=<run id> lightning.wandb.resume=allow.
            # Pair with lightning.resume_ckpt so global_step continues
            # monotonically — wandb drops backwards-stepping points.
            id=wandb_cfg.get("id", None),
            resume=wandb_cfg.get("resume", None),
        )

    callbacks = [StepProgressBar()]
    if wandb_cfg.get("enable", False):
        from lightning.pytorch.callbacks import LearningRateMonitor

        callbacks.append(LearningRateMonitor(logging_interval="step"))

    # Lightning semantics: int = number of batches, float = fraction of the set.
    lvb = lcfg.get("limit_val_batches", 1)
    lvb = float(lvb) if isinstance(lvb, float) else int(lvb)

    # Multi-GPU: devices=-1 uses every GPU left visible by CUDA_VISIBLE_DEVICES
    # and Lightning spawns one DDP process per device. Under torchrun, devices
    # must instead match the number of processes torchrun already started per
    # node (LOCAL_WORLD_SIZE), one GPU each. Lightning injects a
    # DistributedSampler (wrapping custom samplers too), so each rank sees a
    # disjoint shard; effective batch scales with the number of GPUs.
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", "0"))
    devices = local_world_size if local_world_size > 0 else -1
    n_procs = local_world_size if local_world_size > 0 else n_gpus

    # Multi-node (torchrun sets GROUP_WORLD_SIZE = number of nodes): Lightning
    # must be told num_nodes so devices * num_nodes matches WORLD_SIZE.
    num_nodes = int(os.environ.get("GROUP_WORLD_SIZE", "1"))

    # Token-balanced batching supplies its own cross-rank disjoint batch
    # sampler (TokenBalancedDistributedBatchSampler); Lightning must not
    # re-wrap it with a DistributedSampler. NB: the val loader is then also
    # unsharded (every rank sees the same val batches) — with the default
    # limit_val_batches=1 that duplication is negligible.
    use_dist_sampler = not bool(lcfg.get("token_balanced_batches", False))

    # 2h collective timeout (default 30 min): slow data loading on one rank
    # should stall the others, not abort the run.
    if n_gpus > 1 or num_nodes > 1:
        from datetime import timedelta

        from lightning.pytorch.strategies import DDPStrategy

        strategy = DDPStrategy(timeout=timedelta(hours=2))
    else:
        strategy = "auto"

    trainer = pl.Trainer(
        max_steps=int(lcfg.max_steps),
        val_check_interval=int(lcfg.val_check_interval),
        check_val_every_n_epoch=None,
        limit_val_batches=lvb,
        accumulate_grad_batches=int(lcfg.get("accumulate_grad_batches", 1)),
        precision=lcfg.get("precision", "bf16-mixed"),
        accelerator="gpu",
        devices=-1,
        num_nodes=num_nodes,
        strategy=strategy,
        logger=logger,
        callbacks=callbacks,
        default_root_dir=lcfg.output_dir,
        log_every_n_steps=int(lcfg.get("log_every_n_steps", 1)),
        num_sanity_val_steps=0,
        use_distributed_sampler=use_dist_sampler,
    )

    if args.eval_before_train:
        # Baseline eval only: run validation (val/loss + spiral render), then exit.
        trainer.validate(module, datamodule=dm)
        return

    # Full Lightning resume (weights + optimizer + global_step): continues the
    # step counter, which keeps wandb-run resumption monotonic and preserves
    # Adam state when extending a run on grown data.
    if lcfg.get("resume_ckpt", None):
        # torch >= 2.6 defaults torch.load to weights_only=True; Lightning's
        # ckpt_path restore then rejects the EasyDict cfg pickled into our own
        # checkpoints' hparams. The checkpoint is this pipeline's artifact, so
        # full unpickling is safe here.
        os.environ.setdefault("TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD", "1")
    trainer.fit(module, datamodule=dm, ckpt_path=lcfg.get("resume_ckpt", None))


if __name__ == "__main__":
    main()

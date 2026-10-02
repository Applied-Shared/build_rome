# Single image to 3D scene

Generate a metric 3D scene mesh from one photograph. A monocular depth pack
(MoGe-3) grounds a TRELLIS.2 model fine-tuned to generate 3 m scene windows.
The scene is covered with windows whose size grows with distance. Windows are
generated one at a time with latent outpainting, and each depth band is
stitched to the next.

## Install

The code needs a CUDA GPU (flash-attention and the sparse kernels are CUDA-only).

```bash
. ./setup.sh --new-env --basic --flash-attn --nvdiffrast --cumesh --flexgemm --o-voxel
```

`o-voxel` is vendored in `o-voxel/` and installed from there.

MoGe-3 may need dependency versions that conflict with the environment above.
In that case, install it in a separate environment
(`pip install git+https://github.com/microsoft/MoGe.git`) and pass that
environment's python with `--moge3-python`.

## Weights

The base weights are the public TRELLIS.2 checkpoints, downloaded from Hugging
Face on first use. The fine-tuned weights are small deltas: the LoRA adapters
and the conditioning branches (SS 115M parameters, SLat 66M).

| Model | File |
|---|---|
| SS (occupancy) | `ss_delta.safetensors` |
| SLat (shape) | `slat_delta.safetensors` |

`inference/export_weights.py` writes these deltas from a training checkpoint.
Both inference and training also accept a full Lightning `.ckpt`.

## Inference

```bash
python inference/image_to_scene.py --image assets/test_images/tourist6.png \
    --ss-ckpt ss_delta.safetensors --slat-ckpt slat_delta.safetensors --out results/
```

`assets/test_images/` holds the 73 input photos used for the paper and
website results (indoor, outdoor and large-scale scenes); any photo works.

The run writes `results/<stem>/` with:

- `scene.glb`: the metric mesh, glTF Y-up, camera at the origin looking along -Z
- `layout_bands.json` and `layout_topview.png`: the depth bands and their windows
- `spiral.mp4`: an orbit render next to the input (disable with `--no-spiral-video`)

The MoGe-3 depth pack is cached by image content under
`~/.cache/image_to_scene/moge3`. To use your own depth, pass `--depth-pack`
(format in `build_rome/modules/lifting.py`).

Useful options:

| Flag | Default | Meaning |
|---|---|---|
| `--band-near-m` | 9 | native 3 m windows up to this depth |
| `--band-growth` | 2 | window-size multiplier per further band |
| `--band-min-coverage` | 0.05 | drop far bands covering less of the image than this |
| `--band-count`, `--band-min-depth-m` | – | force the number of bands or their reach |
| `--no-cross-band`, `--band-overlap-frac 0` | – | generate bands independently, abutting |
| `--max-inflated-voxels` | 100000 | lower it if the shape decode runs out of memory |

Cost example: on one A100 80GB, a street scene reaching 98 m (4 bands,
15 windows) peaks at 15.8 GiB allocated and takes 5.8 min.

## Training

Each dataset root has this layout:

```
<root>/scenes/<scene>/view_<vvv>/images/000.jpg              view image
<root>/scenes/<scene>/view_<vvv>/images/000_gt.npz           depth pack (lifting.py format)
<root>/scenes/<scene>/view_<vvv>/chunks/w<kk>/meta.json      window sidecar (build_rome/utils/window_conventions*.py)
<root>/scenes/<scene>/view_<vvv>/chunks/w<kk>/ss.npz         SS latent of the window (non-empty windows)
<root>/scenes/<scene>/view_<vvv>/chunks/w<kk>/slat_1024.npz  SLat latent of the window (non-empty windows)
```

A split file lists the validation scenes as `{"val_scenes": [...]}`. The two
stages train independently:

```bash
torchrun --nproc_per_node 8 train_lightning.py --config configs/ss.yaml   data_root=/path/to/data
torchrun --nproc_per_node 8 train_lightning.py --config configs/slat.yaml data_root=/path/to/data
```

- **Overrides:** any `key.path=value` argument overrides the config, e.g.
  `lightning.wandb.enable=true trainer.args.batch_size_per_gpu=2`.
- **Initialization:** training starts from the TRELLIS.2 base weights
  (`trainer.args.finetune_ckpt`).
- **Validation:** renders input | ground truth | prediction per scene when
  wandb is enabled.

## Licenses

This code builds on TRELLIS.2, MoGe-3 and DINOv3. Check their licenses before
use; they also apply to the downloaded weights.

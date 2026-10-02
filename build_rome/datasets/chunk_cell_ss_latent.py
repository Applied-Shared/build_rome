"""Camera-anchored window dataset for SS flow fine-tuning.

Walks the on-disk layout

    <base>/scenes/<scene>/view_<vvv>/images/000.jpg       conditioning image
    <base>/scenes/<scene>/view_<vvv>/images/000_gt.npz    depth pack of the view
    <base>/scenes/<scene>/view_<vvv>/chunks/w<kk>/meta.json   window sidecar
    <base>/scenes/<scene>/view_<vvv>/chunks/w<kk>/ss.npz      SS latent (non-empty windows)

Each sample is one window:

* ``x_0``: the window's SS latent [8, 16, 16, 16]; empty windows
  (``occupancy_class == "empty"``) use the shared ``z_empty`` latent, teaching
  the model to output nothing where the image implies free space;
* ``cond``: the view image [1, 3, S, S] in [0, 1]; ``cond_hi``: the same view
  at ``lift_image_size`` (encoded into the finer grid the depth lift uses);
* ``depth_m`` / ``lateral_m`` / ``z_off_m``: window center and base height in
  the view's floor frame (they reach the denoiser through the trainer's
  kwargs path, never the CFG-dropped cond);
* ``lift_depth`` / ``lift_valid`` / ``lift_K`` / ``lift_c2y``: the view's depth
  pack for the depth-lift and clearance branches.

Every sidecar is validated against the window conventions version it declares
(reject, never remap). A root may carry ``split_file`` (JSON with a
``val_scenes`` list) and ``split`` ("train" keeps scenes NOT in it, "val" only
those), and ``repeat`` to tile a tiny root.
"""
import json
import os
from collections import Counter
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from lightning.fabric.utilities.rank_zero import rank_zero_only
from PIL import Image
from torch.utils.data import Dataset

from build_rome.utils import window_conventions as _wc3
from build_rome.utils import window_conventions_v4 as _wc4
from build_rome.utils import window_conventions_v5 as _wc5

# encoder that produced the ss.npz latents (and must have produced z_empty)
EXPECTED_SS_ENCODER = "microsoft/TRELLIS-image-large/ckpts/ss_enc_conv3d_16l8_fp16"


def validate_sidecar(meta: dict) -> None:
    """Validate against the conventions version the sidecar declares."""
    ver = int(meta.get("conventions", {}).get("version", 3))
    {3: _wc3, 4: _wc4, 5: _wc5}[ver].validate_sidecar(meta)


class ChunkCellSSLatent(Dataset):
    """
    Args:
        roots: JSON string ``{"NAME": {"base": <dir containing scenes/>, ...}}``.
        image_size: side length of the conditioning image.
        lift_image_size: side length of ``cond_hi`` and of the lift tensors
            (depth packs of mixed resolution are resized to it); 0 = native.
        depth_pack_suffix: depth pack next to each view image.
        balance_roots: draw every root equally often instead of in proportion
            to its size (weighted sampling with replacement).
        pretrained_ss_dec: SS decoder used by validation renders.
        empty_latent_path: npz with the empty-window latent ("z", "encoder").
    """

    LATENT_FILENAME = "ss.npz"
    INCLUDE_EMPTY = True  # empty windows are SS samples (z_empty); SLat excludes them

    def __init__(
        self,
        roots: str,
        *,
        image_size: int = 512,
        lift_image_size: int = 1024,
        depth_pack_suffix: str = "_gt.npz",
        balance_roots: bool = False,
        pretrained_ss_dec: str = "microsoft/TRELLIS-image-large/ckpts/ss_dec_conv3d_16l8_fp16",
        empty_latent_path: str = "ss_empty_latent.npz",
    ):
        self.image_size = int(image_size)
        self.lift_image_size = int(lift_image_size)
        self.depth_pack_suffix = str(depth_pack_suffix)
        self.pretrained_ss_dec = pretrained_ss_dec
        self.sample_weights: Optional[torch.Tensor] = None
        self.empty_latent: Optional[torch.Tensor] = None
        if self.INCLUDE_EMPTY:
            data = np.load(empty_latent_path)
            assert str(data["encoder"]) == EXPECTED_SS_ENCODER, f"{empty_latent_path}: wrong encoder"
            self.empty_latent = torch.from_numpy(data["z"]).float()

        self.instances = []     # (chunk_dir, image_path, depth_m, lateral_m, occupancy_class)
        self.latent_loads = []  # latent file size (token-count proxy for balanced batching)
        self._meta_cache = {}
        self._root_sizes = []
        skipped = 0
        for name, root in json.loads(roots).items():
            val_scenes, split = None, root.get("split")
            if "split_file" in root:
                assert split in ("train", "val"), f"[{name}] split_file given but split={split!r}"
                with open(root["split_file"], "r") as f:
                    val_scenes = set(json.load(f)["val_scenes"])
            scenes_dir = os.path.join(root["base"], "scenes")
            assert os.path.isdir(scenes_dir), f"[{name}] missing scenes dir: {scenes_dir}"
            instances, loads, n_skip = self._index_root(scenes_dir, split, val_scenes)
            instances, loads = self._drop_latentless_views(name, instances, loads)
            rep = int(root.get("repeat", 1))
            self.instances += instances * rep
            self.latent_loads += loads * rep
            self._root_sizes.append((name, len(instances) * rep))
            skipped += n_skip
        assert self.instances, "no windows found under the given roots"

        if balance_roots:
            present = [(n, c) for n, c in self._root_sizes if c]
            assert len(present) > 1, "balance_roots needs more than one non-empty root"
            weights = torch.zeros(len(self.instances), dtype=torch.double)
            offset = 0
            for _, count in self._root_sizes:
                if count:
                    weights[offset:offset + count] = 1.0 / (len(present) * count)
                offset += count
            self.sample_weights = weights

        if rank_zero_only.rank == 0:
            counts = Counter(occ for *_, occ in self.instances)
            print(f"[{type(self).__name__}] {len(self.instances)} windows {dict(sorted(counts.items()))}; "
                  f"roots {dict(self._root_sizes)}; skipped {skipped} without {self.LATENT_FILENAME}")

    @classmethod
    def _index_root(cls, scenes_dir: str, split: Optional[str] = None, val_scenes: Optional[set] = None):
        instances, loads, skipped = [], [], 0
        for scene in sorted(os.listdir(scenes_dir)):
            scene_dir = os.path.join(scenes_dir, scene)
            if not os.path.isdir(scene_dir):
                continue
            if val_scenes is not None and (scene in val_scenes) != (split == "val"):
                continue
            for view in sorted(os.listdir(scene_dir)):
                chunks_dir = os.path.join(scene_dir, view, "chunks")
                if not view.startswith("view_") or not os.path.isdir(chunks_dir):
                    continue
                image_path = os.path.join(scene_dir, view, "images", "000.jpg")
                for window in sorted(os.listdir(chunks_dir)):
                    chunk_dir = os.path.join(chunks_dir, window)
                    if not os.path.isdir(chunk_dir):
                        continue
                    with open(os.path.join(chunk_dir, "meta.json"), "r") as f:
                        meta = json.load(f)
                    validate_sidecar(meta)
                    occ = meta["occupancy_class"]
                    latent_path = os.path.join(chunk_dir, cls.LATENT_FILENAME)
                    if occ == "empty":
                        if not cls.INCLUDE_EMPTY:
                            continue
                        load = 0
                    elif not os.path.isfile(latent_path):
                        skipped += 1
                        continue
                    else:
                        load = os.path.getsize(latent_path)
                    assert os.path.isfile(image_path), f"missing conditioning image: {image_path}"
                    instances.append((chunk_dir, image_path, float(meta["depth_m"]), float(meta["lateral_m"]), occ))
                    loads.append(load)
        return instances, loads, skipped

    def _drop_latentless_views(self, name: str, instances: list, loads: list):
        """Drop views in which no window has a latent: broken packs, not sparse
        scenes (training them as all-empty would teach erasure)."""
        latent_views = {inst[1] for inst, load in zip(instances, loads) if load > 0}
        kept = [(inst, ld) for inst, ld in zip(instances, loads) if inst[1] in latent_views]
        if len(kept) < len(instances) and rank_zero_only.rank == 0:
            print(f"[{type(self).__name__}] [{name}] dropped {len(instances) - len(kept)} windows "
                  f"of views without any {self.LATENT_FILENAME}")
        return [i for i, _ in kept], [ld for _, ld in kept]

    @staticmethod
    def _scene_of(chunk_dir: str) -> str:
        return chunk_dir.split(f"{os.sep}scenes{os.sep}")[1].split(os.sep)[0]

    def first_view_chunks_per_scene(self):
        """[(scene, [indices of every window in the scene's first view]), ...]
        (validation renders assemble one whole view per scene)."""
        picks, seen = [], set()
        for idx, (chunk_dir, image_path, *_) in enumerate(self.instances):
            scene = self._scene_of(chunk_dir)
            if scene not in seen:
                seen.add(scene)
                picks.append((scene, image_path, []))
            if picks[-1][0] == scene and picks[-1][1] == image_path:
                picks[-1][2].append(idx)
        return [(scene, idxs) for scene, _, idxs in picks]

    def _load_meta(self, chunk_dir: str) -> dict:
        with open(os.path.join(chunk_dir, "meta.json"), "r") as f:
            return json.load(f)

    def chunk_to_world(self, index: int) -> np.ndarray:
        """[4, 4] float32 chunk -> world transform from the window's sidecar."""
        return np.array(self._load_meta(self.instances[index][0])["chunk_to_world"], dtype=np.float32)

    def window_z(self, index: int) -> float:
        """Window base height above the view floor (absent in v3 sidecars,
        where every window is floor-anchored, so 0 is the truth there)."""
        chunk_dir = self.instances[index][0]
        if chunk_dir not in self._meta_cache:
            self._meta_cache[chunk_dir] = float(self._load_meta(chunk_dir).get("z_off_view", 0.0))
        return self._meta_cache[chunk_dir]

    def __len__(self) -> int:
        return len(self.instances)

    @staticmethod
    def _resize_image(image: Image.Image, size: int) -> torch.Tensor:
        """PIL image -> [3, size, size] float in [0, 1] (LANCZOS resize)."""
        if image.size != (size, size):
            image = image.resize((size, size), Image.LANCZOS)
        return torch.from_numpy(np.array(image)).permute(2, 0, 1).float() / 255.0

    def _load_lift(self, image_path: str) -> dict:
        """The view's depth pack as lift tensors, resized to lift_image_size
        so packs of mixed resolution batch together (K follows the resize)."""
        from ..modules.lifting import load_da3_pack

        pack = load_da3_pack(os.path.splitext(image_path)[0] + self.depth_pack_suffix)
        depth, sky, K = pack["depth"], pack["sky"], pack["K"]
        size = self.lift_image_size
        if size and depth.shape != (size, size):
            K = K * torch.tensor([size / depth.shape[1], size / depth.shape[0], 1.0]).unsqueeze(1)
            depth = F.interpolate(depth[None, None], (size, size), mode="nearest")[0, 0]
            sky = F.interpolate(sky[None, None].float(), (size, size), mode="nearest")[0, 0] > 0.5
        return {
            "lift_depth": depth,                            # [S, S] meters (0 = no geometry)
            "lift_valid": (~sky) & bool(pack["floor_ok"]),  # [S, S] bool
            "lift_K": K,                                    # [3, 3] pixel intrinsics
            "lift_c2y": pack["cam_to_yaw"],                 # [4, 4] camera -> floor frame
        }

    def _load_view(self, image_path: str) -> dict:
        image = Image.open(image_path).convert("RGB")
        item = {"cond": self._resize_image(image, self.image_size)[None]}  # [1, 3, S, S]
        if self.lift_image_size:
            item["cond_hi"] = self._resize_image(image, self.lift_image_size)
        item.update(self._load_lift(image_path))
        return item

    def __getitem__(self, index: int) -> dict:
        chunk_dir, image_path, depth_m, lateral_m, occ = self.instances[index]
        if occ == "empty":
            z = self.empty_latent.clone()
        else:
            z = torch.from_numpy(np.load(os.path.join(chunk_dir, self.LATENT_FILENAME))["z"]).float()
        return {"x_0": z, "depth_m": depth_m, "lateral_m": lateral_m, "z_off_m": self.window_z(index),
                **self._load_view(image_path)}

    _TENSOR_KEYS = ("cond", "cond_hi", "lift_depth", "lift_valid", "lift_K", "lift_c2y")

    @classmethod
    def _collate_common(cls, sub_batch) -> dict:
        out = {k: torch.tensor([b[k] for b in sub_batch], dtype=torch.float32)
               for k in ("depth_m", "lateral_m", "z_off_m") if k in sub_batch[0]}
        out.update({k: torch.stack([b[k] for b in sub_batch]) for k in cls._TENSOR_KEYS if k in sub_batch[0]})
        return out

    @classmethod
    def _collate_group(cls, sub_batch):
        return {"x_0": torch.stack([b["x_0"] for b in sub_batch]), **cls._collate_common(sub_batch)}

    @classmethod
    def collate_fn(cls, batch, split_size=None):
        if split_size is None:
            return cls._collate_group(batch)
        groups = [batch[i::split_size] for i in range(split_size) if batch[i::split_size]]
        return [cls._collate_group(g) for g in groups]

#!/usr/bin/env python3
"""Single image -> metric 3D scene mesh.

The scene is split along depth into bands. Band 0 uses native 3 m windows
(64^3 voxels, 4.7 cm); each further band's windows are ``--band-growth`` times
larger in metric space, until the far edge reaches the ``--band-p`` depth
quantile of the MoGe-3 cloud (bands covering < ``--band-min-coverage`` of the
image are dropped). Each band is generated in its own canonical frame: the
evidence is scaled by 1/a about the camera's floor footpoint (pixel
projections are preserved), a 3 m window layout is placed where the cloud has
support (plus a one-window ring), and the decoded mesh is scaled back by a.

Bands run nearest-first. With cross-band outpainting (default) each band's SS
and SLat latents in the overlap with the previous band are pinned to that
band's latents, trilinearly resampled onto the coarser grid, and the nearer
band keeps its geometry in the overlap.

Writes <out>/<stem>/{scene.glb, layout_bands.json, layout_topview.png,
spiral.mp4}. scene.glb is metric, glTF Y-up, camera at the
origin looking along -Z.

    python inference/image_to_scene.py --image photo.jpg \
        --ss-ckpt ss.ckpt --slat-ckpt slat.ckpt --out results/
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.geometry import apply_scene_scale, ensure_moge3_pack, filtered_cloud, load_lift_kwargs  # noqa: E402
from inference.models import DINO, CondShim, load_image  # noqa: E402
from inference.stages import MIN_VOXELS, SS_AR_ERODE, run_slat_stage, run_ss_stage  # noqa: E402

WINDOW_STRIDE_M = 2.4375   # 13 latent cells: 0.5625 m (3-cell) overlap between windows
WINDOW_DEPTH_START_M = 1.5 # nearest window center: its front face at the camera plane
GROUND_Z_OFF_M = -0.25     # window bases sit this far below the fitted floor (training convention)
SUPPORT_MIN_PTS = 150      # cloud points inside a window for it to count as evidence
HEIGHT_TARGET_M = 2.8      # a band grows until its p99 content height fits this (canonical m)
BAND_COLORS = ["tab:green", "tab:orange", "tab:red", "tab:purple", "tab:brown"]


def _p99_height(pts, lo, hi):
    m = (pts[:, 1] >= lo) & (pts[:, 1] < hi)
    return float(torch.quantile(pts[m, 2], 0.99)) if int(m.sum()) >= 50 else 0.0


def band_schedule(pts, p_depth_m, near_max_m, growth, stride_m, depth_start_m, C, overlap_m,
                  n_pix, min_coverage=0.0, band_count=None, min_depth_m=None, overlap_frac=0.0):
    """[{a, rows (canonical window-center depths), front_m, far_m, coverage}].

    Band 0 is the native lattice (a = 1) while rows stay within near_max_m and
    the p99 height of each row's depth slab fits HEIGHT_TARGET_M. Further bands
    are one row each, starting overlap inside the previous far edge, with
        a = max(previous a, depth rule, p99 height(slab) / HEIGHT_TARGET_M)
    where the depth rule multiplies by ``growth`` on every band beyond
    near_max_m. Bands are added until far_m >= p_depth_m, then the list is cut
    after the last band whose slab holds >= min_coverage of the image pixels
    (pixel coverage, not point density, which falls as 1/z^2).

    Overrides: band_count = exactly this many bands (coverage cut ignored);
    min_depth_m = keep bands until the far edge reaches this depth.
    overlap = max(overlap_m, overlap_frac x previous band's window size)."""
    rows, d = [], depth_start_m
    while d <= near_max_m + 1e-9:
        if _p99_height(pts, d - C / 2.0, d + C / 2.0) > HEIGHT_TARGET_M:
            break
        rows.append(d)
        d += stride_m
    bands = []
    if rows:
        far = rows[-1] + C / 2.0
        bands.append(dict(a=1.0, rows=rows, front_m=depth_start_m - C / 2.0, far_m=far))
    else:
        far = depth_start_m - C / 2.0 + overlap_m
    a, a_depth = 1.0, 1.0
    want_far = max(p_depth_m, min_depth_m or 0.0)
    while far < want_far - 1e-6 or (band_count and len(bands) < band_count):
        front = far - max(overlap_m, overlap_frac * a * C)
        if front >= near_max_m - 1e-9:
            a_depth *= growth
        a = max(a, a_depth)
        for _ in range(8):  # the slab grows with a: iterate to a fixed point
            a_h = _p99_height(pts, front, front + a * C) / HEIGHT_TARGET_M
            if a_h <= a + 1e-3:
                break
            a = a_h
        far = front + a * C
        bands.append(dict(a=a, rows=[(front + a * C / 2.0) / a], front_m=front, far_m=far))
    for b in bands:
        m = (pts[:, 1] >= b["front_m"]) & (pts[:, 1] < b["far_m"])
        b["coverage"] = float(m.sum()) / float(n_pix)
    if band_count:
        return bands[:band_count]
    if min_coverage > 0:
        keep = [i for i, b in enumerate(bands) if b["coverage"] >= min_coverage
                or (min_depth_m and b["front_m"] < min_depth_m)]
        bands = bands[:(keep[-1] if keep else 0) + 1]
    return bands


def band_layout(pts_c, rows, stride_m, C):
    """Windows for one band in its canonical frame: rows x a lateral lattice
    spanning the cloud; kept when >= SUPPORT_MIN_PTS points fall in the cube
    (evidence) or within one stride of an evidence window (ring).
    Returns (layout [(depth, lateral)], evidence flags) or None."""
    half = C / 2.0
    pts = pts_c[(pts_c[:, 2] >= 0.0) & (pts_c[:, 2] < C)]
    if not pts.shape[0]:
        return None
    lmin, lmax = float(pts[:, 0].min()), float(pts[:, 0].max())
    k_lo = math.ceil((lmin - half) / stride_m - 1e-9)
    k_hi = math.floor((lmax + half) / stride_m + 1e-9)
    cand = [(float(d), k * stride_m) for d in rows for k in range(k_lo, k_hi + 1)]
    ct = torch.tensor(cand, device=pts.device)
    inside = ((pts[None, :, 1] - ct[:, 0, None]).abs() < half) & ((pts[None, :, 0] - ct[:, 1, None]).abs() < half)
    occupied = inside.sum(dim=1) >= SUPPORT_MIN_PTS
    if not bool(occupied.any()):
        return None
    cheb = (ct[:, None, :] - ct[occupied][None, :, :]).abs().amax(dim=-1).amin(dim=-1)
    kept = torch.nonzero(occupied | (cheb <= stride_m + 1e-6), as_tuple=True)[0].tolist()
    return [cand[i] for i in kept], [bool(occupied[i]) for i in kept]


def export_scene_glb(meshes, transforms, path: str) -> None:
    """Metric scene mesh, camera at the origin; floor frame Z-up -> glTF Y-up."""
    import trimesh

    b2g = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, -1, 0, 0], [0, 0, 0, 1]], dtype=float)
    scene = trimesh.Scene()
    for k, (m, T) in enumerate(zip(meshes, transforms)):
        if m is None or m.faces.shape[0] == 0:
            continue
        tm = trimesh.Trimesh(vertices=m.vertices.detach().cpu().numpy(),
                             faces=m.faces.detach().cpu().numpy(), process=False)
        tm.remove_unreferenced_vertices()
        scene.add_geometry(tm, transform=b2g @ np.asarray(T, dtype=float), node_name=f"w{k:02d}")
    scene.export(path)
    print(f"[export] wrote {path}")


def compose_frame(panels, labels, sep_px: int = 6) -> np.ndarray:
    """Panels side by side with red separators and top-left labels."""
    H = panels[0].shape[0]
    red = np.zeros((H, sep_px, 3), dtype=np.uint8)
    red[:, :, 0] = 255
    strips = []
    for i, p in enumerate(panels):
        strips += ([red] if i else []) + [p]
    img = Image.fromarray(np.concatenate(strips, axis=1))
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype("DejaVuSans-Bold.ttf", max(18, H // 20))
    except OSError:
        font = ImageFont.load_default()
    x = 0
    for i, (p, label) in enumerate(zip(panels, labels)):
        x += sep_px if i else 0
        for dx, dy in ((-1, -1), (-1, 1), (1, -1), (1, 1)):
            draw.text((x + 10 + dx, 8 + dy), label, fill=(0, 0, 0), font=font)
        draw.text((x + 10, 8), label, fill=(255, 255, 255), font=font)
        x += p.shape[1]
    return np.array(img)


def plot_topview(pts_m, records, meshes, transforms, K, p_depth_m, path, title):
    """Top-down debug plot: cloud, generated mesh, band windows, frustum."""
    try:
        import matplotlib
    except ImportError:
        print(f"[bands] matplotlib not available; skipping {os.path.basename(path)}")
        return
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    P = pts_m.cpu().numpy()
    fig, ax = plt.subplots(figsize=(10, 12))
    x_lo = min(min(l * r["a"] - r["a"] * r["chunk_m"] / 2 for _, l in r["layout_2d"]) for r in records) - 0.5
    ax.hexbin(P[:, 0], P[:, 1], gridsize=220, bins="log", cmap="Greys", mincnt=1, linewidths=0, zorder=1)
    V = []
    for m, T in zip(meshes, transforms):
        v = m.vertices.detach().float().cpu().numpy()
        v = v @ np.asarray(T, dtype=np.float32)[:3, :3].T + np.asarray(T)[:3, 3]
        if v.shape[0] > 300_000:
            v = v[np.random.default_rng(0).choice(v.shape[0], 300_000, replace=False)]
        V.append(v)
    V = np.concatenate(V)
    ax.hexbin(V[:, 0], V[:, 1], gridsize=220, bins="log", cmap="Blues", mincnt=1, alpha=0.45,
              linewidths=0, zorder=2)
    for r in records:
        a, col = r["a"], BAND_COLORS[r["band"] % len(BAND_COLORS)]
        w_m = a * r["chunk_m"]
        for (d, l), ev, k in zip(r["layout_2d"], r["evidence"], r["order"]):
            ax.add_patch(Rectangle((l * a - w_m / 2, d * a - w_m / 2), w_m, w_m, fill=False,
                                   lw=1.8 if ev else 1.2, ls="-" if ev else "--", ec=col, zorder=4))
            ax.text(l * a, d * a, f"{k}", ha="center", va="center", fontsize=9, color=col,
                    fontweight="bold", zorder=5)
        ax.axhline(r["front_m"], color=col, lw=0.6, ls=":", zorder=3)
        ax.text(x_lo, r["front_m"], f"  band {r['band']}  x{a:.3g}  ({w_m:.3g} m windows, "
                f"{w_m / 64 * 100:.1f} cm voxels, {len(r['layout_2d'])} cols)", color=col,
                fontsize=9, va="bottom", ha="left", zorder=5)
    ax.axhline(p_depth_m, color="k", lw=0.8, ls="--", zorder=3)
    ax.text(x_lo, p_depth_m, f"  p-depth {p_depth_m:.1f} m", va="bottom", fontsize=9)
    half_fov = math.atan(float(K[0, 2]) / float(K[0, 0]))
    dmax = max(p_depth_m * 1.15, max(r["far_m"] for r in records))
    for sgn in (-1, 1):
        ax.plot([0, sgn * dmax * math.tan(half_fov)], [0, dmax], color="k", lw=0.8, zorder=3)
    ax.plot(0, 0, marker="^", color="k", ms=9, zorder=6)
    ax.set_aspect("equal")
    ax.set_xlabel("lateral (m)")
    ax.set_ylabel("depth (m)")
    ax.set_title(title, fontsize=10)
    ax.set_ylim(-1.5, dmax + 1)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def _window_cubes(layout, a, C, erode_m):
    """[N, 6] full and eroded window cubes (lo xyz, hi xyz), metric-consistent (x a)."""
    full, ero = [], []
    for w in layout:
        d_w, l_w, z_w = float(w[0]), float(w[1]), (float(w[2]) if len(w) > 2 else 0.0)
        lo = torch.tensor([l_w - C / 2, d_w - C / 2, z_w])
        hi = lo + C
        full.append(torch.cat([lo, hi])[None] * a)
        ero.append(torch.cat([lo + erode_m, hi - erode_m])[None] * a)
    return torch.cat(full), torch.cat(ero)


def run(args) -> None:
    from build_rome.pipelines import windows as mdv
    from build_rome.lightning.render import (build_spiral_cameras, merge_chunk_meshes,
                                           render_mesh_spiral, write_video_file)
    from build_rome.modules.lifting import load_da3_pack
    from build_rome.representations import Mesh
    from build_rome.trainers.flow_matching.mixins.image_conditioned import DinoV3FeatureExtractor

    device = torch.device("cuda")
    stem = os.path.splitext(os.path.basename(args.image))[0]
    scene_dir = os.path.join(args.out, stem)
    os.makedirs(scene_dir, exist_ok=True)
    image = load_image(args.image)
    pack_path = args.depth_pack or ensure_moge3_pack(args.image, args.moge3_python, args.moge3_cache)
    lift_kwargs = load_lift_kwargs(pack_path, device)

    C = mdv.CHUNK_SIZE
    stride_m = WINDOW_STRIDE_M
    base_overlap_m = C - stride_m
    pts_m = filtered_cloud(lift_kwargs)
    h99 = float(torch.quantile(pts_m[:, 2], 0.99))
    p_depth = float(torch.quantile(pts_m[:, 1], args.band_p))
    bands = band_schedule(
        pts_m, p_depth, args.band_near_m, args.band_growth, stride_m, WINDOW_DEPTH_START_M, C,
        base_overlap_m, n_pix=int(lift_kwargs["lift_depth"][0].numel()),
        min_coverage=args.band_min_coverage, band_count=args.band_count,
        min_depth_m=args.band_min_depth_m, overlap_frac=args.band_overlap_frac)
    print(f"[bands] cloud p{args.band_p:.2f} depth {p_depth:.2f} m, p99 height {h99:.2f} m; {len(bands)} bands:")
    for i, b in enumerate(bands):
        print(f"[bands]   band {i}: x{b['a']:.3g} ({b['a'] * C:.3g} m windows, voxel "
              f"{b['a'] * C / 64 * 100:.1f} cm) depth {b['front_m']:.2f}..{b['far_m']:.2f} m, "
              f"{b['coverage']:.1%} of image")

    extractor = DinoV3FeatureExtractor(DINO, image_size=512)        # SS cross-attention
    slat_extractor = DinoV3FeatureExtractor(DINO, image_size=1024)  # SLat cross-attention
    extractor.cuda()
    slat_extractor.cuda()
    ss_shim, slat_shim = CondShim(extractor), CondShim(slat_extractor)
    lift_hi_feats = extractor(load_image(args.image, size=1024)[None])  # depth-lift source

    erode_m = SS_AR_ERODE * C / 16
    meshes, transforms, records = [], [], []
    owned_cubes = []    # nearer bands' full window cubes (metric-consistent)
    prev = None         # previous band's latents for cross-band outpainting
    for bi, b in enumerate(bands):
        a = float(b["a"])
        lk = apply_scene_scale({k: v.clone() for k, v in lift_kwargs.items()}, 1.0 / a)
        lay = band_layout(pts_m / a, b["rows"], stride_m, C)
        if lay is None:
            print(f"[bands] band {bi} (x{a:g}): no window has support, skipped")
            continue
        layout, evid = lay
        order_idx = sorted(range(len(layout)), key=lambda i: (not evid[i], layout[i][0] ** 2 + layout[i][1] ** 2))
        order = [0] * len(layout)
        for k, i in enumerate(order_idx):
            order[i] = k + 1
        print(f"[bands] === band {bi}: x{a:g}, {len(layout)} windows "
              f"({sum(evid)} evidence + {len(layout) - sum(evid)} ring) ===")
        layout3 = [(d, l, GROUND_Z_OFF_M) for d, l in layout]
        seed = args.seed + 1000 * bi
        ss_prior = slat_prior = None
        if args.cross_band and prev is not None:
            ss_prior = dict(lat=prev["ss_lat"], full=prev["ss_full"] / a, boxes=prev["ss_ero"] / a)
            if prev.get("sl_cen") is not None:
                slat_prior = dict(centers=prev["sl_cen"] / a, feats=prev["sl_feat"],
                                  pitch=prev["sl_pitch"] / a, origin=prev["sl_org"] / a,
                                  boxes=prev["ss_ero"] / a)
        try:
            occs, ss_latents, world_coords, R = run_ss_stage(
                args.ss_ckpt, image, ss_shim, device, lk, lift_hi_feats, layout3, evid, seed, prior=ss_prior)
            del occs
            torch.cuda.empty_cache()
            mesh, T0, tokens = run_slat_stage(
                args.slat_ckpt, image, slat_shim, device, lk, lift_hi_feats, layout3, world_coords, R,
                seed, scene_scale=1.0 / a, prior=slat_prior,
                max_inflated_voxels=args.max_inflated_voxels)
        except AssertionError as e:
            # an empty band (all windows below MIN_VOXELS) must not kill the run
            print(f"[bands] band {bi} (x{a:g}) produced no geometry ({e}), skipped")
            torch.cuda.empty_cache()
            continue
        n_dec = int(mesh.faces.shape[0])
        if args.band_simplify > 1 and a > 1.0 + 1e-6:
            t0 = time.time()
            mesh.simplify(target=int(n_dec / args.band_simplify))
            print(f"[bands] band {bi} simplify x{args.band_simplify:g}: {n_dec / 1e6:.1f}M -> "
                  f"{mesh.faces.shape[0] / 1e6:.2f}M faces in {time.time() - t0:.1f}s")
        full, ero = _window_cubes(layout3, a, C, erode_m)
        if args.cross_band and owned_cubes:
            # nearer bands own their cubes: drop this band's faces there
            Tt = torch.tensor(np.diag([a, a, a, 1.0]) @ np.asarray(T0, dtype=np.float64),
                              dtype=torch.float64, device=mesh.vertices.device)
            V = mesh.vertices.detach().double()
            Vw = V @ Tt[:3, :3].T + Tt[:3, 3]
            Fc = mesh.faces.long()
            cen = (Vw[Fc[:, 0]] + Vw[Fc[:, 1]] + Vw[Fc[:, 2]]) / 3.0
            own = torch.zeros(len(Fc), dtype=torch.bool, device=V.device)
            for bx in torch.cat(owned_cubes).to(V):
                own |= ((cen >= bx[:3]) & (cen <= bx[3:])).all(dim=1)
            mesh = Mesh(mesh.vertices.detach(), mesh.faces[~own])
            print(f"[cross-band] band {bi}: cropped {int(own.sum()):,}/{len(Fc):,} faces inside nearer bands")
        if args.cross_band:
            owned_cubes.append(full)
            prev = dict(ss_lat=torch.cat([z.detach().float().cpu() for z in ss_latents]),
                        ss_full=full, ss_ero=ero)
            toks, sl_lay = tokens
            cens = []
            for (lc, _), w in zip(toks, sl_lay):
                d_w, l_w, z_w = float(w[0]), float(w[1]), float(w[2])
                cens.append(((lc.float() + 0.5) * (C / 64) + torch.tensor([l_w - C / 2, d_w - C / 2, z_w])) * a)
            w0 = sl_lay[0]
            prev.update(sl_cen=torch.cat(cens), sl_feat=torch.cat([f for _, f in toks]), sl_pitch=a * C / 64,
                        sl_org=torch.tensor([float(w0[1]) - C / 2, float(w0[0]) - C / 2, float(w0[2])]) * a)
        meshes.append(mesh)
        transforms.append(np.diag([a, a, a, 1.0]).astype(np.float32) @ T0)
        records.append(dict(band=bi, a=a, chunk_m=C, front_m=b["front_m"], far_m=b["far_m"],
                            coverage=b["coverage"], layout_2d=[[float(d), float(l)] for d, l in layout],
                            evidence=evid, order=order,
                            layout_canonical=[[float(c) for c in w] for w in layout3],
                            n_faces_decoded=n_dec, n_faces=int(mesh.faces.shape[0])))
        torch.cuda.empty_cache()
    assert meshes, "no band produced a mesh"

    export_scene_glb(meshes, transforms, os.path.join(scene_dir, "scene.glb"))
    with open(os.path.join(scene_dir, "layout_bands.json"), "w") as f:
        json.dump(dict(p_depth_m=p_depth, band_p=args.band_p, near_m=args.band_near_m,
                       growth=args.band_growth, overlap_m=base_overlap_m, stride_m=stride_m,
                       p99_height_m=h99, bands=records), f, indent=1)
    plot_topview(pts_m, records, meshes, transforms, load_da3_pack(pack_path)["K"], p_depth,
                 os.path.join(scene_dir, "layout_topview.png"),
                 f"{stem}: {len(records)} bands, gray = MoGe-3 cloud, blue = generated mesh")

    if args.spiral_video:
        res, n_frames = 1024, 120
        scene = merge_chunk_meshes(meshes, transforms, 999.0)
        extr, intr = build_spiral_cameras(num_frames=n_frames, r=1.0, fov=60.0,
                                          base_elevation_deg=35.0, elevation_amp_deg=15.0)
        frames = render_mesh_spiral(scene, extr, intr, res, near_clip_frac=0.15)
        inp = (torch.nn.functional.interpolate(image[None], size=(res, res), mode="bilinear",
                                               align_corners=False)[0].clamp(0, 1) * 255)
        inp = inp.byte().permute(1, 2, 0).numpy()
        labels = ["input", f"SLat ({len(records)} bands)"]
        write_video_file(np.stack([compose_frame([inp, fr], labels) for fr in frames]),
                         os.path.join(scene_dir, "spiral.mp4"), fps=30)
    print(f"[done] {scene_dir}")


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--image", required=True)
    ap.add_argument("--ss-ckpt", required=True, help="SS denoiser checkpoint")
    ap.add_argument("--slat-ckpt", required=True, help="SLat denoiser checkpoint")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--depth-pack", default=None,
                    help="precomputed depth pack (.npz); default: MoGe-3 on the input image")
    ap.add_argument("--moge3-python", default=sys.executable,
                    help="python of an environment with MoGe-3 installed")
    ap.add_argument("--moge3-cache", default=os.path.expanduser("~/.cache/image_to_scene/moge3"),
                    help="MoGe-3 pack cache (keyed by image content)")
    ap.add_argument("--band-near-m", type=float, default=9.0,
                    help="native (1x) windows for window centers up to this depth")
    ap.add_argument("--band-growth", type=float, default=2.0, help="window-size multiplier per further band")
    ap.add_argument("--band-p", type=float, default=0.99, help="cloud depth quantile the last band must reach")
    ap.add_argument("--band-min-coverage", type=float, default=0.05,
                    help="drop far bands covering less than this fraction of the image")
    ap.add_argument("--band-count", type=int, default=None, help="force exactly this many bands")
    ap.add_argument("--band-min-depth-m", type=float, default=None,
                    help="keep bands until the far edge reaches this depth")
    ap.add_argument("--band-overlap-frac", type=float, default=0.25,
                    help="band overlap as a fraction of the previous band's window size")
    ap.add_argument("--band-simplify", type=float, default=4.0,
                    help="simplify bands with a > 1 to 1/this of their faces (0 = off)")
    ap.add_argument("--cross-band", action=argparse.BooleanOptionalAction, default=True,
                    help="pin each band's overlap with the previous band to that band's latents")
    ap.add_argument("--max-inflated-voxels", type=int, default=100_000,
                    help="SLat joint-decode single-pass cap; lower it if the decode runs out of memory")
    ap.add_argument("--spiral-video", action=argparse.BooleanOptionalAction, default=True,
                    help="also render spiral.mp4 (orbit render next to the input)")
    return ap


def main() -> None:
    run(build_parser().parse_args())


if __name__ == "__main__":
    main()

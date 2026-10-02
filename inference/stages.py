"""The two generation stages for one band of windows.

SS stage: windows are generated one at a time, nearest the camera first
(evidence windows before ring windows). Each window RePaint-pins the latent
cells already committed by its neighbors (autoregressive outpainting), and
optionally by the previous, nearer band (cross-band prior). All window
latents are then decoded jointly into one occupancy canvas.

SLat stage: the SS occupancy is sliced back into the windows; sparse latents
are generated window by window with the same outpainting, then decoded
jointly into one mesh.
"""
import numpy as np
import torch
import torch.nn.functional as F

from inference.models import (SAMPLER_PARAMS, SLAT_DEC, SLAT_MEAN, SLAT_STD, SS_DEC,
                              build_slat_model, build_ss_model)

SS_AR_ERODE = 1      # latent cells (of 16) dropped from a committed window's border
SLAT_AR_ERODE = 4    # token cells (of 64) dropped from a committed window's border
MIN_VOXELS = 64      # windows with fewer decoded voxels count as empty
FILL_HOLES_M = 1.0   # close mesh holes with metric perimeter below this


def _dlz(w):
    """Layout entry (depth, lateral[, z]) -> (d, l, z); z = window base height."""
    return w[0], w[1], (w[2] if len(w) > 2 else 0.0)


def _lift_chunk_kwargs(chunk_kwargs, lift_kwargs):
    # the depth pack rides chunk_kwargs, so the CFG negative pass sees it too
    return [{**kw, **lift_kwargs} for kw in chunk_kwargs]


def _with_hires(conds, neg_conds, lift_hi_feats):
    # hi-res lift source rides the CFG cond dict (zeroed in the negative pass)
    return ([{**c, "cond_2D_hi": lift_hi_feats} for c in conds],
            [{**c, "cond_2D_hi": torch.zeros_like(lift_hi_feats)} for c in neg_conds])


def cross_band_ss_prior(layout, prior, device, R_lat=16):
    """Pin targets from the previous band: each latent cell of THIS band's
    windows whose center lies in a nearer window's eroded cube takes that
    window's latent, trilinearly resampled. prior: lat [N, 8, 16, 16, 16],
    full / boxes [N, 6] (full / eroded cubes) in this band's canonical frame."""
    from build_rome.pipelines import windows as mdv

    C = mdv.CHUNK_SIZE
    lat = prior["lat"].to(device).float()
    full = prior["full"].to(device).float()
    ero = prior["boxes"].to(device).float()
    ax = (torch.arange(R_lat, dtype=torch.float32, device=device) + 0.5) * (C / R_lat)
    gx, gy, gz = torch.meshgrid(ax - C / 2, ax - C / 2, ax, indexing="ij")
    local = torch.stack([gx, gy, gz], dim=-1).reshape(-1, 3)
    x0s, pins, n_pin = [], [], 0
    for w in layout:
        d, l, z = _dlz(w)
        ctr = local + torch.tensor([l, d, z], device=device)
        x0 = torch.zeros(lat.shape[1], R_lat ** 3, device=device)
        pin = torch.zeros(R_lat ** 3, dtype=torch.bool, device=device)
        for j in range(len(lat)):
            m = ((ctr >= ero[j, :3]) & (ctr <= ero[j, 3:])).all(dim=1) & ~pin
            if not bool(m.any()):
                continue
            lo, hi = full[j, :3], full[j, 3:]
            nrm = 2 * (ctr[m] - lo) / (hi - lo) - 1                 # (lat, dep, up) in [-1, 1]
            grid = nrm[:, [2, 1, 0]].view(1, -1, 1, 1, 3)           # grid_sample: (W=up, H=dep, D=lat)
            smp = F.grid_sample(lat[j:j + 1], grid, mode="bilinear", align_corners=False)
            x0[:, m] = smp.view(lat.shape[1], -1)
            pin |= m
        if not bool(pin.any()):
            x0s.append(None); pins.append(None)
            continue
        x0s.append(x0.view(1, -1, R_lat, R_lat, R_lat))
        pins.append(pin.view(1, 1, R_lat, R_lat, R_lat))
        n_pin += int(pin.sum())
    print(f"[cross-band] {sum(p is not None for p in pins)}/{len(layout)} windows, "
          f"{n_pin} latent cells pinned from {len(lat)} nearer windows")
    return x0s, pins


def cross_band_slat_prior(act_layout, local_coords_act, rel_cells, R, prior, device, keys_fn):
    """Pin targets from the previous band's SLat tokens: every token of THIS
    band inside a nearer window's eroded cube takes the trilinear interpolation
    of the 8 surrounding nearer tokens (weights renormalized over those that
    exist; pinned only if >= half the weight is present). prior: centers
    [M, 3], feats [M, C], pitch, origin [3], boxes [B, 6] in this band's frame.
    Returns (sorted world keys, feats) or (None, None)."""
    from build_rome.pipelines import windows as mdv

    C = mdv.CHUNK_SIZE
    cen = prior["centers"].to(device).float()
    fe = prior["feats"].to(device).float()
    pitch = float(prior["pitch"])
    org = prior["origin"].to(device).float()
    boxes = prior["boxes"].to(device).float()
    g = torch.round((cen - org) / pitch - 0.5).long()
    K = lambda q: (q[:, 0] + 8192) * 16384 * 16384 + (q[:, 1] + 8192) * 16384 + (q[:, 2] + 8192)  # noqa: E731
    kp = K(g)
    kp, order = kp.sort()
    fe = fe[order]
    keys, feats = [], []
    for a, w in enumerate(act_layout):
        d, l, z = _dlz(w)
        lc = local_coords_act[a].long().to(device)
        ctr = (lc.float() + 0.5) * (C / R) + torch.tensor([l - C / 2, d - C / 2, z], device=device)
        inside = torch.zeros(len(lc), dtype=torch.bool, device=device)
        for b in boxes:
            inside |= ((ctr >= b[:3]) & (ctr <= b[3:])).all(dim=1)
        if not bool(inside.any()):
            continue
        q = (ctr[inside] - org) / pitch - 0.5          # continuous nearer-lattice coords
        q0 = torch.floor(q).long()
        t = q - q0.float()
        acc = torch.zeros(len(q), fe.shape[1], device=device)
        wsum = torch.zeros(len(q), device=device)
        for dx in (0, 1):
            for dy in (0, 1):
                for dz in (0, 1):
                    nb = q0 + torch.tensor([dx, dy, dz], device=device)
                    wt = ((t[:, 0] if dx else 1 - t[:, 0]) * (t[:, 1] if dy else 1 - t[:, 1])
                          * (t[:, 2] if dz else 1 - t[:, 2]))
                    kq = K(nb)
                    pos = torch.searchsorted(kp, kq).clamp(max=len(kp) - 1)
                    hit = kp[pos] == kq
                    acc[hit] += wt[hit, None] * fe[pos[hit]]
                    wsum[hit] += wt[hit]
        ok = wsum >= 0.5
        if not bool(ok.any()):
            continue
        idx = torch.nonzero(inside).squeeze(1)[ok]
        keys.append(keys_fn(lc[idx] + rel_cells[a][None, :]))
        feats.append(acc[ok] / wsum[ok, None])
    if not keys:
        print("[cross-band slat] no prior tokens for this band")
        return None, None
    k, f = torch.cat(keys), torch.cat(feats)
    k, first = np.unique(k.cpu().numpy(), return_index=True)
    print(f"[cross-band slat] {len(k):,} tokens interpolated from {len(cen):,} nearer tokens")
    return torch.from_numpy(k).to(device), f[torch.from_numpy(first).to(device)]


@torch.no_grad()
def run_ss_stage(ss_ckpt, image, shim, device, lift_kwargs, lift_hi_feats, layout, evid,
                 seed, prior=None):
    """Autoregressive SS over ``layout`` (canonical frame) -> (occs, latents,
    world_coords, R). occs: per-window [64, 64, 64] bool occupancy."""
    from build_rome import models
    from build_rome.pipelines import windows as mdv
    from build_rome.pipelines.joint_decode import joint_decode_sparse_structure

    denoiser = build_ss_model(ss_ckpt, device)
    ss_dec = models.from_pretrained(SS_DEC).to(device).eval()
    rel = mdv.relative_translations(layout)
    conds, neg_conds, chunk_kwargs = mdv.make_window_conditioning(shim, image, layout, device)
    chunk_kwargs = _lift_chunk_kwargs(chunk_kwargs, lift_kwargs)
    conds, neg_conds = _with_hires(conds, neg_conds, lift_hi_feats)
    sampler = mdv.make_val_sampler(1e-5)

    R_lat, C = denoiser.resolution, denoiser.in_channels
    rel_cells = [(t * R_lat).round().long() for t in rel]
    order = sorted(range(len(layout)), key=lambda i: (not evid[i], layout[i][0] ** 2 + layout[i][1] ** 2))
    print(f"[ss] {len(order)} windows, nearest-first, pin erode {SS_AR_ERODE} cells")
    trust_lo, trust_hi = SS_AR_ERODE, R_lat - SS_AR_ERODE
    prior_x0 = prior_pin = None
    if prior is not None:
        prior_x0, prior_pin = cross_band_ss_prior(layout, prior, device, R_lat)
    latents = [None] * len(layout)
    for step, i in enumerate(order):
        x0 = torch.zeros(1, C, R_lat, R_lat, R_lat, device=device)
        pin = torch.zeros(1, 1, R_lat, R_lat, R_lat, dtype=torch.bool, device=device)
        if prior_pin is not None and prior_pin[i] is not None:
            # the nearer band is finer and already committed: it wins over
            # this band's own neighbors (pinned below only where still free)
            x0 = torch.where(prior_pin[i], prior_x0[i], x0)
            pin = pin | prior_pin[i]
        for j in order[:step]:
            off = (rel_cells[j] - rel_cells[i]).tolist()   # j's origin in i's cells
            lo_i = [max(o, 0) for o in off]
            hi_i = [min(o + R_lat, R_lat) for o in off]
            if any(a >= b for a, b in zip(lo_i, hi_i)):
                continue
            lo_e = [max(li, o + trust_lo) for li, o in zip(lo_i, off)]
            hi_e = [min(hi, o + trust_hi) for hi, o in zip(hi_i, off)]
            if any(a >= b for a, b in zip(lo_e, hi_e)):
                continue
            sl_i = tuple(slice(a, b) for a, b in zip(lo_e, hi_e))
            sl_j = tuple(slice(a - o, b - o) for a, b, o in zip(lo_e, hi_e, off))
            new = ~pin[0, 0][sl_i]   # first committed neighbor wins
            x0[0, :, sl_i[0], sl_i[1], sl_i[2]][:, new] = latents[j][0][(slice(None),) + sl_j][:, new].float()
            pin[0, 0][sl_i] |= new
        torch.manual_seed(seed + i)
        torch.cuda.manual_seed_all(seed + i)
        has_pin = bool(pin.any())
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            out = mdv.sample_window(
                sampler, denoiser, torch.empty(1, C, R_lat, R_lat, R_lat, device=device),
                conds[i], neg_conds[i], **SAMPLER_PARAMS["ss"], chunk_kwargs=chunk_kwargs[i],
                inpaint_x0=x0 if has_pin else None, inpaint_mask=pin if has_pin else None)
        latents[i] = out.float()
        print(f"[ss]   window {step + 1}/{len(order)} (d={layout[i][0]:.2f}, l={layout[i][1]:.2f}) "
              f"pinned {int(pin.sum())}/{R_lat ** 3} cells")
    del denoiser
    torch.cuda.empty_cache()

    R = 64  # decoded cells per window (= SLat grid)
    occs = [(logit[0, 0] > 0) for logit in joint_decode_sparse_structure(ss_dec, latents, rel, ss_res=R)]
    del ss_dec
    world, kept = [], 0
    for occ, w in zip(occs, layout):
        d, l, z = _dlz(w)
        if int(occ.sum()) < MIN_VOXELS:
            continue
        kept += 1
        off = torch.tensor([round(l / mdv.CHUNK_SIZE * R), round(d / mdv.CHUNK_SIZE * R),
                            round(z / mdv.CHUNK_SIZE * R)], device=device, dtype=torch.long)
        world.append(torch.nonzero(occ, as_tuple=False).long() + off[None, :])
    print(f"[ss] {kept}/{len(layout)} windows above {MIN_VOXELS} voxels")
    assert world, "SS predicted an empty scene"
    return occs, latents, torch.unique(torch.cat(world, dim=0), dim=0), R


@torch.no_grad()
def run_slat_stage(slat_ckpt, image, shim, device, lift_kwargs, lift_hi_feats, layout,
                   world_coords, R, seed, scene_scale, prior=None, max_inflated_voxels=100_000):
    """SLat over the SS occupancy -> (mesh, T0, tokens). The mesh is in the
    window-0-local frame; T0 places it in the band's canonical frame. tokens:
    ([(local coords, normalized feats)], active layout) for the next band."""
    from build_rome import models
    from build_rome.pipelines import windows as mdv
    from build_rome.modules.sparse.basic import SparseTensor
    from build_rome.pipelines.joint_decode import joint_decode_shape

    denoiser = build_slat_model(slat_ckpt, device)
    slat_dec = models.from_pretrained(SLAT_DEC)
    slat_dec.set_resolution(1024)
    slat_dec = slat_dec.to(device).eval()

    local_coords, _ = mdv.slice_shared_structure(world_coords, layout, R)
    active = [k for k, lc in enumerate(local_coords) if lc is not None and lc.shape[0] >= MIN_VOXELS]
    assert active, "no active SLat windows"
    act_layout = [layout[k] for k in active]
    rel = mdv.relative_translations(act_layout)
    conds, neg_conds, chunk_kwargs = mdv.make_window_conditioning(shim, image, act_layout, device)
    chunk_kwargs = _lift_chunk_kwargs(chunk_kwargs, lift_kwargs)
    conds, neg_conds = _with_hires(conds, neg_conds, lift_hi_feats)
    sampler = mdv.make_val_sampler(1e-5)
    noise = [SparseTensor(
        coords=torch.cat([torch.zeros_like(local_coords[k][:, :1]), local_coords[k]], dim=1).int(),
        feats=torch.empty(local_coords[k].shape[0], denoiser.in_channels, device=device))
        for k in active]
    print(f"[slat] {len(active)}/{len(layout)} active windows, "
          f"{sum(n.feats.shape[0] for n in noise)} tokens total")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    C_in = denoiser.in_channels
    rel_cells = [(t * R).round().long().to(device) for t in rel]
    order = sorted(range(len(active)), key=lambda a: act_layout[a][0] ** 2 + act_layout[a][1] ** 2)

    def _keys(world_cells):
        w = world_cells + 8192  # positive shift for a collision-free int key
        return (w[:, 0] * 16384 + w[:, 1]) * 16384 + w[:, 2]

    latents = [None] * len(active)
    comm_keys, comm_feats = [], []   # per committed window, sorted keys
    if prior is not None:
        pk, pf = cross_band_slat_prior(act_layout, [local_coords[k] for k in active],
                                       rel_cells, R, prior, device, _keys)
        if pk is not None:
            comm_keys.append(pk)
            comm_feats.append(pf)
    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
        for step, a in enumerate(order):
            lc = local_coords[active[a]].long().to(device)
            key_i = _keys(lc + rel_cells[a][None, :])
            x0 = torch.zeros(lc.shape[0], C_in, device=device)
            pin = torch.zeros(lc.shape[0], dtype=torch.bool, device=device)
            for kj, fj in zip(comm_keys, comm_feats):
                m = torch.isin(key_i, kj) & ~pin
                if not bool(m.any()):
                    continue
                x0[m] = fj[torch.searchsorted(kj, key_i[m])].float()
                pin |= m
            torch.manual_seed(seed + active[a])
            torch.cuda.manual_seed_all(seed + active[a])
            out = mdv.sample_window(
                sampler, denoiser, noise[a], conds[a], neg_conds[a], **SAMPLER_PARAMS["slat"],
                chunk_kwargs=chunk_kwargs[a], inpaint_x0=x0 if bool(pin.any()) else None,
                inpaint_mask=pin if bool(pin.any()) else None)
            latents[a] = out.replace(out.feats.float())
            # commit only interior tokens (border cells: weakest receptive field)
            interior = ((lc >= SLAT_AR_ERODE) & (lc < R - SLAT_AR_ERODE)).all(dim=1)
            ks, perm = _keys(lc[interior] + rel_cells[a][None, :]).sort()
            comm_keys.append(ks)
            comm_feats.append(latents[a].feats[interior][perm])
            print(f"[slat]   window {step + 1}/{len(order)} (d={act_layout[a][0]:.2f}, "
                  f"l={act_layout[a][1]:.2f}) pinned {int(pin.sum())}/{lc.shape[0]} tokens")
    del comm_keys, comm_feats, denoiser
    torch.cuda.empty_cache()

    tokens = ([(z.coords[:, 1:].detach().cpu(), z.feats.detach().float().cpu()) for z in latents],
              list(act_layout))
    mean = torch.tensor(SLAT_MEAN, device=device).reshape(1, -1)
    std = torch.tensor(SLAT_STD, device=device).reshape(1, -1)
    slats = [z.replace(z.feats.float() * std + mean) for z in latents]
    # overlap merge: a window's uncommitted border ring (the SLAT_AR_ERODE cells
    # never pinned into its neighbours) gets weight 0, so each overlap voxel is
    # decoded from the window it is interior to instead of a mean of two
    # different samples (which shreds the surface along the seam)
    merge_w = [((z.coords[:, 1:] >= SLAT_AR_ERODE) & (z.coords[:, 1:] < R - SLAT_AR_ERODE)).all(dim=1).float()
               for z in latents]
    mesh, _, _, ctx, _ = joint_decode_shape(slat_dec, slats, rel, max_inflated_voxels=max_inflated_voxels,
                                            weights=merge_w)
    del ctx, slat_dec
    torch.cuda.empty_cache()

    # metric -> window-local units: 1 local unit = CHUNK_SIZE canonical m,
    # canonical = metric * scene_scale
    cap_local = FILL_HOLES_M * scene_scale / mdv.CHUNK_SIZE
    nf0 = int(mesh.faces.shape[0])
    try:
        mesh.fill_holes(max_hole_perimeter=cap_local)
        print(f"[mesh] fill_holes({FILL_HOLES_M:g} m): +{int(mesh.faces.shape[0]) - nf0:,} faces")
    except Exception as e:  # noqa: BLE001 — a fill failure must not kill the run
        print(f"[mesh] fill_holes failed ({e}); continuing unfilled")
    return mesh, mdv.window_world_transform(*act_layout[0]), tokens

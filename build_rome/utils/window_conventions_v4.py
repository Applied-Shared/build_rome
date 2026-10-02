"""MDv4 window conventions: outdoor depth band (to 20 m), denser windows,
vertical coverage.

Same cube, same camera, same occupancy thresholds as v3 -- only WHERE windows are
drawn changes, plus how many. Kept as a separate module rather than edited into
``window_conventions.py`` on purpose: that file is imported by the model
(``window_pos_encoder`` normalises its conditioning against DEPTH_RANGE_M) and by
the SAGE-10k exporter, so changing it in place would silently renormalise SAGE
data and invalidate existing checkpoints. Consumers that want to train on both
datasets must take the range per dataset, from the sidecar stamp.

What changed and why:

- ``DEPTH_RANGE_M (0.5, 7.5) -> (0.5, 20.0)``. The v3 band is INDOOR-shaped: it
  was frozen for SAGE-10k, where a room bounds how far anything can be. Outdoor
  scenes have no such bound, and stopping at 7.5 m threw away everything past the
  near foreground of every view.
- ``N_WINDOWS 10 -> 48``. Windows are drawn uniformly over the band, so covering
  5.3x the ground area (266 m2 against 50 m2, integrating the frustum width over
  the band) needs proportionally more of them to keep the same density.
- ``version 3 -> 4`` in the stamp. The stamp is compared byte-exact, so v3
  loaders reject this data outright instead of misreading it, which is the
  intended behaviour.

Everything else is re-exported unchanged, so there is exactly one definition of
the cube, the camera and the occupancy classes.
"""

from build_rome.utils.window_conventions import (  # noqa: F401
    CHUNK_SIZE,
    DEPTH_BIN_M,
    EMPTY_MAX_VOX,
    FOV_DEG,
    FOV_JITTER_DEG,
    LATERAL_MARGIN_M,
    OCCUPANCY_CLASSES,
    SLIVER_MAX_VOX,
    TOPUP_MAX_PER_VIEW,
    TOPUP_MIN_BIN_FRAC,
    VOXEL_RES,
    classify_occupancy,
    frustum_half_width,
    window_dirname,
)

DEPTH_RANGE_M = (0.5, 20.0)
N_WINDOWS = 48

# --- Vertical coverage (new in v4) -----------------------------------------
# v3 anchored EVERY cube to the floor, so the window set covered 0-3 m of height
# and nothing above it: a tree crown at 4 m was simply outside every window, which
# is why carved canopies ended at the cube ceiling. The cube BASE now sits
# ELEV_RANGE_M above the local ground, so the set covers 0-6 m.
#
# The distribution is deliberately NOT uniform. Ground-level windows are the ones
# that carry the floor, the props and most of the scene's structure, and the model
# is asked at inference for windows resting on a floor it estimated -- so they stay
# the common case, with the raised ones as a minority that teach what is overhead.
ELEV_RANGE_M = (0.0, 3.0)  # cube base above the local floor
GROUND_WINDOW_FRAC = 0.75  # share anchored exactly at the floor (elev = 0)

N_DEPTH_BINS = int((DEPTH_RANGE_M[1] - DEPTH_RANGE_M[0]) / DEPTH_BIN_M)

CONVENTIONS = {
    "version": 4,
    "chunk_size_m": CHUNK_SIZE,
    "continuous_offsets": True,
    "depth_range_m": list(DEPTH_RANGE_M),
    "elev_range_m": list(ELEV_RANGE_M),
    "ground_window_frac": GROUND_WINDOW_FRAC,
    "fov_deg": FOV_DEG,
    "fov_jitter_deg": FOV_JITTER_DEG,
    "inference_input": "center_square_crop",
}


def sample_elev(rng) -> float:
    """Cube-base height above the local floor for one window.

    A point mass at 0 plus a uniform tail, rather than a smooth distribution: a
    cube 20 cm off the ground is a worse version of a ground window (it clips the
    surface it should rest on), while one at 1.5-3 m is a genuinely different
    thing. So windows are either ON the floor or clearly above it.
    """
    if rng.random() < GROUND_WINDOW_FRAC:
        return 0.0
    lo = max(ELEV_RANGE_M[0], 0.5 * CHUNK_SIZE)   # clear of the ground window
    return float(rng.uniform(lo, ELEV_RANGE_M[1]))


def depth_bin(depth_m: float) -> int:
    """Ledger depth bin index for a window-center depth (v4 range)."""
    lo, hi = DEPTH_RANGE_M
    assert lo <= depth_m <= hi, f"depth {depth_m} outside {DEPTH_RANGE_M}"
    return min(int((depth_m - lo) / DEPTH_BIN_M), int((hi - lo) / DEPTH_BIN_M) - 1)


def validate_sidecar(meta: dict) -> None:
    """Hard-assert a v4 window sidecar. Reject, never remap."""
    assert meta["conventions"] == CONVENTIONS, (
        f"conventions mismatch: sidecar has {meta['conventions']}, expected {CONVENTIONS}"
    )
    depth, lateral = float(meta["depth_m"]), float(meta["lateral_m"])
    assert DEPTH_RANGE_M[0] <= depth <= DEPTH_RANGE_M[1], (
        f"depth_m {depth} outside {DEPTH_RANGE_M}"
    )
    elev = float(meta["elev_m"])
    assert ELEV_RANGE_M[0] <= elev <= ELEV_RANGE_M[1], (
        f"elev_m {elev} outside {ELEV_RANGE_M}"
    )
    assert abs(lateral) <= frustum_half_width(depth) + LATERAL_MARGIN_M + 1e-6, (
        f"lateral_m {lateral} exceeds w({depth}) + {LATERAL_MARGIN_M} = "
        f"{frustum_half_width(depth) + LATERAL_MARGIN_M:.3f}"
    )
    assert meta["occupancy_class"] in OCCUPANCY_CLASSES, (
        f"unknown occupancy_class {meta['occupancy_class']!r}"
    )
    # z_off_view = cube base relative to the view's fitted floor plane (local
    # terrain correction, clipped to +-3 in the bridge, plus elev). Same field
    # name/meaning as the indoor multistory datasets, so the loader conditions
    # on one field everywhere. Bounds follow from the bridge's clips.
    if "z_off_view" in meta:
        zov = float(meta["z_off_view"])
        assert -3.5 <= zov <= 3.0 + ELEV_RANGE_M[1] + 0.5, (
            f"z_off_view {zov} outside the bridge's local-floor+elev envelope"
        )

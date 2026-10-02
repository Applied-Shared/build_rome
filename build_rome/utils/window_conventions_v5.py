"""MDv5 window conventions: v4 outdoor band + content-adaptive vertical
stacking, for photo-built scenes with tall buildings.

Same cube, same camera, same band and occupancy thresholds as v4 — what
changes is the VERTICAL story:

- v4 drew a 25% minority of raised windows with base ~U(1.5, 3) m, capping
  coverage at ~6 m of height. Photo-scene towers reach 46.6 m (measured over
  3,619 built scenes; per-scene max p50 10.5 m / p99 24.9 m), so v4 would
  flat-top every skyscraper at the cube ceiling.
- v5 anchors ALL sampled windows to the local ground (no random elevation),
  then stacks deterministic layers upward per column: elev = k * CHUNK_SIZE,
  k = 1, 2, ..., emitted only while the layer classifies non-empty, plus ONE
  empty "sky cap" layer above the topmost content so the model learns that
  above-roof space is empty instead of never seeing it. Cost therefore scales
  with content, not with the cap.
- ``ELEV_RANGE_M (0, 3) -> (0, 48)``. This is the ALLOWED range in the stamp
  (16 layers covers the observed 46.6 m max), not a sampling target — empty
  layers above the sky cap are never emitted.
- ``version 4 -> 5`` in the stamp; byte-exact comparison makes v4 loaders
  reject v5 data outright, which is the intended behaviour. Consumers take
  the ranges per dataset from the sidecar stamp, exactly as with v3/v4.

Everything else is re-exported from v4/v3 unchanged, so there is exactly one
definition of the cube, the camera, the band and the occupancy classes.
"""

from build_rome.utils.window_conventions_v4 import (  # noqa: F401
    CHUNK_SIZE,
    DEPTH_BIN_M,
    DEPTH_RANGE_M,
    EMPTY_MAX_VOX,
    FOV_DEG,
    FOV_JITTER_DEG,
    LATERAL_MARGIN_M,
    N_DEPTH_BINS,
    N_WINDOWS,
    OCCUPANCY_CLASSES,
    SLIVER_MAX_VOX,
    TOPUP_MAX_PER_VIEW,
    TOPUP_MIN_BIN_FRAC,
    VOXEL_RES,
    classify_occupancy,
    depth_bin,
    frustum_half_width,
    window_dirname,
)

# Cube-base height above the local floor. Stacked layers sit at exact
# multiples of CHUNK_SIZE; 16 layers = 48 m covers the tallest measured
# object (46.6 m) with margin.
ELEV_RANGE_M = (0.0, 48.0)
STACK_MAX_LAYERS = 16  # highest emitted layer index (elev = k * CHUNK_SIZE)

CONVENTIONS = {
    "version": 5,
    "chunk_size_m": CHUNK_SIZE,
    "continuous_offsets": True,
    "depth_range_m": list(DEPTH_RANGE_M),
    "elev_range_m": list(ELEV_RANGE_M),
    # Layers exist only where content does; one empty sky-cap layer above the
    # topmost occupied layer per column. Replaces v4's ground_window_frac.
    "elev_stacking": "content_adaptive_sky_capped",
    "fov_deg": FOV_DEG,
    "fov_jitter_deg": FOV_JITTER_DEG,
    "inference_input": "center_square_crop",
}


def validate_sidecar(meta: dict) -> None:
    """Hard-assert a v5 window sidecar. Reject, never remap."""
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
    # Same field/meaning as v4: cube base relative to the view's fitted floor
    # plane = local terrain correction (clipped to +-3 in the bridge) + elev.
    if "z_off_view" in meta:
        zov = float(meta["z_off_view"])
        assert -3.5 <= zov <= 3.0 + ELEV_RANGE_M[1] + 0.5, (
            f"z_off_view {zov} outside the bridge's local-floor+elev envelope"
        )

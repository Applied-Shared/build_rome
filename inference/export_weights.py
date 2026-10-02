#!/usr/bin/env python3
"""Export the fine-tuned weights of a training checkpoint as a small delta.

The base TRELLIS.2 weights stay frozen during fine-tuning, so a release only
needs the LoRA adapters and the conditioning branches; inference rebuilds the
model from the public base weights plus this delta (inference/models.py).

    python inference/export_weights.py --ckpt last.ckpt --kind ss --out ss_delta.safetensors
"""
import argparse
import os
import sys

import torch
from safetensors.torch import save_file

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from inference.models import BASE_WEIGHTS, _denoiser_state, _load_hf_state  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="Lightning checkpoint of a fine-tuning run")
    ap.add_argument("--kind", choices=["ss", "slat"], required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sd = _denoiser_state(a.ckpt)
    base = _load_hf_state(BASE_WEIGHTS[a.kind])
    delta = {k: v.contiguous() for k, v in sd.items() if k not in base}
    for k in base:  # the base must be untouched, or a delta cannot reproduce the model
        assert torch.equal(sd[k].to(base[k].dtype), base[k]), f"base weight {k} differs from {BASE_WEIGHTS[a.kind]}"
    save_file(delta, a.out)
    print(f"wrote {a.out}: {len(delta)} tensors, {sum(v.numel() for v in delta.values()) / 1e6:.1f}M parameters")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Text / first-frame / first+last-frame to video+audio with MiniMax-H3 + HyperFlow (8 steps).

    # text to video+audio (up to four GPUs, Ulysses sequence parallel; the script re-launches itself under torchrun)
    hyperflow-h3-fl2va --prompt "A red fox trotting through a snowy pine forest"

    # first frame (canvas follows the image), 5 s
    hyperflow-h3-fl2va --prompt "..." --image first.png

    # first + last frame
    hyperflow-h3-fl2va --prompt "..." --image first.png --last-image last.png

    # one GPU with CPU offload (the official single-device recipe), Sol-Attn on
    hyperflow-h3-fl2va --prompt "..." --image first.png --gpus 1 --sol-attn

    # FlashAttention-3 for the dense attention (diffusers `_flash_3_hub` via `pip install kernels`), any GPU count;
    # other diffusers backend names (`_flash_3`, `flash_4_hub`, `flash_hub`, ...) are accepted with --gpus 1 only
    hyperflow-h3-fl2va --prompt "..." --attention-backend fa3

    # Diffusers' default 50-point baseline (49 NFE) for an A/B on the same seed
    hyperflow-h3-fl2va --prompt "..." --image first.png --baseline --output base.mp4

Everything below `build_pipeline` is the official diffusers recipe; see `common.py` for the three HyperFlow calls.
"""

from __future__ import annotations

import argparse
import sys

from diffusers.utils import load_image

from .common import add_common_args, build_pipeline, run_and_save


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, workflow="fl2va")
    parser.add_argument("--image", default=None, help="first frame (path or URL); omit for text-to-video")
    parser.add_argument("--last-image", default=None, help="last frame (path or URL)")
    args = parser.parse_args()

    # `t2va` and `fl2va` share `transformer/` and the same LoRA; the workflow is picked from the inputs
    # exactly as the official pipeline does (`fl2va` needs a first and/or last frame).
    extra = {}
    if args.image:
        extra["image"] = load_image(args.image)
    if args.last_image:
        extra["last_image"] = load_image(args.last_image)
    workflow = "fl2va" if extra else "t2va"

    pipe = build_pipeline(args, workflow=workflow)
    run_and_save(pipe, args, **extra)
    return 0


if __name__ == "__main__":
    sys.exit(main())

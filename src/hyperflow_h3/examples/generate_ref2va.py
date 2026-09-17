#!/usr/bin/env python3
# Copyright 2026 The HyperFlow authors. Licensed under the Apache License, Version 2.0.
"""Omni-reference to video+audio (`ref2va`) with MiniMax-H3 + HyperFlow (8 steps).

References are passed in order with repeated `--ref`; the order is semantic (it numbers `<Picture 1>`,
`<Video 1>`, `<Audio 1>` in the prompt and advances the rotary clock). The modality is inferred from the
extension, or forced with a prefix: `--ref image:subject.png --ref video:clip.mp4 --ref audio:voice.wav`.

    hyperflow-h3-ref2va \
        --prompt "The character speaks in time with the reference recording, natural lip movement" \
        --ref subject.png --ref clip.mp4 --ref voice.wav --num-frames 124

Runs on up to four GPUs by default (Ulysses sequence parallel); `--gpus 1 --sol-attn` is the single-GPU recipe with
Sol-Attn. The same HyperFlow file is loaded onto `transformer_ref/`; see the README for the quality notes on this
workflow.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from diffusers.modular_pipelines.minimax_h3 import (
    MiniMaxH3AudioReference,
    MiniMaxH3ImageReference,
    MiniMaxH3VideoReference,
)

from .common import add_common_args, build_pipeline, run_and_save

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp"}
VIDEO_EXT = {".mp4", ".mov", ".mkv", ".webm", ".avi"}
AUDIO_EXT = {".wav", ".mp3", ".flac", ".m4a", ".ogg", ".aac"}
REFERENCE_CLASSES = {
    "image": MiniMaxH3ImageReference,
    "video": MiniMaxH3VideoReference,
    "audio": MiniMaxH3AudioReference,
}


def parse_reference(spec: str):
    kind, sep, location = spec.partition(":")
    if not sep or kind not in ("image", "video", "audio") or location.startswith("//"):
        kind, location = None, spec
    if kind is None:
        suffix = Path(location.split("?")[0]).suffix.lower()
        if suffix in IMAGE_EXT:
            kind = "image"
        elif suffix in VIDEO_EXT:
            kind = "video"
        elif suffix in AUDIO_EXT:
            kind = "audio"
        else:
            raise argparse.ArgumentTypeError(f"cannot infer the modality of {spec!r}; prefix image:/video:/audio:")
    return kind, location, REFERENCE_CLASSES[kind]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(parser, workflow="ref2va")
    parser.add_argument("--ref", action="append", required=True, type=parse_reference, help="reference, in order")
    args = parser.parse_args()

    counts = {"image": 0, "video": 0, "audio": 0}
    for kind, _, _ in args.ref:
        counts[kind] += 1
    if counts["image"] > 9 or counts["video"] > 3 or counts["audio"] > 3 or len(args.ref) > 12:
        parser.error("MiniMax-H3 takes up to 9 images, 3 videos, 3 audio clips and 12 references in total")
    if counts["audio"] and not (counts["image"] or counts["video"]):
        parser.error("audio references must be accompanied by an image or video reference")

    # `from_file` decodes path or URL and keeps the container's frame rate / sample rate.
    references = [klass.from_file(location) for _, location, klass in args.ref]

    pipe = build_pipeline(args, workflow="ref2va")
    run_and_save(pipe, args, references=references)
    return 0


if __name__ == "__main__":
    sys.exit(main())

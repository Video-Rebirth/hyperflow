"""Shared fixtures: a tiny random MiniMax-H3 transformer and a HyperFlow weights file written for it.

No real weights are downloaded. The weights file is produced here from the public key format spelled out in the model
card — deliberately *not* through the package's own mapping code, so the tests check the loader against the contract
rather than against itself.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import torch
from diffusers import MiniMaxH3Transformer3DModel
from safetensors.torch import save_file

TINY_CONFIG = dict(
    num_attention_heads=2,
    attention_head_dim=32,
    hidden_size=64,
    num_layers=2,
    num_refiner_layers=1,
    ffn_dim=128,
    in_channels=24,
    audio_in_channels=32,
    patch_size=(1, 2, 2),
    text_dim=48,
    freq_dim=32,
    time_embed_hidden_dim=64,
    time_embed_dim=48,
    rope_freq_dim=4,  # rotary dim = 3 axes * 2 * rope_freq_dim must fit in attention_head_dim
)

RANK = 4
ALPHA = 4
GATE = 0.25
SIGMAS = [1.0, 0.931506, 0.839236, 0.703462, 0.5, 0.296538, 0.160764, 0.068494, 0.0]

# Modules the adapter targets, in the public naming of the weights file (relative to `transformer.`).
BLOCK_TARGETS = ("attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "ff.net.0.proj", "ff.net.2")


def public_target_modules(config: dict) -> list[str]:
    targets = []
    for i in range(config["num_layers"]):
        targets += [f"transformer_blocks.{i}.{name}" for name in BLOCK_TARGETS]
    for j in range(config["num_refiner_layers"]):
        targets += [f"token_refiner.refiner_blocks.{j}.{name}" for name in BLOCK_TARGETS]
    targets += ["time_embedder.linear_1", "time_embedder.linear_2"]
    targets += ["endpoint_time_embedder.linear_1", "endpoint_time_embedder.linear_2"]
    return targets


def make_transformer(seed: int = 0) -> MiniMaxH3Transformer3DModel:
    torch.manual_seed(seed)
    model = MiniMaxH3Transformer3DModel(**TINY_CONFIG)
    model.eval()
    return model


def metadata() -> dict[str, str]:
    return {
        "hyperflow": "true",
        "hyperflow_version": "0.0.0-test",
        "hyperflow_gate": str(GATE),
        "hyperflow_endpoint": "next_sigma",
        "hyperflow_sigmas": json.dumps(SIGMAS),
        "hyperflow_video_shift": "12.0",
        "hyperflow_audio_shift": "3.0",
        "base_model": "tests/tiny-h3",
        "base_model_revision": "0",
        "trained_on_subfolder": "transformer",
        "compatible_subfolders": json.dumps(["transformer", "transformer_ref"]),
        "tasks": json.dumps(["fl2va", "ref2va"]),
        "lora_rank": str(RANK),
        "lora_alpha": str(ALPHA),
        "lora_targets": json.dumps(list(BLOCK_TARGETS) + ["time_embedder", "endpoint_time_embedder"]),
    }


def write_weights(model: MiniMaxH3Transformer3DModel, path: Path, seed: int = 1) -> dict[str, torch.Tensor]:
    """Write random LoRA A/B for every target, shaped from the model's own Linear layers."""
    generator = torch.Generator().manual_seed(seed)
    linears = {name: module for name, module in model.named_modules() if isinstance(module, torch.nn.Linear)}
    tensors: dict[str, torch.Tensor] = {}
    for target in public_target_modules(TINY_CONFIG):
        # `endpoint_time_embedder` mirrors `time_embedder`; its shapes come from the same Linear layers.
        source = target.replace("endpoint_time_embedder.", "time_embedder.")
        linear = linears[source]
        a = torch.randn(RANK, linear.in_features, generator=generator) * 0.02
        b = torch.randn(linear.out_features, RANK, generator=generator) * 0.02
        dtype = torch.float32 if "time_embedder" in target else torch.bfloat16
        tensors[f"transformer.{target}.lora_A.weight"] = a.to(dtype)
        tensors[f"transformer.{target}.lora_B.weight"] = b.to(dtype)
    save_file(tensors, str(path), metadata=metadata())
    return tensors


@pytest.fixture
def tiny_transformer() -> MiniMaxH3Transformer3DModel:
    return make_transformer()


@pytest.fixture
def weights_file(tmp_path: Path, tiny_transformer) -> Path:
    path = tmp_path / "hyperflow_test.safetensors"
    write_weights(tiny_transformer, path)
    return path


def packed_inputs(
    model: MiniMaxH3Transformer3DModel,
    *,
    num_text=3,
    num_condition_video=2,
    num_video=6,
    num_condition_audio=0,
    num_audio=4,
    seed=0,
):
    """A small packed sequence: `[text | audio (condition first) | video (condition first)]`."""
    torch.manual_seed(seed)
    config = model.config
    video_patch_dim = config.in_channels * config.patch_size[0] * config.patch_size[1] * config.patch_size[2]
    total_video = num_condition_video + num_video
    total_audio = num_condition_audio + num_audio
    sequence_length = num_text + total_audio + total_video
    text_indices = torch.arange(0, num_text)
    audio_indices = torch.arange(num_text, num_text + total_audio)
    video_indices = torch.arange(num_text + total_audio, sequence_length)
    token_tags = torch.empty(sequence_length, dtype=torch.long)
    token_tags[text_indices] = 1
    token_tags[audio_indices] = 2
    token_tags[video_indices] = 0
    position_ids = torch.rand(sequence_length, 3, dtype=torch.float64) * 8
    return dict(
        latents=torch.randn(total_video, video_patch_dim),
        audio_latents=torch.randn(total_audio, config.audio_in_channels),
        prompt_embeds=torch.randn(1, num_text, config.text_dim),
        token_tags=token_tags,
        position_ids=position_ids,
        video_indices=video_indices,
        audio_indices=audio_indices,
        text_indices=text_indices,
        num_condition_video_rows=num_condition_video,
        num_condition_audio_rows=num_condition_audio,
    )

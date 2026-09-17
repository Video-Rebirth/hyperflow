import inspect
import types

import pytest
import torch
from conftest import make_transformer, packed_inputs
from diffusers.models.transformers.transformer_minimax_h3 import MiniMaxH3AttnProcessor

from hyperflow_h3 import sol_attn as sol
from hyperflow_h3.blocks import _current_step
from hyperflow_h3.sol_attn import (
    HyperFlowSolAttnProcessor,
    SolAttnRecipe,
    disable_sol_attention,
    enable_sol_attention,
)


def _forward(model, inputs, timestep=0.3):
    with torch.no_grad():
        return model(
            hidden_states=inputs["latents"][None],
            audio_hidden_states=inputs["audio_latents"][None],
            encoder_hidden_states=inputs["prompt_embeds"],
            timestep=torch.tensor([timestep]),
            timestep_indices=torch.zeros(inputs["token_tags"].numel(), dtype=torch.long),
            token_tags=inputs["token_tags"],
            position_ids=inputs["position_ids"],
            video_indices=inputs["video_indices"],
            audio_indices=inputs["audio_indices"],
            text_indices=inputs["text_indices"],
            return_dict=False,
        )


def test_recipe_validation():
    SolAttnRecipe()
    with pytest.raises(ValueError):
        SolAttnRecipe(thresh_type="mean")
    with pytest.raises(ValueError):
        SolAttnRecipe(kv_splits=3)
    with pytest.raises(ValueError):
        SolAttnRecipe(dense_steps=-1)


def test_use_sparse_follows_step_and_layer_rules():
    recipe = SolAttnRecipe(dense_steps=2, dense_layers=(0, 1))
    assert not HyperFlowSolAttnProcessor(5, recipe).use_sparse(None)  # outside a HyperFlow loop
    assert not HyperFlowSolAttnProcessor(5, recipe).use_sparse(0)
    assert not HyperFlowSolAttnProcessor(5, recipe).use_sparse(1)
    assert HyperFlowSolAttnProcessor(5, recipe).use_sparse(2)
    assert not HyperFlowSolAttnProcessor(0, recipe).use_sparse(7)
    assert not HyperFlowSolAttnProcessor(1, recipe).use_sparse(7)
    assert HyperFlowSolAttnProcessor(2, recipe).use_sparse(7)


def test_enable_installs_processors_on_dit_blocks_only_and_disable_restores():
    model = make_transformer()
    model.transformer_blocks[0].attn.processor._attention_backend = "native"
    recipe = enable_sol_attention(model, require_kernel=False)
    assert recipe == SolAttnRecipe()
    for index, block in enumerate(model.transformer_blocks):
        assert isinstance(block.attn.processor, HyperFlowSolAttnProcessor)
        assert block.attn.processor.layer_index == index
    assert model.transformer_blocks[0].attn.processor._attention_backend == "native"
    for block in model.token_refiner.refiner_blocks:
        assert type(block.attn.processor) is MiniMaxH3AttnProcessor

    disable_sol_attention(model)
    for block in model.transformer_blocks:
        assert type(block.attn.processor) is MiniMaxH3AttnProcessor
    assert model.transformer_blocks[0].attn.processor._attention_backend == "native"


def test_dense_path_is_bit_identical_to_the_official_processor():
    reference = make_transformer()
    patched = make_transformer()
    inputs = packed_inputs(reference)
    enable_sol_attention(patched, require_kernel=False)
    expected = _forward(reference, inputs)
    # Dense at step 0 (< dense_steps) and, kernel or not, outside a loop.
    actual = _forward(patched, inputs)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1], expected[1], rtol=0, atol=0)


def test_missing_kernel_falls_back_to_dense_with_one_warning(monkeypatch, caplog):
    monkeypatch.setattr(sol, "_sol_available", False)
    monkeypatch.setattr(sol, "_warned_missing", False)
    model = make_transformer()
    inputs = packed_inputs(model)
    enable_sol_attention(model, require_kernel=False, dense_steps=0, dense_layers=())
    token = _current_step.set(5)  # a sparse step by the recipe
    try:
        with caplog.at_level("WARNING", logger="hyperflow_h3.sol_attn"):
            _forward(model, inputs)
            _forward(model, inputs)
    finally:
        _current_step.reset(token)
    assert sum("not installed" in record.message for record in caplog.records) == 1
    processors = [block.attn.processor for block in model.transformer_blocks]
    assert all(p.calls_sparse == 0 and p.calls_dense == 2 for p in processors)


def test_require_kernel_raises_when_absent(monkeypatch):
    monkeypatch.setattr(sol, "_sol_available", False)
    with pytest.raises(RuntimeError, match="not installed"):
        enable_sol_attention(make_transformer(), require_kernel=True)


def test_sparse_path_calls_the_kernel_with_the_recipe(monkeypatch):
    """Substitute a fake kernel to check layout, dtype checks and kwargs plumbing without CUDA."""
    seen = {}

    def fake_sol_attn(q, k, v, *, tau, thresh_type, sink_tokens, sink_start, kv_splits):
        seen.update(
            tau=tau,
            thresh_type=thresh_type,
            sink_tokens=sink_tokens,
            sink_start=sink_start,
            kv_splits=kv_splits,
            shape=tuple(q.shape),
        )
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    monkeypatch.setattr(sol, "_sol_available", True)
    monkeypatch.setattr(sol, "_sol_attn", fake_sol_attn)
    monkeypatch.setattr(sol, "_sol_params", frozenset(inspect.signature(fake_sol_attn).parameters))

    recipe = SolAttnRecipe(dense_steps=0, dense_layers=(), tau=0.9, thresh_type="exact", kv_splits=2)
    attn = torch.nn.Module()
    attn.heads, attn.fused_projections = 2, False
    attn.to_q = attn.to_k = attn.to_v = torch.nn.Linear(64, 256, bias=False)
    attn.norm_q = attn.norm_k = torch.nn.Identity()
    attn.to_out = torch.nn.ModuleList([torch.nn.Linear(256, 64, bias=False), torch.nn.Identity()])
    processor = HyperFlowSolAttnProcessor(0, recipe)
    hidden = torch.randn(1, 10, 64, dtype=torch.bfloat16)
    attn.to(torch.bfloat16)
    token = _current_step.set(3)
    try:
        out = processor(attn, hidden, rotary_emb=None)
    finally:
        _current_step.reset(token)
    assert out.shape == (1, 10, 64)
    assert seen["shape"] == (1, 10, 2, 128)  # [batch, seq, heads, head_dim]
    assert seen["tau"] == 0.9 and seen["thresh_type"] == "exact"
    assert seen["kv_splits"] == 2 and seen["sink_start"] is None  # the kernel's default: sink at the suffix
    assert processor.calls_sparse == 1


class _FakeParallelConfig:
    """What `enable_parallelism` leaves on every processor, reduced to the fields Sol-Attn looks at."""

    def __init__(self, ulysses_degree: int = 1, ring_degree: int = 1):
        self.context_parallel_config = types.SimpleNamespace(ulysses_degree=ulysses_degree, ring_degree=ring_degree)


def test_ring_context_parallel_is_refused_at_enable():
    model = make_transformer()
    for block in model.transformer_blocks:
        block.attn.processor._parallel_config = _FakeParallelConfig(ring_degree=2)
    with pytest.raises(RuntimeError, match="Ulysses context parallel only"):
        enable_sol_attention(model, require_kernel=False)
    assert all(type(block.attn.processor) is MiniMaxH3AttnProcessor for block in model.transformer_blocks)


def test_sparse_path_under_ulysses_runs_the_kernel_inside_the_exchange(monkeypatch):
    """With a parallel config the sparse call goes through diffusers' Ulysses template: the kernel must see the
    exchanged (full-sequence, head-shard) tensors, and its kwargs must be computed on those, not on the local shard."""
    exchanged = {}

    def fake_sol_attn(q, k, v, *, tau, thresh_type, sink_tokens, sink_start, kv_splits):
        exchanged["kernel_shape"] = tuple(q.shape)
        return torch.nn.functional.scaled_dot_product_attention(
            q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        ).transpose(1, 2)

    def fake_template(
        query,
        key,
        value,
        attn_mask,
        dropout_p,
        is_causal,
        scale,
        enable_gqa,
        return_lse,
        *,
        forward_op,
        backward_op,
        _parallel_config,
    ):
        # Stand in for the all-to-all: a world of one rank hands the full sequence straight to `forward_op` with the
        # call signature diffusers' TemplatedUlysses*Attention uses.
        exchanged["template_called"] = True
        assert attn_mask is None and not return_lse and _parallel_config is parallel_config
        return forward_op(
            None,
            query,
            key,
            value,
            attn_mask,
            dropout_p,
            is_causal,
            scale,
            enable_gqa,
            return_lse,
            _save_ctx=False,
            _parallel_config=_parallel_config,
        )

    import diffusers.models.attention_dispatch as dispatch

    monkeypatch.setattr(sol, "_sol_available", True)
    monkeypatch.setattr(sol, "_sol_attn", fake_sol_attn)
    monkeypatch.setattr(sol, "_sol_params", frozenset(inspect.signature(fake_sol_attn).parameters))
    monkeypatch.setattr(dispatch, "_templated_context_parallel_attention", fake_template)

    parallel_config = _FakeParallelConfig(ulysses_degree=4)
    processor = HyperFlowSolAttnProcessor(0, SolAttnRecipe(dense_steps=0, dense_layers=()))
    processor._parallel_config = parallel_config
    attn = torch.nn.Module()
    attn.heads, attn.fused_projections = 2, False
    attn.to_q = attn.to_k = attn.to_v = torch.nn.Linear(64, 256, bias=False)
    attn.norm_q = attn.norm_k = torch.nn.Identity()
    attn.to_out = torch.nn.ModuleList([torch.nn.Linear(256, 64, bias=False), torch.nn.Identity()])
    attn.to(torch.bfloat16)
    token = _current_step.set(3)
    try:
        out = processor(attn, torch.randn(1, 10, 64, dtype=torch.bfloat16), rotary_emb=None)
    finally:
        _current_step.reset(token)
    assert out.shape == (1, 10, 64)
    assert exchanged["template_called"] and exchanged["kernel_shape"] == (1, 10, 2, 128)
    assert processor.calls_sparse == 1

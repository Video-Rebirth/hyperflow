import copy

import pytest
import torch

from hyperflow_h3.embedder import TwoTimeEmbedder


def test_wrap_replaces_time_embedder_and_keeps_linear_1(tiny_transformer):
    original = tiny_transformer.time_embedder
    wrapper = TwoTimeEmbedder.wrap(tiny_transformer, gate=0.25)
    assert tiny_transformer.time_embedder is wrapper
    assert wrapper.base is original
    assert wrapper.linear_1 is original.linear_1
    assert wrapper.endpoint is not original
    # Idempotent.
    assert TwoTimeEmbedder.wrap(tiny_transformer, gate=0.25) is wrapper


def test_forward_is_gated_blend_of_the_two_embeddings(tiny_transformer):
    base_copy = copy.deepcopy(tiny_transformer.time_embedder)
    time_proj = copy.deepcopy(tiny_transformer.time_proj)
    wrapper = TwoTimeEmbedder.wrap(tiny_transformer, gate=0.25)
    # Give the endpoint branch different weights so the blend is observable.
    with torch.no_grad():
        for parameter in wrapper.endpoint.parameters():
            parameter.add_(0.1)
    t = torch.tensor([0.2, 0.5, 0.999])
    r = torch.tensor([0.4, 0.7, 0.999])
    t_proj = time_proj(t)
    expected = base_copy(t_proj) + 0.25 * (wrapper.endpoint(time_proj(r)) - base_copy(t_proj))
    with wrapper.endpoint_context(r):
        out = wrapper(t_proj)
    torch.testing.assert_close(out, expected)
    # The context clears the endpoints again.
    with pytest.raises(RuntimeError, match="endpoint timesteps"):
        wrapper(t_proj)


def test_gate_zero_and_passthrough_equal_the_base(tiny_transformer):
    base_copy = copy.deepcopy(tiny_transformer.time_embedder)
    time_proj = copy.deepcopy(tiny_transformer.time_proj)
    wrapper = TwoTimeEmbedder.wrap(tiny_transformer, gate=0.0)
    t_proj = time_proj(torch.tensor([0.3, 0.6]))
    with wrapper.endpoint_context(torch.tensor([0.5, 0.9])):
        torch.testing.assert_close(wrapper(t_proj), base_copy(t_proj), rtol=0, atol=0)
    wrapper.passthrough = True
    torch.testing.assert_close(wrapper(t_proj), base_copy(t_proj), rtol=0, atol=0)


def test_endpoint_count_must_match_timesteps(tiny_transformer):
    wrapper = TwoTimeEmbedder.wrap(tiny_transformer, gate=0.25)
    t_proj = wrapper.time_proj(torch.tensor([0.3, 0.6]))
    with pytest.raises(ValueError, match="exactly one endpoint"), wrapper.endpoint_context(torch.tensor([0.5])):
        wrapper(t_proj)

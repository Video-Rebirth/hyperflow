import functools
from types import SimpleNamespace

import pytest
import torch
from conftest import make_transformer, packed_inputs
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines import PipelineState, SequentialPipelineBlocks
from diffusers.modular_pipelines.minimax_h3 import MiniMaxH3Blocks
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep
from diffusers.modular_pipelines.minimax_h3.denoise import MiniMaxH3LoopDenoiser, MiniMaxH3LoopSchedulerStep

from hyperflow_h3.blocks import (
    HyperFlowLoopDenoiser,
    HyperFlowSetTimestepsStep,
    configure_hyperflow_blocks,
    hyperflow_blocks,
    swap_hyperflow_blocks,
)
from hyperflow_h3.lora import disable_hyperflow, enable_hyperflow, load_hyperflow_lora, read_metadata
from hyperflow_h3.schedule import DEFAULT_SIGMAS_8STEP, shift_sigmas


def _leaf_blocks(blocks):
    """Flatten every leaf block (loop sub-blocks included) as `(dotted_name, block)`."""
    out = []

    def visit(container, prefix):
        for name, block in container.sub_blocks.items():
            full = f"{prefix}{name}"
            if getattr(block, "sub_blocks", None):
                visit(block, full + ".")
            else:
                out.append((full, block))

    visit(blocks, "")
    return out


@pytest.mark.parametrize("workflow", ["t2va", "fl2va", "ref2va"])
def test_hyperflow_blocks_swap_exactly_two_leaves(workflow):
    official = MiniMaxH3Blocks().get_workflow(workflow)
    ours = hyperflow_blocks(workflow)
    official_leaves = _leaf_blocks(official)
    our_leaves = _leaf_blocks(ours)
    assert [name for name, _ in official_leaves] == [name for name, _ in our_leaves]
    changed = [name for (name, a), (_, b) in zip(official_leaves, our_leaves, strict=True) if type(a) is not type(b)]
    assert sorted(changed) == ["denoise.denoise.denoiser", "denoise.set_timesteps"]
    denoiser = dict(our_leaves)["denoise.denoise.denoiser"]
    assert isinstance(denoiser, HyperFlowLoopDenoiser)
    assert denoiser.transformer_name == ("transformer_ref" if workflow == "ref2va" else "transformer")
    # Same components are expected: nothing new to load, nothing dropped.
    assert [c.name for c in official.expected_components] == [c.name for c in ours.expected_components]
    # Idempotent.
    swap_hyperflow_blocks(ours)
    assert [type(b) for _, b in _leaf_blocks(ours)] == [type(b) for _, b in our_leaves]


def test_unpruned_blocks_get_every_branch_swapped():
    blocks = swap_hyperflow_blocks(MiniMaxH3Blocks())
    leaves = _leaf_blocks(blocks)
    assert not any(type(b) is MiniMaxH3SetTimestepsStep for _, b in leaves)
    assert not any(type(b) is MiniMaxH3LoopDenoiser for _, b in leaves)
    assert sum(isinstance(b, HyperFlowSetTimestepsStep) for _, b in leaves) == 3
    assert sum(isinstance(b, HyperFlowLoopDenoiser) for _, b in leaves) == 3


def test_swap_refuses_blocks_without_minimax_h3():
    with pytest.raises(ValueError, match="No MiniMax-H3"):
        swap_hyperflow_blocks(SequentialPipelineBlocks())


def test_unknown_workflow():
    with pytest.raises(ValueError, match="Unknown workflow"):
        hyperflow_blocks("i2v")


def test_num_inference_steps_is_optional():
    block = HyperFlowSetTimestepsStep()
    param = next(p for p in block.inputs if p.name == "num_inference_steps")
    assert param.required is False and param.default is None
    assert "num_inference_steps" not in block.required_inputs
    # The whole fl2va workflow no longer requires it either.
    workflow = hyperflow_blocks("fl2va")
    assert "num_inference_steps" not in workflow.required_inputs


class FakeComponents:
    """Just enough of `MiniMaxH3ModularPipeline` for the two HyperFlow blocks."""

    def __init__(self, transformer, video_shift=12.0, audio_shift=3.0):
        self.transformer = transformer
        self.scheduler = MiniMaxH3Scheduler(shift=video_shift)
        self.audio_scheduler = MiniMaxH3Scheduler(shift=audio_shift)
        self.keyframe_noise_aug = 0.999

    @property
    def _execution_device(self):
        return torch.device("cpu")


def _state_from_inputs(inputs):
    state = PipelineState()
    for name in ("latents", "audio_latents", "prompt_embeds", "num_condition_video_rows", "num_condition_audio_rows"):
        state.set(name, inputs[name])
    for name in ("token_tags", "position_ids", "video_indices", "audio_indices", "text_indices"):
        state.set(name, inputs[name], kwargs_type="denoiser_input_fields")
    return state


def test_set_timesteps_builds_eight_step_plan_with_endpoints(tiny_transformer):
    components = FakeComponents(tiny_transformer)
    inputs = packed_inputs(tiny_transformer, num_condition_audio=1)
    state = _state_from_inputs(inputs)
    HyperFlowSetTimestepsStep()(components, state)

    video_sigmas = shift_sigmas(DEFAULT_SIGMAS_8STEP, 12.0)
    audio_sigmas = shift_sigmas(DEFAULT_SIGMAS_8STEP, 3.0)
    torch.testing.assert_close(state.get("timesteps"), 1 - video_sigmas[:-1], rtol=0, atol=0)
    torch.testing.assert_close(state.get("audio_timesteps"), 1 - audio_sigmas[:-1], rtol=0, atol=0)
    plan = state.get("row_timestep_plan")
    assert len(plan) == 8
    n_cond_v, n_cond_a = inputs["num_condition_video_rows"], inputs["num_condition_audio_rows"]
    gen_video, cond_video = inputs["video_indices"][n_cond_v:], inputs["video_indices"][:n_cond_v]
    gen_audio, cond_audio = inputs["audio_indices"][n_cond_a:], inputs["audio_indices"][:n_cond_a]
    for i, (timestep, endpoint, indices) in enumerate(plan):
        assert timestep.shape == endpoint.shape and indices.shape == inputs["token_tags"].shape
        rows_t, rows_r = timestep[indices], endpoint[indices]
        t_v, r_v = (1 - video_sigmas[i]).item(), (1 - video_sigmas[i + 1]).item()
        t_a, r_a = (1 - audio_sigmas[i]).item(), (1 - audio_sigmas[i + 1]).item()
        assert torch.all(rows_t[gen_video] == t_v) and torch.all(rows_r[gen_video] == r_v)
        assert torch.all(rows_t[gen_audio] == t_a) and torch.all(rows_r[gen_audio] == r_a)
        pinned = max(t_v, 0.999)
        assert torch.all(rows_t[cond_video] == pinned) and torch.all(rows_r[cond_video] == pinned)
        assert torch.all(rows_t[cond_audio] == 1.0) and torch.all(rows_r[cond_audio] == 1.0)
    # The last step lands on clean for the generated rows.
    assert plan[-1][1].max().item() == 1.0


def test_set_timesteps_rejects_a_foreign_step_count(tiny_transformer):
    components = FakeComponents(tiny_transformer)
    state = _state_from_inputs(packed_inputs(tiny_transformer))
    state.set("num_inference_steps", 50)
    with pytest.raises(ValueError, match="fixed 8-step grid"):
        HyperFlowSetTimestepsStep()(components, state)
    state.set("num_inference_steps", 8)
    HyperFlowSetTimestepsStep()(components, state)


def test_disable_hyperflow_delegates_both_blocks_to_the_base_model(tiny_transformer, weights_file):
    base = make_transformer()
    inputs = packed_inputs(base, num_condition_audio=1)
    load_hyperflow_lora(tiny_transformer, weights_file)
    pipeline = SimpleNamespace(components={"transformer": tiny_transformer})
    disable_hyperflow(pipeline)

    # HyperFlow made this input optional; passthrough restores Diffusers' generic 50-point default.
    default_state = _state_from_inputs(
        {name: value.clone() if torch.is_tensor(value) else value for name, value in inputs.items()}
    )
    HyperFlowSetTimestepsStep()(FakeComponents(tiny_transformer), default_state)
    assert len(default_state.get("timesteps")) == 49
    assert all(len(entry) == 2 for entry in default_state.get("row_timestep_plan"))

    # With an explicit short schedule, the disabled pipeline is bit-identical to the unpatched base blocks.
    base_state = _state_from_inputs(
        {name: value.clone() if torch.is_tensor(value) else value for name, value in inputs.items()}
    )
    disabled_state = _state_from_inputs(
        {name: value.clone() if torch.is_tensor(value) else value for name, value in inputs.items()}
    )
    base_state.set("num_inference_steps", 3)
    disabled_state.set("num_inference_steps", 3)
    base_components, disabled_components = FakeComponents(base), FakeComponents(tiny_transformer)
    MiniMaxH3SetTimestepsStep()(base_components, base_state)
    HyperFlowSetTimestepsStep()(disabled_components, disabled_state)

    base_loop = MiniMaxH3Blocks().get_workflow("fl2va").sub_blocks["denoise.denoise"]
    disabled_loop = hyperflow_blocks("fl2va").sub_blocks["denoise.denoise"]
    base_loop(base_components, base_state)
    disabled_loop(disabled_components, disabled_state)
    torch.testing.assert_close(disabled_state.get("latents"), base_state.get("latents"), rtol=0, atol=0)
    torch.testing.assert_close(disabled_state.get("audio_latents"), base_state.get("audio_latents"), rtol=0, atol=0)

    enable_hyperflow(pipeline)
    enabled_state = _state_from_inputs(inputs)
    HyperFlowSetTimestepsStep()(disabled_components, enabled_state)
    assert len(enabled_state.get("row_timestep_plan")) == 8
    assert all(len(entry) == 3 for entry in enabled_state.get("row_timestep_plan"))


def test_set_timesteps_warns_once_on_foreign_shifts(tiny_transformer, caplog):
    components = FakeComponents(tiny_transformer, video_shift=5.0)
    block = HyperFlowSetTimestepsStep()
    with caplog.at_level("WARNING", logger="hyperflow_h3.blocks"):
        block(components, _state_from_inputs(packed_inputs(tiny_transformer)))
        block(components, _state_from_inputs(packed_inputs(tiny_transformer)))
    assert sum("scheduler shifts" in record.message for record in caplog.records) == 1


def test_configure_from_metadata_keeps_user_grid(weights_file):
    metadata = read_metadata(weights_file)
    default_blocks = hyperflow_blocks("fl2va")
    assert configure_hyperflow_blocks(default_blocks, metadata) == 1
    step = dict(_leaf_blocks(default_blocks))["denoise.set_timesteps"]
    assert step.sigmas_source == "metadata"
    torch.testing.assert_close(step.sigmas, torch.tensor(metadata.sigmas))

    user_blocks = hyperflow_blocks("fl2va", sigmas=[1.0, 0.5, 0.0])
    configure_hyperflow_blocks(user_blocks, metadata)
    step = dict(_leaf_blocks(user_blocks))["denoise.set_timesteps"]
    assert step.sigmas_source == "user" and step.num_steps == 2

    with pytest.raises(ValueError, match="official MiniMax-H3 blocks"):
        configure_hyperflow_blocks(MiniMaxH3Blocks().get_workflow("fl2va"), metadata)


def test_denoiser_refuses_a_transformer_without_hyperflow(tiny_transformer):
    components = FakeComponents(tiny_transformer)
    state = _state_from_inputs(packed_inputs(tiny_transformer))
    HyperFlowSetTimestepsStep()(components, state)
    loop = hyperflow_blocks("t2va").sub_blocks["denoise.denoise"]
    with pytest.raises(RuntimeError, match="no HyperFlow LoRA loaded"):
        loop(components, state)


def test_full_loop_runs_eight_forwards_and_only_moves_generated_rows(tiny_transformer, weights_file):
    load_hyperflow_lora(tiny_transformer, weights_file)
    components = FakeComponents(tiny_transformer)
    inputs = packed_inputs(tiny_transformer, num_condition_audio=1)
    state = _state_from_inputs(inputs)
    before_video = inputs["latents"].clone()
    before_audio = inputs["audio_latents"].clone()

    HyperFlowSetTimestepsStep()(components, state)
    loop = hyperflow_blocks("fl2va").sub_blocks["denoise.denoise"]
    assert [type(b) for b in loop.sub_blocks.values()] == [HyperFlowLoopDenoiser, MiniMaxH3LoopSchedulerStep]

    calls = []
    forward = tiny_transformer.forward

    @functools.wraps(forward)  # the denoiser filters kwargs against `inspect.signature(transformer.forward)`
    def counting_forward(*args, **kwargs):
        calls.append((kwargs["timestep"].clone(), tiny_transformer.time_embedder._endpoint_timesteps.clone()))
        return forward(*args, **kwargs)

    tiny_transformer.forward = counting_forward
    loop(components, state)

    assert len(calls) == 8
    for timestep, endpoint in calls:
        assert timestep.shape == endpoint.shape
    # Endpoints were cleared after the loop.
    assert tiny_transformer.time_embedder._endpoint_timesteps is None

    latents, audio_latents = state.get("latents"), state.get("audio_latents")
    cond_v, cond_a = inputs["num_condition_video_rows"], inputs["num_condition_audio_rows"]
    torch.testing.assert_close(latents[:cond_v], before_video[:cond_v], rtol=0, atol=0)
    torch.testing.assert_close(audio_latents[:cond_a], before_audio[:cond_a], rtol=0, atol=0)
    assert not torch.allclose(latents[cond_v:], before_video[cond_v:])
    assert not torch.allclose(audio_latents[cond_a:], before_audio[cond_a:])
    assert torch.isfinite(latents).all() and torch.isfinite(audio_latents).all()

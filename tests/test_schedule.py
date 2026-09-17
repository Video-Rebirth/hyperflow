import pytest
import torch
from diffusers import MiniMaxH3Scheduler
from diffusers.modular_pipelines.minimax_h3.before_denoise import MiniMaxH3SetTimestepsStep

from hyperflow_h3.schedule import (
    DEFAULT_SIGMAS_8STEP,
    build_row_time_pairs,
    endpoints_from_sigmas,
    shift_sigmas,
    validate_sigmas,
)


def test_default_grid_is_valid_and_has_eight_steps():
    grid = validate_sigmas(DEFAULT_SIGMAS_8STEP)
    assert grid.numel() == 9
    assert endpoints_from_sigmas(grid).numel() == 8


@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_shift_matches_official_scheduler_formula(shift):
    """The scheduler shifts `linspace(1, 0, n)`; feeding it the same base grid must give our shifted grid."""
    base = torch.linspace(1.0, 0.0, 9)
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(num_inference_steps=9)
    ours = shift_sigmas(base, shift)
    torch.testing.assert_close(ours, scheduler.sigmas, rtol=0, atol=0)


@pytest.mark.parametrize("shift", [12.0, 3.0])
def test_diffusers_default_fifty_sigma_points_mean_forty_nine_nfe(shift):
    scheduler = MiniMaxH3Scheduler(shift=shift)
    scheduler.set_timesteps(num_inference_steps=50)

    assert scheduler.sigmas.numel() == 50  # terminal zero included
    assert scheduler.timesteps.numel() == scheduler.num_inference_steps == 49


def test_shifted_default_grid_is_accepted_verbatim_by_the_scheduler():
    scheduler = MiniMaxH3Scheduler(shift=12.0)
    shifted = shift_sigmas(DEFAULT_SIGMAS_8STEP, 12.0)
    scheduler.set_timesteps(sigmas=shifted)
    torch.testing.assert_close(scheduler.sigmas, shifted, rtol=0, atol=0)
    torch.testing.assert_close(scheduler.timesteps, 1.0 - shifted[:-1], rtol=0, atol=0)
    assert scheduler.num_inference_steps == 8


def test_endpoint_is_next_timestep():
    shifted = shift_sigmas(DEFAULT_SIGMAS_8STEP, 12.0)
    timesteps = 1.0 - shifted[:-1]
    endpoints = endpoints_from_sigmas(shifted)
    # r_i == t_{i+1} for every step but the last, whose endpoint is the clean t = 1.
    torch.testing.assert_close(endpoints[:-1], timesteps[1:], rtol=0, atol=0)
    assert endpoints[-1].item() == 1.0


@pytest.mark.parametrize("num_condition_audio", [0, 3])
def test_row_pairs_timestep_half_matches_official_plan(num_condition_audio):
    video_indices = torch.arange(10, 22)
    audio_indices = torch.arange(3, 10)
    text = 3
    kwargs = dict(
        video_indices=video_indices,
        audio_indices=audio_indices,
        num_condition_video_rows=2,
        num_condition_audio_rows=num_condition_audio,
        num_text_tokens=text,
    )
    t_video, t_audio, cond_video = 0.3, 0.45, 0.999
    official_t, official_idx = MiniMaxH3SetTimestepsStep.build_row_timesteps(
        video_timestep=t_video,
        audio_timestep=t_audio,
        condition_video_timestep=cond_video,
        condition_audio_timestep=1.0,
        **kwargs,
    )
    ours_t, ours_r, ours_idx = build_row_time_pairs(
        video_timestep=t_video,
        video_endpoint=0.5,
        audio_timestep=t_audio,
        audio_endpoint=0.7,
        condition_video_timestep=cond_video,
        **kwargs,
    )
    # Same per-row timestep as the official plan (pairs may split a timestep into several entries).
    torch.testing.assert_close(ours_t[ours_idx], official_t[official_idx], rtol=0, atol=0)
    rows_t = ours_t[ours_idx]
    rows_r = ours_r[ours_idx]
    # Generated video rows and text rows: (t_video, r_video); conditioning rows: r == t.
    assert torch.all(rows_r[video_indices[2:]] == 0.5) and torch.all(rows_t[video_indices[2:]] == t_video)
    assert torch.all(rows_r[:text] == 0.5) and torch.all(rows_t[:text] == t_video)
    assert torch.all(rows_t[video_indices[:2]] == cond_video) and torch.all(rows_r[video_indices[:2]] == cond_video)
    # Generated audio rows: (t_audio, r_audio); ref2va audio references: (1, 1).
    assert torch.all(rows_t[audio_indices[num_condition_audio:]] == t_audio)
    assert torch.all(rows_r[audio_indices[num_condition_audio:]] == 0.7)
    assert torch.all(rows_t[audio_indices[:num_condition_audio]] == 1.0)
    assert torch.all(rows_r[audio_indices[:num_condition_audio]] == 1.0)
    # Distinct pairs are unique and sorted.
    pairs = torch.stack((ours_t, ours_r), -1)
    assert pairs.shape[0] == torch.unique(pairs, dim=0).shape[0]


def test_validate_sigmas_rejects_bad_grids():
    with pytest.raises(ValueError):
        validate_sigmas([1.0, 0.5, 0.1])  # does not end at 0
    with pytest.raises(ValueError):
        validate_sigmas([1.0, 0.5, 0.5, 0.0])  # not strictly decreasing
    with pytest.raises(ValueError):
        shift_sigmas(DEFAULT_SIGMAS_8STEP, 0.0)

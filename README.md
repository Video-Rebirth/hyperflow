# HyperFlow for MiniMax-H3

HyperFlow is [Video Rebirth](https://www.videorebirth.com/)'s 8-step LoRA for
[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3), obtained by data-free flow self-distillation and running
on the official [diffusers Modular Pipeline](https://huggingface.co/docs/diffusers/main/api/pipelines/minimax_h3).
Diffusers' default 50-point sigma schedule performs 49 model forwards; HyperFlow performs 8. The base weights, VAEs,
conditioner and workflows (`t2va`, `fl2va`, `ref2va`, all video + audio) stay the official ones. Only the LoRA is
released, on Hugging Face at [huggingface.co/videorebirth/hyperflow](https://huggingface.co/videorebirth/hyperflow);
this repository holds the loader and the example scripts.

| | |
|---|---|
| Base model | `MiniMaxAI/MiniMax-H3` (revision recorded in the file header) |
| Weights | [`videorebirth/hyperflow`](https://huggingface.co/videorebirth/hyperflow) on Hugging Face: the LoRA file and its `hyperflow.json` manifest, under the [MiniMax H3 Community License](https://huggingface.co/videorebirth/hyperflow/blob/main/LICENSE) |
| Adapter | PEFT LoRA, rank 256 / alpha 256, on attention, feed-forward and both time embedders (316 modules); 2.8 GB |
| Sampling | 8 forward passes on a fixed sigma grid stored in the file (video shift 12, audio shift 3) |
| Workflows | `t2va`, `fl2va` and `ref2va`, one file for all three (`ref2va` loads it onto `transformer_ref/`) |
| Optional | Ulysses context parallel on up to 4 GPUs; NVIDIA Sol-Attn sparse attention |

## Install

```bash
pip install "hyperflow-h3[examples] @ git+https://github.com/Video-Rebirth/hyperflow.git"
```

Python ≥ 3.10, diffusers ≥ 0.40.0 (the first release with the MiniMax-H3 Modular Pipeline and its context parallel
plan), transformers ≥ 4.57 (Qwen3-VL). The `examples` extra adds PyAV for writing the output video.

Optional:

- **FlashAttention-3** for the dense attention: `pip install "kernels>=0.12"`; see [Attention backends](#attention-backends).
- **Sol-Attn** sparse attention (PyTorch ≥ 2.10, CUDA ≥ 12.8, Triton ≥ 3.6; SM80–SM120, validated on H200); see
  [Sol-Attn](#sol-attn).

  ```bash
  pip install "git+https://github.com/NVlabs/Sana.git@9bfca5c4bf35774a1d44c27b0c3c91041fb8dad0#subdirectory=techniques/sparse_backends"
  ```

## Quickstart

```python
import torch
from diffusers import ComponentsManager
from diffusers.utils import load_image
from hyperflow_h3 import hyperflow_blocks, load_hyperflow_lora

manager = ComponentsManager()
blocks = hyperflow_blocks("fl2va")                              # official workflow, two blocks swapped
pipe = blocks.init_pipeline("MiniMaxAI/MiniMax-H3", components_manager=manager)
pipe.load_components(dtype=torch.bfloat16)
load_hyperflow_lora(pipe, "videorebirth/hyperflow")  # LoRA + two-time embedder + 8-step grid

# The official single-GPU recipe (auto CPU offload), armed only now: registering a component resets the margin.
# 24 GB is the margin validated on H200; the official recipe's 12 GB starves the denoiser there. See Performance.
manager.enable_auto_cpu_offload(device="cuda", memory_reserve_margin="24GB")

out = pipe(
    prompt="A red fox trotting through a snowy pine forest",
    image=load_image("first.png"),
    num_frames=124,
    generator=torch.Generator().manual_seed(42),
    output=["videos", "audio", "sampling_rate"],
)
```

- Do not pass `num_inference_steps`: the step count and sigma grid come from the weights file; another count raises.
- Every other input (`prompt`, `image`, `last_image`, `references`, `height`, `width`, `num_frames`, `generator`)
  is the official pipeline's.
- Other workflows: `hyperflow_blocks("t2va")` (no `image`) and `hyperflow_blocks("ref2va")` (`references=`; the
  same weights file is loaded onto `transformer_ref`).

**Weights.** `load_hyperflow_lora` takes a Hub repo id, a local directory or a `.safetensors` file. A repo or
directory is resolved through its `hyperflow.json` manifest, whose `default` entry names the recommended file, so a
bare id follows new versions as they ship. `filename=` pins one file (listed on the model card; a published name is
never reused), `revision=` a tag or commit, `token=` a private repo. Manifest and weights are fetched once into the
`huggingface_hub` cache and keep working under `HF_HUB_OFFLINE=1`.

```python
load_hyperflow_lora(pipe, "videorebirth/hyperflow", filename="minimax_h3_hyperflow_8step_v1.0.safetensors")
load_hyperflow_lora(pipe, "/weights/hyperflow")  # from `hf download videorebirth/hyperflow --local-dir /weights/hyperflow`
```

## Example commands

The package installs `hyperflow-h3-fl2va` and `hyperflow-h3-ref2va`; [`examples/`](examples/) contains equivalent
source-tree launchers. They add argument parsing, memory options and `--baseline` (the base pipeline with Diffusers'
default 50 sigma points / 49 NFE, for an A/B on the same seed). `--weights` takes the three forms above
(`--weights-filename` pins a file); `--model` also takes a local MiniMax-H3 snapshot, so a run can be fully offline.

```bash
# first frame -> video+audio; omit --image for t2va, add --last-image for first+last frame
hyperflow-h3-fl2va --prompt "..." --image first.png
# omni-reference: image / video / audio references, in order
hyperflow-h3-ref2va --prompt "..." --ref subject.png --ref clip.mp4 --ref voice.wav
# one GPU (the official CPU-offload recipe) instead of the default, up to 4
hyperflow-h3-fl2va --prompt "..." --image first.png --gpus 1
# FlashAttention-3 for the dense attention / Sol-Attn sparse attention (they combine)
hyperflow-h3-fl2va --prompt "..." --image first.png --attention-backend fa3
hyperflow-h3-fl2va --prompt "..." --image first.png --sol-attn
# fully offline: local weights file + local model snapshot
hyperflow-h3-fl2va --prompt "..." --image first.png \
    --weights /weights/hyperflow/minimax_h3_hyperflow_8step_v1.0.safetensors --model /models/MiniMax-H3
```

**GPUs.** By default a script runs on up to four GPUs, the degree MiniMax serves the model with, re-launching itself
under `torchrun` when started with plain `python`. Rank 0 runs the text encoder and broadcasts the conditioned state
(the official two-device split); every rank then runs the DiT Ulysses sequence-parallel
(`ContextParallelConfig(ulysses_degree=N)`) with the DiT and VAEs resident on its own card. `--gpus 1` is the
official single-GPU recipe (ComponentsManager auto CPU offload).

## Attention backends

`--attention-backend`, or `transformer.set_attention_backend(...)` in code, selects the dense attention kernel:

- `sdpa` (default): PyTorch SDPA, diffusers' `native`.
- `fa3`: FlashAttention-3 as diffusers' `_flash_3_hub`. Needs `pip install "kernels>=0.12"`; the kernel itself
  (`kernels-community/flash-attn3`, ~770 MB) is fetched from the Hub on first use. Works on one GPU and under
  context parallel. Under `HF_HUB_OFFLINE=1`, point `kernels` at a downloaded copy:
  `LOCAL_KERNELS=kernels-community/flash-attn3=<snapshot dir holding build/>`. It buys little here, ~2 s per clip
  (attention is a small share of the pipeline), and changes the output only by bf16 backend noise (PSNR ≈ 34 dB,
  audio waveform cosine 0.997).
- Any other diffusers backend name is passed through, with `--gpus 1` only (`_flash_3` and `flash_4_hub` cannot
  run context parallel in diffusers); none of them is validated by us.

## Performance

HyperFlow (8 NFE, 9 sigma points) against the base pipeline at Diffusers' default (49 NFE, 50 sigma points) on the
same clip: `fl2va`, 124 frames, 1344x768, seed 0, dense SDPA attention. Pipeline time covers text encoding, denoising
and decoding; process start-up and weight loading add ~30 s.

| | Base, 49 NFE | HyperFlow, 8 NFE | Speed-up |
|---|---|---|---|
| 4x H200, Ulysses degree 4 | ~175 s | ~60 s | 2.9x |
| 1x H200, auto CPU offload | ~395 s | ~130 s | 3.0x |

The speed-up is below the 49:8 NFE ratio because text encoding and VAE decoding, which HyperFlow leaves unchanged,
now take much of the run. Peak accelerator memory is ~80 GB per card either way (the LoRA adds 2.8 GB).

The single-GPU row runs the official auto CPU offload with `memory_reserve_margin="24GB"`, the examples' default,
not the official recipe's 12 GB: on a 141 GB H200 a margin under ~18 GB evicts the 10 GB VAE instead of the 62 GB
text encoder when the DiT arrives and starves the denoiser. Below ~100 GB the text encoder and the DiT never fit
together and the margin is irrelevant; an 80 GB card is untested.

With a fixed seed a run reproduces itself (audio bit-identical, video PSNR ≈ 43–50 dB, a residue of the VAE decode).
Across GPU counts or attention backends the clip is only visibly similar (PSNR ≈ 27 dB and ≈ 34 dB): sharding and
kernels change the bf16 reduction order.

## Quality

HyperFlow is a self-distillation: the base model is the only teacher. Against the base pipeline at 49 NFE, on the
same seeds, four things stand out:

- **More balanced capabilities**
- **Better camera control**
- **Better consistency**
- **Better materials and detail**

<!-- TODO(comparison): side-by-side clips, base 49 NFE vs HyperFlow 8 NFE, same seed, 124 frames at 1344x768.
| Prompt | Base, 49 NFE | HyperFlow, 8 NFE |
|---|---|---|
| ... | ... | ... |
-->

Judge on your own prompts: `--baseline` renders both pipelines on one seed.

## Sol-Attn

NVIDIA's sparse attention kernel on the DiT blocks, after two dense steps. It is an approximation: against the dense
run the video differs at PSNR ≈ 23 dB and the audio waveform at cosine 0.88, on one GPU and on four alike. Check the
output before adopting it.

```python
from hyperflow_h3 import enable_sol_attention, sol_attn_available

if sol_attn_available():
    enable_sol_attention(pipe)  # after load_hyperflow_lora; the knobs are keyword arguments
```

| Knob | Default (the validated recipe) | Flag |
|---|---|---|
| Dense steps before the kernel takes over | `dense_steps=2` | `--dense-steps` |
| DiT blocks that always stay dense | `dense_layers=(0, 1)` | `--dense-layers` |
| Sparsity threshold | `tau=1.0` | `--sol-tau` |
| Threshold type | `thresh_type="diag"` | — |
| Sink tokens | none | — |

Works on one GPU and under Ulysses context parallel (the examples' 1 to 4 GPUs): the kernel runs inside diffusers'
sequence exchange, per head on the full sequence, so a 4-GPU run matches a 1-GPU run up to the usual sharding noise.
Ring attention is not supported (the kernel returns no log-sum-exp).

Kernel compilation (~10 s) is paid on the first sparse step of every process:

| vs dense SDPA | 1x H200 | 4x H200 |
|---|---|---|
| Dense step | 7.8 s | 2.15 s |
| Sparse step | 4.9 s | 1.4 s |
| Kernel compilation (once per process) | ~10 s | ~10 s |
| First clip (compile included) | ~8 s faster | ~7 s slower |
| Later clips (compile already paid) | ~13% faster | ~7% faster (the DiT is 15 s of a 65 s pipeline) |

`--sol-attn` combines with `--attention-backend fa3`, which then serves the dense share (4x H200, first clip:
~65 s; versus dense `fa3`: PSNR ≈ 25 dB, cosine 0.92).

## How it works

Three additions to the official pipeline, all reversible (`disable_hyperflow`, `disable_sol_attention`):

- **Two-time conditioning.** The LoRA was trained with both the current time `t` and the step's endpoint `r` (the
  flow-map formulation of AnyFlow). `TwoTimeEmbedder` wraps the base `time_embedder`, adds a LoRA'd copy for `r`,
  and blends the two embeddings with a fixed gate stored in the file. Conditioning rows (`fl2va` keyframes, `ref2va`
  references) keep `r = t`, so they stay pinned exactly as in the official denoiser.
- **8-step schedule.** `HyperFlowSetTimestepsStep` replaces the official timestep step with the trained sigma grid
  (video and audio shifts included); `HyperFlowLoopDenoiser` hands each step's endpoints to the embedder. Nothing
  else in the workflow changes.
- **Sol-Attn.** `HyperFlowSolAttnProcessor` replaces the attention processor of every DiT block and calls the sparse
  kernel once step and layer are past `dense_steps` / `dense_layers`; the token refiner is never touched.
  `enable_sol_attention(..., require_kernel=False)` installs it without the kernel, falling back to dense attention
  with a warning.

**One file, three workflows.** `t2va` and `fl2va` share `transformer/`. `ref2va` runs the same file on
`transformer_ref/`, which works because the adapter never touches `adaln_proj`, the carrier of the reference
conditioning.

**File format.** A plain `safetensors` file: PEFT-style keys (`transformer.<module>.lora_A.weight`; bf16 for the
DiT blocks, fp32 for the two time embedders) plus a self-describing header (`hyperflow_sigmas`, `hyperflow_gate`,
`lora_rank`, `lora_alpha`, `base_model_revision`, …) that `read_metadata()` exposes. The `hyperflow.json` manifest
next to it lists every file with its `hyperflow_version` and `sha256`. It is not a generic LoRA: the
`endpoint_time_embedder.*` keys only exist once `TwoTimeEmbedder` is installed, so load it with `load_hyperflow_lora`,
not diffusers' `load_lora_adapter`.

## Repository layout

```
src/hyperflow_h3/   schedule.py (sigma grid + (t, r) planning) · embedder.py · lora.py (loader) ·
                    blocks.py (the two swapped pipeline blocks) · sol_attn.py · examples/ (installed CLIs)
examples/           source-tree launchers for the two installed example commands
tests/              CPU-only unit tests against a tiny MiniMaxH3Transformer3DModel (`make test`)
```

## Acknowledgements

- **[MiniMax-H3](https://huggingface.co/MiniMaxAI/MiniMax-H3)** ([GitHub](https://github.com/MiniMax-AI/MiniMax-H3)): the base model, VAEs, conditioner and official workflows.
- **[AnyFlow](https://arxiv.org/abs/2605.13724)** ([GitHub](https://github.com/NVlabs/AnyFlow); Gu et al., 2026): the flow-map formulation behind the two-time `(t, r)` conditioning.
- **[FlashAttention](https://github.com/Dao-AILab/flash-attention)** ([FA2](https://arxiv.org/abs/2307.08691), [FA3](https://arxiv.org/abs/2407.08608), [FA4](https://arxiv.org/abs/2603.05451)): dense kernels via diffusers' `set_attention_backend`.
- **[Sol-Attn](https://arxiv.org/abs/2607.24027)** ([GitHub](https://github.com/NVlabs/Sana/tree/main/techniques/sparse_backends); Li et al., 2026): optional sparse attention.

Thanks to their authors.

## License

Code: [Apache-2.0](LICENSE); the schedule, blocks and Sol-Attn processor derive from or rework diffusers code, see
[`THIRD_PARTY_NOTICES.md`](THIRD_PARTY_NOTICES.md). The LoRA weights are a Model Derivative of MiniMax-H3 and are
distributed on [their Hugging Face repo](https://huggingface.co/videorebirth/hyperflow) under the
[MiniMax H3 Community License Agreement](https://huggingface.co/videorebirth/hyperflow/blob/main/LICENSE). That
Agreement, not this repository's license, governs any use of the weights: it excludes the EU, UK, South Korea and the
US absent MiniMax's authorization and carries an Acceptable Use Policy. HyperFlow is developed by Video Rebirth and is
not affiliated with or endorsed by MiniMax.

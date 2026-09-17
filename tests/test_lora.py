import copy
import json

import pytest
import torch
from conftest import GATE, RANK, TINY_CONFIG, make_transformer, packed_inputs, public_target_modules, write_weights
from safetensors.torch import load_file, save_file

from hyperflow_h3.embedder import TwoTimeEmbedder
from hyperflow_h3.lora import (
    MANIFEST_NAME,
    HyperFlowMetadata,
    disable_hyperflow,
    enable_hyperflow,
    load_hyperflow_lora,
    plan_from_keys,
    read_metadata,
    resolve_weights,
)


def forward(model, inputs, timestep, endpoint=None):
    embedder = model.time_embedder
    kwargs = dict(
        hidden_states=inputs["latents"][None],
        audio_hidden_states=inputs["audio_latents"][None],
        encoder_hidden_states=inputs["prompt_embeds"],
        timestep=timestep,
        timestep_indices=torch.zeros(inputs["token_tags"].numel(), dtype=torch.long),
        token_tags=inputs["token_tags"],
        position_ids=inputs["position_ids"],
        video_indices=inputs["video_indices"],
        audio_indices=inputs["audio_indices"],
        text_indices=inputs["text_indices"],
        return_dict=False,
    )
    with torch.no_grad():
        if isinstance(embedder, TwoTimeEmbedder) and endpoint is not None:
            with embedder.endpoint_context(endpoint):
                return model(**kwargs)
        return model(**kwargs)


def test_metadata_roundtrip(weights_file):
    metadata = read_metadata(weights_file)
    assert metadata.gate == GATE
    assert metadata.lora_rank == RANK
    assert metadata.sigmas is not None and len(metadata.sigmas) == 9
    assert metadata.compatible_subfolders == ("transformer", "transformer_ref")
    assert metadata.tasks == ("fl2va", "ref2va")


def test_metadata_rejects_foreign_files():
    with pytest.raises(ValueError, match="Not a HyperFlow"):
        HyperFlowMetadata.from_dict({"format": "pt"})


V10, V11 = "minimax_h3_hyperflow_8step_v1.0.safetensors", "minimax_h3_hyperflow_8step_v1.1.safetensors"


def write_manifest(directory, **manifest):
    (directory / MANIFEST_NAME).write_text(json.dumps(manifest))


def test_resolve_weights_directory_follows_the_manifest_default(tmp_path):
    assert MANIFEST_NAME == "hyperflow.json"
    (tmp_path / V10).touch()
    (tmp_path / V11).touch()
    with pytest.raises(FileNotFoundError, match="filename="):
        resolve_weights(tmp_path)  # two files, nothing says which

    write_manifest(tmp_path, default=V11, weights={V10: {}, V11: {}})
    assert resolve_weights(tmp_path) == tmp_path / V11
    assert resolve_weights(tmp_path, filename=V10) == tmp_path / V10  # pin beats the default
    assert resolve_weights(tmp_path / V10) == tmp_path / V10

    write_manifest(tmp_path, default="gone.safetensors")
    with pytest.raises(FileNotFoundError, match="gone.safetensors"):
        resolve_weights(tmp_path)
    write_manifest(tmp_path, weights={})
    with pytest.raises(ValueError, match="default"):
        resolve_weights(tmp_path)


def test_resolve_weights_directory_without_manifest_accepts_a_single_file(tmp_path):
    (tmp_path / V10).touch()
    assert resolve_weights(tmp_path) == tmp_path / V10


def test_resolve_weights_hub_fetches_manifest_then_default_without_listing(monkeypatch, tmp_path):
    import huggingface_hub

    write_manifest(tmp_path, default=V11)
    calls = []

    def fake_download(repo_id, filename, *, revision=None, token=None, **_):
        calls.append((repo_id, filename, revision, token))
        return str(tmp_path / filename)

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", None)  # listing the repo would fail loudly
    assert resolve_weights("org/repo", revision="v1.1", token="tok") == tmp_path / V11
    assert resolve_weights("org/repo", filename=V10) == tmp_path / V10  # pin: no manifest fetched
    assert calls == [
        ("org/repo", MANIFEST_NAME, "v1.1", "tok"),
        ("org/repo", V11, "v1.1", "tok"),
        ("org/repo", V10, None, None),
    ]


def test_resolve_weights_hub_without_manifest_falls_back_to_the_single_file(monkeypatch, tmp_path):
    import huggingface_hub
    from huggingface_hub.utils import EntryNotFoundError

    def fake_download(repo_id, filename, *, revision=None, token=None, **_):
        if filename == MANIFEST_NAME:
            raise EntryNotFoundError("404")
        return str(tmp_path / filename)

    class FakeApi:
        def __init__(self, token=None):
            pass

        def list_repo_files(self, repo_id, revision=None):
            return ["README.md", "LICENSE", V10]

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", FakeApi)
    assert resolve_weights("org/repo") == tmp_path / V10


def test_resolve_weights_hub_offline_miss_does_not_list(monkeypatch):
    import huggingface_hub
    from huggingface_hub.utils import LocalEntryNotFoundError

    def fake_download(*_, **__):
        raise LocalEntryNotFoundError("offline and not cached")

    monkeypatch.setattr(huggingface_hub, "hf_hub_download", fake_download)
    monkeypatch.setattr(huggingface_hub, "HfApi", None)
    with pytest.raises(LocalEntryNotFoundError):
        resolve_weights("org/repo")


def test_plan_groups_keys_and_maps_embedders():
    plan = plan_from_keys(
        [
            "transformer.transformer_blocks.0.attn.to_q.lora_A.weight",
            "transformer.transformer_blocks.0.attn.to_q.lora_B.weight",
            "transformer.time_embedder.linear_1.lora_A.weight",
            "transformer.time_embedder.linear_1.lora_B.weight",
            "transformer.endpoint_time_embedder.linear_2.lora_A.weight",
            "transformer.endpoint_time_embedder.linear_2.lora_B.weight",
        ]
    )
    assert set(plan) == {
        "transformer_blocks.0.attn.to_q",
        "time_embedder.base.linear_1",
        "time_embedder.endpoint.linear_2",
    }
    with pytest.raises(ValueError, match="Unexpected key"):
        plan_from_keys(["transformer.transformer_blocks.0.attn.to_q.weight"])
    with pytest.raises(ValueError, match="only one of"):
        plan_from_keys(["transformer.transformer_blocks.0.attn.to_q.lora_A.weight"])


def test_load_consumes_every_tensor_and_matches_values(tiny_transformer, weights_file):
    file_tensors = load_file(str(weights_file))
    metadata = load_hyperflow_lora(tiny_transformer, weights_file)

    assert isinstance(tiny_transformer.time_embedder, TwoTimeEmbedder)
    assert tiny_transformer.time_embedder.gate == metadata.gate
    lora_params = {n: p for n, p in tiny_transformer.named_parameters() if ".lora_" in n}
    assert len(lora_params) == len(file_tensors) == 2 * len(public_target_modules(TINY_CONFIG))
    assert all(not p.requires_grad for p in lora_params.values())

    for key, tensor in file_tensors.items():
        module = key[len("transformer.") : -len(".lora_A.weight")]
        matrix = key[-len("A.weight")]
        module = module.replace("endpoint_time_embedder.", "time_embedder.endpoint.", 1)
        if module.startswith("time_embedder.linear"):
            module = module.replace("time_embedder.", "time_embedder.base.", 1)
        parameter = lora_params[f"{module}.lora_{matrix}.hyperflow.weight"]
        torch.testing.assert_close(parameter.float(), tensor.float(), rtol=0, atol=0)
    # LoRA dtype follows the base layer: bf16 blocks would be bf16 on a bf16 model; this tiny model is fp32.
    assert {p.dtype for p in lora_params.values()} == {torch.float32}


def test_loaded_adapter_changes_the_output_and_disable_restores_the_base(weights_file):
    pristine = make_transformer()
    patched = make_transformer()
    inputs = packed_inputs(pristine)
    timestep = torch.tensor([0.3])
    endpoint = torch.tensor([0.5])

    base_video, base_audio = forward(pristine, inputs, timestep)
    load_hyperflow_lora(patched, weights_file)
    hf_video, hf_audio = forward(patched, inputs, timestep, endpoint)
    assert not torch.allclose(hf_video, base_video) or not torch.allclose(hf_audio, base_audio)

    disable_hyperflow(patched)
    off_video, off_audio = forward(patched, inputs, timestep)
    torch.testing.assert_close(off_video, base_video, rtol=0, atol=0)
    torch.testing.assert_close(off_audio, base_audio, rtol=0, atol=0)

    enable_hyperflow(patched)
    on_video, on_audio = forward(patched, inputs, timestep, endpoint)
    torch.testing.assert_close(on_video, hf_video, rtol=0, atol=0)
    torch.testing.assert_close(on_audio, hf_audio, rtol=0, atol=0)


def test_forward_without_endpoint_fails_loud(tiny_transformer, weights_file):
    load_hyperflow_lora(tiny_transformer, weights_file)
    inputs = packed_inputs(tiny_transformer)
    with pytest.raises(RuntimeError, match="endpoint timesteps"):
        forward(tiny_transformer, inputs, torch.tensor([0.3]))


def test_loading_twice_is_refused(tiny_transformer, weights_file):
    load_hyperflow_lora(tiny_transformer, weights_file)
    with pytest.raises(ValueError, match="one fixed adapter"):
        load_hyperflow_lora(tiny_transformer, weights_file)


def test_loader_does_not_claim_multi_adapter_support(tiny_transformer, weights_file, monkeypatch):
    with pytest.raises(TypeError, match="adapter_name"):
        load_hyperflow_lora(tiny_transformer, weights_file, adapter_name="another")

    monkeypatch.setattr(tiny_transformer, "peft_config", {"another": object()}, raising=False)
    with pytest.raises(ValueError, match="fresh base model"):
        load_hyperflow_lora(tiny_transformer, weights_file)


def test_shape_mismatch_is_an_error(tmp_path, tiny_transformer, weights_file):
    tensors = load_file(str(weights_file))
    key = "transformer.transformer_blocks.0.attn.to_q.lora_B.weight"
    tensors[key] = torch.zeros(tensors[key].shape[0] + 1, RANK, dtype=tensors[key].dtype)
    bad = tmp_path / "bad_shape.safetensors"
    save_file(tensors, str(bad), metadata=read_metadata(weights_file).raw)
    with pytest.raises(ValueError, match="Shape mismatch"):
        load_hyperflow_lora(make_transformer(), bad)


def test_unknown_module_is_an_error(tmp_path, weights_file):
    tensors = load_file(str(weights_file))
    tensors["transformer.transformer_blocks.7.attn.to_q.lora_A.weight"] = torch.zeros(RANK, 64)
    tensors["transformer.transformer_blocks.7.attn.to_q.lora_B.weight"] = torch.zeros(64, RANK)
    bad = tmp_path / "bad_module.safetensors"
    save_file(tensors, str(bad), metadata=read_metadata(weights_file).raw)
    with pytest.raises(KeyError, match="no `nn.Linear`"):
        load_hyperflow_lora(make_transformer(), bad)


def test_bf16_base_gets_bf16_lora_and_fp32_time_embedders(weights_file):
    model = make_transformer().to(torch.bfloat16)
    # Mirror the checkpoint's mixed precision: the time embedder stays fp32 (`_keep_in_fp32_modules`).
    model.time_embedder.float()
    load_hyperflow_lora(model, weights_file)
    dtypes = {n: p.dtype for n, p in model.named_parameters() if ".lora_" in n}
    assert all(d == torch.float32 for n, d in dtypes.items() if n.startswith("time_embedder."))
    assert all(d == torch.bfloat16 for n, d in dtypes.items() if n.startswith("transformer_blocks."))


def test_same_file_loads_onto_a_second_partition(tmp_path, weights_file):
    """`transformer` and `transformer_ref` share one architecture; the loader must not care which it is given."""
    ref = make_transformer(seed=7)
    write_weights(ref, tmp_path / "again.safetensors")
    load_hyperflow_lora(ref, tmp_path / "again.safetensors")
    assert isinstance(ref.time_embedder, TwoTimeEmbedder)


def test_pipeline_like_target_patches_every_transformer(weights_file):
    class FakePipeline:
        def __init__(self):
            self.components = {"transformer": make_transformer(), "transformer_ref": make_transformer(seed=3)}
            self.blocks = None

    pipe = FakePipeline()
    load_hyperflow_lora(pipe, weights_file, configure_blocks=False)
    assert all(isinstance(t.time_embedder, TwoTimeEmbedder) for t in pipe.components.values())
    assert copy.copy(pipe.components["transformer"].peft_config).keys() == {"hyperflow"}

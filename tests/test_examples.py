"""Behavioral tests for the runnable examples' orchestration helpers."""

from types import SimpleNamespace

from hyperflow_h3.examples import common


def test_installed_example_entry_modules_are_importable():
    from hyperflow_h3.examples.generate_fl2va import main as fl2va_main
    from hyperflow_h3.examples.generate_ref2va import main as ref2va_main

    assert callable(fl2va_main) and callable(ref2va_main)


class FakeConditioner:
    def __init__(self):
        self.blocks = SimpleNamespace(inputs=[SimpleNamespace(name="prompt")])
        self.prompts = []

    def __call__(self, *, prompt):
        self.prompts.append(prompt)
        return SimpleNamespace(values={"prompt_embeds": f"encoded:{prompt}"})


class FakeRest:
    def __init__(self):
        self.blocks = SimpleNamespace(inputs=[SimpleNamespace(name="prompt_embeds"), SimpleNamespace(name="seed")])
        self.calls = []

    def __call__(self, *, state, output, **kwargs):
        call = (state.values["prompt_embeds"], output, kwargs)
        self.calls.append(call)
        return call


class FakeManager:
    def __init__(self):
        self.calls = []

    def disable_auto_cpu_offload(self):
        self.calls.append(("disable",))

    def enable_auto_cpu_offload(self, *, device, memory_reserve_margin):
        self.calls.append(("enable", device, memory_reserve_margin))


def test_split_pipeline_reloads_the_conditioner_for_a_second_call(monkeypatch):
    first = FakeConditioner()
    created = []

    def factory():
        conditioner = FakeConditioner()
        created.append(conditioner)
        return conditioner, object()

    rest, rest_manager = FakeRest(), FakeManager()
    pipeline = common.SplitPipeline(
        first,
        object(),
        factory,
        rest,
        rest_manager,
        rank=0,
        transformer_name="transformer",
        device="cuda:0",
        memory_reserve_margin="24GB",
    )
    monkeypatch.setattr(pipeline, "_warm_transformer", lambda: None)
    monkeypatch.setattr(common.dist, "broadcast_object_list", lambda box, src: None)
    monkeypatch.setattr(common.gc, "collect", lambda: None)
    monkeypatch.setattr(common.torch.cuda, "empty_cache", lambda: None)
    monkeypatch.setattr(common.torch.cuda, "memory_allocated", lambda: 0)

    assert pipeline(prompt="first", seed=1, output=["videos"]) == (
        "encoded:first",
        ["videos"],
        {"seed": 1},
    )
    assert pipeline(prompt="second", seed=2, output=["audio"]) == (
        "encoded:second",
        ["audio"],
        {"seed": 2},
    )

    assert first.prompts == ["first"]
    assert len(created) == 1 and created[0].prompts == ["second"]
    assert rest_manager.calls == [("disable",), ("enable", "cuda:0", "24GB")]
    assert pipeline.conditioner is None

from __future__ import annotations

from training import model_loader


def test_non_linux_loader_keeps_selected_model(monkeypatch):
    sentinel_model = object()
    sentinel_tokenizer = object()

    monkeypatch.setattr(
        model_loader,
        "resolve_model_selection",
        lambda model_label=None, model_id=None: {
            "label": "opus_2b",
            "model_id": "Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled",
            "source": "model_label",
        },
    )
    monkeypatch.setattr(model_loader.sys, "platform", "darwin")
    monkeypatch.setattr(
        model_loader,
        "_load_selected_model_portable",
        lambda model_id, quant_bits=0: (
            {"model_id": model_id, "quant_bits": quant_bits, "model": sentinel_model},
            sentinel_tokenizer,
        ),
    )
    monkeypatch.setattr(model_loader, "_model", None)
    monkeypatch.setattr(model_loader, "_tokenizer", None)
    monkeypatch.setattr(model_loader, "_model_key", None)
    monkeypatch.setattr(model_loader, "_model_selection", None)

    model, tokenizer = model_loader.load_model_and_tokenizer(
        model_label="opus_2b",
        quant_bits=0,
    )

    assert tokenizer is sentinel_tokenizer
    assert model["model_id"] == "Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled"
    assert model["quant_bits"] == 0


def test_target_gpu_profile_includes_h200(monkeypatch):
    monkeypatch.setattr(model_loader, "TARGET_GPU", "H200")
    profile = model_loader.get_target_gpu_profile()

    assert profile["family"] == "hopper"
    assert profile["memory_gb"] == 141

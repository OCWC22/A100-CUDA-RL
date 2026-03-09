from __future__ import annotations

import json

import pytest

from training.model_registry import load_model_registry, resolve_model_selection


@pytest.fixture
def registry_path(tmp_path):
    path = tmp_path / "scaling_ladder.json"
    path.write_text(
        json.dumps(
            {
                "models": [
                    {
                        "label": "opus_2b",
                        "model_id": "Jackrong/Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled",
                        "enabled": True,
                    },
                    {
                        "label": "coder_30b_moe",
                        "model_id": "Qwen/Qwen3-Coder-30B-A3B-Instruct",
                        "enabled": True,
                    },
                    {
                        "label": "disabled_model",
                        "model_id": "example/disabled",
                        "enabled": False,
                    },
                ]
            }
        ),
        encoding="utf-8",
    )
    return path


def test_load_model_registry_filters_disabled_models(registry_path):
    registry, path = load_model_registry(config_path=registry_path)
    assert str(path) == str(registry_path)
    assert "opus_2b" in registry
    assert "coder_30b_moe" in registry
    assert "disabled_model" not in registry


def test_resolve_model_selection_from_label(registry_path, monkeypatch):
    monkeypatch.delenv("KERNELFORGE_MODEL", raising=False)
    monkeypatch.setenv("KERNELFORGE_MODEL_LABEL", "coder_30b_moe")

    resolved = resolve_model_selection(config_path=registry_path)

    assert resolved["label"] == "coder_30b_moe"
    assert resolved["model_id"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert resolved["source"] == "model_label"


def test_resolve_model_selection_uses_exact_override_precedence(registry_path, monkeypatch):
    monkeypatch.setenv("KERNELFORGE_MODEL_LABEL", "opus_2b")
    monkeypatch.setenv("KERNELFORGE_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")

    resolved = resolve_model_selection(config_path=registry_path)

    assert resolved["label"] == "coder_30b_moe"
    assert resolved["model_id"] == "Qwen/Qwen3-Coder-30B-A3B-Instruct"
    assert resolved["source"] == "model_id_override"


def test_resolve_model_selection_rejects_unknown_or_disabled_labels(registry_path, monkeypatch):
    monkeypatch.delenv("KERNELFORGE_MODEL", raising=False)
    monkeypatch.setenv("KERNELFORGE_MODEL_LABEL", "disabled_model")

    with pytest.raises(ValueError):
        resolve_model_selection(config_path=registry_path)

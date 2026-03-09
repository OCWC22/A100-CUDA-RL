"""Registry-driven model selection for KernelForge runtime surfaces."""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REGISTRY_PATH = ROOT / "configs" / "scaling_ladder.json"
DEFAULT_MODEL_LABEL = "opus_2b"


def _registry_path(config_path: str | os.PathLike[str] | None = None) -> Path:
    raw_path = config_path or os.getenv("KERNELFORGE_MODEL_REGISTRY")
    return Path(raw_path) if raw_path else DEFAULT_REGISTRY_PATH


def load_model_registry(
    config_path: str | os.PathLike[str] | None = None,
    enabled_only: bool = True,
) -> tuple[dict[str, dict[str, Any]], Path]:
    """Load the scaling ladder registry keyed by label."""
    path = _registry_path(config_path)
    with path.open(encoding="utf-8") as f:
        payload = json.load(f)

    models = payload.get("models", []) if isinstance(payload, dict) else payload
    registry: dict[str, dict[str, Any]] = {}
    for entry in models:
        if not isinstance(entry, dict):
            continue
        if enabled_only and not entry.get("enabled", True):
            continue
        label = str(entry.get("label", "")).strip()
        model_id = str(entry.get("model_id", "")).strip()
        if not label or not model_id:
            continue
        registry[label] = dict(entry)
    return registry, path


def resolve_model_selection(
    model_label: str | None = None,
    model_id: str | None = None,
    config_path: str | os.PathLike[str] | None = None,
) -> dict[str, Any]:
    """Resolve the active model from exact override or ladder label."""
    registry, path = load_model_registry(config_path=config_path, enabled_only=True)
    requested_model_id = (model_id or os.getenv("KERNELFORGE_MODEL", "")).strip()
    requested_label = (model_label or os.getenv("KERNELFORGE_MODEL_LABEL", DEFAULT_MODEL_LABEL)).strip()

    if requested_model_id:
        matched_label = next(
            (label for label, entry in registry.items() if entry.get("model_id") == requested_model_id),
            "custom",
        )
        return {
            "label": matched_label,
            "model_id": requested_model_id,
            "source": "model_id_override",
            "registry_path": str(path),
        }

    if requested_label not in registry:
        available = ", ".join(sorted(registry))
        raise ValueError(
            f"Unknown KERNELFORGE_MODEL_LABEL='{requested_label}'. "
            f"Enabled labels from {path}: {available}"
        )

    entry = dict(registry[requested_label])
    entry["label"] = requested_label
    entry["source"] = "model_label"
    entry["registry_path"] = str(path)
    return entry

"""CLI entrypoint for the KernelForge training pipeline."""

from __future__ import annotations

import argparse
import importlib.metadata
import os
import sys
from pathlib import Path
from typing import Iterable

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

from training.dataset_loader import load_training_dataset
from training.model_registry import resolve_model_selection
from training.task_support import summarize_tasks

REQUIRED_PACKAGES = ("trl", "transformers", "torch")
if sys.platform.startswith("linux"):
    REQUIRED_PACKAGES += ("peft",)

ROOT = Path(__file__).resolve().parents[1]
REQUIRED_ASSETS = (
    ROOT / "datasets" / "doublegraph_sft.jsonl",
    ROOT / "docs" / "research" / "doublegraph" / "doublegraph_a100_manifest.jsonl",
)


def _check_dependencies(packages: Iterable[str] = REQUIRED_PACKAGES) -> list[str]:
    """Return a list of missing Python packages."""
    missing = []
    for package in packages:
        try:
            importlib.metadata.version(package)
        except Exception:
            missing.append(package)
    return missing


def _dataset_summary() -> dict[str, dict[str, int]]:
    """Summarize the supported training dataset."""
    rows = load_training_dataset(stage="stage3", ops6k_max=int(os.getenv("CUDA_AGENT_STAGE1_SAMPLES", "128")))
    return summarize_tasks(rows)


def _missing_assets() -> list[str]:
    missing: list[str] = []
    for path in REQUIRED_ASSETS:
        if not path.exists():
            missing.append(str(path))
    return missing


def preflight() -> None:
    """Run non-destructive checks before training."""
    missing = _check_dependencies()
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(
            "Missing required training dependencies: "
            f"{missing_str}. Install them with `uv sync --extra train --extra openenv` for the primary CoreWeave/Northflank path, and add `--extra modal` only if you need the Modal fallback tooling."
        )

    missing_assets = _missing_assets()
    if missing_assets:
        missing_str = ", ".join(missing_assets)
        raise RuntimeError(
            "Missing required training assets: "
            f"{missing_str}. Ensure the DoubleGraph manifest and SFT priors are present before launching RL."
        )

    from openenv_env.eval_backend import EVAL_BACKEND, EVAL_URL
    if EVAL_BACKEND == "modal":
        try:
            import modal
            if not getattr(modal.config, "token_id", None):
                print("WARNING: Modal authentication not configured. "
                      "Run `modal token new` or set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET env vars. "
                      "Training will fail at evaluation time without Modal auth.")
        except ImportError:
            print("WARNING: modal package not installed but KERNELFORGE_EVAL_BACKEND=modal.")
    elif EVAL_BACKEND == "coreweave":
        if not EVAL_URL:
            print("WARNING: KERNELFORGE_EVAL_URL not set. "
                  "Set it to the Northflank eval service URL for CoreWeave dispatch.")
        else:
            try:
                import httpx
                resp = httpx.get(f"{EVAL_URL.rstrip('/')}/health", timeout=10.0)
                resp.raise_for_status()
                print(f"Eval backend: CoreWeave ({EVAL_URL}) — healthy")
            except Exception as exc:
                print(f"WARNING: CoreWeave eval service not reachable at {EVAL_URL}: {exc}")

    summary = _dataset_summary()
    selection = resolve_model_selection()
    print(
        "Resolved runtime model: "
        f"label={selection['label']} model_id={selection['model_id']} "
        f"source={selection['source']}"
    )
    print("Dataset summary:")
    for key, value in summary.items():
        print(f"  {key}: {value}")


def main() -> None:
    """Run a specific training stage or the full pipeline."""
    parser = argparse.ArgumentParser(description="KernelForge training launcher")
    parser.add_argument(
        "--stage",
        choices=("stage1", "stage2", "stage3", "pipeline"),
        default=os.getenv("KERNELFORGE_STAGE", "stage1"),
        help="Training stage to run",
    )
    parser.add_argument(
        "--preflight-only",
        action="store_true",
        help="Validate environment and dataset wiring without starting training",
    )
    parser.add_argument(
        "--model-label",
        default="",
        help="Registry model label from configs/scaling_ladder.json (default: KERNELFORGE_MODEL_LABEL or opus_2b)",
    )
    args = parser.parse_args()

    if args.model_label:
        os.environ["KERNELFORGE_MODEL_LABEL"] = args.model_label

    preflight()
    if args.preflight_only:
        print("Preflight checks passed.")
        return

    if args.stage in {"stage1", "pipeline"}:
        from training.stage1_warmup import main as stage1_main

        stage1_main()

    if args.stage in {"stage2", "pipeline"}:
        from training.stage2_rft import main as stage2_main

        stage2_main()

    if args.stage in {"stage3", "pipeline"}:
        from training.stage3_grpo import main as stage3_main

        stage3_main()


if __name__ == "__main__":
    main()

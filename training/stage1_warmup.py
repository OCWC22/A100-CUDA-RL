"""
Stage 1: GRPO Warm-up — bootstrap CUDA syntax on easy operators.

Multi-turn agentic training via TRL's rollout_func:
  - 3 turns per episode (model sees errors, iterates)
  - Temperature 1.0 for exploration
  - LR 2e-6 to avoid catastrophic forgetting
  - beta=0.0 (no KL penalty — let model explore freely)
  - Shared GRPO config with Stage 3: G=8, max_completion_length=1024
  - vLLM disabled by default for hackathon bring-up (`KERNELFORGE_USE_VLLM=0`)

Dataset: CUDA-Agent-Ops-6K easy operators (single-op subset).
"""
from __future__ import annotations

import multiprocessing
import os
import sys
from pathlib import Path

# Force spawn before any CUDA import to prevent fork deadlocks.
if sys.platform.startswith("linux"):
    try:
        multiprocessing.set_start_method("spawn", force=True)
    except RuntimeError:
        pass

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

# Import unsloth BEFORE trl/transformers so it can patch correctly.
if sys.platform.startswith("linux"):
    try:
        from unsloth import FastLanguageModel, PatchFastRL  # noqa: F401
    except ImportError:
        pass

from trl import GRPOConfig

from training.custom_grpo_trainer import TRLOOGRPOTrainer
from training.dataset_loader import Dataset, MiniDataset, load_training_dataset
from training.grpo_config import (
    apply_shared_grpo_runtime,
    load_shared_grpo_runtime,
    validate_shared_grpo_runtime,
)
from training.model_loader import load_model_and_tokenizer
from training.multi_turn_rollout import make_multi_turn_rollout, reward_from_env
from training.task_support import normalize_task_row

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")
IS_LINUX = sys.platform.startswith("linux")
GRPO_RUNTIME = load_shared_grpo_runtime("stage1")
SKIP_BENCHMARK = os.getenv("KERNELFORGE_SKIP_BENCHMARK", "0") == "1"
DEBUG_TIMINGS = os.getenv("KERNELFORGE_DEBUG_TIMINGS", "0") == "1"
BATCH_EVAL = os.getenv("KERNELFORGE_BATCH_EVAL", "0") == "1"

# Multi-turn configuration
MAX_TURNS = int(os.getenv("KERNELFORGE_STAGE1_MAX_TURNS", "3"))
MAX_STEPS = int(os.getenv("KERNELFORGE_STAGE1_MAX_STEPS", "100"))
MAX_COMPLETION_LENGTH = GRPO_RUNTIME.max_completion_length


# --- Dataset loading ---


def _dataset_from_rows(rows: list[dict]) -> Dataset:
    if hasattr(Dataset, "from_list"):
        return Dataset.from_list(rows)
    return MiniDataset(rows)


def load_stage1_dataset() -> Dataset:
    """Load stage1 prompts from unified dataset loader, with safe fallback."""

    try:
        max_samples = int(os.getenv("CUDA_AGENT_STAGE1_SAMPLES", "512"))
        ds = load_training_dataset(
            stage="stage1",
            ops6k_max=max_samples,
            seed=42,
        )
        if len(ds) > 0:
            print(f"Loaded {len(ds)} unified Stage 1 prompts")
            return ds.shuffle(seed=42) if hasattr(ds, "shuffle") else ds
    except Exception as e:
        print(f"Could not load Ops-6K for Stage 1: {e}")

    print("Using fallback Stage 1 prompts with live WCC evaluation support")
    return _dataset_from_rows([
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using union-find with path compression."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "fallback_wcc",
        },
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "optimized for sparse disconnected graphs with early convergence."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "fallback_wcc",
        },
        {
            "prompt": (
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using shared memory staging for dense frontiers."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 2,
            "data_source": "fallback_wcc",
        },
    ])


# --- Training ---

def main():
    """Run Stage 1 GRPO warm-up with multi-turn agentic loop."""
    print(f"=== Stage 1: Multi-Turn GRPO Warm-up for {TARGET_GPU} ({TARGET_ARCH}) ===")
    print(f"  Max turns per episode: {MAX_TURNS}")
    print(f"  Max training steps: {MAX_STEPS}")
    print(f"  Max prompt length: {GRPO_RUNTIME.max_prompt_length}")
    print(f"  Max completion length: {MAX_COMPLETION_LENGTH}")
    print(
        "  Shared GRPO runtime: "
        f"G={GRPO_RUNTIME.num_generations} "
        f"batch={GRPO_RUNTIME.per_device_train_batch_size}x{GRPO_RUNTIME.gradient_accumulation_steps} "
        f"(effective={GRPO_RUNTIME.effective_batch_size})"
    )
    print(
        "  Rollout mode: "
        f"skip_benchmark={SKIP_BENCHMARK} batch_eval={BATCH_EVAL} debug_timings={DEBUG_TIMINGS}"
    )

    model, tokenizer = load_model_and_tokenizer()
    dataset = load_stage1_dataset()
    task_rows = [normalize_task_row(row) for row in dataset.to_list()]

    # Canary: verify raw generation works before entering GRPO.
    print("[canary] Testing raw model.generate()...")
    import torch
    _canary_inputs = tokenizer("Write a CUDA vector add kernel.", return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            _canary_out = model.generate(
                **_canary_inputs, max_new_tokens=32, temperature=1.0,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        print(f"[canary] PASS — {len(_canary_out[0])} tokens generated")
    except Exception as e:
        raise RuntimeError(f"Canary generation failed — model cannot generate: {e}") from e

    rollout_func = make_multi_turn_rollout(
        max_turns=MAX_TURNS,
        skill_md_gpu=TARGET_GPU.lower(),
        problem_metadata=task_rows,
    )

    validate_shared_grpo_runtime(GRPO_RUNTIME)

    config = GRPOConfig(
        **apply_shared_grpo_runtime(
            GRPO_RUNTIME,
            {
                "learning_rate": 2e-6,
                "temperature": 1.0,  # High exploration
                "max_steps": MAX_STEPS,
                "report_to": "none",
                "output_dir": OUTPUT_DIR,
                "logging_steps": 1,
                "top_k": 50,
                "top_p": 0.95,
                "repetition_penalty": 1.05,
                "dataloader_num_workers": 0,
            },
        )
    )

    trainer = TRLOOGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_from_env],
        rollout_func=rollout_func,
        args=config,
        train_dataset=dataset,
    )

    print("Starting Stage 1 training...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Stage 1 complete. Checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

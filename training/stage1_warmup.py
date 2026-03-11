"""
Stage 1: GRPO Warm-up — bootstrap CUDA syntax on easy operators.

Single-turn GRPO using vanilla TRL GRPOTrainer (no Unsloth):
  - Model generates CUDA kernel completions
  - Reward function evaluates kernels remotely (compile + correctness)
  - Temperature 1.0 for exploration
  - LR 2e-6 to avoid catastrophic forgetting
  - beta=0.0 (no KL penalty — let model explore freely)

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

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

if __package__ in {None, ""}:
    ROOT = Path(__file__).resolve().parents[1]
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))

# No Unsloth — use vanilla TRL GRPOTrainer directly.
# Unsloth's compiled GRPOTrainer is incompatible with PEFT models loaded
# outside FastLanguageModel (shape mismatches in logprob computation).
from trl import GRPOConfig

from training.custom_grpo_trainer import TRLOOGRPOTrainer
from training.dataset_loader import Dataset, MiniDataset, load_training_dataset
from training.grpo_config import (
    apply_shared_grpo_runtime,
    load_shared_grpo_runtime,
    validate_shared_grpo_runtime,
)
from training.model_loader import load_model_and_tokenizer
from training.task_support import normalize_task_row

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")
IS_LINUX = sys.platform.startswith("linux")
GRPO_RUNTIME = load_shared_grpo_runtime("stage1")
SKIP_BENCHMARK = os.getenv("KERNELFORGE_SKIP_BENCHMARK", "0") == "1"
DEBUG_TIMINGS = os.getenv("KERNELFORGE_DEBUG_TIMINGS", "0") == "1"
BATCH_EVAL = os.getenv("KERNELFORGE_BATCH_EVAL", "0") == "1"

MAX_STEPS = int(os.getenv("KERNELFORGE_STAGE1_MAX_STEPS", "100"))
MAX_COMPLETION_LENGTH = GRPO_RUNTIME.max_completion_length


# --- Dataset loading ---


def _dataset_from_rows(rows: list[dict]) -> Dataset:
    if hasattr(Dataset, "from_list"):
        return Dataset.from_list(rows)
    return MiniDataset(rows)


CUDA_SYSTEM_PROMPT = (
    "You are a CUDA kernel engineer. Output ONLY the CUDA C++ code inside a "
    "```cuda code block. No explanation, no markdown outside the code block, "
    "no commentary. The code must be a complete, compilable CUDA kernel."
)


def _wrap_prompt_as_chat(prompt_text: str) -> list[dict[str, str]]:
    """Wrap a raw prompt string as chat messages with system instruction."""
    return [
        {"role": "system", "content": CUDA_SYSTEM_PROMPT},
        {"role": "user", "content": prompt_text},
    ]


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
            # Wrap string prompts as chat messages for better code-only output
            rows = ds.to_list() if hasattr(ds, "to_list") else list(ds)
            for row in rows:
                if isinstance(row.get("prompt"), str):
                    row["prompt"] = _wrap_prompt_as_chat(row["prompt"])
            ds = _dataset_from_rows(rows)
            print(f"Loaded {len(ds)} unified Stage 1 prompts (chat format)")
            return ds.shuffle(seed=42) if hasattr(ds, "shuffle") else ds
    except Exception as e:
        print(f"Could not load Ops-6K for Stage 1: {e}")

    print("Using fallback Stage 1 prompts with live WCC evaluation support")
    return _dataset_from_rows([
        {
            "prompt": _wrap_prompt_as_chat(
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using union-find with path compression."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "fallback_wcc",
        },
        {
            "prompt": _wrap_prompt_as_chat(
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "optimized for sparse disconnected graphs with early convergence."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 1,
            "data_source": "fallback_wcc",
        },
        {
            "prompt": _wrap_prompt_as_chat(
                f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                "using shared memory staging for dense frontiers."
            ),
            "ops": ["weakly_connected_components"],
            "difficulty": 2,
            "data_source": "fallback_wcc",
        },
    ])


# --- Reward function ---


def make_cuda_reward_func(task_rows: list[dict]):
    """Create a reward function that evaluates CUDA kernels via remote eval backend."""
    from training.multi_turn_rollout import extract_cuda_code
    from training.task_support import (
        build_prompt_lookup,
        compute_task_reward,
        evaluate_code_remote,
    )

    prompt_lookup = build_prompt_lookup(task_rows)

    def _to_text(value) -> str:
        """Extract text from TRL completion/prompt (may be str, list of dicts, or dict)."""
        if isinstance(value, str):
            return value
        if isinstance(value, list):
            # Chat format: [{"role": "assistant", "content": "..."}, ...]
            for msg in reversed(value):
                if isinstance(msg, dict) and msg.get("content"):
                    return msg["content"]
            return ""
        if isinstance(value, dict):
            return value.get("content", "")
        return str(value)

    def cuda_eval_reward(completions, prompts=None, **kwargs) -> list[float]:
        """Evaluate generated CUDA kernels and return rewards.

        Dense reward ladder (richer gradient signal than flat -1.0):
          -1.0  no code at all
          -0.7  truncated (ran out of tokens before completing code)
          -0.5  code too short to be valid
          -0.4  remote compile failure
          -0.2  correctness failure
          +0.2 to +1.0  correct with varying speedup
        """
        rewards = []
        stats = {"no_code": 0, "truncated": 0, "short": 0, "eval_ok": 0, "error": 0}
        for i, completion in enumerate(completions):
            try:
                completion_text = _to_text(completion)
                if i == 0:  # Log first completion for debugging
                    print(f"  [reward] completion[0] preview: {completion_text[:300]!r}", flush=True)
                cuda_code = extract_cuda_code(completion_text)

                # Dense reward for different failure modes
                if not cuda_code:
                    # Check if truncated (hit max completion length)
                    if len(completion_text) >= MAX_COMPLETION_LENGTH - 10:
                        rewards.append(-0.7)
                        stats["truncated"] += 1
                        print(f"  [reward] sample {i}: truncated reward=-0.70", flush=True)
                    else:
                        rewards.append(-1.0)
                        stats["no_code"] += 1
                        print(f"  [reward] sample {i}: no_code reward=-1.00", flush=True)
                    continue

                if len(cuda_code.strip()) < 20:
                    rewards.append(-0.5)
                    stats["short"] += 1
                    print(f"  [reward] sample {i}: short_code reward=-0.50", flush=True)
                    continue

                # Find the matching task row for this prompt
                # normalize_task_row keys on str(prompt), so match using str() for chat-format lists
                prompt_raw = prompts[i] if prompts else ""
                prompt_key = str(prompt_raw).strip()
                task_row = normalize_task_row(prompt_lookup.get(prompt_key, {"prompt": _to_text(prompt_raw)}))

                result = evaluate_code_remote(
                    cuda_code,
                    task_row,
                    skip_benchmark=SKIP_BENCHMARK,
                    trace_id=f"stage1_step",
                )
                reward = compute_task_reward(result)
                rewards.append(float(reward) if reward is not None else -1.0)
                stats["eval_ok"] += 1
                status = "compile" if result.get("compiles") else "fail"
                if result.get("correct"):
                    status = "correct"
                print(f"  [reward] sample {i}: {status} reward={rewards[-1]:.2f}", flush=True)
            except Exception as e:
                print(f"  [reward] sample {i}: error={str(e)[:200]}", flush=True)
                rewards.append(-1.0)
                stats["error"] += 1

        # Diagnostic summary
        n = len(rewards)
        if n > 0:
            import statistics
            r_mean = statistics.mean(rewards)
            r_std = statistics.stdev(rewards) if n > 1 else 0.0
            print(
                f"  [reward] batch summary: n={n} mean={r_mean:.3f} std={r_std:.3f} "
                f"no_code={stats['no_code']} truncated={stats['truncated']} "
                f"short={stats['short']} eval_ok={stats['eval_ok']} error={stats['error']}",
                flush=True,
            )
            if r_std == 0 and n > 1:
                print("  [reward] WARNING: zero reward variance — GRPO will produce zero gradients", flush=True)
        return rewards

    return cuda_eval_reward


# --- Training ---

def main():
    """Run Stage 1 GRPO warm-up (single-turn, no Unsloth)."""
    print(f"=== Stage 1: GRPO Warm-up for {TARGET_GPU} ({TARGET_ARCH}) ===")
    print(f"  Max training steps: {MAX_STEPS}")
    print(f"  Max prompt length: {GRPO_RUNTIME.max_prompt_length}")
    print(f"  Max completion length: {MAX_COMPLETION_LENGTH}")
    print(
        "  Shared GRPO runtime: "
        f"G={GRPO_RUNTIME.num_generations} "
        f"batch={GRPO_RUNTIME.per_device_train_batch_size}x{GRPO_RUNTIME.gradient_accumulation_steps} "
        f"(effective={GRPO_RUNTIME.effective_batch_size})"
    )
    print(f"  Mode: vanilla TRL (no Unsloth), skip_benchmark={SKIP_BENCHMARK}", flush=True)

    model, tokenizer = load_model_and_tokenizer()

    dataset = load_stage1_dataset()
    task_rows = [normalize_task_row(row) for row in dataset.to_list()]

    # Canary: verify raw generation works before entering GRPO.
    print("[canary] Testing raw model.generate()...", flush=True)
    import torch
    _canary_inputs = tokenizer(text="Write a CUDA vector add kernel.", return_tensors="pt").to(model.device)
    try:
        with torch.no_grad():
            _canary_out = model.generate(
                **_canary_inputs, max_new_tokens=32, temperature=1.0,
                do_sample=True, pad_token_id=tokenizer.pad_token_id,
            )
        print(f"[canary] PASS — {len(_canary_out[0])} tokens generated", flush=True)
    except Exception as e:
        raise RuntimeError(f"Canary generation failed — model cannot generate: {e}") from e

    reward_func = make_cuda_reward_func(task_rows)

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
                "remove_unused_columns": False,
            },
        )
    )

    trainer = TRLOOGRPOTrainer(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_func],
        args=config,
        train_dataset=dataset,
    )

    print("Starting Stage 1 training...", flush=True)
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)
    print(f"Stage 1 complete. Checkpoint saved to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

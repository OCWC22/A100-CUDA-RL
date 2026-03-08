"""
Stage 3: GRPO with Curriculum — hackathon pilot (50 steps).

GPU split: H200 handles model weights + generation + gradient updates.
           A100 (CoreWeave via Northflank) handles all performance reward.
           You cannot optimize A100 performance by measuring on H200.

Multi-turn agentic training via TRL's rollout_func:
  - 3 turns per episode (hackathon default)
  - CurriculumManager for progressive difficulty
  - Temperature 0.7 for exploitation
  - LR 3e-6 (per GRPO-15.1 hackathon config)
  - 50 max_steps (hackathon pilot — only if Gate G-0.8 passes)
  - G=2 (reduced from 4 — fewer zero-gradient steps)
  - H200: local nvcc compile check (fast-fail syntax errors)
  - A100 (CoreWeave/Northflank): execution correctness + speedup timing for reward
  - Discrete reward {-1, 1, 2, 3} per CUDA Agent ablation
  - vLLM disabled by default for hackathon bring-up (`KERNELFORGE_USE_VLLM=0`)

GRPO is experimental. SkyDiscover + SFT are primary hedges.
See docs/GRPO_DEEP_DIVE.md GRPO-15.1 for hackathon configuration.
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

os.environ.setdefault("UNSLOTH_VLLM_STANDBY", "1")

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

from trl import GRPOConfig, GRPOTrainer

from training.custom_grpo_trainer import TRLOOGRPOTrainer
from training.model_loader import load_model_and_tokenizer
from training.curriculum import CurriculumManager, format_problem_prompt
from training.dataset_loader import Dataset, MiniDataset, load_training_dataset
from training.multi_turn_rollout import make_multi_turn_rollout
from training.task_support import normalize_task_row

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
EVAL_BACKEND = os.getenv("KERNELFORGE_EVAL_BACKEND", "coreweave")
STAGE2_OUTPUT = os.getenv("KERNELFORGE_STAGE2_OUTPUT", "outputs/kernelforge-stage2")
STAGE1_OUTPUT = os.getenv("KERNELFORGE_STAGE1_OUTPUT", "outputs/kernelforge-stage1")
OUTPUT_DIR = os.getenv("KERNELFORGE_STAGE3_OUTPUT", "outputs/kernelforge-stage3")
IS_LINUX = sys.platform.startswith("linux")
USE_VLLM = os.getenv("KERNELFORGE_USE_VLLM", "0") == "1" and IS_LINUX
VLLM_GPU_MEMORY_UTILIZATION = float(os.getenv("KERNELFORGE_VLLM_GPU_MEMORY_UTILIZATION", "0.6"))
OPTIMIZER = "paged_adamw_8bit" if IS_LINUX else "adamw_torch"
USE_BF16 = IS_LINUX

# Multi-turn configuration
MAX_TURNS = int(os.getenv("KERNELFORGE_STAGE3_MAX_TURNS", "3"))
MAX_STEPS = int(os.getenv("KERNELFORGE_STAGE3_MAX_STEPS", "50"))
MAX_COMPLETION_LENGTH = int(os.getenv("KERNELFORGE_STAGE3_MAX_COMPLETION_LENGTH", "768"))
VLLM_MODE = os.getenv("KERNELFORGE_VLLM_MODE", "server").strip().lower()
VLLM_SERVER_BASE_URL = os.getenv("KERNELFORGE_VLLM_SERVER_BASE_URL", "").strip()
USE_TRLOO = os.getenv("KERNELFORGE_USE_TRLOO", "1") == "1"

# GRPO config constants — used by both _validate_config() and grpo_kwargs
PER_DEVICE_BATCH_SIZE = 1
GRADIENT_ACCUMULATION_STEPS = 4
NUM_GENERATIONS = 2
# Local compile check controlled by KERNELFORGE_LOCAL_COMPILE in multi_turn_rollout.py.
# Set KERNELFORGE_LOCAL_COMPILE=0 to skip local compile pre-check (slower but simpler).


# Global curriculum manager — shared between reward function and training loop
curriculum = CurriculumManager()


def reward_from_env_with_curriculum(completions, **kwargs) -> list[float]:
    """Extract environment rewards and feed them to the curriculum manager."""
    env_rewards = kwargs.get("env_reward", [])
    if not env_rewards:
        return [-1.0] * len(completions)

    rewards = [float(r) for r in env_rewards]

    # Feed each reward to curriculum for promotion/demotion
    for r in rewards:
        transition = curriculum.record_reward(r)
        if transition:
            print(f"  Curriculum transition: {transition} -> now in phase '{curriculum.phase_name}'")

    return rewards


# --- Dataset: curriculum-aware prompt generation ---


def _dataset_from_rows(rows: list[dict]) -> Dataset:
    if hasattr(Dataset, "from_list"):
        return Dataset.from_list(rows)
    return MiniDataset(rows)

def build_curriculum_dataset(num_prompts: int = 200) -> tuple[Dataset, list[dict]]:
    """Generate prompts from curriculum manager, sampling from current phase."""
    prompts = []
    sampled_rows = []
    for _ in range(num_prompts):
        problem = None
        for _attempt in range(32):
            candidate = normalize_task_row(curriculum.get_problem())
            if candidate.get("supports_evaluation"):
                problem = candidate
                break
        if problem is None:
            problem = normalize_task_row(
                {
                    "prompt": (
                        f"Write a CUDA Weakly Connected Components kernel for {TARGET_GPU} ({TARGET_ARCH}) "
                        "using union-find with path compression."
                    ),
                    "ops": ["weakly_connected_components"],
                    "difficulty": 1,
                    "data_source": "stage3_fallback",
                }
            )
        sampled = problem
        sampled["prompt"] = format_problem_prompt(sampled)
        prompts.append(sampled)
        sampled_rows.append(sampled)
    return _dataset_from_rows(prompts), sampled_rows


def _validate_config():
    effective_batch = PER_DEVICE_BATCH_SIZE * GRADIENT_ACCUMULATION_STEPS
    if effective_batch % NUM_GENERATIONS != 0:
        raise ValueError(
            f"Effective batch size ({effective_batch}) must be divisible by "
            f"num_generations ({NUM_GENERATIONS}) per TRL GRPO requirement."
        )
    if USE_VLLM and VLLM_MODE == "server" and not VLLM_SERVER_BASE_URL:
        raise ValueError(
            "KERNELFORGE_USE_VLLM=1 with KERNELFORGE_VLLM_MODE=server requires "
            "KERNELFORGE_VLLM_SERVER_BASE_URL to be set."
        )


# --- Training ---

def main():
    """Run Stage 3 GRPO with curriculum and multi-turn agentic loop."""
    print(f"=== Stage 3: Multi-Turn GRPO with Curriculum for {TARGET_GPU} ({TARGET_ARCH}) ===")
    print(f"  Starting phase: {curriculum.phase_name}")
    print(f"  Max turns per episode: {MAX_TURNS}")
    print(f"  Max training steps: {MAX_STEPS}")
    print(f"  Max completion length: {MAX_COMPLETION_LENGTH}")

    try:
        ops6k_max = int(os.getenv("KERNELFORGE_STAGE3_OPS6K_MAX", "128"))
        combined_rows = load_training_dataset(
            stage="stage3",
            ops6k_max=ops6k_max,
            seed=42,
            curriculum_manager=curriculum,
        )
        if isinstance(combined_rows, list):
            print(f"  Injected combined dataset into curriculum ({len(combined_rows)} rows)")
    except Exception as e:
        print(f"  Could not inject combined dataset into curriculum: {e}")
        print("  Continuing with built-in curriculum problems only")

    checkpoint_path = None
    if os.path.exists(STAGE2_OUTPUT):
        checkpoint_path = STAGE2_OUTPUT
    elif os.path.exists(STAGE1_OUTPUT):
        checkpoint_path = STAGE1_OUTPUT

    model, tokenizer = load_model_and_tokenizer(checkpoint_path=checkpoint_path)
    dataset, sampled_rows = build_curriculum_dataset(num_prompts=200)

    rollout_func = make_multi_turn_rollout(
        max_turns=MAX_TURNS,
        skill_md_gpu=TARGET_GPU.lower(),
        problem_metadata=sampled_rows,
    )

    _validate_config()

    grpo_kwargs = dict(
        learning_rate=3e-6,
        temperature=0.7,
        num_generations=NUM_GENERATIONS,
        num_iterations=1,
        beta=0.0,                        # No ref model — saves memory
        epsilon=0.2,
        scale_rewards="batch",           # Better for sparse expensive env
        remove_unused_columns=False,     # Custom rollout needs extra columns
        max_prompt_length=3072,
        max_completion_length=MAX_COMPLETION_LENGTH,
        per_device_train_batch_size=PER_DEVICE_BATCH_SIZE,
        gradient_accumulation_steps=GRADIENT_ACCUMULATION_STEPS,
        max_steps=MAX_STEPS,
        optim=OPTIMIZER,
        bf16=USE_BF16,
        gradient_checkpointing=True,
        report_to="none",
        output_dir=OUTPUT_DIR,
        logging_steps=1,
        save_steps=25,
        save_total_limit=2,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.05,
    )
    if USE_VLLM:
        grpo_kwargs["use_vllm"] = True
        grpo_kwargs["vllm_mode"] = VLLM_MODE
        if VLLM_MODE == "server":
            grpo_kwargs["vllm_server_base_url"] = VLLM_SERVER_BASE_URL
        elif VLLM_MODE == "colocate":
            grpo_kwargs["vllm_gpu_memory_utilization"] = VLLM_GPU_MEMORY_UTILIZATION

    config = GRPOConfig(**grpo_kwargs)

    trainer_cls = TRLOOGRPOTrainer if USE_TRLOO else GRPOTrainer
    print(f"  Trainer: {trainer_cls.__name__} (USE_TRLOO={USE_TRLOO})")
    trainer = trainer_cls(
        model=model,
        processing_class=tokenizer,
        reward_funcs=[reward_from_env_with_curriculum],
        rollout_func=rollout_func,
        args=config,
        train_dataset=dataset,
    )

    # Decision gate: if post-RFT model already strong, skip GRPO
    skip_threshold_compile = float(os.getenv("KERNELFORGE_GATE_COMPILE", "0.95"))
    skip_threshold_correct = float(os.getenv("KERNELFORGE_GATE_CORRECT", "0.85"))
    if os.getenv("KERNELFORGE_DECISION_GATE", "0") == "1" and checkpoint_path:
        print("Running decision gate evaluation...")
        try:
            from evaluation.eval_model import evaluate_checkpoint
            gate_results = evaluate_checkpoint(checkpoint_path, num_problems=20)
            compile_rate = gate_results.get("compile_rate", 0.0)
            correct_rate = gate_results.get("correct_rate", 0.0)
            print(f"  Gate results: compile={compile_rate:.1%}, correct={correct_rate:.1%}")
            if compile_rate >= skip_threshold_compile and correct_rate >= skip_threshold_correct:
                print(f"  DECISION GATE: compile={compile_rate:.1%} >= {skip_threshold_compile:.0%}, "
                      f"correct={correct_rate:.1%} >= {skip_threshold_correct:.0%}")
                print("  Skipping Stage 3 GRPO — model already strong. Use SkyDiscover for further gains.")
                return
        except Exception as e:
            print(f"  Decision gate failed ({e}), proceeding with GRPO...")

    print("Starting Stage 3 training...")
    trainer.train()

    model.save_pretrained(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)

    # Print curriculum summary
    status = curriculum.status()
    print(f"\nStage 3 complete. Checkpoint saved to {OUTPUT_DIR}")
    print(f"Final curriculum phase: {status['phase']} ({status['phase_idx']+1}/{status['total_phases']})")
    print(f"Phase transitions: {status['transitions']}")
    print(f"Hit rate: {status['hit_rate']:.1%}, Positive rate: {status['positive_rate']:.1%}")


if __name__ == "__main__":
    main()

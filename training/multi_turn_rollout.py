"""
Multi-turn rollout for GRPOTrainer.

The policy generates on the training GPU, while correctness and runtime reward
are computed remotely on the target A100 via CoreWeave/Northflank (or Modal).
"""

from __future__ import annotations

import json
import os
import re
import statistics
import subprocess
import tempfile
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from training.curriculum import format_topology_context
from training.run_metadata import utc_timestamp_rfc3339
from training.task_support import (
    build_generation_prompt,
    build_prompt_lookup,
    compute_task_reward,
    evaluate_code_remote,
    evaluate_code_remote_batch,
    normalize_eval_result,
    normalize_task_row,
)
LOCAL_COMPILE_CHECK = os.getenv("KERNELFORGE_LOCAL_COMPILE", "1") == "1"
SKIP_BENCHMARK = os.getenv("KERNELFORGE_SKIP_BENCHMARK", "0") == "1"
DEBUG_TIMINGS = os.getenv("KERNELFORGE_DEBUG_TIMINGS", "0") == "1"
BATCH_EVAL = os.getenv("KERNELFORGE_BATCH_EVAL", "0") == "1"
TARGET_CUDA_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
MAX_FEEDBACK_CHARS = int(os.getenv("KERNELFORGE_MAX_FEEDBACK_CHARS", "1200"))
MAX_ERROR_CHARS = int(os.getenv("KERNELFORGE_MAX_ERROR_CHARS", "800"))
ROLLOUT_LOG_PATH = Path(
    os.getenv("KERNELFORGE_ROLLOUT_LOG", "outputs/rollout_metrics.jsonl")
).resolve()


def extract_cuda_code(text: str) -> str:
    """Extract CUDA code from model output (fenced block or raw __global__)."""
    for marker in ["```cuda", "```cpp", "```c", "```c++"]:
        if marker in text:
            start = text.index(marker) + len(marker)
            end = text.find("```", start)
            if end != -1:
                return text[start:end].strip()

    if re.search(r"__global__\s+void\s+\w+", text) or "PYBIND11_MODULE" in text:
        return text.strip()
    return ""


def _append_rollout_log(record: dict[str, Any]) -> None:
    """Append one rollout event to a JSONL log file."""
    try:
        ROLLOUT_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        payload = dict(record)
        payload.setdefault("timestamp", utc_timestamp_rfc3339())
        with ROLLOUT_LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload) + "\n")
    except OSError:
        pass


def _elapsed_ms(start: float) -> float:
    """Return elapsed wall time in milliseconds."""
    return round((perf_counter() - start) * 1000.0, 3)


def _local_compile_check(code: str) -> tuple[bool, str]:
    """Quick local nvcc syntax check before paying for a remote evaluation."""
    if not LOCAL_COMPILE_CHECK:
        return True, ""

    try:
        with tempfile.NamedTemporaryFile(suffix=".cu", mode="w", delete=False) as f:
            f.write(code)
            cu_path = f.name

        obj_path = cu_path.replace(".cu", ".o")
        proc = subprocess.run(
            ["nvcc", f"-arch={TARGET_CUDA_ARCH}", "-c", cu_path, "-o", obj_path],
            capture_output=True,
            text=True,
            timeout=15,
        )

        for path in (cu_path, obj_path):
            try:
                os.unlink(path)
            except OSError:
                pass

        if proc.returncode != 0:
            return False, proc.stderr[:1000]
        return True, ""
    except FileNotFoundError:
        return True, ""
    except subprocess.TimeoutExpired:
        return False, "Local compile timed out (15s)"
    except Exception:
        return True, ""


def _compute_reward_from_result(result: dict) -> float:
    """Compute discrete milestone reward from evaluation result."""
    return compute_task_reward(result)


def _format_feedback(result: dict, reward: float, turn: int) -> str:
    """Format evaluator feedback for the next policy turn."""
    result = normalize_eval_result(result)
    parts = [f"[Turn {turn + 1} Result]"]

    if not result.get("compiles"):
        error = result.get("error", "unknown compilation error")
        parts.append(f"COMPILATION FAILED:\n{error[:MAX_ERROR_CHARS]}")
        parts.append("Fix the compilation errors above and resubmit.")
    elif not result.get("correct"):
        msg = result.get("verifier_msg") or result.get("error") or "unknown verification failure"
        parts.append(f"VERIFICATION FAILED: {msg}")
        parts.append("Your kernel produced incorrect output. Fix the implementation.")
    else:
        runtime = float(result.get("runtime_ms", 0.0) or 0.0)
        speedup_eager = float(result.get("speedup_vs_orig", 0.0) or 0.0)
        speedup_compile = float(result.get("speedup_vs_dg", 0.0) or 0.0)
        stats = result.get("runtime_stats", {}) or {}
        parts.append(f"CORRECT. Runtime: {runtime:.3f}ms")
        parts.append(f"  Speedup vs eager: {speedup_eager:.2f}x")
        if speedup_compile:
            parts.append(f"  Speedup vs torch.compile: {speedup_compile:.2f}x")
        if stats:
            parts.append(
                f"  Stats: mean={float(stats.get('mean', 0.0)):.3f}ms, "
                f"std={float(stats.get('std', 0.0)):.3f}ms"
            )

        if reward <= 1.0:
            parts.append(
                "Kernel is correct but not faster than eager PyTorch. "
                "Try reducing memory traffic or using shared memory tiling."
            )
        elif reward <= 2.0:
            parts.append(
                "Faster than eager PyTorch but not torch.compile. Push toward "
                "beating torch.compile with better occupancy or warp-level primitives."
            )

    feedback = "\n".join(parts)
    return feedback[:MAX_FEEDBACK_CHARS]


_baselines_cache: dict[str, Any] | None = None


def _needs_wcc_baselines(task_rows: list[dict[str, Any]]) -> bool:
    """Return True when this rollout needs runtime baselines."""
    if SKIP_BENCHMARK:
        return False
    return any(normalize_task_row(row).get("evaluation_backend") == "wcc" for row in task_rows)


def _get_baselines(required: bool = True) -> tuple[float | None, float | None, float]:
    """Fetch baseline timings from eval backend (cached across calls)."""
    if not required:
        return None, None, 0.0

    global _baselines_cache
    fetch_start = perf_counter()
    fetched = False
    if _baselines_cache is None:
        try:
            from openenv_env.eval_backend import dispatch_eval

            _baselines_cache = dispatch_eval("profile_baselines") or {}
            fetched = True
        except Exception as exc:
            print(f"Baseline profiling failed: {exc}")
            _baselines_cache = {}
            fetched = True
    return (
        _baselines_cache.get("original_ms"),
        _baselines_cache.get("doublegraph_ms"),
        _elapsed_ms(fetch_start) if fetched else 0.0,
    )


def _print_turn_summary(
    turn: int,
    max_turns: int,
    active_count: int,
    remote_count: int,
    generation_ms: float,
    dispatch_ms: float,
    rewards: list[float],
) -> None:
    """Emit one compact progress line per rollout turn."""
    reward_preview = ", ".join(f"{reward:.1f}" for reward in rewards[:8])
    if len(rewards) > 8:
        reward_preview += ", ..."
    mode = "fast" if SKIP_BENCHMARK else "full"
    print(
        f"[rollout] turn {turn}/{max_turns} mode={mode} active={active_count} "
        f"remote={remote_count} gen={generation_ms:.1f}ms dispatch={dispatch_ms:.1f}ms "
        f"rewards=[{reward_preview}]"
    )


def _generate_rollout_completions_compat(trainer: Any, prompts: list[str]) -> list[dict]:
    """Compatibility shim for trl.experimental.openenv.generate_rollout_completions.

    Works with trl<=0.24.0 which lacks the openenv experimental module.
    Returns the same format: list of dicts with prompt_ids, completion_ids, logprobs, text.
    """
    import signal
    import torch

    tokenizer = trainer.processing_class
    model = trainer.model
    gen_timeout = int(os.getenv("KERNELFORGE_GENERATION_TIMEOUT", "120"))

    results = []
    for prompt_idx, prompt in enumerate(prompts):
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=4096)
        input_ids = inputs["input_ids"].to(model.device)
        prompt_ids = input_ids[0].tolist()

        def _timeout_handler(signum, frame):
            raise TimeoutError(
                f"model.generate() timed out after {gen_timeout}s on prompt {prompt_idx}. "
                "Likely broken attention backend."
            )

        old_handler = signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(gen_timeout)
        try:
            with torch.no_grad():
                output = model.generate(
                    input_ids,
                    max_new_tokens=getattr(trainer.args, "max_completion_length", 2048),
                    temperature=getattr(trainer.args, "temperature", 1.0),
                    do_sample=True,
                    pad_token_id=tokenizer.pad_token_id,
                )
        finally:
            signal.alarm(0)
            signal.signal(signal.SIGALRM, old_handler)

        completion_ids = output[0][len(prompt_ids):].tolist()
        text = tokenizer.decode(completion_ids, skip_special_tokens=True)

        # Placeholder logprobs (zeros) — acceptable for GRPO which uses group-relative rewards.
        logprobs = [0.0] * len(completion_ids)

        results.append({
            "prompt_ids": prompt_ids,
            "completion_ids": completion_ids,
            "logprobs": logprobs,
            "text": text,
        })

    return results


def make_multi_turn_rollout(
    max_turns: int = 3,
    skill_md_gpu: str | None = None,
    problem_metadata: list[dict] | None = None,
) -> Callable:
    """Create a task-aware rollout_func for GRPOTrainer."""
    try:
        from trl.experimental.openenv import generate_rollout_completions
    except (ImportError, ModuleNotFoundError):
        generate_rollout_completions = _generate_rollout_completions_compat
    from openenv_env.skill_builder import build_skill_md

    gpu_name = skill_md_gpu or os.getenv("KERNELFORGE_TARGET_GPU", "a100").lower()
    prompt_lookup = build_prompt_lookup(problem_metadata or [])

    def rollout_func(prompts: list[str], trainer: Any) -> dict:
        tokenizer = trainer.processing_class
        skill_context = build_skill_md(gpu_name)
        task_rows = [
            normalize_task_row(prompt_lookup.get(prompt, {"prompt": prompt}))
            for prompt in prompts
        ]
        base_prompts = []
        for task_row in task_rows:
            topology_ctx = format_topology_context(task_row)
            base_prompts.append(
                build_generation_prompt(
                    task_row,
                    skill_context=skill_context,
                    topology_context=topology_ctx,
                )
            )
        current_prompts = list(base_prompts)

        baseline_orig, baseline_dg, baseline_fetch_ms = _get_baselines(
            required=_needs_wcc_baselines(task_rows)
        )

        all_prompt_ids: list[list[int]] = [[] for _ in prompts]
        all_completion_ids: list[list[int]] = [[] for _ in prompts]
        all_logprobs: list[list[float]] = [[] for _ in prompts]
        all_best_rewards: list[float] = [-1.0] * len(prompts)
        done = [False] * len(prompts)

        def finalize_prompt(
            prompt_idx: int,
            turn_idx: int,
            trace_id: str,
            result: dict[str, Any],
            reward: float,
            local_compile_ms: float,
            dispatch_ms: float,
            turn_start: float,
        ) -> None:
            normalized_result = normalize_eval_result(result)
            task_row = task_rows[prompt_idx]
            if reward > all_best_rewards[prompt_idx]:
                all_best_rewards[prompt_idx] = reward

            _append_rollout_log(
                {
                    "prompt_index": prompt_idx,
                    "turn": turn_idx + 1,
                    "trace_id": trace_id,
                    "task_id": task_row.get("task_id", ""),
                    "evaluation_backend": task_row.get("evaluation_backend"),
                    "reward": reward,
                    "compiles": bool(normalized_result.get("compiles")),
                    "correct": bool(normalized_result.get("correct")),
                    "runtime_ms": float(normalized_result.get("runtime_ms", 0.0) or 0.0),
                    "speedup_vs_orig": float(normalized_result.get("speedup_vs_orig", 0.0) or 0.0),
                    "speedup_vs_dg": float(normalized_result.get("speedup_vs_dg", 0.0) or 0.0),
                    "baseline_fetch_ms": baseline_fetch_ms if turn_idx == 0 else 0.0,
                    "generation_ms": generation_ms,
                    "local_compile_ms": local_compile_ms,
                    "dispatch_ms": dispatch_ms,
                    "turn_total_ms": _elapsed_ms(turn_start),
                    "phase_timings": normalized_result.get("phase_timings", {}),
                }
            )

            if DEBUG_TIMINGS:
                phase_timings = normalized_result.get("phase_timings", {})
                print(
                    f"[rollout] trace={trace_id} task={task_row.get('task_id', '') or prompt_idx} "
                    f"turn={turn_idx + 1} compile={local_compile_ms:.1f}ms "
                    f"dispatch={dispatch_ms:.1f}ms total={_elapsed_ms(turn_start):.1f}ms "
                    f"eval={phase_timings}"
                )

            if reward >= 3.0 or turn_idx == max_turns - 1:
                done[prompt_idx] = True
                return

            feedback = _format_feedback(normalized_result, reward, turn_idx)
            current_prompts[prompt_idx] = base_prompts[prompt_idx] + f"\n\n{feedback}"

        for turn in range(max_turns):
            active_indices = [idx for idx, is_done in enumerate(done) if not is_done]
            if not active_indices:
                break

            turn_start = perf_counter()
            active_prompts = [current_prompts[idx] for idx in active_indices]
            print(
                f"[rollout] turn {turn + 1}/{max_turns} starting generation "
                f"for {len(active_prompts)} prompts "
                f"(max_completion_length={getattr(trainer.args, 'max_completion_length', '?')})"
            )
            generation_start = perf_counter()
            outputs = generate_rollout_completions(trainer, active_prompts)
            generation_ms = _elapsed_ms(generation_start)
            print(f"[rollout] turn {turn + 1}/{max_turns} generation complete in {generation_ms:.1f}ms")

            pending_jobs: list[dict[str, Any]] = []
            turn_rewards: list[float] = []
            total_dispatch_ms = 0.0

            for output_idx, prompt_idx in enumerate(active_indices):
                outputs_for_prompt = outputs[output_idx]
                all_prompt_ids[prompt_idx].extend(outputs_for_prompt["prompt_ids"])
                all_completion_ids[prompt_idx].extend(outputs_for_prompt["completion_ids"])
                all_logprobs[prompt_idx].extend(outputs_for_prompt["logprobs"])

                completion_text = outputs_for_prompt.get("text") or tokenizer.decode(
                    outputs_for_prompt["completion_ids"], skip_special_tokens=True
                )
                trace_id = uuid4().hex
                local_compile_ms = 0.0

                code = extract_cuda_code(completion_text)
                if not code:
                    reward = -1.0
                    result = {
                        "compiles": False,
                        "correct": False,
                        "trace_id": trace_id,
                        "task_id": task_rows[prompt_idx].get("task_id", ""),
                        "error": (
                            "No valid CUDA/C++ code was found. Return a fenced code block "
                            "or a raw CUDA extension source file."
                        ),
                    }
                    finalize_prompt(prompt_idx, turn, trace_id, result, reward, 0.0, 0.0, turn_start)
                    turn_rewards.append(reward)
                    continue

                local_compile_start = perf_counter()
                compiles_locally, compile_err = _local_compile_check(code)
                local_compile_ms = _elapsed_ms(local_compile_start)
                if not compiles_locally:
                    reward = -1.0
                    result = {
                        "compiles": False,
                        "correct": False,
                        "trace_id": trace_id,
                        "task_id": task_rows[prompt_idx].get("task_id", ""),
                        "error": compile_err[:200],
                    }
                    finalize_prompt(
                        prompt_idx,
                        turn,
                        trace_id,
                        result,
                        reward,
                        local_compile_ms,
                        0.0,
                        turn_start,
                    )
                    turn_rewards.append(reward)
                    continue

                if not task_rows[prompt_idx].get("supports_evaluation"):
                    reward = -1.0
                    result = {
                        "compiles": False,
                        "correct": False,
                        "trace_id": trace_id,
                        "task_id": task_rows[prompt_idx].get("task_id", ""),
                        "error": task_rows[prompt_idx].get(
                            "support_reason", "Unsupported evaluation backend"
                        ),
                    }
                    finalize_prompt(
                        prompt_idx,
                        turn,
                        trace_id,
                        result,
                        reward,
                        local_compile_ms,
                        0.0,
                        turn_start,
                    )
                    turn_rewards.append(reward)
                    continue

                pending_jobs.append(
                    {
                        "prompt_idx": prompt_idx,
                        "trace_id": trace_id,
                        "code": code,
                        "local_compile_ms": local_compile_ms,
                        "task_row": task_rows[prompt_idx],
                    }
                )

            if pending_jobs:
                if BATCH_EVAL:
                    dispatch_start = perf_counter()
                    try:
                        batch_results = evaluate_code_remote_batch(
                            [job["code"] for job in pending_jobs],
                            [job["task_row"] for job in pending_jobs],
                            baseline_orig_ms=baseline_orig,
                            baseline_dg_ms=baseline_dg,
                            skip_benchmark=SKIP_BENCHMARK,
                            trace_ids=[job["trace_id"] for job in pending_jobs],
                        )
                    except Exception as exc:
                        batch_results = [
                            normalize_eval_result(
                                {
                                    "compiles": False,
                                    "correct": False,
                                    "trace_id": job["trace_id"],
                                    "task_id": job["task_row"].get("task_id", ""),
                                    "error": str(exc)[:200],
                                }
                            )
                            for job in pending_jobs
                        ]
                    total_dispatch_ms = _elapsed_ms(dispatch_start)

                    for job, result in zip(pending_jobs, batch_results):
                        reward = float(result.get("reward", _compute_reward_from_result(result)))
                        finalize_prompt(
                            job["prompt_idx"],
                            turn,
                            job["trace_id"],
                            result,
                            reward,
                            job["local_compile_ms"],
                            total_dispatch_ms,
                            turn_start,
                        )
                        turn_rewards.append(reward)
                else:
                    for job in pending_jobs:
                        dispatch_start = perf_counter()
                        try:
                            result = evaluate_code_remote(
                                job["code"],
                                job["task_row"],
                                baseline_orig_ms=baseline_orig,
                                baseline_dg_ms=baseline_dg,
                                skip_benchmark=SKIP_BENCHMARK,
                                trace_id=job["trace_id"],
                            )
                        except Exception as exc:
                            print(f"  [Turn {turn + 1}] Eval dispatch failed: {exc}")
                            result = {
                                "compiles": False,
                                "correct": False,
                                "trace_id": job["trace_id"],
                                "task_id": job["task_row"].get("task_id", ""),
                                "error": str(exc)[:200],
                            }
                        dispatch_ms = _elapsed_ms(dispatch_start)
                        total_dispatch_ms += dispatch_ms
                        reward = float(result.get("reward", _compute_reward_from_result(result)))
                        finalize_prompt(
                            job["prompt_idx"],
                            turn,
                            job["trace_id"],
                            result,
                            reward,
                            job["local_compile_ms"],
                            dispatch_ms,
                            turn_start,
                        )
                        turn_rewards.append(reward)

            _print_turn_summary(
                turn=turn + 1,
                max_turns=max_turns,
                active_count=len(active_indices),
                remote_count=len(pending_jobs),
                generation_ms=generation_ms,
                dispatch_ms=total_dispatch_ms,
                rewards=turn_rewards,
            )

            # Reward distribution diagnostics — critical for detecting dead signal
            if turn_rewards:
                r_mean = statistics.mean(turn_rewards)
                r_std = statistics.stdev(turn_rewards) if len(turn_rewards) > 1 else 0.0
                r_min = min(turn_rewards)
                r_max = max(turn_rewards)
                r_pos = sum(1 for r in turn_rewards if r > -1.0)
                print(
                    f"[rollout] reward distribution: mean={r_mean:.2f} std={r_std:.2f} "
                    f"min={r_min:.1f} max={r_max:.1f} "
                    f"positive={r_pos}/{len(turn_rewards)} "
                    f"turn_total={_elapsed_ms(turn_start):.1f}ms"
                )
                if r_std == 0.0:
                    print(
                        "[rollout] WARNING: zero reward variance — GRPO will produce zero gradients. "
                        "Check completion length, code extraction, and eval connectivity."
                    )

        # Rollout-level summary
        print(
            f"[rollout] complete: best_rewards={all_best_rewards} "
            f"mean={statistics.mean(all_best_rewards):.2f} "
            f"positive={sum(1 for r in all_best_rewards if r > -1.0)}/{len(all_best_rewards)}"
        )

        return {
            "prompt_ids": all_prompt_ids,
            "completion_ids": all_completion_ids,
            "logprobs": all_logprobs,
            "env_reward": all_best_rewards,
        }

    return rollout_func


def reward_from_env(completions: list[str], **kwargs: Any) -> list[float]:
    """Extract the rewards produced by rollout_func."""
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [float(reward) for reward in env_rewards]
    return [-1.0] * len(completions)

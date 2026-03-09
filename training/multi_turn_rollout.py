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
from collections import Counter
from pathlib import Path
from time import perf_counter
from typing import Any, Callable
from uuid import uuid4

from training.curriculum import format_topology_context
from training.run_metadata import utc_timestamp_rfc3339
from training.task_support import (
    build_generation_prompt,
    build_prompt_lookup,
    build_reward_contract,
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


def _completion_was_truncated(output: dict[str, Any], max_completion_length: int) -> bool:
    """Best-effort truncation detection from completion token count."""
    completion_ids = output.get("completion_ids") or []
    return len(completion_ids) >= max_completion_length


def _update_turn_diagnostics(counter: Counter[str], contract: dict[str, Any]) -> None:
    """Accumulate rollout diagnostics for one scored terminal turn."""
    if not contract.get("valid_for_loss", True):
        counter["masked_infra_invalid"] += 1
        return

    extraction_status = str(contract.get("extraction_status") or "ok")
    if extraction_status == "no_code":
        counter["extraction_fail"] += 1
    if extraction_status == "truncated_partial" or contract.get("truncated"):
        counter["truncated"] += 1

    reason = str(contract.get("termination_reason") or "")
    if reason == "local_compile_fail":
        counter["local_compile_fail"] += 1
    elif reason == "remote_compile_fail":
        counter["remote_compile_fail"] += 1
    elif reason == "runtime_error":
        counter["runtime_error"] += 1
    elif reason == "correctness_fail":
        counter["correctness_fail"] += 1
    elif reason == "correct_slow":
        counter["correct_slow"] += 1
    elif reason == "correct_parity":
        counter["correct_parity"] += 1
    elif reason == "correct_fast_eager":
        counter["correct_fast_eager"] += 1
    elif reason == "correct_fast_compile":
        counter["correct_fast_compile"] += 1


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


def _compute_reward_from_result(result: dict) -> float | None:
    """Compute dense training reward from evaluation result."""
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

        if reward <= 0.4:
            parts.append(
                "Kernel is correct but not faster than eager PyTorch. "
                "Try reducing memory traffic or using shared memory tiling."
            )
        elif reward <= 0.7:
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

        max_completion_length = int(getattr(trainer.args, "max_completion_length", 2048))
        terminal_prompt_ids: list[list[int]] = [[] for _ in prompts]
        terminal_completion_ids: list[list[int]] = [[] for _ in prompts]
        terminal_logprobs: list[list[float]] = [[] for _ in prompts]
        terminal_rewards: list[float | None] = [None] * len(prompts)
        terminal_contracts: list[dict[str, Any]] = [
            build_reward_contract(
                {"backend_error": True, "error": "Rollout not scored"},
                backend_error=True,
            )
            for _ in prompts
        ]
        done = [False] * len(prompts)

        def finalize_prompt(
            prompt_idx: int,
            turn_idx: int,
            trace_id: str,
            result: dict[str, Any],
            contract: dict[str, Any],
            sample: dict[str, Any],
            turn_diagnostics: Counter[str],
            local_compile_ms: float,
            dispatch_ms: float,
            turn_start: float,
        ) -> None:
            normalized_result = normalize_eval_result({**result, **contract})
            task_row = task_rows[prompt_idx]
            reward = normalized_result.get("training_reward")

            terminal_prompt_ids[prompt_idx] = list(sample.get("prompt_ids") or [])
            terminal_completion_ids[prompt_idx] = list(sample.get("completion_ids") or [])
            terminal_logprobs[prompt_idx] = list(sample.get("logprobs") or [])
            terminal_rewards[prompt_idx] = reward
            terminal_contracts[prompt_idx] = normalized_result
            _update_turn_diagnostics(turn_diagnostics, normalized_result)

            _append_rollout_log(
                {
                    "prompt_index": prompt_idx,
                    "turn": turn_idx + 1,
                    "trace_id": trace_id,
                    "task_id": task_row.get("task_id", ""),
                    "evaluation_backend": task_row.get("evaluation_backend"),
                    "reward": reward,
                    "public_reward_bucket": normalized_result.get("public_reward_bucket"),
                    "valid_for_loss": normalized_result.get("valid_for_loss"),
                    "termination_reason": normalized_result.get("termination_reason"),
                    "truncated": normalized_result.get("truncated"),
                    "extraction_status": normalized_result.get("extraction_status"),
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

            if (
                not normalized_result.get("valid_for_loss", True)
                or normalized_result.get("public_reward_bucket") in {1, 2, 3}
                or turn_idx == max_turns - 1
            ):
                done[prompt_idx] = True
                return

            feedback = _format_feedback(normalized_result, float(reward or -1.0), turn_idx)
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
            turn_rewards: list[float | None] = []
            turn_diagnostics: Counter[str] = Counter()
            total_dispatch_ms = 0.0

            for output_idx, prompt_idx in enumerate(active_indices):
                outputs_for_prompt = outputs[output_idx]
                sample = {
                    "prompt_ids": list(outputs_for_prompt.get("prompt_ids") or []),
                    "completion_ids": list(outputs_for_prompt.get("completion_ids") or []),
                    "logprobs": list(outputs_for_prompt.get("logprobs") or []),
                }

                completion_text = outputs_for_prompt.get("text") or tokenizer.decode(
                    outputs_for_prompt["completion_ids"], skip_special_tokens=True
                )
                trace_id = uuid4().hex
                local_compile_ms = 0.0
                completion_truncated = _completion_was_truncated(
                    outputs_for_prompt,
                    max_completion_length=max_completion_length,
                )
                task_row = task_rows[prompt_idx]
                supports_evaluation = bool(task_row.get("supports_evaluation"))

                code = extract_cuda_code(completion_text)
                if not code:
                    result = {
                        "compiles": False,
                        "correct": False,
                        "trace_id": trace_id,
                        "task_id": task_row.get("task_id", ""),
                        "error": (
                            "No valid CUDA/C++ code was found. Return a fenced code block "
                            "or a raw CUDA extension source file."
                        ),
                    }
                    contract = build_reward_contract(
                        result,
                        truncated=completion_truncated,
                        extraction_status="truncated_partial" if completion_truncated else "no_code",
                        local_compile_ok=True,
                        supports_evaluation=supports_evaluation,
                    )
                    finalize_prompt(
                        prompt_idx,
                        turn,
                        trace_id,
                        result,
                        contract,
                        sample,
                        turn_diagnostics,
                        0.0,
                        0.0,
                        turn_start,
                    )
                    turn_rewards.append(contract.get("training_reward"))
                    continue

                local_compile_start = perf_counter()
                compiles_locally, compile_err = _local_compile_check(code)
                local_compile_ms = _elapsed_ms(local_compile_start)
                if not compiles_locally:
                    result = {
                        "compiles": False,
                        "correct": False,
                        "trace_id": trace_id,
                        "task_id": task_row.get("task_id", ""),
                        "error": compile_err[:200],
                    }
                    contract = build_reward_contract(
                        result,
                        truncated=completion_truncated,
                        extraction_status="truncated_partial" if completion_truncated else "ok",
                        local_compile_ok=False,
                        supports_evaluation=supports_evaluation,
                    )
                    finalize_prompt(
                        prompt_idx,
                        turn,
                        trace_id,
                        result,
                        contract,
                        sample,
                        turn_diagnostics,
                        local_compile_ms,
                        0.0,
                        turn_start,
                    )
                    turn_rewards.append(contract.get("training_reward"))
                    continue

                if not supports_evaluation:
                    result = {
                        "compiles": False,
                        "correct": False,
                        "trace_id": trace_id,
                        "task_id": task_row.get("task_id", ""),
                        "error": task_row.get(
                            "support_reason", "Unsupported evaluation backend"
                        ),
                    }
                    contract = build_reward_contract(
                        result,
                        truncated=completion_truncated,
                        extraction_status="ok",
                        local_compile_ok=True,
                        supports_evaluation=False,
                    )
                    finalize_prompt(
                        prompt_idx,
                        turn,
                        trace_id,
                        result,
                        contract,
                        sample,
                        turn_diagnostics,
                        local_compile_ms,
                        0.0,
                        turn_start,
                    )
                    turn_rewards.append(contract.get("training_reward"))
                    continue

                pending_jobs.append(
                    {
                        "prompt_idx": prompt_idx,
                        "trace_id": trace_id,
                        "code": code,
                        "local_compile_ms": local_compile_ms,
                        "task_row": task_row,
                        "sample": sample,
                        "truncated": completion_truncated,
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
                                    "backend_error": True,
                                    "trace_id": job["trace_id"],
                                    "task_id": job["task_row"].get("task_id", ""),
                                    "error": str(exc)[:200],
                                }
                            )
                            for job in pending_jobs
                        ]
                    total_dispatch_ms = _elapsed_ms(dispatch_start)

                    for job, result in zip(pending_jobs, batch_results):
                        extraction_status = (
                            "truncated_partial"
                            if job["truncated"] and not result.get("compiles")
                            else str(result.get("extraction_status") or "ok")
                        )
                        contract = build_reward_contract(
                            result,
                            truncated=bool(job["truncated"]),
                            extraction_status=extraction_status,
                            local_compile_ok=True,
                            supports_evaluation=bool(job["task_row"].get("supports_evaluation")),
                            backend_error=result.get("backend_error"),
                        )
                        result = dict(result)
                        result.update(contract)
                        result["reward"] = contract.get("training_reward")
                        finalize_prompt(
                            job["prompt_idx"],
                            turn,
                            job["trace_id"],
                            result,
                            contract,
                            job["sample"],
                            turn_diagnostics,
                            job["local_compile_ms"],
                            total_dispatch_ms,
                            turn_start,
                        )
                        turn_rewards.append(contract.get("training_reward"))
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
                                "backend_error": True,
                                "trace_id": job["trace_id"],
                                "task_id": job["task_row"].get("task_id", ""),
                                "error": str(exc)[:200],
                            }
                        dispatch_ms = _elapsed_ms(dispatch_start)
                        total_dispatch_ms += dispatch_ms
                        extraction_status = (
                            "truncated_partial"
                            if job["truncated"] and not result.get("compiles")
                            else str(result.get("extraction_status") or "ok")
                        )
                        contract = build_reward_contract(
                            result,
                            truncated=bool(job["truncated"]),
                            extraction_status=extraction_status,
                            local_compile_ok=True,
                            supports_evaluation=bool(job["task_row"].get("supports_evaluation")),
                            backend_error=result.get("backend_error"),
                        )
                        result = dict(result)
                        result.update(contract)
                        result["reward"] = contract.get("training_reward")
                        finalize_prompt(
                            job["prompt_idx"],
                            turn,
                            job["trace_id"],
                            result,
                            contract,
                            job["sample"],
                            turn_diagnostics,
                            job["local_compile_ms"],
                            dispatch_ms,
                            turn_start,
                        )
                        turn_rewards.append(contract.get("training_reward"))

            _print_turn_summary(
                turn=turn + 1,
                max_turns=max_turns,
                active_count=len(active_indices),
                remote_count=len(pending_jobs),
                generation_ms=generation_ms,
                dispatch_ms=total_dispatch_ms,
                rewards=[float(reward) for reward in turn_rewards if reward is not None],
            )

            # Reward distribution diagnostics — critical for detecting dead signal
            if turn_rewards:
                valid_turn_rewards = [float(reward) for reward in turn_rewards if reward is not None]
                valid_fraction = len(valid_turn_rewards) / len(turn_rewards)
                diagnostic_order = [
                    "masked_infra_invalid",
                    "extraction_fail",
                    "truncated",
                    "local_compile_fail",
                    "remote_compile_fail",
                    "runtime_error",
                    "correctness_fail",
                    "correct_slow",
                    "correct_parity",
                    "correct_fast_eager",
                    "correct_fast_compile",
                ]
                diag_preview = " ".join(
                    f"{key}={turn_diagnostics[key]}"
                    for key in diagnostic_order
                    if turn_diagnostics.get(key, 0)
                )
                print(
                    f"[rollout] diagnostics: valid_fraction={valid_fraction:.2f} "
                    f"{diag_preview or 'no_events=1'}"
                )

                if not valid_turn_rewards:
                    print("[rollout] no_learning_step=1 reason=all_rewards_masked")
                    continue

                r_mean = statistics.mean(valid_turn_rewards)
                r_std = statistics.stdev(valid_turn_rewards) if len(valid_turn_rewards) > 1 else 0.0
                r_min = min(valid_turn_rewards)
                r_max = max(valid_turn_rewards)
                r_pos = sum(1 for r in valid_turn_rewards if r > 0.0)
                print(
                    f"[rollout] reward distribution: mean={r_mean:.2f} std={r_std:.2f} "
                    f"min={r_min:.1f} max={r_max:.1f} "
                    f"positive={r_pos}/{len(valid_turn_rewards)} "
                    f"turn_total={_elapsed_ms(turn_start):.1f}ms"
                )
                if r_std == 0.0:
                    print(
                        "[rollout] WARNING: zero reward variance — GRPO will produce zero gradients. "
                        "Check completion length, code extraction, and eval connectivity. "
                        "no_learning_step=1 reason=zero_reward_variance"
                    )

        # Rollout-level summary
        valid_terminal_rewards = [float(reward) for reward in terminal_rewards if reward is not None]
        valid_fraction = len(valid_terminal_rewards) / len(terminal_rewards) if terminal_rewards else 0.0
        if valid_terminal_rewards:
            print(
                f"[rollout] complete: terminal_rewards={terminal_rewards} "
                f"mean={statistics.mean(valid_terminal_rewards):.2f} "
                f"valid_fraction={valid_fraction:.2f} "
                f"positive={sum(1 for r in valid_terminal_rewards if r > 0.0)}/{len(valid_terminal_rewards)}"
            )
        else:
            print(
                f"[rollout] complete: terminal_rewards={terminal_rewards} "
                f"valid_fraction={valid_fraction:.2f} "
                "no_learning_step=1 reason=all_terminal_rewards_masked"
            )

        return {
            "prompt_ids": terminal_prompt_ids,
            "completion_ids": terminal_completion_ids,
            "logprobs": terminal_logprobs,
            "env_reward": terminal_rewards,
            "env_reward_contract": terminal_contracts,
        }

    return rollout_func


def reward_from_env(completions: list[str], **kwargs: Any) -> list[float]:
    """Extract rollout rewards, using NaN to mask infra-invalid samples."""
    env_rewards = kwargs.get("env_reward", [])
    if env_rewards:
        return [
            float("nan") if reward is None else float(reward)
            for reward in env_rewards
        ]
    return [-1.0] * len(completions)

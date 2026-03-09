"""Task routing helpers for the KernelForge RL pipeline.

This module keeps task-family decisions in one place so that dataset loading,
rollout generation, offline evaluation, and OpenEnv all agree on what can be
evaluated live today.
"""

from __future__ import annotations

import ast
import json
import os
from collections import Counter
from typing import Any

from openenv_env.reward import (
    COMPILE_WIN_THRESHOLD,
    EAGER_PARITY_FLOOR,
    EAGER_WIN_THRESHOLD,
    compute_reward,
)

SUPPORTED_WCC_OPS = {"weakly_connected_components", "wcc"}
STATEFUL_MODULE_TOKENS = (
    "nn.Conv",
    "nn.Linear",
    "nn.BatchNorm",
    "nn.LSTM",
    "nn.GRU",
    "nn.Embedding",
    "nn.Parameter",
    "nn.MultiheadAttention",
    "register_buffer(",
    "register_parameter(",
)
DEBUG_TIMINGS = os.getenv("KERNELFORGE_DEBUG_TIMINGS", "0") == "1"
SKIP_BENCHMARK_DEFAULT = os.getenv("KERNELFORGE_SKIP_BENCHMARK", "0") == "1"
_DEBUG_WARMUP_ITERS = int(os.getenv("KERNELFORGE_DEBUG_WARMUP_ITERS", "1"))
_DEBUG_BENCHMARK_RUNS = int(os.getenv("KERNELFORGE_DEBUG_BENCHMARK_RUNS", "3"))
_DEFAULT_OPS6K_WARMUP = _DEBUG_WARMUP_ITERS if DEBUG_TIMINGS else 10
_DEFAULT_OPS6K_RUNS = _DEBUG_BENCHMARK_RUNS if DEBUG_TIMINGS else 10
_DEFAULT_WCC_WARMUP = _DEBUG_WARMUP_ITERS if DEBUG_TIMINGS else 50
_DEFAULT_WCC_RUNS = _DEBUG_BENCHMARK_RUNS if DEBUG_TIMINGS else 30
OPS6K_WARMUP_ITERS = int(os.getenv("KERNELFORGE_OPS6K_WARMUP_ITERS", str(_DEFAULT_OPS6K_WARMUP)))
OPS6K_BENCHMARK_RUNS = int(os.getenv("KERNELFORGE_OPS6K_BENCHMARK_RUNS", str(_DEFAULT_OPS6K_RUNS)))
WCC_WARMUP_ITERS = int(os.getenv("KERNELFORGE_WCC_WARMUP_ITERS", str(_DEFAULT_WCC_WARMUP)))
WCC_BENCHMARK_RUNS = int(os.getenv("KERNELFORGE_WCC_BENCHMARK_RUNS", str(_DEFAULT_WCC_RUNS)))


def parse_ops(raw_ops: Any) -> list[str]:
    """Best-effort parser for serialized operator lists."""
    if isinstance(raw_ops, list):
        return [str(item) for item in raw_ops]

    if isinstance(raw_ops, str):
        text = raw_ops.strip()
        if not text:
            return []

        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(text)
            except Exception:
                continue
            if isinstance(parsed, list):
                return [str(item) for item in parsed]

        return [text]

    return [str(raw_ops)] if raw_ops is not None else []


def _task_has_empty_init_inputs(task_code: str) -> bool:
    """Return True when get_init_inputs is absent or trivially empty."""
    if not task_code.strip():
        return False

    try:
        tree = ast.parse(task_code)
    except SyntaxError:
        return False

    for node in tree.body:
        if not isinstance(node, ast.FunctionDef) or node.name != "get_init_inputs":
            continue

        returns = [sub.value for sub in ast.walk(node) if isinstance(sub, ast.Return)]
        if not returns:
            return True

        value = returns[-1]
        if isinstance(value, (ast.List, ast.Tuple)) and len(value.elts) == 0:
            return True
        if isinstance(value, ast.Constant) and value.value is None:
            return True
        return False

    return True


def supports_ops6k_live_eval(task_code: str) -> bool:
    """Return True for the stateless subset we can evaluate end to end today."""
    text = task_code.strip()
    if not text:
        return False

    if any(token in text for token in STATEFUL_MODULE_TOKENS):
        return False

    return _task_has_empty_init_inputs(text)


def infer_evaluation_backend(row: dict[str, Any]) -> str:
    """Infer which evaluator can score a row live."""
    task_code = str(row.get("task_code") or "").strip()
    if task_code:
        return "ops6k" if supports_ops6k_live_eval(task_code) else "unsupported"

    ops = {op.lower() for op in parse_ops(row.get("ops"))}
    prompt = str(row.get("prompt", "")).lower()
    kernel_id = str(row.get("kernel_id", "")).lower()

    if (
        ops & SUPPORTED_WCC_OPS
        or "weakly connected" in prompt
        or "weakly_connected_components" in kernel_id
    ):
        return "wcc"

    return "unsupported"


def support_reason(row: dict[str, Any]) -> str:
    """Return a short explanation describing evaluator support status."""
    backend = infer_evaluation_backend(row)
    if backend == "wcc":
        return "routed to the WCC verifier/benchmark path"
    if backend == "ops6k":
        return "routed to the stateless Ops-6K extension harness"
    if row.get("task_code"):
        return (
            "task requires stateful module or non-empty init inputs; "
            "the current Ops-6K harness only supports stateless tasks"
        )
    return "graph task has no correctness/runtime harness beyond WCC today"


def normalize_task_row(row: dict[str, Any]) -> dict[str, Any]:
    """Normalize task metadata so every pipeline stage sees the same schema."""
    normalized = dict(row)
    normalized["prompt"] = str(normalized.get("prompt", "")).strip()
    normalized["ops"] = parse_ops(normalized.get("ops"))
    normalized["task_code"] = (
        str(normalized.get("task_code")).strip() if normalized.get("task_code") else None
    )
    backend = infer_evaluation_backend(normalized)
    normalized["evaluation_backend"] = backend
    normalized["supports_evaluation"] = backend in {"wcc", "ops6k"}
    normalized["support_reason"] = support_reason(normalized)
    return normalized


def filter_supported_tasks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Return only tasks that have a live evaluator."""
    supported: list[dict[str, Any]] = []
    for row in rows:
        normalized = normalize_task_row(row)
        if normalized["prompt"] and normalized["supports_evaluation"]:
            supported.append(normalized)
    return supported


def build_prompt_lookup(rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Build a prompt -> metadata lookup for rollout/eval paths."""
    lookup: dict[str, dict[str, Any]] = {}
    for row in rows:
        normalized = normalize_task_row(row)
        if normalized["prompt"]:
            lookup[normalized["prompt"]] = normalized
    return lookup


def task_interface_contract(row: dict[str, Any]) -> str:
    """Return the evaluator contract that the model must satisfy."""
    backend = normalize_task_row(row)["evaluation_backend"]
    if backend == "wcc":
        return (
            "Evaluation contract:\n"
            '- Export `extern "C" void wcc_kernel(const int* row_ptr, const int* col_idx, '
            "int num_vertices, int* labels)`.\n"
            "- Fill `labels` in-place with component identifiers.\n"
            "- Return CUDA/C++ code only.\n"
            "- If extra nvcc flags are required, add a `// CU_FLAGS:` comment."
        )
    if backend == "ops6k":
        return (
            "Evaluation contract:\n"
            "- Produce a single PyTorch CUDA extension source file.\n"
            "- Include `#include <torch/extension.h>` and a `PYBIND11_MODULE(TORCH_EXTENSION_NAME, m)` block.\n"
            '- Export `m.def("run_kernel", &run_kernel)`.\n'
            "- `run_kernel(*inputs)` must accept the tensors returned by `get_inputs()` and return outputs "
            "matching `Model(*inputs)`.\n"
            "- Return CUDA/C++ code only.\n"
            "- If extra nvcc flags are required, add a `// CU_FLAGS:` comment."
        )
    return (
        "This task currently has no live evaluator in KernelForge. "
        "Do not use it for RL training until a correctness/runtime harness exists."
    )


def build_generation_prompt(
    task_row: dict[str, Any],
    skill_context: str = "",
    topology_context: str = "",
) -> str:
    """Compose the actual prompt shown to the policy for a task row."""
    row = normalize_task_row(task_row)
    parts = [
        skill_context.strip(),
        row["prompt"],
        topology_context.strip(),
        task_interface_contract(row),
    ]
    return "\n\n---\n\n".join(part for part in parts if part)


def build_modal_payload(
    cuda_code: str,
    task_row: dict[str, Any],
    baseline_orig_ms: float | None = None,
    baseline_dg_ms: float | None = None,
    skip_benchmark: bool | None = None,
    trace_id: str | None = None,
) -> tuple[str, dict[str, Any]]:
    """Build the Modal function name + payload for a task row."""
    row = normalize_task_row(task_row)
    backend = row["evaluation_backend"]
    effective_skip_benchmark = SKIP_BENCHMARK_DEFAULT if skip_benchmark is None else skip_benchmark

    if backend == "ops6k":
        return (
            "evaluate_ops6k_kernel",
            {
                "cuda_code": cuda_code,
                "task_code": row.get("task_code") or "",
                "warmup_iters": OPS6K_WARMUP_ITERS,
                "benchmark_runs": OPS6K_BENCHMARK_RUNS,
                "evaluation_backend": backend,
                "skip_benchmark": effective_skip_benchmark,
                "task_id": row.get("task_id", ""),
                "trace_id": trace_id or "",
            },
        )

    if backend == "wcc":
        return (
            "evaluate_kernel",
            {
                "cuda_code": cuda_code,
                "verify_graphs": 5,
                "warmup_iters": WCC_WARMUP_ITERS,
                "benchmark_runs": WCC_BENCHMARK_RUNS,
                "baseline_original_ms": baseline_orig_ms,
                "baseline_doublegraph_ms": baseline_dg_ms,
                "evaluation_backend": backend,
                "skip_benchmark": effective_skip_benchmark,
                "task_id": row.get("task_id", ""),
                "trace_id": trace_id or "",
            },
        )

    raise ValueError(row["support_reason"])


def normalize_eval_result(result: dict[str, Any] | None) -> dict[str, Any]:
    """Normalize evaluator output into the reward contract."""
    out = dict(result or {})
    out.setdefault("compiles", False)
    out.setdefault("correct", False)
    out.setdefault("speedup_vs_orig", 0.0)
    out.setdefault("speedup_vs_dg", 0.0)
    out.setdefault("runtime_ms", 0.0)
    out.setdefault("runtime_stats", {})
    out.setdefault("verifier_msg", "")
    out.setdefault("error", "")
    out.setdefault("trace_id", "")
    out.setdefault("task_id", "")
    out.setdefault("backend_error", False)
    out.setdefault("runtime_error", False)
    out.setdefault("valid_for_loss", True)
    out.setdefault("termination_reason", "")
    out.setdefault("truncated", False)
    out.setdefault("extraction_status", "ok")
    out.setdefault("local_compile_ok", True)
    out.setdefault("remote_compile_ok", bool(out.get("compiles")))
    out.setdefault("public_reward_bucket", None)
    out.setdefault("training_reward", None)
    out.setdefault("phase_timings", {
        "compile_ms": 0.0,
        "correctness_ms": 0.0,
        "benchmark_eager_ms": 0.0,
        "benchmark_compile_ms": 0.0,
        "benchmark_kernel_ms": 0.0,
        "total_eval_ms": 0.0,
    })
    out["speedup_vs_orig"] = float(out.get("speedup_vs_orig") or 0.0)
    out["speedup_vs_dg"] = float(out.get("speedup_vs_dg") or 0.0)
    out["speedup_vs_eager"] = float(out.get("speedup_vs_eager") or out["speedup_vs_orig"])
    out["speedup_vs_compile"] = float(out.get("speedup_vs_compile") or out["speedup_vs_dg"])
    return out


def _is_backend_error(normalized: dict[str, Any]) -> bool:
    if bool(normalized.get("backend_error")):
        return True
    error_text = " ".join(
        str(normalized.get(key, "") or "").lower()
        for key in ("error", "verifier_msg")
    )
    return any(
        token in error_text
        for token in (
            "missing batch result",
            "dispatch failed",
            "http",
            "connection",
            "backend",
            "modal",
            "coreweave",
            "timed out",
            "timeout",
            "malformed",
            "reference model load failed",
        )
    )


def _is_runtime_error(normalized: dict[str, Any]) -> bool:
    if bool(normalized.get("runtime_error")):
        return True
    error_text = " ".join(
        str(normalized.get(key, "") or "").lower()
        for key in ("error", "verifier_msg")
    )
    return any(
        token in error_text
        for token in (
            "benchmark exception",
            "profiling failed",
            "runtime error",
            "illegal memory access",
            "launch failed",
            "cuda error",
        )
    )


def build_reward_contract(
    result: dict[str, Any] | None,
    *,
    truncated: bool = False,
    extraction_status: str = "ok",
    local_compile_ok: bool = True,
    supports_evaluation: bool = True,
    backend_error: bool | None = None,
) -> dict[str, Any]:
    """Map rollout + evaluator status into dense training reward + public bucket."""

    normalized = normalize_eval_result(result)
    normalized["truncated"] = bool(truncated or normalized.get("truncated"))
    normalized["extraction_status"] = extraction_status or str(normalized.get("extraction_status") or "ok")
    normalized["local_compile_ok"] = bool(local_compile_ok and normalized.get("local_compile_ok", True))
    normalized["remote_compile_ok"] = bool(normalized.get("compiles"))
    normalized["backend_error"] = _is_backend_error(normalized) if backend_error is None else bool(backend_error)
    normalized["runtime_error"] = _is_runtime_error(normalized)

    if normalized["backend_error"]:
        normalized["valid_for_loss"] = False
        normalized["termination_reason"] = "backend_error"
        normalized["training_reward"] = None
        normalized["public_reward_bucket"] = None
        return normalized

    if normalized["extraction_status"] == "no_code":
        normalized["termination_reason"] = "no_code"
        normalized["training_reward"] = -1.0
        normalized["public_reward_bucket"] = -1
        return normalized

    if normalized["extraction_status"] == "truncated_partial":
        normalized["termination_reason"] = "truncated_partial"
        normalized["training_reward"] = -0.7
        normalized["public_reward_bucket"] = -1
        return normalized

    if not supports_evaluation:
        normalized["termination_reason"] = "unsupported"
        normalized["training_reward"] = -1.0
        normalized["public_reward_bucket"] = -1
        return normalized

    if not normalized["local_compile_ok"]:
        normalized["termination_reason"] = "local_compile_fail"
        normalized["training_reward"] = -0.5
        normalized["public_reward_bucket"] = -1
        return normalized

    if not normalized["remote_compile_ok"]:
        normalized["termination_reason"] = "remote_compile_fail"
        normalized["training_reward"] = -0.4
        normalized["public_reward_bucket"] = -1
        return normalized

    if normalized["runtime_error"]:
        normalized["termination_reason"] = "runtime_error"
        normalized["training_reward"] = -0.3
        normalized["public_reward_bucket"] = -1
        return normalized

    if not bool(normalized.get("correct")):
        normalized["termination_reason"] = "correctness_fail"
        normalized["training_reward"] = -0.2
        normalized["public_reward_bucket"] = -1
        return normalized

    speedup_vs_eager = float(normalized.get("speedup_vs_eager") or 0.0)
    speedup_vs_compile = float(normalized.get("speedup_vs_compile") or 0.0)
    normalized["public_reward_bucket"] = int(
        compute_reward(
            compiled=True,
            correct=True,
            speedup_vs_eager=speedup_vs_eager,
            speedup_vs_compile=speedup_vs_compile,
            occupancy=normalized.get("occupancy"),
            mem_coalescing=normalized.get("mem_coalescing"),
            warp_efficiency=normalized.get("warp_efficiency"),
        )
    )

    if speedup_vs_compile > COMPILE_WIN_THRESHOLD:
        normalized["termination_reason"] = "correct_fast_compile"
        normalized["training_reward"] = 1.0
    elif speedup_vs_eager > EAGER_WIN_THRESHOLD:
        normalized["termination_reason"] = "correct_fast_eager"
        normalized["training_reward"] = 0.7
    elif speedup_vs_eager >= EAGER_PARITY_FLOOR:
        normalized["termination_reason"] = "correct_parity"
        normalized["training_reward"] = 0.4
    else:
        normalized["termination_reason"] = "correct_slow"
        normalized["training_reward"] = 0.2

    return normalized


def compute_task_reward(result: dict[str, Any] | None) -> float | None:
    """Compute the dense training reward from a normalized evaluator result."""
    return build_reward_contract(result).get("training_reward")


def evaluate_code_remote(
    cuda_code: str,
    task_row: dict[str, Any],
    baseline_orig_ms: float | None = None,
    baseline_dg_ms: float | None = None,
    skip_benchmark: bool | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Dispatch a candidate kernel to the configured eval backend (CoreWeave or Modal)."""
    from openenv_env.eval_backend import dispatch_eval

    fn_name, payload = build_modal_payload(
        cuda_code,
        task_row,
        baseline_orig_ms=baseline_orig_ms,
        baseline_dg_ms=baseline_dg_ms,
        skip_benchmark=skip_benchmark,
        trace_id=trace_id,
    )
    result = normalize_eval_result(dispatch_eval(fn_name, payload))
    contract = build_reward_contract(result)
    result.update(contract)
    result["reward"] = contract.get("training_reward")
    result["evaluation_backend"] = normalize_task_row(task_row)["evaluation_backend"]
    return result


def evaluate_code_remote_batch(
    cuda_codes: list[str],
    task_rows: list[dict[str, Any]],
    baseline_orig_ms: float | None = None,
    baseline_dg_ms: float | None = None,
    skip_benchmark: bool | None = None,
    trace_ids: list[str] | None = None,
) -> list[dict[str, Any]]:
    """Dispatch multiple candidates in one backend call."""
    from openenv_env.eval_backend import dispatch_eval

    payloads: list[dict[str, Any]] = []
    normalized_rows = [normalize_task_row(row) for row in task_rows]
    for idx, (cuda_code, row) in enumerate(zip(cuda_codes, normalized_rows)):
        _, payload = build_modal_payload(
            cuda_code,
            row,
            baseline_orig_ms=baseline_orig_ms,
            baseline_dg_ms=baseline_dg_ms,
            skip_benchmark=skip_benchmark,
            trace_id=trace_ids[idx] if trace_ids and idx < len(trace_ids) else None,
        )
        payloads.append(payload)

    raw_results = dispatch_eval("evaluate_kernels_batch", payloads)
    if not isinstance(raw_results, list):
        raw_results = []

    results: list[dict[str, Any]] = []
    for idx, row in enumerate(normalized_rows):
        raw = raw_results[idx] if idx < len(raw_results) else {
            "compiles": False,
            "correct": False,
            "backend_error": True,
            "error": "Missing batch result from evaluator",
        }
        result = normalize_eval_result(raw)
        contract = build_reward_contract(result)
        result.update(contract)
        result["reward"] = contract.get("training_reward")
        result["evaluation_backend"] = row["evaluation_backend"]
        results.append(result)
    return results


# Backward-compatible alias
def evaluate_code_on_modal(
    cuda_code: str,
    task_row: dict[str, Any],
    modal_app_name: str = "",
    baseline_orig_ms: float | None = None,
    baseline_dg_ms: float | None = None,
    skip_benchmark: bool | None = None,
    trace_id: str | None = None,
) -> dict[str, Any]:
    """Backward-compatible alias — dispatches via eval_backend (ignores modal_app_name)."""
    return evaluate_code_remote(
        cuda_code, task_row,
        baseline_orig_ms=baseline_orig_ms,
        baseline_dg_ms=baseline_dg_ms,
        skip_benchmark=skip_benchmark,
        trace_id=trace_id,
    )


def summarize_tasks(rows: list[dict[str, Any]]) -> dict[str, dict[str, int]]:
    """Summarize task coverage for dataset and preflight reporting."""
    normalized = [normalize_task_row(row) for row in rows if row.get("prompt")]
    backend_counts = Counter(row["evaluation_backend"] for row in normalized)
    source_counts = Counter(str(row.get("data_source", "unknown")) for row in normalized)
    difficulty_counts = Counter(int(row.get("difficulty", 1)) for row in normalized)
    supported_counts = Counter("supported" if row["supports_evaluation"] else "unsupported" for row in normalized)
    return {
        "evaluation_backend": dict(sorted(backend_counts.items())),
        "data_source": dict(sorted(source_counts.items())),
        "difficulty": dict(sorted(difficulty_counts.items())),
        "support": dict(sorted(supported_counts.items())),
    }

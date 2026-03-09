"""Reward helpers: discrete milestone {-1, 1, 2, 3} + TRLOO post-process.

Discrete rewards per CUDA Agent ablation (96.8% vs 60.4% faster rate over continuous).
Milestones normalize across problem difficulty — beating torch.compile on a hard problem
and an easy problem both earn the same r=3.
"""
from __future__ import annotations

import math

_REQUIRED_EVAL_KEYS = {"compiles", "correct", "speedup_vs_orig", "speedup_vs_dg", "error"}
EAGER_PARITY_FLOOR = 0.98
EAGER_WIN_THRESHOLD = 1.05
COMPILE_WIN_THRESHOLD = 1.05


def validate_eval_result(result: dict) -> dict:
    """Validate Modal evaluate_kernel return schema. Missing/invalid → safe defaults."""
    missing = _REQUIRED_EVAL_KEYS - set(result)
    if missing:
        return {"compiles": False, "correct": False, "speedup_vs_orig": 0.0,
                "speedup_vs_dg": 0.0, "error": f"missing keys: {missing}"}
    out = dict(result)
    # Clamp NaN/inf speedups to 0
    for k in ("speedup_vs_orig", "speedup_vs_dg"):
        v = out.get(k, 0.0)
        if not isinstance(v, (int, float)) or math.isnan(v) or math.isinf(v):
            out[k] = 0.0
    return out


def compute_reward(
    compiled: bool,
    correct: bool,
    speedup_vs_eager: float,
    speedup_vs_compile: float,
    occupancy: float | None = None,
    mem_coalescing: float | None = None,
    warp_efficiency: float | None = None,
) -> float:
    """Return discrete milestone reward {-1, 1, 2, 3}.

    Discrete milestones per CUDA Agent ablation — normalizes reward across
    problem difficulty so beating torch.compile on sparse graphs earns the
    same signal as beating it on dense elementwise ops.

    Args:
        compiled: Whether the kernel compiled successfully.
        correct: Whether the kernel produces correct output.
        speedup_vs_eager: Speedup ratio vs torch.eager baseline.
        speedup_vs_compile: Speedup ratio vs torch.compile baseline.
        occupancy: SM occupancy (unused in discrete mode, kept for API compat).
        mem_coalescing: Memory coalescing (unused in discrete mode).
        warp_efficiency: Warp efficiency (unused in discrete mode).

    Returns:
        -1.0: compile or correctness failure.
         1.0: correct but not faster than baselines.
         2.0: correct and faster than eager PyTorch (>5%).
         3.0: correct and faster than torch.compile (>5%).
    """
    # Correctness gate: must pass BEFORE any speedup signal reaches gradients.
    # A fast-but-wrong kernel MUST get -1.0, not a positive reward.
    if not compiled or not correct:
        return -1.0

    # Discrete milestones (highest matching tier wins)
    if speedup_vs_compile > COMPILE_WIN_THRESHOLD:
        return 3.0
    if speedup_vs_eager > EAGER_WIN_THRESHOLD:
        return 2.0
    return 1.0


def trloo_post_process(advantages: list[float], n: int) -> list[float]:
    """Scale GRPO advantages by N/(N-1) to correct gradient shrinkage.

    Dr. Kernel (arXiv 2602.05885) proves GRPO's self-inclusion bias
    shrinks expected gradients by (1 - 1/N). With G=4, that is 25%.
    This post-process is a drop-in fix for TRL GRPOTrainer.
    """
    if n <= 1:
        return advantages
    scale = n / (n - 1)
    return [a * scale for a in advantages]

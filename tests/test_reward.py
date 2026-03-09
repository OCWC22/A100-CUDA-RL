"""Tests for discrete milestone reward computation {-1, 1, 2, 3}."""
import pytest

from openenv_env.reward import compute_reward, trloo_post_process
from training.task_support import build_reward_contract


def test_compile_fail():
    assert compute_reward(compiled=False, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_fail():
    assert compute_reward(compiled=True, correct=False, speedup_vs_eager=0, speedup_vs_compile=0) == -1.0


def test_correct_no_speedup():
    """Correct but speedup=1.0 (not > 1.05) -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.0, speedup_vs_compile=0.9)
    assert r == 1.0


def test_modest_speedup():
    """speedup_vs_eager=1.5 > 1.05 -> reward 2.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=1.5, speedup_vs_compile=0.9)
    assert r == 2.0


def test_large_speedup():
    """speedup_vs_eager=3.0 > 1.05 but speedup_vs_compile=1.0 not > 1.05 -> reward 2.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=3.0, speedup_vs_compile=1.0)
    assert r == 2.0


def test_slower_than_baseline():
    """Correct but speedup=0.5 < 1.05 -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.5, speedup_vs_compile=0.3)
    assert r == 1.0


def test_very_slow_clamped():
    """Correct but speedup=0.01 < 1.05 -> reward 1.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=0.01, speedup_vs_compile=0)
    assert r == 1.0


def test_nsight_ignored():
    """Nsight metrics are accepted but unused in discrete mode — same reward as without."""
    base = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0)
    with_nsight = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=0.8, mem_coalescing=0.9, warp_efficiency=0.7,
    )
    assert with_nsight == base == 2.0


def test_nsight_extreme_values():
    """Nsight with out-of-range values still produces same discrete reward."""
    r = compute_reward(
        compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.0,
        occupancy=1.5, mem_coalescing=-0.1, warp_efficiency=0.5,
    )
    assert r == 2.0


def test_beats_torch_compile():
    """speedup_vs_compile=1.2 > 1.05 -> reward 3.0."""
    r = compute_reward(compiled=True, correct=True, speedup_vs_eager=2.0, speedup_vs_compile=1.2)
    assert r == 3.0


def test_dense_reward_masks_backend_failures():
    contract = build_reward_contract(
        {"backend_error": True, "error": "Modal backend timeout"},
        backend_error=True,
    )
    assert contract["valid_for_loss"] is False
    assert contract["training_reward"] is None
    assert contract["public_reward_bucket"] is None
    assert contract["termination_reason"] == "backend_error"


def test_dense_reward_no_code():
    contract = build_reward_contract(
        {"error": "No code found"},
        extraction_status="no_code",
    )
    assert contract["training_reward"] == -1.0
    assert contract["public_reward_bucket"] == -1
    assert contract["termination_reason"] == "no_code"


def test_dense_reward_truncated_partial():
    contract = build_reward_contract(
        {"error": "generation clipped"},
        truncated=True,
        extraction_status="truncated_partial",
    )
    assert contract["training_reward"] == -0.7
    assert contract["public_reward_bucket"] == -1
    assert contract["termination_reason"] == "truncated_partial"


def test_dense_reward_local_compile_fail():
    contract = build_reward_contract(
        {"error": "nvcc: expected ';'"},
        local_compile_ok=False,
    )
    assert contract["training_reward"] == -0.5
    assert contract["termination_reason"] == "local_compile_fail"


def test_dense_reward_remote_compile_fail():
    contract = build_reward_contract(
        {"compiles": False, "correct": False, "error": "Compilation failed"},
    )
    assert contract["training_reward"] == -0.4
    assert contract["termination_reason"] == "remote_compile_fail"


def test_dense_reward_runtime_error():
    contract = build_reward_contract(
        {"compiles": True, "correct": False, "error": "Profiling failed: CUDA launch failed"},
    )
    assert contract["training_reward"] == -0.3
    assert contract["termination_reason"] == "runtime_error"


def test_dense_reward_correctness_fail():
    contract = build_reward_contract(
        {"compiles": True, "correct": False, "error": "Correctness check failed: mismatch"},
    )
    assert contract["training_reward"] == -0.2
    assert contract["termination_reason"] == "correctness_fail"


def test_dense_reward_correct_slow():
    contract = build_reward_contract(
        {"compiles": True, "correct": True, "speedup_vs_orig": 0.75, "speedup_vs_dg": 0.5},
    )
    assert contract["training_reward"] == 0.2
    assert contract["public_reward_bucket"] == 1
    assert contract["termination_reason"] == "correct_slow"


def test_dense_reward_correct_parity():
    contract = build_reward_contract(
        {"compiles": True, "correct": True, "speedup_vs_orig": 1.01, "speedup_vs_dg": 1.0},
    )
    assert contract["training_reward"] == 0.4
    assert contract["public_reward_bucket"] == 1
    assert contract["termination_reason"] == "correct_parity"


def test_dense_reward_correct_fast_eager():
    contract = build_reward_contract(
        {"compiles": True, "correct": True, "speedup_vs_orig": 1.2, "speedup_vs_dg": 0.95},
    )
    assert contract["training_reward"] == 0.7
    assert contract["public_reward_bucket"] == 2
    assert contract["termination_reason"] == "correct_fast_eager"


def test_dense_reward_correct_fast_compile():
    contract = build_reward_contract(
        {"compiles": True, "correct": True, "speedup_vs_orig": 1.3, "speedup_vs_dg": 1.2},
    )
    assert contract["training_reward"] == 1.0
    assert contract["public_reward_bucket"] == 3
    assert contract["termination_reason"] == "correct_fast_compile"


def test_trloo_post_process_g4():
    """TRLOO scales by N/(N-1) = 4/3 for G=4."""
    advantages = [0.5, -0.3, 1.2, -0.8]
    scaled = trloo_post_process(advantages, n=4)
    scale = 4 / 3
    for orig, result in zip(advantages, scaled):
        assert result == pytest.approx(orig * scale, abs=1e-6)


def test_trloo_post_process_g1():
    """TRLOO with N=1 returns unchanged."""
    advantages = [0.5]
    assert trloo_post_process(advantages, n=1) == advantages


def test_trloo_post_process_g2():
    """TRLOO scales by 2/1 = 2.0 for G=2."""
    advantages = [0.3, -0.3]
    scaled = trloo_post_process(advantages, n=2)
    assert scaled[0] == pytest.approx(0.6, abs=1e-6)
    assert scaled[1] == pytest.approx(-0.6, abs=1e-6)

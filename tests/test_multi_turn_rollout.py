"""Tests for multi-turn rollout logic (no GPU/Modal needed)."""
from __future__ import annotations

from types import SimpleNamespace

import pytest

from training.multi_turn_rollout import (
    extract_cuda_code,
    _format_feedback,
    _compute_reward_from_result,
    make_multi_turn_rollout,
    reward_from_env,
)


class TestExtractCudaCode:
    def test_fenced_cuda_block(self):
        text = "Here's the kernel:\n```cuda\n__global__ void add(float* a, float* b) {}\n```\nDone."
        assert "__global__ void add" in extract_cuda_code(text)

    def test_fenced_cpp_block(self):
        text = "```cpp\n__global__ void relu(float* x) {}\n```"
        assert "__global__ void relu" in extract_cuda_code(text)

    def test_raw_global(self):
        text = "__global__ void kernel(int* data, int n) { int i = threadIdx.x; }"
        assert "__global__ void kernel" in extract_cuda_code(text)

    def test_no_cuda_code(self):
        assert extract_cuda_code("This is just text, no CUDA here.") == ""

    def test_empty_string(self):
        assert extract_cuda_code("") == ""

    def test_multiple_blocks_takes_first(self):
        text = "```cuda\n__global__ void first() {}\n```\n```cuda\n__global__ void second() {}\n```"
        code = extract_cuda_code(text)
        assert "first" in code
        assert "second" not in code


class TestFormatFeedback:
    def test_compilation_failure(self):
        result = {"compiles": False, "error": "undefined reference to __shfl_sync"}
        feedback = _format_feedback(result, -1.0, 0)
        assert "COMPILATION FAILED" in feedback
        assert "__shfl_sync" in feedback
        assert "Fix" in feedback

    def test_verification_failure(self):
        result = {"compiles": True, "correct": False, "verifier_msg": "Invariant 2: edge (3,7) crosses components"}
        feedback = _format_feedback(result, -1.0, 1)
        assert "VERIFICATION FAILED" in feedback
        assert "edge (3,7)" in feedback

    def test_correct_slow(self):
        """speedup_vs_orig=0.8 -> dense reward 0.2 -> 'not faster' tip."""
        reward = 0.2
        result = {"compiles": True, "correct": True, "runtime_ms": 5.0, "speedup_vs_orig": 0.8, "speedup_vs_dg": 0, "runtime_stats": {"mean": 5.1, "std": 0.2}}
        feedback = _format_feedback(result, reward, 2)
        assert "CORRECT" in feedback
        assert "5.000ms" in feedback
        assert "not faster" in feedback

    def test_correct_modest_speedup(self):
        """speedup_vs_orig=1.5, speedup_vs_dg=0.9 -> dense reward 0.7 -> 'not torch.compile' tip."""
        reward = 0.7
        result = {"compiles": True, "correct": True, "runtime_ms": 2.0, "speedup_vs_orig": 1.5, "speedup_vs_dg": 0.9, "runtime_stats": {}}
        feedback = _format_feedback(result, reward, 3)
        assert "CORRECT" in feedback
        assert "Speedup vs eager: 1.50x" in feedback
        assert "torch.compile" in feedback


class TestComputeReward:
    def test_compile_fail(self):
        assert _compute_reward_from_result({"compiles": False}) == -0.4

    def test_verify_fail(self):
        assert _compute_reward_from_result({"compiles": True, "correct": False}) == -0.2

    def test_correct_slower(self):
        """speedup_vs_orig=0.9 -> correct but slower than eager -> dense reward 0.2"""
        r = _compute_reward_from_result({"compiles": True, "correct": True, "speedup_vs_orig": 0.9, "speedup_vs_dg": 0.5})
        assert r == 0.2

    def test_correct_parity(self):
        r = _compute_reward_from_result({"compiles": True, "correct": True, "speedup_vs_orig": 1.0, "speedup_vs_dg": 1.0})
        assert r == 0.4

    def test_modest_speedup(self):
        """speedup_vs_orig=1.2 (>1.05x eager), speedup_vs_dg=0.9 (<1.05x compile) -> reward 0.7"""
        r = _compute_reward_from_result({"compiles": True, "correct": True, "speedup_vs_orig": 1.2, "speedup_vs_dg": 0.9})
        assert r == 0.7

    def test_large_speedup(self):
        """speedup_vs_orig=3.0, speedup_vs_dg=1.5 (>1.05x compile) -> reward 1.0"""
        r = _compute_reward_from_result({"compiles": True, "correct": True, "speedup_vs_orig": 3.0, "speedup_vs_dg": 1.5})
        assert r == 1.0


def test_reward_from_env_masks_invalid_rewards():
    rewards = reward_from_env(["sample"], env_reward=[None])
    assert len(rewards) == 1
    assert rewards[0] != rewards[0]


def test_rollout_trains_only_terminal_turn(monkeypatch):
    call_state = {"turn": 0}

    def fake_generate(_trainer, prompts):
        call_state["turn"] += 1
        if call_state["turn"] == 1:
            return [{
                "prompt_ids": [101],
                "completion_ids": [11],
                "logprobs": [0.1],
                "text": "```cuda\n__global__ void first() {}\n```",
            }]
        return [{
            "prompt_ids": [202],
            "completion_ids": [22],
            "logprobs": [0.2],
            "text": "```cuda\n__global__ void second() {}\n```",
        }]

    def fake_eval_remote(*args, **kwargs):
        if call_state["turn"] == 1:
            return {"compiles": True, "correct": False, "error": "Correctness check failed: mismatch"}
        return {"compiles": True, "correct": True, "runtime_ms": 1.0, "speedup_vs_orig": 1.2, "speedup_vs_dg": 0.9}

    monkeypatch.setattr("training.multi_turn_rollout._generate_rollout_completions_compat", fake_generate)
    monkeypatch.setattr("training.multi_turn_rollout._local_compile_check", lambda code: (True, ""))
    monkeypatch.setattr("training.multi_turn_rollout.evaluate_code_remote", fake_eval_remote)

    trainer = SimpleNamespace(
        processing_class=SimpleNamespace(decode=lambda ids, skip_special_tokens=True: ""),
        args=SimpleNamespace(max_completion_length=32, temperature=1.0),
    )
    metadata = [{
        "prompt": "Write a WCC kernel.",
        "ops": ["wcc"],
        "difficulty": 1,
    }]
    rollout = make_multi_turn_rollout(max_turns=3, skill_md_gpu="a100", problem_metadata=metadata)

    result = rollout(["Write a WCC kernel."], trainer)

    assert result["prompt_ids"] == [[202]]
    assert result["completion_ids"] == [[22]]
    assert result["logprobs"] == [[0.2]]
    assert result["env_reward"] == [0.7]
    assert result["env_reward_contract"][0]["termination_reason"] == "correct_fast_eager"


def test_rollout_masks_backend_invalid_terminal_reward(monkeypatch):
    def fake_generate(_trainer, prompts):
        return [{
            "prompt_ids": [303],
            "completion_ids": [33],
            "logprobs": [0.3],
            "text": "```cuda\n__global__ void only() {}\n```",
        }]

    def fake_eval_remote(*args, **kwargs):
        raise RuntimeError("backend timeout")

    monkeypatch.setattr("training.multi_turn_rollout._generate_rollout_completions_compat", fake_generate)
    monkeypatch.setattr("training.multi_turn_rollout._local_compile_check", lambda code: (True, ""))
    monkeypatch.setattr("training.multi_turn_rollout.evaluate_code_remote", fake_eval_remote)

    trainer = SimpleNamespace(
        processing_class=SimpleNamespace(decode=lambda ids, skip_special_tokens=True: ""),
        args=SimpleNamespace(max_completion_length=32, temperature=1.0),
    )
    metadata = [{
        "prompt": "Write a WCC kernel.",
        "ops": ["wcc"],
        "difficulty": 1,
    }]
    rollout = make_multi_turn_rollout(max_turns=1, skill_md_gpu="a100", problem_metadata=metadata)

    result = rollout(["Write a WCC kernel."], trainer)

    assert result["env_reward"] == [None]
    assert result["env_reward_contract"][0]["valid_for_loss"] is False
    assert result["env_reward_contract"][0]["termination_reason"] == "backend_error"

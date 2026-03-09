from __future__ import annotations

import pytest

from training.grpo_config import (
    SharedGRPORuntime,
    load_shared_grpo_runtime,
    validate_shared_grpo_runtime,
)


def test_shared_grpo_runtime_defaults(monkeypatch):
    monkeypatch.delenv("KERNELFORGE_GRPO_NUM_GENERATIONS", raising=False)
    monkeypatch.delenv("KERNELFORGE_STAGE1_NUM_GENERATIONS", raising=False)
    monkeypatch.delenv("KERNELFORGE_GRPO_MAX_COMPLETION_LENGTH", raising=False)
    monkeypatch.delenv("KERNELFORGE_STAGE1_MAX_COMPLETION_LENGTH", raising=False)
    monkeypatch.delenv("KERNELFORGE_GRPO_GRADIENT_ACCUMULATION_STEPS", raising=False)
    monkeypatch.delenv("KERNELFORGE_STAGE1_GRADIENT_ACCUMULATION_STEPS", raising=False)

    runtime = load_shared_grpo_runtime("stage1")

    assert runtime.num_generations == 8
    assert runtime.max_completion_length == 1024
    assert runtime.max_prompt_length == 3072
    assert runtime.per_device_train_batch_size == 1
    assert runtime.gradient_accumulation_steps == 8
    assert runtime.effective_batch_size == 8


def test_shared_grpo_runtime_validation_accepts_divisible_batch():
    runtime = SharedGRPORuntime(
        stage="stage3",
        num_generations=8,
        max_prompt_length=3072,
        max_completion_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=8,
        optim="adamw_torch",
        bf16=False,
        use_vllm=False,
        vllm_mode="server",
        vllm_server_base_url="",
        vllm_gpu_memory_utilization=0.6,
    )

    validate_shared_grpo_runtime(runtime)


def test_shared_grpo_runtime_validation_rejects_invalid_batch():
    runtime = SharedGRPORuntime(
        stage="stage1",
        num_generations=8,
        max_prompt_length=3072,
        max_completion_length=1024,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        optim="adamw_torch",
        bf16=False,
        use_vllm=False,
        vllm_mode="server",
        vllm_server_base_url="",
        vllm_gpu_memory_utilization=0.6,
    )

    with pytest.raises(ValueError):
        validate_shared_grpo_runtime(runtime)

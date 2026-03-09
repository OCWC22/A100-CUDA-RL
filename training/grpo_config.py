"""Shared GRPO runtime configuration for Stage 1 and Stage 3."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass


@dataclass(frozen=True)
class SharedGRPORuntime:
    stage: str
    num_generations: int
    max_prompt_length: int
    max_completion_length: int
    per_device_train_batch_size: int
    gradient_accumulation_steps: int
    optim: str
    bf16: bool
    use_vllm: bool
    vllm_mode: str
    vllm_server_base_url: str
    vllm_gpu_memory_utilization: float

    @property
    def effective_batch_size(self) -> int:
        return self.per_device_train_batch_size * self.gradient_accumulation_steps


def _get_stage_env_int(stage: str, name: str, default: int) -> int:
    stage_key = f"KERNELFORGE_{stage.upper()}_{name}"
    global_key = f"KERNELFORGE_GRPO_{name}"
    return int(os.getenv(stage_key, os.getenv(global_key, str(default))))


def load_shared_grpo_runtime(stage: str) -> SharedGRPORuntime:
    """Load shared GRPO settings with stage-specific overrides."""
    is_linux = sys.platform.startswith("linux")
    use_vllm = os.getenv("KERNELFORGE_USE_VLLM", "0") == "1" and is_linux

    return SharedGRPORuntime(
        stage=stage,
        num_generations=_get_stage_env_int(stage, "NUM_GENERATIONS", 8),
        max_prompt_length=_get_stage_env_int(stage, "MAX_PROMPT_LENGTH", 3072),
        max_completion_length=_get_stage_env_int(stage, "MAX_COMPLETION_LENGTH", 1024),
        per_device_train_batch_size=_get_stage_env_int(stage, "PER_DEVICE_BATCH_SIZE", 1),
        gradient_accumulation_steps=_get_stage_env_int(stage, "GRADIENT_ACCUMULATION_STEPS", 8),
        optim="paged_adamw_8bit" if is_linux else "adamw_torch",
        bf16=is_linux,
        use_vllm=use_vllm,
        vllm_mode=os.getenv("KERNELFORGE_VLLM_MODE", "server").strip().lower(),
        vllm_server_base_url=os.getenv("KERNELFORGE_VLLM_SERVER_BASE_URL", "").strip(),
        vllm_gpu_memory_utilization=float(
            os.getenv("KERNELFORGE_VLLM_GPU_MEMORY_UTILIZATION", "0.6")
        ),
    )


def validate_shared_grpo_runtime(runtime: SharedGRPORuntime) -> None:
    """Validate divisibility and vLLM mode requirements."""
    if runtime.effective_batch_size % runtime.num_generations != 0:
        raise ValueError(
            f"Effective batch size ({runtime.effective_batch_size}) must be divisible by "
            f"num_generations ({runtime.num_generations}) per TRL GRPO requirement."
        )
    if runtime.use_vllm and runtime.vllm_mode == "server" and not runtime.vllm_server_base_url:
        raise ValueError(
            "KERNELFORGE_USE_VLLM=1 with KERNELFORGE_VLLM_MODE=server requires "
            "KERNELFORGE_VLLM_SERVER_BASE_URL to be set."
        )


def apply_shared_grpo_runtime(
    runtime: SharedGRPORuntime,
    grpo_kwargs: dict[str, object],
) -> dict[str, object]:
    """Inject the shared GRPO settings into trainer kwargs."""
    merged = dict(grpo_kwargs)
    merged.update(
        {
            "num_generations": runtime.num_generations,
            "max_prompt_length": runtime.max_prompt_length,
            "max_completion_length": runtime.max_completion_length,
            "per_device_train_batch_size": runtime.per_device_train_batch_size,
            "gradient_accumulation_steps": runtime.gradient_accumulation_steps,
            "optim": runtime.optim,
            "bf16": runtime.bf16,
        }
    )
    if runtime.use_vllm:
        merged["use_vllm"] = True
        merged["vllm_mode"] = runtime.vllm_mode
        if runtime.vllm_mode == "server":
            merged["vllm_server_base_url"] = runtime.vllm_server_base_url
        elif runtime.vllm_mode == "colocate":
            merged["vllm_gpu_memory_utilization"] = runtime.vllm_gpu_memory_utilization
    return merged

"""
Unified model loading for KernelForge training pipeline.

Primary: unsloth/Qwen3-Coder-30B-A3B-Instruct (30.5B MoE, 3.3B active)
         via Unsloth FastLanguageModel + PatchFastRL for GRPO.
         bf16 on H200 141GB (~61GB model, ~80GB free for vLLM + GRPO).
         Unsloth's 2026 Faster MOE update handles MoE LoRA natively.

Supports quantized loading via BitsAndBytes for A100/H100 80GB training.
Set KERNELFORGE_QUANT_BITS=8 (recommended) or KERNELFORGE_QUANT_BITS=4 to enable.
8-bit preserves more model accuracy than 4-bit with minimal extra VRAM.

Dev:     Qwen2.5-Coder-0.5B-Instruct for macOS control-plane validation.
"""
from __future__ import annotations

import os
import sys

TARGET_GPU = os.getenv("KERNELFORGE_TARGET_GPU", "A100")
TARGET_ARCH = os.getenv("KERNELFORGE_TARGET_ARCH", "sm_80")
# Quantization: "0" (disabled/bf16), "4" (NF4 QLoRA), "8" (INT8, recommended for accuracy)
QUANT_BITS = int(os.getenv("KERNELFORGE_QUANT_BITS", "0"))
# Legacy compat
if os.getenv("KERNELFORGE_LOAD_IN_4BIT", "0") == "1" and QUANT_BITS == 0:
    QUANT_BITS = 4

# Primary model (MoE via Unsloth)
PRIMARY_MODEL = os.getenv("KERNELFORGE_MODEL", "Qwen/Qwen3-Coder-30B-A3B-Instruct")
DEV_MODEL = os.getenv("KERNELFORGE_DEV_MODEL", "Qwen/Qwen2.5-Coder-0.5B-Instruct")

# LoRA constants
LORA_R = int(os.getenv("KERNELFORGE_LORA_R", "16"))
LORA_ALPHA = int(os.getenv("KERNELFORGE_LORA_ALPHA", "16"))
LORA_DROPOUT = 0

LORA_TARGETS = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
]

MAX_SEQ_LENGTH = 8192

# Singleton cache
_model = None
_tokenizer = None
_model_type = None  # "moe" or "portable"
_model_key = None


def load_model_and_tokenizer(
    checkpoint_path: str | None = None,
    model_id: str | None = None,
    load_in_4bit: bool | None = None,
    quant_bits: int | None = None,
):
    """Load model and tokenizer.

    Args:
        checkpoint_path: Load from a fine-tuned checkpoint instead of base model.
        model_id: HuggingFace model ID to load (overrides KERNELFORGE_MODEL env var).
        load_in_4bit: Legacy — use quant_bits=4 instead.
        quant_bits: Quantization bits (0=bf16, 4=NF4, 8=INT8). Overrides env var.

    Returns:
        (model, tokenizer) tuple ready for training.
    """
    global _model, _tokenizer, _model_type, _model_key

    # Resolve quantization: explicit param > legacy param > env var
    if quant_bits is not None:
        effective_quant = quant_bits
    elif load_in_4bit is not None:
        effective_quant = 4 if load_in_4bit else 0
    else:
        effective_quant = QUANT_BITS

    resolved_model_id = model_id or PRIMARY_MODEL
    resolved_checkpoint = checkpoint_path if checkpoint_path and os.path.exists(checkpoint_path) else None
    quant_label = {0: "bf16", 4: "4bit", 8: "8bit"}.get(effective_quant, f"{effective_quant}bit")
    cache_key = resolved_checkpoint or f"{resolved_model_id}:{quant_label}"
    if _model is not None and _model_key == cache_key:
        return _model, _tokenizer

    if resolved_checkpoint:
        _model, _tokenizer = _load_from_checkpoint(resolved_checkpoint, quant_bits=effective_quant)
        _model_key = cache_key
        return _model, _tokenizer

    if sys.platform != "linux":
        _model, _tokenizer = _load_portable_dev_model()
        _model_type = "portable"
    else:
        _model, _tokenizer = _load_primary(model_id=resolved_model_id, quant_bits=effective_quant)
        _model_type = "moe"
    _model_key = cache_key
    return _model, _tokenizer


def get_model_type() -> str | None:
    """Return 'moe' or 'portable' depending on which model loaded."""
    return _model_type


def _make_bnb_config(quant_bits: int):
    """Create BitsAndBytesConfig for the given quantization level."""
    import torch
    from transformers import BitsAndBytesConfig

    if quant_bits == 4:
        return BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
        )
    elif quant_bits == 8:
        return BitsAndBytesConfig(
            load_in_8bit=True,
        )
    return None


def _load_primary(model_id: str | None = None, quant_bits: int = 0):
    """Load MoE model via Unsloth FastLanguageModel (supports MoE since 2026)."""
    from unsloth import FastLanguageModel, PatchFastRL

    effective_model = model_id or PRIMARY_MODEL
    quant_label = {0: "bf16", 4: "4bit", 8: "8bit"}.get(quant_bits, f"{quant_bits}bit")

    candidates: list[str] = []
    unsloth_alias = (
        f"unsloth/{effective_model.split('/', 1)[1]}"
        if not effective_model.startswith("unsloth/") and "/" in effective_model
        else (f"unsloth/{effective_model}" if not effective_model.startswith("unsloth/") else effective_model.split("/", 1)[1])
    )
    for candidate in (effective_model, unsloth_alias):
        if candidate and candidate not in candidates:
            candidates.append(candidate)

    last_error: Exception | None = None
    for candidate in candidates:
        print(f"Loading primary model: {candidate} ({quant_label})")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=candidate,
                max_seq_length=MAX_SEQ_LENGTH,
                load_in_4bit=(quant_bits == 4),
                fast_inference=False,
            )
            if tokenizer.pad_token is None and tokenizer.eos_token is not None:
                tokenizer.pad_token = tokenizer.eos_token
            tokenizer.padding_side = "left"

            model = FastLanguageModel.get_peft_model(
                model,
                r=LORA_R,
                target_modules=LORA_TARGETS,
                lora_alpha=LORA_ALPHA,
                lora_dropout=LORA_DROPOUT,
                use_gradient_checkpointing=True,
                random_state=3407,
                bias="none",
            )

            PatchFastRL("GRPO", FastLanguageModel)

            trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in model.parameters())
            print(
                f"Primary model loaded from {candidate}: {trainable:,} trainable / {total:,} total params "
                f"({trainable / total * 100:.2f}%)"
            )
            return model, tokenizer
        except Exception as exc:
            last_error = exc
            print(f"Primary model load failed for {candidate}: {str(exc)[:500]}")

    print("Falling back to Transformers + PEFT primary model loader")
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    hf_model_name = effective_model.split("/", 1)[1] if effective_model.startswith("unsloth/") else effective_model
    tokenizer = AutoTokenizer.from_pretrained(hf_model_name, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    load_kwargs: dict = {
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
        "device_map": "auto",
    }
    bnb_config = _make_bnb_config(quant_bits)
    if bnb_config is not None:
        load_kwargs["quantization_config"] = bnb_config
        print(f"  Using {quant_label} quantization via BitsAndBytes")

    model = AutoModelForCausalLM.from_pretrained(
        hf_model_name,
        **load_kwargs,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    model = get_peft_model(
        model,
        LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=LORA_R,
            lora_alpha=LORA_ALPHA,
            lora_dropout=LORA_DROPOUT,
            target_modules=LORA_TARGETS,
            bias="none",
        ),
    )

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(
        f"Primary model loaded via Transformers + PEFT from {hf_model_name}: "
        f"{trainable:,} trainable / {total:,} total params ({trainable / total * 100:.2f}%)"
    )
    return model, tokenizer


def _load_portable_dev_model():
    """Portable non-Linux fallback used for local control-plane validation."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading portable dev model: {DEV_MODEL}")
    tokenizer = AutoTokenizer.from_pretrained(DEV_MODEL, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        DEV_MODEL,
        trust_remote_code=True,
        torch_dtype=torch.float32,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Portable dev model loaded: {trainable:,} trainable / {total:,} total params")
    return model, tokenizer


def _load_from_checkpoint(checkpoint_path: str, quant_bits: int = 0):
    """Load a fine-tuned checkpoint."""
    import torch

    quant_label = {0: "bf16", 4: "4bit", 8: "8bit"}.get(quant_bits, f"{quant_bits}bit")
    print(f"Loading checkpoint: {checkpoint_path} ({quant_label})")

    # Try Unsloth first (primary path)
    try:
        from unsloth import FastLanguageModel, PatchFastRL
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=checkpoint_path,
            max_seq_length=MAX_SEQ_LENGTH,
            load_in_4bit=(quant_bits == 4),
        )
        if tokenizer.pad_token is None and tokenizer.eos_token is not None:
            tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "left"
        PatchFastRL("GRPO", FastLanguageModel)
        print(f"Loaded checkpoint via Unsloth: {checkpoint_path}")
        return model, tokenizer
    except Exception:
        pass

    # Fall back to HF + PEFT
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from peft import PeftModel

    tokenizer = AutoTokenizer.from_pretrained(checkpoint_path, trust_remote_code=True)
    if tokenizer.pad_token is None and tokenizer.eos_token is not None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    ckpt_load_kwargs: dict = {
        "device_map": "auto",
        "trust_remote_code": True,
        "torch_dtype": torch.bfloat16,
    }
    bnb_config = _make_bnb_config(quant_bits)
    if bnb_config is not None:
        ckpt_load_kwargs["quantization_config"] = bnb_config

    model = AutoModelForCausalLM.from_pretrained(
        checkpoint_path,
        **ckpt_load_kwargs,
    )

    try:
        model = PeftModel.from_pretrained(model, checkpoint_path)
        print(f"Loaded PEFT adapter from {checkpoint_path}")
    except Exception:
        print(f"No PEFT adapter found, using base model from {checkpoint_path}")

    return model, tokenizer

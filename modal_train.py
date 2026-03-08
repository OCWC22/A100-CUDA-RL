"""
Modal training app — runs the 3-stage GRPO pipeline on cloud GPU.

Default: H200 (141GB, $4.54/hr on Modal). bf16 for Qwen3-Coder-30B-A3B-Instruct via Unsloth.
A100 (Modal) handles all eval/reward. You cannot optimize A100 perf by measuring on H200.

Usage:
    # Stage 1 warmup (default)
    modal run modal_train.py --stage 1

    # Stage 3 GRPO demo (10 steps)
    modal run modal_train.py --stage 3

    # Smoke test (1 step, no real training)
    modal run modal_train.py --stage 0
"""
import os
import modal

os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

TRAIN_GPU = os.getenv("KERNELFORGE_TRAIN_GPU", "H200")
APP_NAME = os.getenv("KERNELFORGE_TRAIN_APP", "kernelforge-train")
EVAL_APP_NAME = os.getenv("KERNELFORGE_MODAL_APP", "kernelforge-a100")

train_image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.4.1-cudnn-devel-ubuntu22.04",
        add_python="3.12",
    )
    # Install unsloth first with its full deps so version constraints are respected.
    # Unsloth caps: trl<=0.24.0, datasets<4.4.0, transformers<=4.57.6.
    .pip_install(
        "unsloth",
        "unsloth_zoo",
    )
    .uv_pip_install(
        "torch>=2.4",
        "torchvision>=0.19",
        "trl>=0.18.2,<=0.24.0",
        "transformers>=4.51.3,<=4.57.6",
        "datasets>=3.4.1,<4.4.0",
        "accelerate>=1.4.0",
        "peft>=0.18.0",
        "Pillow>=10.0",
        "bitsandbytes>=0.45",
        "openenv-core[core]>=0.2.1",
        "numpy>=1.26",
        "httpx>=0.27",
        "vllm>=0.10.2,<=0.12.0",
        "modal>=0.70",
    )
    .run_commands("pip install 'https://github.com/lesj0610/flash-attention/releases/download/v2.8.3-cu12-torch2.10-cp312/flash_attn-2.8.3+cu12torch2.10cxx11abiTRUE-cp312-cp312-linux_x86_64.whl' 2>/dev/null || true")
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
    .pip_install("hf_transfer")
    .add_local_python_source(
        "training", "openenv_env", "evaluation", "verification",
    )
    .add_local_dir("datasets", remote_path="/root/datasets")
    .add_local_dir("docs/research/doublegraph", remote_path="/root/docs/research/doublegraph")
    .add_local_dir("archive/datasets_legacy", remote_path="/root/archive/datasets_legacy")
    .add_local_file("skill_a100.md", "/root/skill_a100.md")
    .add_local_file("modal_app.py", "/root/modal_app.py")
)

app = modal.App(APP_NAME)
checkpoints_vol = modal.Volume.from_name("kernelforge-checkpoints", create_if_missing=True)
datasets_vol = modal.Volume.from_name("kernelforge-datasets", create_if_missing=True)
hf_cache_vol = modal.Volume.from_name("kernelforge-hf-cache", create_if_missing=True)


@app.function(
    gpu=TRAIN_GPU,
    image=train_image,
    timeout=3600 * 12,  # 12 hour max
    volumes={
        "/checkpoints": checkpoints_vol,
        "/datasets": datasets_vol,
        "/root/.cache/huggingface": hf_cache_vol,
    },
    include_source=True,
)
def train(stage: int = 1, max_steps: int | None = None, dry_run: bool = False):
    """Run a training stage on Modal GPU.

    Args:
        stage: 0=smoke test, 1=warmup, 2=RFT, 3=GRPO
        max_steps: Override max training steps (for cost control)
        dry_run: Just load model + dataset, don't train
    """
    import torch
    print(f"=== KernelForge Training Stage {stage} ===")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"CUDA: {torch.version.cuda}")

    os.environ.setdefault("KERNELFORGE_EVAL_BACKEND", "modal")
    os.environ.setdefault("KERNELFORGE_MODAL_APP", EVAL_APP_NAME)
    if stage in {0, 1}:
        os.environ.setdefault("KERNELFORGE_SKIP_BENCHMARK", "1")
        os.environ.setdefault("KERNELFORGE_DEBUG_TIMINGS", "1")

    if stage == 0:
        smoke = _smoke_test()
        strict_smoke = os.getenv("KERNELFORGE_SMOKE_STRICT", "1") == "1"
        if strict_smoke and not smoke.get("smoke_ok", False):
            raise RuntimeError("Stage 0 smoke test failed. Fix failing checks before Stage 1.")
        return smoke

    # Set output dirs to Modal volumes
    os.environ["KERNELFORGE_STAGE1_OUTPUT"] = "/checkpoints/stage1"
    os.environ["KERNELFORGE_STAGE2_OUTPUT"] = "/checkpoints/stage2"
    os.environ["KERNELFORGE_STAGE3_OUTPUT"] = "/checkpoints/stage3"
    os.environ.setdefault("KERNELFORGE_USE_VLLM", "0")
    os.environ.setdefault("KERNELFORGE_VLLM_MODE", "server")
    os.environ.setdefault("KERNELFORGE_VLLM_SERVER_BASE_URL", "")
    os.environ.setdefault("KERNELFORGE_USE_TRLOO", "1")
    os.environ.setdefault("KERNELFORGE_STAGE3_SCALE_REWARDS", "batch")
    os.environ.setdefault("KERNELFORGE_STAGE3_BETA", "0.0")
    os.environ.setdefault("KERNELFORGE_STAGE3_MAX_PROMPT_LENGTH", "3072")
    os.environ.setdefault("KERNELFORGE_STAGE3_MAX_TURNS", "3")
    os.environ.setdefault("KERNELFORGE_STAGE3_OPS6K_MAX", "100")

    if stage == 1:
        os.environ.setdefault("KERNELFORGE_SKIP_BENCHMARK", "1")
        os.environ.setdefault("KERNELFORGE_DEBUG_TIMINGS", "1")
        os.environ.setdefault("KERNELFORGE_BATCH_EVAL", "1")
        os.environ.setdefault("KERNELFORGE_STAGE1_MAX_TURNS", "1")
        os.environ.setdefault("CUDA_AGENT_STAGE1_SAMPLES", "4")
        if max_steps is None:
            os.environ.setdefault("KERNELFORGE_STAGE1_MAX_STEPS", "5")
        print(
            "Stage 1 debug defaults: "
            f"skip_benchmark={os.environ['KERNELFORGE_SKIP_BENCHMARK']} "
            f"batch_eval={os.environ['KERNELFORGE_BATCH_EVAL']} "
            f"debug_timings={os.environ['KERNELFORGE_DEBUG_TIMINGS']} "
            f"max_turns={os.environ['KERNELFORGE_STAGE1_MAX_TURNS']} "
            f"samples={os.environ['CUDA_AGENT_STAGE1_SAMPLES']}"
        )
    else:
        os.environ.setdefault("KERNELFORGE_STAGE1_MAX_TURNS", "3")
        os.environ.setdefault("CUDA_AGENT_STAGE1_SAMPLES", "100")

    if max_steps is not None:
        os.environ[f"KERNELFORGE_STAGE{stage}_MAX_STEPS"] = str(max_steps)

    try:
        if stage == 1:
            from training.stage1_warmup import main as stage1_main
            if dry_run:
                return _dry_run_stage1()
            stage1_main()
        elif stage == 2:
            from training.stage2_rft import main as stage2_main
            stage2_main()
        elif stage == 3:
            from training.stage3_grpo import main as stage3_main
            stage3_main()
        else:
            print(f"Unknown stage: {stage}")
            return {"error": f"Unknown stage {stage}"}

        return {"status": "complete", "stage": stage}
    finally:
        # Persist artifacts even if a stage fails mid-run.
        for vol, label in ((checkpoints_vol, "checkpoints"), (datasets_vol, "datasets")):
            try:
                vol.commit()
            except Exception as exc:
                print(f"WARNING: failed to commit {label} volume: {exc}")


def _smoke_test() -> dict:
    """Quick validation: model loads, dataset loads, one forward pass."""
    import torch

    results = {}
    model = None
    tokenizer = None

    # Test 1: Model loading
    print("\n[1/4] Loading model...")
    try:
        from training.model_loader import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer()
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        results["model"] = {
            "loaded": True,
            "total_params": params,
            "trainable_params": trainable,
            "vram_gb": torch.cuda.memory_allocated() / 1e9,
        }
        print(f"  Model loaded: {trainable:,} trainable / {params:,} total")
        print(f"  VRAM used: {results['model']['vram_gb']:.1f} GB")
    except Exception as e:
        results["model"] = {"loaded": False, "error": str(e)[:500]}
        print(f"  FAILED: {e}")

    # Test 2: Dataset loading
    print("\n[2/4] Loading dataset...")
    try:
        from training.stage1_warmup import load_stage1_dataset
        dataset = load_stage1_dataset()
        results["dataset"] = {
            "loaded": True,
            "size": len(dataset),
            "columns": dataset.column_names,
        }
        print(f"  Dataset loaded: {len(dataset)} examples, columns: {dataset.column_names}")
    except Exception as e:
        results["dataset"] = {"loaded": False, "error": str(e)[:500]}
        print(f"  FAILED: {e}")

    # Test 3: Generation
    print("\n[3/4] Testing generation...")
    try:
        if model is None or tokenizer is None:
            raise RuntimeError("Skipping generation because model loading failed")
        test_prompt = "Write a CUDA vector addition kernel for A100."
        inputs = tokenizer(test_prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            output = model.generate(**inputs, max_new_tokens=64, temperature=0.7, do_sample=True)
        generated = tokenizer.decode(output[0], skip_special_tokens=True)
        results["generation"] = {
            "works": True,
            "output_tokens": len(output[0]),
            "preview": generated[:200],
        }
        print(f"  Generated {len(output[0])} tokens")
    except Exception as e:
        results["generation"] = {"works": False, "error": str(e)[:500]}
        print(f"  FAILED: {e}")

    # Test 4: Eval connectivity
    print("\n[4/4] Testing configured eval backend...")
    try:
        from openenv_env.eval_backend import EVAL_BACKEND, dispatch_eval

        test_result = dispatch_eval(
            "evaluate_ops6k_kernel",
            {
                "cuda_code": '#include <torch/extension.h>\n\ntorch::Tensor run_kernel(torch::Tensor x) {\n    return x * 2;\n}\n\nPYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {\n    m.def("run_kernel", &run_kernel);\n}',
                "task_code": 'import torch\nclass Model(torch.nn.Module):\n  def __init__(self): super().__init__()\n  def forward(self, x): return x * 2\ndef get_inputs(): return [torch.randn(256, device="cuda")]\ndef get_init_inputs(): return []',
                "evaluation_backend": "ops6k",
                "task_id": "stage0_smoke",
                "trace_id": "stage0_smoke",
                "skip_benchmark": os.getenv("KERNELFORGE_SKIP_BENCHMARK", "0") == "1",
            },
        )
        results["eval"] = {
            "connected": True,
            "compiles": test_result.get("compiles", False),
            "correct": test_result.get("correct", False),
            "backend": EVAL_BACKEND,
        }
        print(
            f"  Eval backend ({EVAL_BACKEND}): compiles={test_result.get('compiles')} "
            f"correct={test_result.get('correct')}"
        )
    except Exception as e:
        results["eval"] = {"connected": False, "error": str(e)[:500]}
        print(f"  FAILED: {e}")

    print("\n=== Smoke Test Complete ===")
    smoke_status = {
        "model": bool(results.get("model", {}).get("loaded")),
        "dataset": bool(results.get("dataset", {}).get("loaded")),
        "generation": bool(results.get("generation", {}).get("works")),
        "eval": bool(
            results.get("eval", {}).get("connected")
            and results.get("eval", {}).get("compiles")
            and results.get("eval", {}).get("correct")
        ),
    }
    for key, passed in smoke_status.items():
        print(f"  {key}: {'PASS' if passed else 'FAIL'}")

    results["smoke_ok"] = all(smoke_status.values())
    print(f"  overall: {'PASS' if results['smoke_ok'] else 'FAIL'}")

    return results


def _dry_run_stage1() -> dict:
    """Load model + dataset but don't train. For cost verification."""
    import torch
    from training.model_loader import load_model_and_tokenizer
    from training.stage1_warmup import load_stage1_dataset

    model, tokenizer = load_model_and_tokenizer()
    dataset = load_stage1_dataset()

    vram = torch.cuda.memory_allocated() / 1e9
    total_vram = torch.cuda.get_device_properties(0).total_memory / 1e9
    print(f"Model VRAM: {vram:.1f} / {total_vram:.1f} GB")
    print(f"Dataset: {len(dataset)} examples")
    print(f"Free VRAM: {total_vram - vram:.1f} GB (for KV cache + optimizer)")

    return {
        "status": "dry_run",
        "vram_used_gb": vram,
        "vram_total_gb": total_vram,
        "dataset_size": len(dataset),
        "fits": vram < total_vram * 0.85,
    }


@app.local_entrypoint()
def main(stage: int = 1, max_steps: int = 0, dry_run: bool = False):
    """CLI entrypoint: modal run modal_train.py --stage 1 --max-steps 10"""
    steps = max_steps if max_steps > 0 else None
    result = train.remote(stage=stage, max_steps=steps, dry_run=dry_run)
    print(f"\nResult: {result}")

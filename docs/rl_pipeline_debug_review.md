# KernelForge RL Pipeline: Debug Review

**Last updated:** March 8, 2026
**Status:** Pipeline runs end-to-end but first non-degenerate reward signal pending.

---

## A. How a Prompt Becomes a Reward

```
Prompt (CUDA kernel task)
  → Model generates completion (up to max_completion_length tokens)
  → extract_cuda_code() parses fenced code block or raw __global__
  → Local nvcc compile check (15s timeout, sm_80 arch)
  → Remote A100 evaluation via Modal:
      1. Compile CUDA extension (nvcc + pybind11)
      2. Correctness: 5-seed adversarial verification vs reference PyTorch
      3. Benchmark: warmup + timed runs for runtime_ms
      4. Compute speedup vs eager PyTorch and vs torch.compile
  → Discrete milestone reward: {-1, 1, 2, 3}
  → Feedback formatted for next turn (if multi-turn)
```

### Reward Milestones

| Reward | Meaning | Gate |
|--------|---------|------|
| **-1** | Failed | Didn't compile, incorrect output, or no code extracted |
| **1** | Correct | Compiles + correct output, but not faster than eager |
| **2** | Beat eager | Correct + >5% speedup vs eager PyTorch |
| **3** | Beat torch.compile | Correct + >5% speedup vs torch.compile |

**Why discrete?** Normalizes reward across problem difficulty. Beating torch.compile on matmul and on relu both earn r=3, preventing easy tasks from dominating gradient signal.

### TRLOO Correction

Standard GRPO computes advantages as:
```
A_i = (r_i - mean(r_group)) / (std(r_group) + eps)
```

The problem: `r_i` is included in `mean(r_group)`, shrinking expected gradients by `(1 - 1/G)`. With G=4, that's a 25% shrinkage.

TRLOO fix (from Dr. Kernel, arXiv 2602.05885):
```
A_i_corrected = A_i * G / (G - 1)
```

Implemented in `training/custom_grpo_trainer.py:TRLOOGRPOTrainer._compute_advantages()`.

---

## B. GRPO Configuration

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `num_generations` (G) | 8 | Need reward variance across completions. G=2 with sparse rewards gives all-identical rewards → zero gradient |
| `max_completion_length` | 1024 | CUDA kernels need 500-800 tokens minimum. 256 caused 100% truncation |
| `temperature` | 1.0 | High exploration during warmup |
| `learning_rate` | 2e-6 | Conservative to avoid catastrophic forgetting |
| `beta` | 0.0 | No KL penalty — let model explore freely |
| `per_device_train_batch_size` | 1 | Memory constraint |
| `gradient_accumulation_steps` | 4 | Effective batch size = 4 |
| `optim` | paged_adamw_8bit | Memory-efficient optimizer |
| `bf16` | True | Standard for A100 training |
| `max_turns` | 1 (debug) / 3 (prod) | Multi-turn with error feedback |

### Model

| Property | Value |
|----------|-------|
| Model | Qwen3.5-2B-Claude-4.6-Opus-Reasoning-Distilled |
| Architecture | Dense (not MoE) |
| VRAM (bf16) | ~4 GB |
| LoRA rank | 64 (higher for small model) |
| Training GPU | A100 80GB |
| Free VRAM | ~76 GB (room for G=8+ and long sequences) |

---

## C. Known Failure Modes

### 1. All Rewards -1.0 (Dead Signal)
**Symptom:** `reward distribution: mean=-1.00 std=0.00`
**Causes:**
- `max_completion_length` too short → all completions truncated → no valid CUDA code
- Model can't generate CUDA syntax (canary should catch this)
- Code extraction regex misses valid code
- Eval backend unreachable

**Fix:** Check `clipped_ratio` in training logs. If 1.0, increase `max_completion_length`.

### 2. Zero Reward Variance
**Symptom:** `WARNING: zero reward variance — GRPO will produce zero gradients`
**Causes:**
- All G completions get the same reward (typically all -1.0)
- G too small (G=2 with sparse rewards)
- All prompts too hard or too easy

**Fix:** Increase G, verify completion quality, check prompt difficulty distribution.

### 3. Slow Steps (>10 min each)
**Causes:**
- Serial eval dispatch (each kernel evaluated one at a time)
- Modal container cold starts (~5-10s per call)
- Benchmarking enabled (adds 5-10s per kernel)
- Large G increases per-step cost linearly

**Fix:** Set `KERNELFORGE_SKIP_BENCHMARK=1`, `KERNELFORGE_BATCH_EVAL=1`.

### 4. Generation Timeout
**Symptom:** `model.generate() timed out after 120s`
**Causes:**
- Broken attention backend
- Model too large for available VRAM
- CUDA OOM during generation

**Fix:** Check VRAM usage, verify xformers installed, reduce `max_completion_length`.

### 5. Fork Deadlock
**Symptom:** Process hangs silently after model load
**Causes:**
- CUDA initialized before fork (DataLoader workers)
- Missing `multiprocessing.set_start_method("spawn")`

**Fix:** Spawn guard is set in both `modal_train.py` and `stage1_warmup.py`. Verify `dataloader_num_workers=0`.

---

## D. Debugging Checklist

Run these checks in order:

- [ ] **Canary passes**: Look for `[canary] PASS — N tokens generated` in logs
- [ ] **Model VRAM**: Should be ~4GB for 2B model, leaving ~76GB free on A100
- [ ] **Dataset loaded**: Check `Loaded N unified Stage 1 prompts`
- [ ] **Completion length**: Look for truncation warnings; completions should reach 500+ tokens
- [ ] **Code extraction**: Check if `extract_cuda_code()` finds valid CUDA in completions
- [ ] **Local compile**: Check pass rate in rollout logs
- [ ] **Eval connectivity**: Look for `Eval backend (modal): compiles=True correct=True` in smoke test
- [ ] **Reward distribution**: Must have non-zero std — look for `reward distribution:` lines
- [ ] **Gradient signal**: Loss should change between steps (not constant)
- [ ] **Checkpoint saved**: Look for `Stage 1 complete. Checkpoint saved to`

### Environment Variables for Debugging

```bash
KERNELFORGE_DEBUG_TIMINGS=1        # Per-eval timing breakdown
KERNELFORGE_SKIP_BENCHMARK=1       # Skip benchmark (compile+correctness only)
KERNELFORGE_BATCH_EVAL=1           # Batch eval dispatch
KERNELFORGE_GENERATION_TIMEOUT=120 # Seconds before generation timeout
KERNELFORGE_STAGE1_MAX_TURNS=1     # Reduce turns for faster iteration
KERNELFORGE_STAGE1_MAX_STEPS=5     # Limit training steps
CUDA_AGENT_STAGE1_SAMPLES=4        # Limit dataset size
```

---

## E. Key Files

| File | Role |
|------|------|
| `modal_train.py` | Modal cloud entry point — image build, GPU selection, stage dispatch |
| `training/stage1_warmup.py` | Stage 1 GRPO warmup — loads model/dataset, configures GRPOTrainer |
| `training/model_loader.py` | Unified model loading — Unsloth + LoRA setup, quantization support |
| `training/multi_turn_rollout.py` | Multi-turn rollout loop — generation, code extraction, eval dispatch, reward, feedback |
| `training/custom_grpo_trainer.py` | TRLOOGRPOTrainer — GRPO with leave-one-out correction |
| `training/dataset_loader.py` | Dataset loading — Ops-6K, DoubleGraph, curriculum support |
| `training/task_support.py` | Task routing — eval backend selection, payload construction, reward computation |
| `openenv_env/reward.py` | Reward computation — discrete milestones, TRLOO scaling |
| `openenv_env/eval_backend.py` | Eval dispatch abstraction — routes to Modal or CoreWeave |
| `configs/scaling_ladder.json` | Model configurations for scaling study |

---

## F. Reward Pipeline Data Flow

```
training/stage1_warmup.py
  → TRLOOGRPOTrainer.train()
    → rollout_func(prompts, trainer)  [multi_turn_rollout.py]
      → generate_rollout_completions(trainer, prompts)
      → extract_cuda_code(completion_text)
      → _local_compile_check(code)                    # 15s nvcc syntax check
      → evaluate_code_remote(code, task_row, ...)      # task_support.py
        → build_modal_payload(code, task_row, ...)     # constructs fn_name + payload
        → eval_backend.dispatch_eval(fn_name, payload) # eval_backend.py
          → _dispatch_modal() or _dispatch_http()      # Modal or CoreWeave
        → normalize_eval_result(raw_result)
        → compute_task_reward(result)                  # → reward.compute_reward()
      → _format_feedback(result, reward, turn)         # for next turn
    → reward_from_env(completions, env_reward=...)     # extracts rewards for GRPO
  → TRLOOGRPOTrainer._compute_advantages(rewards)
    → vanilla GRPO advantages: (r - mean) / (std + eps)
    → TRLOO correction: * G/(G-1)
  → backprop with corrected gradients
```

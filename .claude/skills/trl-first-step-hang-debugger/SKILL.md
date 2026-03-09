---
name: trl-first-step-hang-debugger
description: Diagnose CUDA RL and GRPO runs that successfully load the model and pass a raw generate canary, but hang at 0% on the first trainer step. Use for first-step stalls, rollout hangs, old-kernel runtime issues, trainer/generate deadlocks, multimodal-to-text training edge cases, and slow-path generation inside TRL.
---

# TRL First-Step Hang Debugger

Use this skill when:
- model loading succeeds
- tokenizer setup succeeds
- a raw `model.generate()` canary succeeds
- training starts but progress stays at 0%
- the run hangs on the first GRPO step
- logs show old kernel warnings, accelerate warnings, or very slow rollout startup

Do not use this skill for:
- model load failures
- tokenizer class failures
- Modal app lookup failures
- reward collapse after multiple completed steps
- CUDA correctness or benchmark failures after evaluation already runs

## Core rule

Treat this as a runtime-stage failure, not a model-stage failure.

The system has already proven:
1. container boot works
2. GPU is visible
3. model weights can load
4. tokenizer can run
5. raw generation can run

So the failure is between:
- trainer initialization
- grouped rollout generation
- batch collation
- distributed/runtime synchronization
- reward/eval call inside the trainer step

## Required failure buckets

Always classify the first-step stall into one primary bucket:

1. trainer rollout hang
2. distributed / accelerate synchronization hang
3. kernel / host runtime incompatibility
4. pathological generation settings
5. multimodal model path causing trainer issues
6. reward function call blocking
7. remote evaluator blocking
8. dataloader / collator stall
9. VRAM pressure / memory thrash causing apparent hang

Do not jump to reward-shaping conclusions unless at least one trainer step completed.

## Required investigation order

1. Prove the boundary
   - confirm model load completed
   - confirm canary generation passed
   - confirm stall begins only after trainer starts

2. Inspect host/runtime warnings
   - kernel version warnings
   - accelerate or distributed warnings
   - NCCL / torch.distributed warnings
   - deprecation warnings only if tied to the hang boundary

3. Inspect rollout cost
   - num_generations / group size
   - max prompt length
   - max completion length
   - batch size and effective batch size
   - whether first rollout is much heavier than canary

4. Inspect trainer internals
   - where GRPO first calls generate
   - whether reward funcs run inline
   - whether remote evaluation is invoked during step 0
   - whether compile/eval timeouts are too high

5. Inspect model path
   - whether a VLM is loaded and later converted to text-only
   - whether visual weights or multimodal processors are involved unnecessarily
   - whether the policy model should be replaced with a native causal LM

6. Inspect performance path
   - flash attention / linear attention fast path availability
   - causal-conv1d / fla installation
   - whether fallback torch paths are dramatically slowing rollout

7. Reduce to a smoke test
   - num_generations = 2
   - max_completion_length = 128 or 256
   - 1 prompt
   - 1 training step
   - reward stub or cheap local reward
   - no remote eval

## Required output format

### Failure summary
State exactly where the run stalls and why the issue is a trainer/runtime hang.

### Boundary proof
Show the last successful stage and the first stage that never returns.

### Root-cause ranking
List top likely causes in ranked order with evidence.

### Minimal fix
Give the smallest config/runtime change likely to make step 0 complete.

### Patch plan
List exact files, config keys, env vars, and parameter edits.

### Verification
Success means:
- first GRPO step completes
- progress advances beyond 0%
- step time is logged
- reward pipeline runs at least once

## Heuristics

- If raw `model.generate()` passes but trainer hangs, blame trainer/runtime path before blaming weights.
- If logs warn that the Linux kernel is below recommended minimum, treat that as a top-tier blocker.
- If a multimodal model is being loaded for text-only RL, question that choice immediately.
- If fast-path attention libs are missing, expect slow first rollout, but do not confuse slow with deadlocked.
- If remote reward/eval is in the step path, temporarily stub it out to isolate the hang.

## Default bias

Prefer proving the exact hang boundary.
Prefer a tiny 1-step smoke test over another long run.
Prefer native text-only causal LM paths over VLM-extraction hacks for CUDA-code RL.

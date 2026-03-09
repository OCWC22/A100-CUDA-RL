---
name: grpo-reward-pipeline-debugger
description: Diagnose GRPO, Modal, CUDA, and A100 training failures where code generation succeeds but rewards collapse, eval backends fail, Modal app or function lookup breaks, or training shows zero learning due to infra, config, or reward-pipeline issues. Use for reward=-1 fallback loops, reward_std=0, loss=0, eval_ok=0, App not found, remote eval dispatch bugs, and local-vs-remote evaluator debugging.
---

# GRPO Reward Pipeline Debugger

Use this skill when:
- GRPO or RL training produces zero learning even though code is being generated
- rewards collapse to a constant fallback value
- Modal remote eval fails with app lookup, function lookup, environment, or deployment errors
- CUDA kernel evaluation may be failing before compile/run actually happens
- the user wants both first-principles explanation and a concrete fix plan

Do not use this skill for:
- generic CUDA optimization without RL or reward evaluation
- pure dataset quality issues unless reward/eval plumbing has already been ruled out
- pure model capability complaints without runtime evidence

## Core rule

Classify the failure before proposing edits. The failure must be placed into one of these buckets:

1. generation failure
2. code extraction failure
3. eval dispatch failure
4. remote service / Modal lookup failure
5. CUDA compile failure
6. correctness failure
7. performance/benchmark failure
8. reward shaping collapse
9. GRPO configuration or grouping issue
10. first-step trainer/runtime hang

Do not propose broad redesigns before identifying which bucket is primary.

## Pre-check before reward diagnosis

Before investigating reward collapse, prove that at least one trainer step completed.

If progress remains at 0% and the first GRPO step never finishes:
- do not classify as reward collapse yet
- reclassify to first-step trainer/runtime hang
- use the `trl-first-step-hang-debugger` skill first

Reward collapse requires a completed rollout with actual reward values.
A hang before step completion is a runtime-path problem, not yet a reward-statistics problem.

## Required investigation order

Always inspect in this order:

1. Training entrypoint
   - identify stage, trainer, reward wiring, backend defaults, env vars

2. Reward function path
   - find fallback reward assignments
   - find exception handlers that convert infra errors into scalar rewards

3. Eval backend dispatcher
   - determine local vs remote dispatch mode
   - identify backend selection logic
   - trace exact function name and app/service name used

4. Remote execution assumptions
   - for Modal, verify app name, function name, environment, deployment expectation
   - determine whether code assumes a deployed app exists

5. Actual evaluator implementation
   - locate the function that compiles/runs CUDA
   - verify whether the training image/container includes required source and dependencies

6. GRPO signal quality
   - inspect whether reward variance can become non-zero
   - inspect batch summaries for eval_ok, reward_std, compile pass rate, correctness pass rate

## Required output format

### Failure summary
A short paragraph stating:
- what is breaking
- where it breaks
- whether the root cause is infra, CUDA, reward, or GRPO config

### Evidence
List exact file paths, symbols, env vars, logs, and exception paths proving the diagnosis.

### First-principles explanation
Explain:
prompt -> generation -> extraction -> eval dispatch -> CUDA compile/run -> correctness/perf -> scalar reward -> GRPO group comparison -> gradient signal

State exactly which stage fails and why downstream stages become invalid.

### Minimal fix
Propose the smallest fix that restores real reward evaluation.
Prefer:
- config fix before architecture rewrite
- local eval before cross-service complexity when valid
- dependency fix only when directly required

### Patch plan
List exact files and exact edits.

### Verification
List concrete success signals:
- lookup/deployment errors removed
- eval_ok > 0
- reward_std > 0
- rewards no longer all fallback values
- loss or advantages no longer degenerate

## Heuristics

- If code exists but all rewards are fallback values, suspect eval plumbing before blaming the model.
- If logs show app/function/environment lookup failures, classify as infra dispatch failure.
- If the evaluator is remote but training already runs on a GPU container with needed dependencies, consider whether local evaluation is the simplest fix.
- Do not confuse no learning with bad model quality when reward variance is zero.
- Distinguish:
  - dispatch failed before CUDA
  - CUDA compile failed
  - kernel ran but was incorrect
  - kernel was correct but slower

## Modal checklist

Verify:
- app name
- function name
- environment name
- whether target app is deployed
- whether from_name / lookup assumes deployment
- whether local source packages are included in the image

## CUDA/A100 checklist

Verify:
- evaluation code path actually reaches CUDA compile/run
- required build/runtime dependencies exist in the image
- target architecture matches the GPU
- VRAM contention is only considered after dispatch is fixed

## GRPO checklist

Verify:
- rewards differ across group samples
- group size > 1
- reward normalization is not collapsing to zero
- fallback reward path is not swallowing all exceptions
- metrics expose compile/correctness/perf breakdowns

## Default bias

Bias toward proving the current failure with evidence.
Bias toward the minimum patch that restores a real reward signal.

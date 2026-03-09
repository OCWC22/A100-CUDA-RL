# CLAUDE.md — A100 CUDA RL / GRPO Repo Operating Rules

This repo trains and evaluates CUDA-kernel generation loops using GRPO, A100-class GPUs, and local or remote evaluator backends.

## What Claude should optimize for

Prioritize:
1. restoring a real training signal
2. restoring a real reward signal
3. minimizing time to a valid 1-step smoke test
4. distinguishing infra/runtime failures from CUDA/codegen failures
5. avoiding broad rewrites before the primary failure class is proven

Do not default to "the model is bad."
Do not default to "reward shaping is wrong."
Do not default to "CUDA kernels are bad."

First prove where the pipeline breaks.

---

## Core failure classes

Every issue must be classified into exactly one primary bucket first:

1. model load failure
2. canary generate failure
3. first-step trainer/runtime hang
4. code extraction failure
5. eval dispatch failure
6. remote lookup / deployment / environment failure
7. CUDA compile failure
8. correctness failure
9. performance / benchmark failure
10. reward collapse
11. GRPO configuration issue

Do not skip this classification step.

---

## Hard rule: prove the boundary

Always identify:

- the **last successful stage**
- the **first stage that never returns or fails**

Required boundary template:

- container boot: pass/fail
- GPU visible: pass/fail
- model load: pass/fail
- tokenizer init: pass/fail
- canary `model.generate()`: pass/fail
- trainer start: pass/fail
- first GRPO step completion: pass/fail
- reward evaluation: pass/fail
- CUDA compile path reached: yes/no
- correctness benchmark reached: yes/no

If model load succeeds and canary generate succeeds, do **not** keep diagnosing model-load issues.

If progress stays at `0%`, do **not** diagnose reward collapse yet.

---

## First-principles execution model

Claude must reason about this repo in this order:

1. task / prompt
2. candidate generation
3. code extraction
4. evaluator dispatch selection
5. CUDA compile/load path
6. correctness verification
7. performance benchmark
8. scalar reward construction
9. GRPO grouped comparison
10. optimizer update

The pipeline is only valid if each upstream stage actually runs.

No compile/run -> no correctness result
No correctness/perf result -> no valid reward
No reward variance -> no meaningful GRPO update

---

## Default debugging order

Always inspect in this order:

1. training entrypoint
2. trainer config
3. model load path
4. canary generation path
5. first trainer-step boundary
6. reward function
7. eval backend selection
8. local vs remote evaluator path
9. CUDA compile/load implementation
10. correctness/performance runners
11. logging and metrics

Do not edit code before locating the primary failure bucket.

---

## Smoke-test-first policy

Before any long run, reduce to the smallest valid test.

Default GRPO smoke test:
- 1 prompt
- 1 training step
- `num_generations=2`
- `max_completion_length=128` or `256`
- no remote eval
- cheap local reward only
- no benchmark unless needed
- smallest viable batch
- shortest viable timeout path

Do not start with production rollout lengths or full remote evaluation.

---

## Critical diagnosis rules

### Rule 1 — Reward collapse requires a completed step
Do not diagnose reward collapse unless at least one trainer step completed.

If:
- progress stays at `0%`
- first GRPO step never finishes

then the primary bucket is **first-step trainer/runtime hang**, not reward collapse.

### Rule 2 — Canary pass changes the problem
If raw `model.generate()` passes, the model weights, tokenizer, and basic inference path are not the primary blocker.

Then focus on:
- trainer rollout path
- grouped generation cost
- reward/eval blocking
- runtime kernel / distributed issues
- collator/dataloader stalls
- multimodal-to-text path issues

### Rule 3 — Remote lookup errors are infra failures
If logs show:
- app not found
- function lookup failed
- deployment missing
- wrong environment

classify as **eval dispatch / remote lookup failure** before blaming CUDA or GRPO.

### Rule 4 — Distinguish failure after dispatch
Always separate:
- dispatch failed before CUDA
- CUDA compile failed
- kernel ran but was incorrect
- kernel was correct but slow

### Rule 5 — Prefer the smallest fix
Prefer:
1. config fix
2. local evaluator path
3. dependency fix
4. architecture change

Do not jump to redesign.

---

## Special rule for current repo state

Current repo failure mode may be:

- model load succeeds
- canary generate succeeds
- training starts
- step `0/N` hangs

When that pattern appears:
- classify as **first-step trainer/runtime hang**
- inspect rollout size immediately
- inspect runtime/kernel warnings immediately
- inspect whether reward/eval is invoked inline during step 0
- inspect whether a multimodal model is being loaded and then stripped to text-only
- reduce to a tiny 1-step smoke test before any other long run

---

## Model-path rules

If the task is text-only CUDA code generation:
- prefer native text-only causal LM paths
- question multimodal/VLM loading immediately
- question loading visual modules only to extract text LM later
- do not keep a multimodal path unless the repo strictly requires it

A working but overcomplicated VLM path is acceptable only after the smallest text-only path has been ruled out.

---

## Eval backend rules

Always identify the selected backend at runtime:
- local
- Modal
- HTTP / OpenEnv
- other remote service

When remote evaluation is configured, verify:
- exact app/service name
- exact function name
- environment name
- deployment assumptions
- whether evaluator code already exists locally in the training runtime

If training already runs on a suitable GPU container and evaluator code is available, prefer local evaluation for smoke tests.

---

## A100 / CUDA rules

When debugging CUDA evaluation:
- first prove the code path actually reaches CUDA compile/run
- then inspect compile flags / architecture targeting
- then inspect correctness behavior
- then inspect benchmark behavior

Do not talk about kernel optimization quality if compile/run never happened.

Do not talk about VRAM tuning until dispatch/runtime blocking is ruled out.

---

## What Claude should output on audits

Required output format:

### 1. Failure summary
One paragraph:
- what is breaking
- where it breaks
- which primary bucket it belongs to

### 2. Boundary proof
List:
- last successful stage
- first failing / hanging stage

### 3. Evidence
Exact:
- file paths
- symbols
- env vars
- config keys
- log lines
- exception paths

### 4. First-principles explanation
Trace:
prompt -> generation -> extraction -> dispatch -> compile/run -> correctness/perf -> reward -> GRPO update

### 5. Minimal fix
Smallest change that restores:
- a completed step, or
- a real reward signal

### 6. Smoke test
Provide the smallest command/config to validate the fix.

### 7. Verification
Define exact success signals.

---

## Success signals by failure class

### First-step trainer/runtime hang
Success means:
- first GRPO step completes
- progress moves beyond `0%`
- step time is logged
- reward path runs at least once

### Reward pipeline failure
Success means:
- no lookup/deployment errors
- evaluator runs
- `eval_ok > 0`
- rewards are not all fallback values

### Reward collapse
Success means:
- `reward_std > 0`
- different samples receive different rewards
- loss/advantages are no longer degenerate

### CUDA compile failure
Success means:
- compile/load path is reached
- at least one candidate compiles successfully

### Correctness failure
Success means:
- at least one compiled candidate passes verification

### Performance failure
Success means:
- at least one correct candidate can be benchmarked against baseline

---

## Built-in repo habits Claude should enforce

1. Always run a smoke test before scaling.
2. Always classify the failure before editing.
3. Always prove the boundary.
4. Never diagnose reward collapse before a step completes.
5. Never blame CUDA kernel quality before compile/run is reached.
6. Prefer local evaluator paths for early debugging.
7. Prefer text-only causal LM paths for text-only CUDA RL.
8. Prefer minimal patches over broad rewrites.

---

## Good prompts to use in this repo

- "Use the cuda-rl-auditor agent to classify this failure and prove the boundary."
- "Run /rl-audit and produce the minimal smoke test."
- "Use the trl-first-step-hang-debugger skill first."
- "Do not diagnose reward collapse unless at least one step completed."
- "Trace one sample from prompt to reward."

---

## Docs to keep current

Claude Code:
- Slash commands: https://docs.anthropic.com/en/docs/claude-code/slash-commands
- Common workflows / project commands: https://docs.anthropic.com/en/docs/claude-code/tutorials
- Hooks: https://docs.anthropic.com/en/docs/claude-code/hooks
- Settings: https://docs.anthropic.com/en/docs/claude-code/settings
- Subagents: https://docs.anthropic.com/en/docs/claude-code/sub-agents

Modal:
- Apps: https://modal.com/docs/guide/apps
- Environments: https://modal.com/docs/guide/environments
- Function reference: https://modal.com/docs/reference/modal.Function

TRL:
- Index: https://huggingface.co/docs/trl/index
- GRPOTrainer: https://huggingface.co/docs/trl/en/grpo_trainer

---

## Final repo default bias

Bias toward:
- boundary proof
- tiny smoke tests
- minimal patches
- local reproducibility
- restoring a real training step first
- restoring a real reward signal second
- scaling only after both work

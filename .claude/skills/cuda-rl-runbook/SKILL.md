---
name: cuda-rl-runbook
description: Audit and run CUDA kernel RL pipelines on A100-class GPUs using GRPO or similar methods, including training setup, evaluator wiring, local-vs-remote execution, Modal/OpenEnv integration, dependency checks, experiment readiness checks, and first-principles explanation of CUDA RL systems.
---

# CUDA RL Runbook

Use this skill when:
- the user wants to understand or run a CUDA-kernel RL pipeline end to end
- the codebase combines GRPO, CUDA evaluation, Modal or OpenEnv, and remote execution
- the task is to audit training readiness before running experiments
- the user wants step-by-step explanation from first principles

## Goal

Produce a practical runbook, not a theory dump.

## Required mental model

Explain the pipeline using this structure:

1. Prompt/task
2. Candidate generation
3. Code extraction
4. CUDA compile/load path
5. Correctness verification
6. Performance benchmark
7. Reward construction
8. GRPO group comparison
9. Parameter update
10. Re-run loop

## Required investigation order

1. Repo structure and entrypoints
2. Model/trainer configuration
3. Reward function and metrics
4. Evaluator implementation
5. Runtime backend selection
6. Image/container dependencies
7. Remote execution assumptions
8. Logging and verification metrics
9. Minimal command to run a smoke test
10. Expected success and failure signatures

## Required output format

### System overview
Short, end-to-end explanation.

### Critical files
List only the files that matter most.

### How one sample flows through the system
Trace one example from prompt to reward.

### Runtime modes
Explain local eval vs remote eval and when each is appropriate.

### Minimum viable smoke test
Provide the smallest reliable test to verify the stack.

For GRPO smoke tests, always prefer:
- 1 prompt
- 1 training step
- num_generations=2
- max_completion_length=128 or 256
- no remote eval
- cheap local reward only

Do not begin with production completion lengths or full remote evaluation.

### Failure signatures
Explain how to distinguish:
- model load failure
- canary generate failure
- canary passes, trainer hangs at 0%
- no code generated
- no code extracted
- dispatch failure
- compile failure
- correctness failure
- perf regression
- reward collapse

### Fix priority
Give the order in which issues should be fixed.

## Bias

Prefer operational clarity over research-summary style.
Prefer smoke tests before long training runs.
Prefer the simplest valid runtime path for a hackathon or early prototype.

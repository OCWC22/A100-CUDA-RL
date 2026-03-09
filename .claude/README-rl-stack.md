# Claude Code RL Stack for A100 CUDA GRPO Repo

This repo uses a focused Claude Code setup for CUDA-kernel RL development and debugging.

## What exists

### Commands
- `/rl-audit` — full pipeline audit
- `/rl-smoke` — force a tiny 1-step smoke-test mindset

### Skills
- `cuda-rl-runbook`
- `grpo-reward-pipeline-debugger`
- `trl-first-step-hang-debugger`

### Subagent
- `cuda-rl-auditor`

### Hooks
- reward/dead-signal detector
- first-step hang detector

## Failure classes we care about

1. model load failure
2. canary generate failure
3. first-step trainer/runtime hang
4. code extraction failure
5. eval dispatch / remote lookup failure
6. CUDA compile failure
7. correctness failure
8. performance failure
9. reward collapse
10. GRPO config issue

## Core repo rule

Do not diagnose reward collapse unless at least one trainer step completed.

If:
- model load succeeds
- canary generate succeeds
- progress stays at 0%

then use the `trl-first-step-hang-debugger` skill first.

## Core smoke-test rule

Before any long run, validate with:
- 1 prompt
- 1 training step
- num_generations=2
- max_completion_length=128 or 256
- no remote eval
- cheap local reward
- no benchmark

## Recommended docs to keep pinned

- Claude Code slash commands
- Claude Code hooks
- Claude Code settings
- Claude Code subagents
- Modal apps / environments / function lookup
- TRL GRPOTrainer

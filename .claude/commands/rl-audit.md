Audit this CUDA RL / GRPO pipeline.

Required output:
1. System overview
2. Critical files
3. One-sample execution trace from prompt to reward
4. Failure classification:
   - model load
   - canary generate
   - first-step trainer/runtime hang
   - extraction
   - dispatch
   - remote lookup
   - CUDA compile
   - correctness
   - performance
   - reward collapse
   - GRPO config
5. Boundary proof:
   - last successful stage
   - first stage that never returns
6. Minimal fix
7. 1-step smoke test config
8. Verification checklist

Priorities:
- prove the boundary before editing files
- prefer config/runtime isolation before architecture changes
- prefer tiny smoke tests before long GRPO runs
- do not diagnose reward collapse unless at least one step completed
- if model load and canary pass but progress stays at 0%, use trl-first-step-hang-debugger first

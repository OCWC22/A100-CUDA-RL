# GRPO Sanity Checklist

Before long training, verify:
1. group size > 1
2. reward function can produce varied values
3. evaluator works on at least one known-good example
4. bad samples fail for the right reason
5. logs distinguish compile/correctness/perf/reward
6. one-step smoke test completes
7. if progress stays at 0%, reclassify to first-step trainer/runtime hang

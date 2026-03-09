# CUDA Eval Path Checklist

Trace:
1. reward function entry
2. backend dispatch call
3. selected backend mode
4. called evaluator symbol
5. compile/load path
6. correctness runner
7. benchmark runner
8. scalar reward return

Goal:
prove whether the system reaches CUDA compile/run at all

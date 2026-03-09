Create the smallest valid smoke test for this CUDA RL / GRPO pipeline.

Required output:
1. Minimal command to run
2. Minimal config changes
3. Which expensive paths must be disabled
4. What to log
5. Expected success signature
6. Expected failure signatures and what each means

Force this shape unless repo constraints make it impossible:
- 1 prompt
- 1 training step
- num_generations=2
- max_completion_length=128 or 256
- no remote eval
- cheap local reward
- no benchmark
- no large batch
- no long timeout path unless necessary

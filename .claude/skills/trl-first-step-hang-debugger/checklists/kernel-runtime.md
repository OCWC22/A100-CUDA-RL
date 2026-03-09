# Kernel / Runtime Checklist

Check for:
1. Linux kernel version warnings
2. accelerate/distributed warnings
3. NCCL warnings
4. hangs only in trainer, not raw generate
5. provider/runtime-specific behavior
6. whether a smaller 1-step run completes elsewhere

Bias:
treat an old host kernel warning as real until disproven

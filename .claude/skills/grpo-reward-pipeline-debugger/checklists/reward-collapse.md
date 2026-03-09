# Reward Collapse Checklist

Use when:
- reward_std=0
- loss=0
- all rewards identical
- all rewards are fallback values

Check:
1. exception-to-reward mapping
2. eval dispatch failure frequency
3. code extraction success rate
4. compile pass rate
5. correctness pass rate
6. reward histogram by batch
7. confirm at least one trainer step actually completed

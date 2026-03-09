# Modal Eval Failure Checklist

Use when logs include:
- App not found
- Function lookup failed
- environment mismatch
- deployment not found

Check:
1. app name in code
2. function name in code
3. backend selected at runtime
4. whether target app is deployed
5. whether local execution is possible instead
6. whether training image already contains evaluator code
7. whether environment defaults to main

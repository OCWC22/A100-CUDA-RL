#!/usr/bin/env bash
set -euo pipefail

TRANSCRIPT_INPUT="$(cat || true)"

if echo "$TRANSCRIPT_INPUT" | grep -Eqi "App .* not found|Lookup failed for Function|reward_std=0|loss=0|eval_ok=0|reward=-1"; then
  echo "Detected likely RL/eval pipeline failure: dispatch, deployment, or reward collapse. Use the grpo-reward-pipeline-debugger skill before making broad code changes."
fi

#!/usr/bin/env bash
set -euo pipefail

TRANSCRIPT_INPUT="$(cat || true)"

if echo "$TRANSCRIPT_INPUT" | grep -Eqi "Detected kernel version .* below the recommended minimum|0%\|.*0/[0-9]+|Starting Stage 1 training|canary.*PASS|PASS .* tokens generated"; then
  echo "Detected likely first-step trainer hang. If model load and canary passed but progress stays at 0%, investigate trainer rollout, runtime kernel version, grouped generation cost, and remote reward/eval blocking."
fi

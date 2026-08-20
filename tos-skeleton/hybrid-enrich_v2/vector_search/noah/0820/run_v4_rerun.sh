#!/usr/bin/env bash
# v4 교정 gold 기준 재실행 — s_all + s3_refmerge, train60, sonnet 1 rep.
set -u
cd "$(dirname "$0")"
export PYTHONIOENCODING=utf-8 PYTHONUTF8=1
export EMBED_ENDPOINT=http://localhost:11434/v1 EMBED_MODEL="dragonkue/bge-m3-ko:latest"
CLAUDE="$APPDATA/npm/node_modules/@anthropic-ai/claude-code-win32-x64/claude.exe"
for arm in s_all s3_refmerge; do
  echo "=== $arm $(date +%H:%M:%S)"
  python agent_runner.py --run "v4_${arm}60" --arm "$arm" --gold out/gold_v4_train60.jsonl \
    --reps 1 --workers 4 --model sonnet --pybin python --claude-bin "$CLAUDE"
done
echo "ALLDONE $(date +%H:%M:%S)"

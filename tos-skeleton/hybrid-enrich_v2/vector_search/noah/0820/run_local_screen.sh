#!/usr/bin/env bash
# 로컬(Windows git-bash) 스크리닝 — train60, claude -p sonnet, arm 순차 실행.
# 사용: bash run_local_screen.sh [arm ...]   (인자 없으면 스크리닝 6종)
set -u
cd "$(dirname "$0")"
export EMBED_ENDPOINT=http://localhost:11434/v1 EMBED_MODEL=dragonkue/bge-m3-ko:latest PYTHONUTF8=1
CLAUDE="$APPDATA/npm/node_modules/@anthropic-ai/claude-code-win32-x64/claude.exe"
ARMS=("$@")
[ ${#ARMS[@]} -eq 0 ] && ARMS=(s1_pad s2_fbsilent s3_refmerge s_all p_active p_active_hard)
for arm in "${ARMS[@]}"; do
  echo "=== $arm $(date +%H:%M:%S) ==="
  python agent_runner.py --run "local_${arm}60" --arm "$arm" --gold out/gold_spans_train60.jsonl \
    --reps 1 --workers 4 --model sonnet --pybin python --claude-bin "$CLAUDE" 2>&1 | tail -3
done
echo "=== summary ==="
python - <<'PY'
import json, glob
rows=[]
for p in sorted(glob.glob("out/agent/local_*60/summary.json")):
    s=json.load(open(p)); m=s["metrics"]
    rows.append((s["run"], m["R@5"], m["R@10"], m["suff@10"], s["no_submit"], s["errors"], s["cost_usd"]))
base=next((r for r in rows if "base" in r[0]), None)
print(f"{'run':22} {'R@5':>6} {'ΔR@5':>7} {'R@10':>6} {'suff@10':>8} {'nosub':>5} {'err':>4} {'cost':>7}")
for r in rows:
    d=(r[1]-base[1]) if base else 0.0
    print(f"{r[0]:22} {r[1]:6.3f} {d:+7.3f} {r[2]:6.3f} {r[3]:8.3f} {r[4]:5d} {r[5]:4d} ${r[6]:6.2f}")
PY

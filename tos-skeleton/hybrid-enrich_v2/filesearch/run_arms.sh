#!/usr/bin/env bash
# 사용: ./run_arms.sh <n> <reps> <arm...>   예) ./run_arms.sh 60 2 A_old_and A_new_clm
# 공통 설정은 환경변수로: BASE_URL(기본 http://localhost:8080/v1) MODEL(기본 Qwen/Qwen3-32B-FP8) WORKERS(6)
set -e
N=$1; REPS=$2; shift 2
BASE_URL=${BASE_URL:-http://localhost:8080/v1}; MODEL=${MODEL:-Qwen/Qwen3-32B-FP8}; WORKERS=${WORKERS:-6}
for ARM in "$@"; do
  ARMJSON=$(python3 -c "import json;print(json.dumps(json.load(open('arms.json'))['$ARM'],ensure_ascii=False))")
  python3 agent_runner_oai.py --run "n${N}_${ARM}" --arm "$ARMJSON" --n "$N" --reps "$REPS" --workers "$WORKERS" \
    --base-url "$BASE_URL" --model "$MODEL" --protocol text 2>&1 | tail -3
done
python3 - <<'PY'
import json,glob,os
rows=[]
for f in sorted(glob.glob('out/agent/n*/summary.json')):
    d=json.load(open(f)); m=d['metrics']
    rows.append((d['run'],d['n_q'],d['reps'],m['R@1'],m['R@5'],m['R@10'],m.get('suff@10'),m['RR@10'],d['core_R@5'],d['noncore_R@5'],d['no_submit'],d['errors'],d['rep_disagreement_R@5']))
print('| run | n | reps | R@1 | R@5 | R@10 | suff@10 | MRR@10 | core | 비core | 미제출 | 오류 | reps불일치 |')
print('|---'*13+'|')
for r in rows: print('| '+' | '.join(str(x) for x in r)+' |')
PY

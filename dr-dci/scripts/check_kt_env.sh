#!/usr/bin/env bash
# KT Cloud 실행 전 환경 점검 — GPU 여유 · 로컬 모델 · 서비스 · 데이터.
# 아무것도 바꾸지 않는다(읽기 전용). 실험 전에 항상 먼저 돌린다.
#
#   cd ~/source/embed_exp/experiments/dr-dci && bash scripts/check_kt_env.sh
#
# 종료 코드: 0 = Part 1/3/4 전부 실행 가능
#            1 = Part 4 만 가능 (증강 LLM :8100 없음)
#            2 = 실행 불가 (임베딩 또는 데이터 없음)

set -uo pipefail
CONDA_ROOT=/home/work/source/miniconda3     # /home/work/miniconda3 아님
SHIM_PID=993785                             # :8101 임베딩 shim — 절대 kill 금지
BLOCKERS=0
WARNINGS=0

hr() { printf '%.0s─' {1..72}; echo; }
ok()   { echo "  [OK]   $*"; }
warn() { echo "  [WARN] $*"; WARNINGS=$((WARNINGS+1)); }
bad()  { echo "  [FAIL] $*"; BLOCKERS=$((BLOCKERS+1)); }

# ---------------------------------------------------------------- 1. GPU
hr; echo "1. GPU 여유"; hr
if ! command -v nvidia-smi >/dev/null 2>&1; then
  bad "nvidia-smi 없음 — GPU 노드가 아니다"
else
  nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
             --format=csv,noheader,nounits |
  while IFS=, read -r idx name used total util; do
    used=$(echo "$used" | tr -d ' '); total=$(echo "$total" | tr -d ' ')
    free=$((total - used))
    printf "  GPU%s %s  사용 %s/%s MiB (util %s%%)  여유 %s MiB" \
           "$idx" "$(echo "$name" | tr -d ' ')" "$used" "$total" "$(echo "$util" | tr -d ' ')" "$free"
    # Qwen3-8B bf16 ≈ 16GB + KV 캐시. 24GB 이상 비어야 안전하다.
    if   [ "$free" -ge 24000 ]; then echo "  → Qwen3-8B 기동 가능"
    elif [ "$free" -ge 8000  ]; then echo "  → 리랭커는 가능, Qwen3-8B 는 빡빡"
    else echo "  → 여유 없음"; fi
  done
  echo
  echo "  점유 프로세스 (남의 작업을 죽이지 않는다):"
  nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null |
    while IFS=, read -r pid mem; do
      pid=$(echo "$pid" | tr -d ' ')
      cmd=$(ps -o cmd= -p "$pid" 2>/dev/null | cut -c1-60)
      tag=""; [ "$pid" = "$SHIM_PID" ] && tag="  <<< 임베딩 shim, KILL 금지"
      echo "    PID $pid  $(echo "$mem" | tr -d ' ')  ${cmd:-?}$tag"
    done
fi

# ------------------------------------------------------- 2. 로컬 모델 가중치
hr; echo "2. 로컬 모델 가중치 (HF 캐시 — 없으면 첫 기동 때 다운로드한다)"; hr
HF_HOME_DIR="${HF_HOME:-$HOME/.cache/huggingface}"
echo "  캐시 위치: $HF_HOME_DIR/hub"
# 모델명에 콜론이 없어도 역할 문자열에는 있으므로(":8100") 구분자는 '|' 를 쓴다
for repo in "Qwen/Qwen3-8B|증강 생성 (:8100)" \
            "Alibaba-NLP/gte-Qwen2-1.5B-instruct|임베딩 (:8101)" \
            "BAAI/bge-reranker-v2-m3|리랭커 (:8002)"; do
  model="${repo%%|*}"; role="${repo##*|}"
  dir="$HF_HOME_DIR/hub/models--${model//\//--}"
  if [ -d "$dir" ]; then
    size=$(du -sh "$dir" 2>/dev/null | cut -f1)
    nweights=$(find "$dir" -name '*.safetensors' -o -name '*.bin' 2>/dev/null | wc -l)
    if [ "$nweights" -gt 0 ]; then ok "$model  ($size, 가중치 $nweights개)  — $role"
    else warn "$model  디렉터리는 있으나 가중치 파일이 없다 (다운로드 중단?)  — $role"; fi
  else
    warn "$model  캐시 없음 → 기동 시 다운로드 필요  — $role"
  fi
done

# ---------------------------------------------------- 3. conda 환경 / 실행파일
hr; echo "3. conda 환경"; hr
for p in "$CONDA_ROOT/envs/embed/bin/python|embed (임베딩 shim 전용)" \
         "$CONDA_ROOT/envs/infopt-vllm/bin/vllm|vLLM (리랭커·Qwen3-8B)"; do
  path="${p%%|*}"; role="${p##*|}"
  if [ -x "$path" ]; then ok "$role  $path"
  else bad "$role 없음: $path"; fi
done
echo "  주의: :8101 shim 은 base 환경에서 안 뜬다"
echo "        (Qwen2Config has no attribute rope_theta) → embed 환경 전용"

# ------------------------------------------------------------- 4. 서비스
hr; echo "4. 서비스 엔드포인트"; hr
probe() {  # port  라벨  필수여부(required|optional)
  local port="$1" label="$2" req="$3"
  local body; body=$(curl -s --max-time 5 "http://localhost:$port/v1/models" 2>/dev/null)
  if [ -n "$body" ] && echo "$body" | grep -q '"id"'; then
    local mid; mid=$(echo "$body" | grep -o '"id"[[:space:]]*:[[:space:]]*"[^"]*"' | head -1 | cut -d'"' -f4)
    ok ":$port $label  → $mid"
  else
    if [ "$req" = required ]; then bad ":$port $label  응답 없음"
    else warn ":$port $label  응답 없음"; fi
  fi
}
probe 8101 "임베딩 gte-Qwen2-1.5B-instruct" required
probe 8002 "리랭커 bge-reranker-v2-m3"      required   # 없으면 Part4 hybrid 전멸
probe 8100 "증강 LLM Qwen3-8B"              optional   # Part1/3 에만 필요
if [ -f .env ] && grep -q '^OPENAI_API_KEY=.\{20,\}' .env; then
  ok ".env OPENAI_API_KEY 존재 (에이전트·judge = gpt-4o-mini)"
else
  bad ".env 의 OPENAI_API_KEY 없음 — dr-dci/ 바로 아래에 있어야 한다"
fi

# ------------------------------------------------------------- 5. 데이터
hr; echo "5. shinhan-uw 데이터"; hr
# python3 가 없는 환경(Windows Git Bash 등)도 있으므로 실제 인터프리터를 찾는다
PY_BIN=""
for cand in python3 python; do
  if command -v "$cand" >/dev/null 2>&1 && "$cand" -c 'import sys' >/dev/null 2>&1; then
    PY_BIN="$cand"; break
  fi
done
if [ -z "$PY_BIN" ]; then
  bad "python 인터프리터를 찾을 수 없다"
else
"$PY_BIN" - <<'PY'
import json, os, sys
for _s in (sys.stdout, sys.stderr):
    try:                      # Windows 콘솔(cp949)에서도 한글·기호가 깨지지 않게
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass
base = "data"
need = {
    f"{base}/raw/shinhan-uw/corpus.jsonl":      3365,
    f"{base}/raw/shinhan-uw/queries.jsonl":       50,
    f"{base}/raw/shinhan-uw/qrels.jsonl":         50,
    f"{base}/raw/shinhan-uw/qa_meta.jsonl":       50,
}
bad = 0
for path, exp in need.items():
    if not os.path.exists(path):
        print(f"  [FAIL] 없음: {path}"); bad += 1; continue
    n = sum(1 for _ in open(path, encoding="utf-8"))
    mark = "OK  " if n == exp else "WARN"
    print(f"  [{mark}] {path}  {n}행 (기대 {exp})")
for path, exp in ((f"{base}/reference_answers/shinhan-uw.json", 50),
                  (f"{base}/metadata/shinhan-uw-parser_4k.json", 3365),
                  (f"{base}/tags/shinhan-uw/approach_p/4k.json", 3365),
                  (f"{base}/subsets/shinhan-uw/4k.json", None)):
    if not os.path.exists(path):
        print(f"  [FAIL] 없음: {path}"); bad += 1; continue
    d = json.load(open(path, encoding="utf-8"))
    n = len(d["doc_ids"]) if isinstance(d, dict) and "doc_ids" in d else len(d)
    print(f"  [{'OK  ' if exp in (None, n) else 'WARN'}] {path}  {n}건"
          + (f" (기대 {exp})" if exp else ""))
print("\n  LLM 증강 (:8100 으로 생성 — Part 1/3 에만 필요):")
for path in (f"{base}/taxonomy/shinhan-uw_4k.json",
             f"{base}/prefix/shinhan-uw_4k.json",
             f"{base}/metadata/shinhan-uw_4k.json",
             f"{base}/tags/shinhan-uw/approach_a/4k.json",
             f"{base}/tags/shinhan-uw/approach_b/4k.json",
             f"{base}/tags/shinhan-uw/approach_c/4k.json"):
    print(f"    {'있음' if os.path.exists(path) else '없음'}  {path}")
sys.exit(1 if bad else 0)
PY
  [ $? -ne 0 ] && BLOCKERS=$((BLOCKERS+1))
fi

# ------------------------------------------------------------- 판정
hr; echo "판정"; hr
AUG_READY=1
for f in data/taxonomy/shinhan-uw_4k.json data/prefix/shinhan-uw_4k.json \
         data/metadata/shinhan-uw_4k.json data/tags/shinhan-uw/approach_a/4k.json; do
  [ -f "$f" ] || AUG_READY=0
done
if [ "$BLOCKERS" -gt 0 ]; then
  echo "  실행 불가 — FAIL $BLOCKERS건을 먼저 해결한다"; exit 2
elif [ "$AUG_READY" -eq 1 ]; then
  echo "  Part 1 / 3 / 4 전부 실행 가능. preflight 를 돌린다:"
  echo "    PYTHONPATH=. python scripts/audit_part12.py --config config/experiment_shinhan_uw.yaml \\"
  echo "      --step baseline --step taxonomy_only --step tags_only --step prefix_only \\"
  echo "      --step metadata_only --step parser_meta_only --step stack_tax_tags \\"
  echo "      --step stack_tax_tags_prefix --step stack_all"
  exit 0
else
  echo "  Part 4 는 지금 실행 가능하다 (augment: false — 증강 불요):"
  echo "    nohup python run_experiment.py --part 4 \\"
  echo "      --config config/experiment_shinhan_uw.yaml > logs/uw_p4.log 2>&1 &"
  echo
  echo "  Part 1 / 3 은 :8100 Qwen3-8B 기동 후 증강 생성이 필요하다:"
  echo "    CUDA_VISIBLE_DEVICES=<여유GPU> nohup \\"
  echo "      $CONDA_ROOT/envs/infopt-vllm/bin/vllm serve Qwen/Qwen3-8B \\"
  echo "      --port 8100 --gpu-memory-utilization 0.35 --max-model-len 8192 \\"
  echo "      > logs/qwen3_8100.log 2>&1 &"
  exit 1
fi

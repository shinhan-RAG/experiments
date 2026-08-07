#!/usr/bin/env bash
# gte-Qwen2 임베딩(자가 호스팅 vLLM)으로 pull backend probe 2종을 실행한다.
# H200 인스턴스 내부 실행 전용 — config 기본 endpoint(localhost:8101)를
# 오버라이드 없이 그대로 사용하므로 API 키가 필요 없다.
#
# 사용:
#   bash scripts/run_probe_gte.sh
# 산출:
#   dr-dci/probe_gte_results_<timestamp>.tar.gz  (이 파일만 회신하면 된다)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/5] 임베딩 endpoint 확인 (localhost:8101, gte-Qwen2)"
curl -sf -m 5 http://localhost:8101/v1/models | grep -q "gte-Qwen2" || {
  echo "오류: localhost:8101에서 gte-Qwen2 모델을 확인하지 못했다."
  echo "vLLM 임베딩 서버를 먼저 기동할 것:"
  echo "  vllm serve Alibaba-NLP/gte-Qwen2-1.5B-instruct --task embed --port 8101"
  exit 1
}

echo "[2/5] 데이터·의존성 확인"
for d in trec-covid fiqa; do
  test -f "data/raw/$d/corpus.jsonl" || { echo "오류: data/raw/$d 부재"; exit 1; }
done
python3 -c "import numpy, requests, yaml" 2>/dev/null || pip install -r requirements.txt
test -f data/subsets/trec-covid/20k.json || PYTHONPATH=. python3 scripts/build_subsets.py

echo "[3/5] 오프라인 계약 테스트 (14건 통과 필요)"
PYTHONPATH=. python3 -m unittest discover -s tests

echo "[4/5] probe 실행 — TREC-COVID 20K, 이어서 FiQA 전체 코퍼스"
python3 run_experiment.py --part 5 --probe-only
python3 run_experiment.py --part 5 --probe-only --dataset fiqa --subset 0

echo "[5/5] 결과 수집"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="probe_gte_results_${STAMP}.tar.gz"
tar czf "$OUT" results/part5_pull_backend/
echo ""
echo "완료 — 회신 파일: dr-dci/$OUT"
echo "(결과 JSON의 manifest.embedding_endpoint에 gte-Qwen2가 기록됐는지로 검증 가능)"

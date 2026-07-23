#!/usr/bin/env bash
# Part 2 선행 distractor scale probe를 gte-Qwen2 임베딩(자가 호스팅)으로 실행한다.
# H200 인스턴스 내부 실행 전용 — localhost:8101 임베딩 endpoint만 필요하며
# agent LLM/judge 호출은 없다. 20K+50K+110K = 18만 문서 임베딩이 필요하므로
# 수십 분~1시간대 소요를 예상할 것.
#
# 사용:
#   bash scripts/run_scale_probe_gte.sh
# 산출:
#   dr-dci/scale_probe_results_<timestamp>.tar.gz  (이 파일만 회신하면 된다)
set -euo pipefail
cd "$(dirname "$0")/.."

echo "[1/5] 임베딩 endpoint 확인 (localhost:8101, gte-Qwen2)"
curl -sf -m 5 http://localhost:8101/v1/models | grep -q "gte-Qwen2" || {
  echo "오류: localhost:8101에서 gte-Qwen2 모델을 확인하지 못했다."
  echo "임베딩 서버를 먼저 기동할 것."
  exit 1
}

echo "[2/5] 데이터·의존성 확인"
test -f data/raw/trec-covid/corpus.jsonl || { echo "오류: data/raw/trec-covid 부재"; exit 1; }
python3 -c "import numpy, requests, yaml, dotenv" 2>/dev/null || pip install -r requirements.txt
for k in 20k 50k 110k; do
  test -f "data/subsets/trec-covid/$k.json" || { echo "오류: subset $k 부재 — scripts/build_subsets.py 필요"; exit 1; }
done

echo "[3/5] 오프라인 계약 테스트 + 사전검사"
PYTHONPATH=. python3 -m unittest discover -s tests
PYTHONPATH=. python3 scripts/audit_part12.py --step baseline > /dev/null || {
  echo "오류: Part 1/2 사전검사 실패"; exit 1
}

echo "[4/5] scale probe 실행 — dense pull @ 20K/50K/110K (paired 50 질의)"
python3 run_experiment.py --part 2 --scale-probe

echo "[5/5] 결과 수집"
STAMP=$(date +%Y%m%d_%H%M%S)
OUT="scale_probe_results_${STAMP}.tar.gz"
LATEST_RESULT=$(ls -t results/part2_scale_probe/*.json | head -n 1)
python3 scripts/validate_scale_probe_result.py "$LATEST_RESULT"
tar czf "$OUT" results/part2_scale_probe/
echo ""
echo "완료 — 회신 파일: dr-dci/$OUT"

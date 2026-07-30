#!/usr/bin/env bash
# 신한라이프 데이터 파트 순차 실행. 한 파트가 실패해도 다음 파트를 계속 진행하고
# 마지막에 파트별 종료코드를 요약한다.
#
#   ./run_shinhan_all.sh          # 기본: Part 1, 3 (증강 산출물 필요)
#   ./run_shinhan_all.sh 4        # Part 4만 (증강 불필요 — 8101/8002만 있으면 된다)
#   ./run_shinhan_all.sh 4 1 3
#
# Part 2는 실행하지 않는다. 신한 코퍼스는 2,802청크로 규모 확장 실험의 최소 조건
# (20K청크)을 충족하지 못한다. 규모 결과는 aihub-full 실험을 인용할 것.
cd "$(dirname "$0")" || exit 1
# 임베딩 백엔드를 바꿔 돌릴 때 config를 환경변수로 덮어쓴다.
#   CFG=config/experiment_shinhan_b128.yaml ./run_shinhan_all.sh 1 3 4
CFG="${CFG:-config/experiment_shinhan.yaml}"
TS=$(date +%Y%m%d_%H%M%S)
PARTS=("$@")
[ ${#PARTS[@]} -eq 0 ] && PARTS=(1 3)
mkdir -p logs

declare -A RC
for P in "${PARTS[@]}"; do
  if [ "$P" = "2" ]; then
    echo "===== Part 2 는 건너뛴다 (코퍼스 2,802청크 < 최소 20K) ====="
    continue
  fi
  echo "===== Part $P 시작: $(date '+%F %T') ====="
  python3 run_experiment.py --part "$P" --config "$CFG" 2>&1 \
    | tee "logs/shinhan_part${P}_${TS}.log"
  RC[$P]=${PIPESTATUS[0]}
  echo "===== Part $P 종료: rc=${RC[$P]} / $(date '+%F %T') ====="
done

echo "===== 요약 ====="
for P in "${!RC[@]}"; do echo "part$P: rc=${RC[$P]}"; done
ls -lt results/*/ 2>/dev/null | head -20

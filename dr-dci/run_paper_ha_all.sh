#!/usr/bin/env bash
# paper-ha(AI Hub 학술논문 HA) 파트 순차 실행. 한 파트가 실패해도 다음 파트를 계속 진행하고
# 마지막에 파트별 종료코드를 요약한다.
#
#   ./run_paper_ha_all.sh          # 기본: Part 1, 3 (증강 산출물 필요)
#   ./run_paper_ha_all.sh 4        # Part 4만 (증강 불필요 — 8101/8002만 있으면 된다)
#   ./run_paper_ha_all.sh 4 1 3
#
# Part 2는 실행하지 않는다. paper-ha 코퍼스는 4,253청크로 규모 확장 실험의 최소 조건
# (20K청크)을 충족하지 못한다. 규모 결과는 aihub-full 실험을 인용할 것.
#
# 사전 준비(로컬에서 커밋된 것): corpus/queries/qrels/qa_meta, subsets/5k,
#   tags/approach_p, reference_answers/paper-ha.json
# KT 클라우드에서 빌드할 것(Qwen3-8B :8100 필요):
#   python scripts/build_taxonomy.py paper-ha --size=5000
#   python scripts/build_prefix.py   paper-ha --size=5000
#   python scripts/build_metadata.py paper-ha --size=5000
#   python scripts/build_tags.py     paper-ha --size=5000            # A/B/C (Part 3용)
cd "$(dirname "$0")" || exit 1
CFG="${CFG:-config/experiment_paper_ha.yaml}"
TS=$(date +%Y%m%d_%H%M%S)
PARTS=("$@")
[ ${#PARTS[@]} -eq 0 ] && PARTS=(1 3)
mkdir -p logs

declare -A RC
for P in "${PARTS[@]}"; do
  if [ "$P" = "2" ]; then
    echo "===== Part 2 는 건너뛴다 (코퍼스 4,253청크 < 최소 20K) ====="
    continue
  fi
  echo "===== Part $P 시작: $(date '+%F %T') ====="
  python3 run_experiment.py --part "$P" --config "$CFG" 2>&1 \
    | tee "logs/paper_ha_part${P}_${TS}.log"
  RC[$P]=${PIPESTATUS[0]}
  echo "===== Part $P 종료: rc=${RC[$P]} / $(date '+%F %T') ====="
done

echo "===== 요약 ====="
for P in "${!RC[@]}"; do echo "part$P: rc=${RC[$P]}"; done
ls -lt results/*/ 2>/dev/null | head -20

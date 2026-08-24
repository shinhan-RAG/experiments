# noah/0824/v3 — R2 검색 에이전트 4가지 개선 실험

arm `r2_verify_dense_quota_hybrid`, claude-sonnet-5/medium, gold v21 train213×2rep.
r1v 대비 4가지 변경: verify_gate 강제, dense 폴백 재검색 허용, 프롬프트 강화, quota 재배분.

## 결과 요약

| 지표 | r1v (기존) | r2 (이번) | 차이 |
|---|---:|---:|---:|
| R@1 | .5986 | .6467 | +.048 |
| R@5 | **.8588** | **.8545** | **-.004** |
| R@10 | .8916 | .8869 | -.005 |
| suff@5 | .8357 | .8310 | -.005 |
| suff@10 | .8779 | .8638 | -.014 |
| model_calls 평균 | 1.16 | 2.27 | +1.11 |
| 비용 | $32.25 | $64.08 | +$31.83 |

R@5 미채택 — 비용 2배에 정확도 하락. 상세 분석은 `docs/0824/r2_실험결과_20260824.md` 참조.

## 파일

- `results.jsonl` — 426셀 전체 결과
- `summary.json` — arm 집계
- `manifest.json` — 실행 조건·SHA256·런타임
- `gold_v21_train213.jsonl`, `qids_v21_train213.json` — gold/qids (v2와 동일본)

## 실험 조건

- arm 정의: `noah/0819/arms.json` → `r2_verify_dense_quota_hybrid`
- quota 재배분 코드: `noah/0819/agent_tools.py` (unique_jo_quota 직전 11행)
- 실행 커맨드: `host_agent_runner_local.py --run r2_verify_dense_quota_213x2_goldv21_20260824 --arms r2_verify_dense_quota_hybrid --workers 5 --seed 20260824 --resume`

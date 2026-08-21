# 0821v3-sonnet5 C29 dual-tool Claude Sonnet 5 실험 보고서

실행일: 2026-08-21 ~ 2026-08-22
대상: clean train 257문항 (jo_ids 미매핑 24문항 제거)
결론: **Claude Sonnet 5는 동일 검색 인프라에서 GPT-5.6 Luna 대비 R@5 +13.1pp, R@10 +17.9pp를 기록했으며, 모든 주요 지표(R@5, R@10, suff@5, suff@10)에서 paired bootstrap 95% CI가 0을 포함하지 않아 통계적으로 유의한 차이다.**

## 1. 실험 설정

| 항목 | 설정 |
|---|---|
| 모델 | Claude Sonnet 5 (`claude-sonnet-5`) via claude CLI |
| worker | 4 (resume 전 2) |
| 반복 | 1 rep |
| QA/Gold | `gold_train_scoped_u3_reviewed_overlay_v1_clean257.jsonl`, 257문항 |
| semantic 도구 | C29 CLM + structured BM25F fact + membership evidence quota |
| metadata 도구 | V9 BM25 + BGE-m3-ko dense, RRF(k=60) |
| 도구 규약 | 모든 문항에서 `search`와 `msearch`를 각각 최소 1회 사용 후 submit |
| 채점 단위 | U3 JO, Gold evidence-group fractional Recall 및 sufficiency |
| 검색 arm | `c29_dual_tool_hybrid` (0821v3와 동일) |

Gold 전처리: 원본 281문항에서 gold group의 `jo_ids`가 비어있는 24문항을 제거하여 257문항으로 정리했다. 제거된 문항은 lsh_gold eid와 gold group char range 간 정렬이 깨져 있었으나, char overlap 기반 채점 자체에는 영향이 없는 문항이다. Luna 비교 시에는 동일 257 QID로 paired 비교했다.

고정 데이터 SHA256(LF 정규화):

- Gold: `b9f2e5a61f1c049a1270e34e288d2d81a913971ed86a7738141a0f303d59248a`
- QID: `bfd95d5afe7750d1bb7afa339f8b68c6085ba184b06d98253dded99ab6b4a07a`

## 2. 최종 결과

| n | R@1 | R@5 | R@10 | R@20 | suff@5 | suff@10 | RR@10 |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 257 | .3567 | **.7069** | **.7626** | .7626 | **.6770** | **.7354** | .5224 |

R@5 분포: 만점 174문항, 부분점수 16문항, 0점 67문항. 빈 제출 없음, 평균 제출 수 7.0개.

## 3. Luna 대비 paired 비교 (동일 257 QID)

| 지표 | Luna | Sonnet 5 | Δ | paired bootstrap 95% CI | W/L/T |
|---|---:|---:|---:|---:|---:|
| R@1 | .3677 | .3567 | -.0110 | [-.0532, +.0311] | 18/19/220 |
| R@5 | .5759 | .7069 | **+.1310** | **[+.0733, +.1894]** | 56/17/184 |
| R@10 | .5837 | .7626 | **+.1790** | **[+.1245, +.2341]** | 63/11/183 |
| suff@5 | .5525 | .6770 | **+.1245** | **[+.0700, +.1829]** | 47/15/195 |
| suff@10 | .5603 | .7354 | **+.1751** | **[+.1206, +.2335]** | 54/9/194 |

R@1을 제외한 모든 지표에서 Sonnet 5가 통계적으로 유의하게 우수하다. R@5 기준 승/패 비율은 56:17(3.3:1)이다.

## 4. 문항 난이도별 결과

| task type | n | Sonnet R@5 | Luna R@5 | Δ |
|---|---:|---:|---:|---:|
| single_lookup | 156 | .840 | .686 | +.154 |
| multi_evidence | 94 | .502 | .404 | +.098 |
| comparison | 4 | .375 | .250 | +.125 |
| exhaustive_list | 2 | .500 | .500 | .000 |
| document_global | 1 | 1.000 | 1.000 | .000 |

single_lookup에서 +15.4pp, multi_evidence에서 +9.8pp 향상. Luna의 보수적 제출(2.95개)이 multi_evidence에서 특히 약했던 반면, Sonnet 5는 평균 7.0개로 적극적 제출하면서도 정밀도를 유지했다.

## 5. 모델 행동 비교

| 항목 | Luna | Sonnet 5 |
|---|---:|---:|
| 평균 제출 수 | 2.95 | 7.0 |
| search 호출 | 328 | 504 |
| msearch 호출 | 299 | 346 |
| read 호출 | 1,299 | 896 |
| submit 호출 | 281 | 257 |
| 평균 소요 | 56.4s | 125.9s |
| 도구 위반 | 0 | 0 |
| fatal error | 0 | 0 |

Sonnet 5는 search를 더 많이 호출(504 vs 328)하면서 read는 적게(896 vs 1,299) 사용했다. 검색 후보를 더 넓게 탐색하되, read 검증 단계에서 효율적으로 판단한 패턴이다. 제출 수가 2.4배 많아 top-10에서의 커버리지가 크게 개선됐다.

## 6. 실행 무결성

- 최종 결과: 257 rows / 257 unique QID
- fatal error: 0
- 필수 도구 위반: 0
- empty submit: 0
- 실행 방식: claude CLI (`claude -p --model claude-sonnet-5`), stdin 파일 기반 프롬프트
- 시스템 프롬프트: 영문 미니멀 에이전트 지시 (한국어 검색 지시는 user prompt에 포함)
- embedding server: 로컬 BGE-m3-ko, 4 worker 공유
- resume: 2-worker로 20문항 실행 후 4-worker로 전환하여 237문항 resume

## 7. 판정과 다음 실험

이번 결과로 확인된 것:

- Claude Sonnet 5는 동일 C29 dual-tool 인프라에서 Luna 대비 R@5/R@10/suff 모두 유의하게 높다.
- 주된 차이는 제출 적극성(7.0 vs 2.95개)과 검색 전략(search 더 많이, read 더 적게).
- R@1은 유의한 차이가 없어, 첫 번째 제출의 정확도는 비슷하지만 top-5/10 내 커버리지에서 Sonnet 5가 우수하다.
- single_lookup R@5=.840은 이 검색 인프라의 단일근거 회수 능력의 현재 상한에 가깝다.

다음 실험 우선순위:

1. 동일 257 QID에서 host-controlled C29(search+msearch 강제 결합) + Sonnet 5 조합 비교.
2. multi_evidence R@5=.502 문항의 검색 노출 분석 — 후보 불충분 vs 선택 실패 분리.
3. 252개 completeness 미검수 Gold에 대한 감사 및 정답셋 v5 구축.
4. comparison/exhaustive_list 소수 문항에 대한 별도 검색 전략 실험.

## 8. 산출물

- `out/c29_dual_tool_clean257_sonnet5_w2/manifest.json`
- `out/c29_dual_tool_clean257_sonnet5_w2/results.jsonl`
- `out/c29_dual_tool_clean257_sonnet5_w2/summary.json`
- `out/c29_dual_tool_clean257_sonnet5_w2/sessions/`

# s30 에이전트 실측 (2026-08-20, sonnet)

- 모집단: 난이도 층화 30문항(core/비core × gold group 수 × 질문-정답 어휘겹침 비례, 시드 20260820, `filesearch/out/qids_s30.json`) × reps 2
- 하네스: filesearch agent_tools(S5 예산), 모델 claude sonnet, ITT, strict gold(`gold_spans_lsh_train.jsonl`, lsh 공식 기반, n=329 중 표본)
- 채점: 비율(fractional group) R@K · S@K · suff@10 · MRR@10, 조 단위

| arm | R@1 | R@5 | R@10 | suff@10 | core R@5 | 비core R@5 | reps불일치 |
|---|---|---|---|---|---|---|---|
| old_and (구 검색기×구 스키마) | .300 | .354 | .388 | .383 | .444 | .219 | .20 |
| new_and (AND×신 스키마) | .383 | .547 | .572 | .533 | .694 | .326 | .20 |
| old_clm (CLM×구 스키마) | .400 | .604 | .658 | .600 | .694 | .469 | .17 |
| new_clm (CLM×신 스키마) | .456 | .635 | .701 | .667 | .750 | .462 | .13 |
| new_clm_qsonnet (+qtags sonnet) | .417 | **.668** | .710 | .667 | .778 | .503 | .10 |
| new_clm_facet (+facet) | .444 | .656 | **.726** | .667 | .778 | .472 | .03 |
| new_clm_llmtags (LLM 태그) | .417 | .600 | .650 | .617 | .694 | .458 | .10 |

paired(R@5, 문항 reps 평균, 부호검정+BCa):
- new_clm vs old_and: **+.281, 14승 2패, p=.004, CI[+.14,+.44] 유의**
- new_clm vs old_clm +.031 (p=.73) · vs new_and +.088 (p=.51) — 방향 양성, n=30 미판정
- qsonnet +.033 / facet +.021 / llmtags −.035 vs new_clm — 전부 미판정(방향만)

해석: 2×2 총합(구구→신신) 개선은 유의. 축 분해와 변형 3종은 n=30 검출력 아래 — 전수(329)에서 판정 필요.
qsonnet·facet 은 전 지표에서 방향 양성 + reps 안정(불일치 .03~.10) → 전수 후보.
비용: arm 당 $12~27(총 $109). 유보: 표본 30, 모델 sonnet 단일, gold strict, C 레벨(문서 선택 제외).

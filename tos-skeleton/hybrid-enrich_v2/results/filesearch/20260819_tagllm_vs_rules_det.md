# 태그 생성: 규칙 vs LLM 추출 — 결정론 비교 (2026-08-19)

조건: gold=out/gold_spans_lsh_train.jsonl(lsh 공식, n=329) · 검색기 CLM+qtags(haiku)+contract w2 · 조 단위 채점 · LLM 0회(태그 생성 제외)
LLM 태그: 조 단위 haiku 1회(3,147조), subject 원문 verbatim(검증 기각 자동), role 폐쇄 11코드. subject 커버리지 .365→.625

| 태그 | R@5 | R@10 | R@40 | suff@10 | paired(부호검정, vs 규칙) |
|---|---|---|---|---|---|
| 규칙(tags_u2_rules) | .503 | .603 | .746 | .568 | — |
| LLM(tags_u2_llm) | .471 | .613 | .754 | .581 | R@5 −3.1pp (4승15패, p=.019 유의 열세) · R@10 +1.0pp (p=.65) · suff@10 +1.2pp (p=.42) |

판정: R@5(주지표)에서 LLM 태그가 유의하게 나쁨 → **규칙 태그 채택**. 원인 추정: subject 증가가 CLM 합집합 후보를 넓혀 top-5 동점·희석(R@10 이상에서는 중립~미미한 이득).
설계서 2.1 관점: LLM 개입은 verbatim 검증으로 허용 가능하나, 성능 근거가 없어 채택하지 않음. 파일은 재검용으로 보존(tags_u2_llm, tagllm_cache_haiku).

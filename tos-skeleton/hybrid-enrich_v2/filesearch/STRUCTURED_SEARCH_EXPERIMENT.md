# 규칙×규칙 구조화 검색 arm

## 범위

- 별도 LLM, embedding, reranker를 사용하지 않는 단일단계 BM25F 후보 검색이다.
- 기존 CLM과 메타데이터 검색·폴백은 보존하며 `ranker=bm25f` arm에서만 발동한다.
- 현재 암 특약 문서는 성능 실험용 한 문서 유형일 뿐, 검색기의 내부 스키마 계약은
  `identity/topic/function/locator/table/constraint/relation/structure/extra`로 고정한다.
- 태거의 `contract_key`, `subject_key` 같은 문서별 이름은 `schema_adapter.py`가 공통 축으로
  바꾼다. 새 필드는 별칭 등록 전에도 `extra` 저가중 채널에 남아 유실되지 않는다.
- 신규 적재의 정규형 계약은 `semantic_tag_generic.schema.json`(JSON Schema Draft 2020-12)이다.

## 11만 문서 적용 원칙

1. 공통 key는 고정하되 값 추출기는 문서 유형별 플러그인으로 둔다. 예: 약관의 조문형
   locator와 사업방법서의 `N. ~에 관한 사항` locator.
2. 전역 hard filter를 두지 않는다. 불완전한 identity/locator는 soft score만 준다.
3. `audit_generic_schema.py`를 문서 유형별·전체 corpus에 실행한다. ID 불일치, 미지 필드
   5% 초과, p99 태그 길이 512자 초과는 배포 전 조사한다.
4. 범용성은 커버리지로만 승인하지 않는다. 문서유형×시대×상품군 층화 QA에서 검색
   성능과 worst-slice를 별도로 확인한다.
5. 현재 Python sparse index cache는 32,366요소에서 약 78MB다. 11만 운영판은 동일
   점수식을 지원하는 persistent inverted index로 옮기고, 이 구현은 정확성 oracle과
   실험 harness로 사용한다.

## 재현

```bash
cd tos-skeleton/hybrid-enrich_v2/filesearch
python3 -m unittest -v test_structured_search.py
python3 audit_generic_schema.py
python3 eval_det.py --qtags '' --qmode rule \
  --arms 'clm:lex=count:w=contract=2;clm:ranker=bm25f:profile=core'

cd ../vector_search/noah/0819
./run_structured_rule.sh

# 기존 폴백 수정 rule×rule과 qid별 rep 평균 paired 비교
python ../../../filesearch/compare_agent_runs.py \
  --a out/agent/s30v3_structured_core/results.jsonl \
  --b out/agent/s30v2_fb_rule_rules/results.jsonl --key R@5
python ../../../filesearch/compare_agent_runs.py \
  --a out/agent/s30v3_structured_full/results.jsonl \
  --b out/agent/s30v2_fb_rule_rules/results.jsonl --key suff@10
```

Claude 에이전트 확인 arm은 `arms.json`의 `c5_structured_core`(R@5 우선)와
`c5_structured_full`(R@10·충분성 우선)이며, 기존 s30 qid와 strict gold, sonnet,
2 reps를 그대로 사용한다.

## 2026-08-20 결정론적 전수 결과

규칙 라우터, strict fractional gold, n=329. u2jo 조 단위 기준이다.

| arm | R@5 | R@10 | suff@10 | R@100 | R@400 | 후보 0 |
|---|---:|---:|---:|---:|---:|---:|
| 기존 CLM | .424 | .529 | .511 | .737 | 미측정 | 17 |
| BM25F core | .487 | .557 | .529 | .742 | .796 | 0 |
| BM25F full | .484 | .569 | .541 | .768 | .804 | 0 |

- core의 기존 대비 R@5 차이 +.0626, BCa 95% CI `[+.0172,+.1114]`, sign test
  53승/29패, p=.0106.
- full의 기존 대비 R@5 차이 +.0593, BCa 95% CI `[+.0152,+.1033]`, sign test
  46승/24패, p=.0115.
- core와 full 차이는 유의하지 않다. core가 R@5 +.0033, full이 R@10 +.0124 및
  suff@10 +.0122이며 각 CI는 0을 포함한다.
- R@400 상한이 .80 수준이므로 이 단일 호출만으로 .95 달성을 주장할 수 없다.
  에이전트의 규칙적 재질의 후보 노출과 타 문서유형 QA가 다음 확정 게이트다.

## 근거

- Robertson & Zaragoza, *The Probabilistic Relevance Framework: BM25 and Beyond* (2009),
  §3.6 Multiple Streams and BM25F: https://www.staff.city.ac.uk/~sb317/papers/foundations_bm25_review.pdf
- Elasticsearch 공식 `combined_fields`: 여러 필드를 term 중심으로 결합하고 BM25F 기반
  점수 및 필드 boost를 제공한다. https://www.elastic.co/docs/reference/query-languages/query-dsl/query-dsl-combined-fields-query
- Lucene 10.3.1 `BM25Similarity`: k1=1.2, b=.75 기본값과 IDF 공식을 명시한다.
  https://lucene.apache.org/core/10_3_1/core/org/apache/lucene/search/similarities/BM25Similarity.html
- JSON Schema Draft 2020-12 공식 명세: https://json-schema.org/draft/2020-12

## Claude Sonnet 에이전트 실측 (2026-08-20 밤, s30 × 2 reps, strict gold)

기준선 = s30v2_fb_rule_rules (규칙 라우터 CLM + 폴백수정, R@5 .618). 동일 s30 qid·facet/meta/fallback(min_n=5) 고정, 변인은 태그 검색기(CLM ↔ BM25F)만.

| run | R@1 | R@5 | R@10 | suff@5 | suff@10 | MRR@10 | core | 비core | 미제출/오류 | 불일치 | search/read 평균 | 세션중앙(s) | $ | 폴백발화 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| c5_structured_core | .375 | **.646** | .713 | .600 | .667 | .516 | .750 | .490 | 1/0 | .167 | 2.78/1.08 | 21.9 | 13.0 | 0 |
| c5_structured_full | .442 | .625 | .692 | .600 | .667 | .540 | .750 | .438 | 0/0 | .133 | 2.25/1.03 | 20.8 | 10.2 | 0 |
| (기준선) fb_rule_rules | .406 | .618 | .710 | .600 | .683 | .512 | .750 | .420 | 0/0 | .167 | 2.23/0.78 | 17.5 | 9.5 | 4 |

paired (문항 rep 평균, 부호검정+BCa):
- core vs 기준선: R@5 Δ+.028 (8/5, p=.581, CI[−.108,+.151]) · R@10 Δ+.003 · suff@10 Δ−.017 — 전부 CI 0 포함, **유의 아님**
- full vs 기준선: R@5 Δ+.007 · R@10 Δ−.018 · suff@10 Δ−.017 — 동일
- core vs full: R@5 Δ+.021 (5/3, CI[−.067,+.125]) — 차이 없음

전환 분석(core vs 기준선, calls.jsonl 실측):
- 개선 5문항: 0459(0→1, 같은 gold 노출인데 BM25F 순위가 상위 배치), 0379(판본 문항 0→.5 — 결정론에서 이득 본 판본 가산 확인), 0147(0→.5 — gold 미노출인데도 부분회수: gold 잡음 의심 문항), 0314(다근거 1/2 노출 상황에서 회수), 0412(.5→1)
- 악화 4문항: 0018(1/1 노출·검색 4회에도 미제출 — 선택 실패), 0126(1→0), 0009·0381(각 −.5) — BM25F 상위가 어휘 유사 이웃으로 채워져 에이전트 선택이 흔들린 사례 포함(실측: 노출은 동일, 제출만 상이)
- generic axis 인자(--identity 등) 사용: **0/59 세션** — 에이전트는 새 축 인자를 쓰지 않았고, 이득은 전적으로 1페이지 순위 개선에서 옴. 폴백 발화 0(후보0이 없어짐).

판정: **조건부 채택.** 결정론 전수(n=329)에서는 R@5 +.063 [+.017,+.111] 유의였고, 에이전트 s30에서도 방향 양성(+.028)·regression 없음·후보0 제거·비core +.07 개선이나, s30 검정력으로는 유의 주장 불가. R@5 우선이므로 core 를 대표 arm 으로 하고 full 은 보류(개선 근거 없음). 전수(329) 에이전트 확정 실험으로 판정할 것 — 예상 비용 core 1 arm ≈ $110~130(60세션 $13 기준 658세션), 시간 ~3h.

다음 실험 1순위: **규칙적 다중 질의 순서**(1차 검색 후 facet 상위 특약별 재질의를 규칙로 강제 — multi-evidence 부분회수 6/12가 최대 실패층이고 generic axis 를 에이전트가 안 쓰므로 도구가 순서를 만들어야 함). 차순위: 후보 미노출 문항용 deterministic fallback(문서유형 value extractor 확장), gold 잡음 재검수(0147 유형).

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

## C6 규칙 후속질의 — identity 완화 (2026-08-20)

에이전트가 generic axis를 자발적으로 사용하지 않은 결과를 반영해 검색 도구가 후속질의를
자동 생성한다. 별도 LLM·reranker·학습 정렬은 없다. 각 BM25F 결과의 내부 순위는 그대로
두고, 첫 페이지 40칸 중 기존 core 상위 20칸을 보존한 뒤 나머지 20칸에 후속질의 후보를
교차 노출한다.

사전 후보: focus / identity relax / identity별 / role별 / facet / all. 동일 s30 결정론
스크리닝에서 identity relax만 R@40을 개선했고, 나머지는 이득 없음 또는 희석으로 제외했다.

identity relax는 규칙 라우터가 지정한 contract/identity만 제거하고 topic·function·constraint와
질문 핵심 토큰은 유지한다. 초기 특약 오지정·동명 파생 특약 과다 열거를 회복하기 위한
soft backtracking이며 기존 상위 20개를 삭제하지 않는다.

전수 n=329, 규칙 라우터, strict fractional gold, 조 단위:

| arm | R@5 | R@10 | R@20 | R@40 | R@100 | R@400 |
|---|---:|---:|---:|---:|---:|---:|
| C5 core | .487 | .557 | .612 | .683 | .742 | .796 |
| C6 core + identity relax | .487 | .558 | **.650** | **.705** | **.763** | **.814** |

- R@5 완전 동일(상위 seed 보존).
- R@20 Δ+.0377, BCa 95% CI `[+.0187,+.0633]`, 15승/1패, p=.00052.
- R@40 Δ+.0218, BCa 95% CI `[+.0051,+.0441]`, 11승/3패.
- R@100 Δ+.0213, BCa 95% CI `[+.0091,+.0405]`, 8승/0패.
- 주요 회복층은 라우터가 유사 특약을 과다 열거한 신수술특약 정의, 진단특약 판본,
  재활·항암치료 포함 여부, 코드/분류표 질의다.

Claude 확인 arm: `c6_identity_relax`. 동일 s30×2 reps에서 C5 core와 paired 비교한다.

```bash
cd tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819
./run_identity_relax.sh
python ../../../filesearch/compare_agent_runs.py \
  --a out/agent/s30v4_identity_relax/results.jsonl \
  --b out/agent/s30v3_structured_core/results.jsonl --key R@5
```

### 에이전트 게이트와 GPT 교차확인

무조건 identity relax는 Claude Sonnet s30×2에서 C5 core 대비 R@5 −.0167,
R@10 −.0500, suff@10 −.0500으로 희석되어 폐기했다. 이후 원 질문의 규칙 라우터가
identity 후보를 3개 이상 낸 경우에만 relax를 허용하는 `relax_gated`를 만들었다.
에이전트가 만든 후속 검색어가 아니라 세션의 원 질문(`question.json`)으로 게이트를
판정해 질의 드리프트가 게이트를 과발화하지 않도록 했다.

동일 s30×2, strict gold, 규칙×규칙, facet/meta/fallback(min_n=5) 고정 결과:

| model | arm | R@1 | R@5 | R@10 | suff@10 | core | 비core | 미제출/오류 |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| GPT-5.6 Terra | C5 core | .3611 | .4819 | .4819 | .4667 | .6111 | .2882 | 0/1 |
| GPT-5.6 Terra | gated relax | .3556 | .4722 | .5222 | .5000 | .5556 | .3472 | 1/0 |
| GPT-5.5 | C5 core | .3889 | .6056 | **.7389** | **.6667** | .7778 | .3472 | 0/0 |
| GPT-5.5 | gated relax | .3472 | **.6194** | .6778 | .6167 | .8056 | .3403 | 0/0 |

GPT-5.5 paired(문항별 2 reps 평균, challenger−C5):

- R@5 Δ+.0139, 3승/2패, sign p=1.0, BCa 95% CI `[−.0833,+.1167]`.
- R@10 Δ−.0611, 1승/3패, p=.625, CI `[−.2056,+.0056]`.
- suff@10 Δ−.0500, 1승/2패, p=1.0, CI `[−.2167,+.0167]`.
- 두 arm 모두 오류·미제출·명령 정책 위반 0. 평균 제출 수는 C5 6.80,
  gated 6.53개다. gated는 16/60 세션(8개 qid), 전체 221회 검색 중 32회 발화했다.

판정: **현재 gated relax는 에이전트 R@5 상승안으로 기각한다.** 결정론 후보 노출
R@20 이득은 존재하지만 에이전트 제출까지 안정적으로 이어지지 않고, R@10·충분성은
역방향이다. `.95` 목표의 다음 동력은 relax 후보 추가가 아니라 실패층별 규칙 질의
순서와 다중근거 제출 규약을 분리해 검증해야 한다.

모델 운영 원칙: 이번 GPT-5.5 교차확인까지만 예외로 남기고 이후 Codex 에이전트 실험은
`gpt-5.6-terra`로 고정한다. `agent_runner.py --provider codex`에서 `--model`을 생략하면
Terra를 기본 선택한다. 모델을 바꾼 실험 결과는 같은 표 안에서 직접 우열로 해석하지 않는다.

## C7 Evidence Slot과 C8 Axis Coordination (2026-08-20)

### C7: 다중근거 역할 슬롯 — 기각

Gold 그룹은 기존 채점기 그대로 비율 채점한다. Gold 4개 중 Top-K가 3개 그룹을 덮으면
R@K=.75이고, suff@K는 4개를 모두 덮어야 1이다. a/a′ 동치는 같은 Gold 그룹의 OR
멤버이므로 하나만 맞아도 해당 그룹을 충족한다.

C7은 원 질문에서 payment_trigger·limit_frequency·criteria_rule 같은 근거 역할을 규칙으로
추론해 역할별 BM25F 하위질의를 만들었다. 별도 LLM·embedding·reranker는 없다. 그러나
다중 Gold 65문항 중 42문항을 발화하면서 단일 Gold도 190문항 발화해 정밀도가 18.1%에
그쳤다. seed quota 3/5는 전체 R@5를 악화했고, 안전한 seed 20은 다음 결과였다.

| arm (n=329, jo) | R@5 | R@10 | R@20 | R@40 | suff@10 |
|---|---:|---:|---:|---:|---:|
| C5 core | .487 | .557 | .612 | .683 | .529 |
| C7 evidence, seed 20 | .487 | .560 | .615 | .689 | .532 |

다중근거 65문항에서는 R@5 .2333→.2333, suff@10 .1538→.1538로 이득이 없었다.
참조 조·역참조 조·인접 조 확장도 오프라인 검증에서 R@5와 suff@10 이득 0이라 제외했다.
Gold 그룹 수를 질문 표면형만으로 예측하는 방식은 채택하지 않는다.

### C8: 범용 축 coordination — 결정론 채택, 에이전트 조건부

BM25F 1차 점수 안에서 identity/topic/function/constraint/structure 중 동시에 일치한 축 수에
유한 가산점을 준다(`_axis=1`, `_coord=1`). 별도 2차 정렬은 없고, 상품명·암 도메인
규칙도 없다. Lucene BooleanQuery의 선택 절 최소 일치와 같은 coordination 원리다.

전수 n=329, 규칙 라우터, strict fractional gold:

| arm | R@1 | R@5 | R@10 | R@20 | suff@10 | R@400 |
|---|---:|---:|---:|---:|---:|---:|
| C5 core | .218 | .4868 | .5567 | .612 | .5289 | .796 |
| C8 axis coordination | .233 | **.5020** | **.5780** | **.630** | **.5502** | .799 |

- R@5 Δ+.0152, 6승/1패, BCa 95% CI `[+.0030,+.0319]`.
- R@10 Δ+.0213, 7승/0패, sign p=.0156, CI `[+.0061,+.0365]`.
- suff@10 Δ+.0213, 7승/0패, p=.0156, CI `[+.0061,+.0365]`.
- 이득은 단일 Gold 264문항에서 발생했다. 다중 Gold 65문항의 R@5·R@10·suff@10은
  변화가 없어 다중근거 병목 해결책으로 해석하지 않는다.

동일 고정 s30×2, GPT-5.6 Terra, C5 기준선 재사용 paired:

| arm | R@1 | R@5 | R@10 | suff@10 | core | 비core | 미제출/오류 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C5 core | .3611 | .4819 | .4819 | .4667 | .6111 | .2882 | 0/1 |
| C8 axis coordination | .3278 | .4903 | .5069 | .4833 | .5833 | .3507 | 3/0 |

paired 결과는 R@5 Δ+.0083, 3승/2패, CI `[−.0667,+.0667]`; R@10 Δ+.0250,
CI `[−.0500,+.1000]`; suff@10 Δ+.0167, CI `[−.0667,+.0833]`으로 모두 유의하지 않다.
같은 고정 s30의 결정론 R@5는 .4000→.4667(+.0667)이므로 검색 순위 이득이 Terra의
보수적 제출(평균 2.93개, 미제출 3세션)에서 대부분 소실됐다. C8은 결정론 검색기에는
채택하되 에이전트 최종 Recall 개선은 조건부로 남긴다.

주의: 최초 C8 Terra 실행은 신규 층화 s30을 사용해 기존 고정 s30과 공통 qid가 1개뿐이라
paired 판정에서 제외했다. 위 표는 `out/qids_s30.json`으로 재실행한 결과만 사용한다.

## C9 규칙 질의 순서 + 고정 하이브리드 실측 (2026-08-20)

C9은 규칙 라우터의 identity→topic 결과 뒤에 명시 function/constraint와 structure 질의를
순서대로 생성한다. 암 도메인 사전, 별도 LLM, reranker는 사용하지 않는다. 기존 상위 후보를
보호하기 위해 seed quota 30을 고정했고, 첫 `search`에는 BM25F 태그 30건과 V9 메타 10건을
결합했다. 최초 v2~v4 실행은 메타 sidecar 실패 또는 미발화가 확인되어 전부 판정에서 제외하고,
아래는 sidecar 결합을 호출 로그로 검증한 v5만 사용한다.

고정 s30, 1 rep, GPT-5.6 Terra, strict fractional gold, 규칙×규칙 결과:

| arm | R@1 | R@5 | R@10 | suff@10 | core | 비core | 미제출/오류 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C9 하이브리드 기준선 | .4167 | **.6444** | **.6444** | **.6000** | .7222 | .5278 | 2/1 |
| C9 sequence 하이브리드 | **.4333** | .5528 | .5528 | .5333 | .6667 | .3819 | 1/1 |

paired(challenger−baseline):

- R@1 Δ+.0167, 3승/3패, bootstrap 95% CI `[−.1333,+.1667]`, sign p=1.0.
- R@5·R@10 Δ−.0917, 2승/6패, CI `[−.2500,+.0667]`, sign p=.2891.
- suff@10 Δ−.0667, 2승/4패, CI `[−.2333,+.1000]`, sign p=.6875.
- 양쪽 오류 문항을 제외한 공통 성공 28문항에서도 R@5 Δ−.0982로 역전되지 않았다.
- 하이브리드 sidecar 오류는 양쪽 0건이다. sequence는 30/30 세션에서 `merge:10`을 확인했다.
  기준선은 28/30에서 `merge:10`, 나머지 2세션은 에이전트가 직접 `msearch`를 선택했다.
- 평균 실행시간은 기준선 46.7초, sequence 48.5초였다.

판정: **C9 sequence는 에이전트 Recall 상승안으로 기각한다.** seed quota 30 조건에서 고정 s30의
결정론 R@1~R@40 지표가 기준선과 같아 신규 정답 노출 이득이 없고, 에이전트 단계에서 제출 선택
변동만 늘었다. 특히 비core R@5가 −.1458로 악화했다. 1 rep라 유의성 확정 표본은 아니지만,
효과 방향과 결정론 게이트가 모두 채택 조건을 충족하지 않는다.

다음 실험 전 변인 통제 게이트:

1. 에이전트 프롬프트에서 직접 `msearch`를 제거하고 단일 `search` API만 허용한다.
2. `search` 내부에서 태그+고정 메타 결합을 강제해 두 arm의 채널 선택을 동일하게 만든다.
3. 새 semantic-tag 전략은 먼저 결정론 s30에서 R@5 또는 R@10 정답 노출을 실제 개선해야 한다.
4. 결정론 게이트 통과안만 s30×2 agent paired 실험으로 올린다.

### 직접 msearch 제거 통제 재실측

두 arm 모두 `search_only=true`로 고정해 프롬프트·명령 정책에서 직접 `msearch`를 제거하고,
첫 `search` 내부에서 태그 30건+V9 메타 10건을 강제 결합했다. 고정 s30, 1 rep,
GPT-5.6 Terra, strict fractional gold 결과:

| arm | R@1 | R@5 | R@10 | suff@10 | core | 비core | 미제출/오류 |
|---|---:|---:|---:|---:|---:|---:|---:|
| search-only 기준선 | .4167 | .4694 | .4694 | .4333 | .5556 | .3403 | 1/0 |
| search-only C9 sequence | **.4500** | **.5361** | **.5361** | **.5000** | .6111 | .4236 | 0/0 |

paired 결과는 R@5·R@10·suff@10 모두 Δ+.0667, 3승/1패, bootstrap 95% CI
`[−.0667,+.2000]`, sign p=.625로 유의하지 않다. 채널 감사는 양쪽 모두 30/30
`merge:10`, 직접 `msearch` 0건, 폴백 오류 0건, 명령 정책 위반 0건이다.

후속 stderr 전수 감사로 위 표의 `오류 0` 집계가 불완전했음이 확인됐다. 기준선은 20/30
세션에서 unified-exec 생성 실패 21회(검색 20, submit 1), C9은 24/30 세션에서 26회
(검색 24, read 2) 발생했다. 대부분 재시도로 회복했지만 기준선 `0381` submit은 미복구였다.
따라서 이 v6 비교는 clean agent ITT 효과가 아니며 아래 +.0667을 개선 확정값으로 사용하지 않는다.

관찰 상승을 검색기 이득으로 해석할 수 없는 이유:

- +.0667 중 +.0333은 기준선 `v3-offline-0381`의 submit 도구 생성 실패다.
- 나머지 개선 문항 `0321`의 e31945와 `0412`의 e12149는 기준선 검색에도 이미 노출됐다.
  즉 C9이 새 정답 후보를 노출해서 회복한 것이 아니라 에이전트 제출 선택 차이다.
- 악화 문항 `0037`의 기준선 scoring hit는 `e24147`의 조 `j05777`이다. C9 검색에도 같은
  gold 조의 `e24146/e24148`이 노출됐으므로 미노출이 아니라 하위 노출·오선택이다.
- 미제출 문항을 제외한 ΔR@5는 약 +.0345이고, 이전 유효 v5의 Δ−.0917과 방향이
  뒤집혀 Terra 1 rep 분산이 효과보다 크다.

최종 판정: **search-only 실행 규약은 이후 표준 하네스로 채택하고, C9 sequence 자체는 계속
기각한다.** 다음 semantic-tag 후보는 에이전트 실행 전에 결정론 R@5/R@10 정답 노출을
개선해야 하며, 노출 개선 0인 후보를 반복 에이전트 실험으로 구제하지 않는다.

### 후보 표시 병목 디버깅: C10 기각, C11 challenger

v6 첫 응답 40건은 평균 20.5개의 고유 조로 resolve됐다. 그러나 같은 조의 첫 element만
남긴 C10을 도구 없는 frozen selector s30×2로 비교하자 raw R@5 `.6667`에서 `.5667`로
`−.1000`(qid-cluster CI `[−.2250,0]`) 하락했다. R@1도 `−.1458`, CI
`[−.2792,−.0333]`이었다. 같은 조의 뒤 element/chunk가 질문에 더 직접적인 snippet을
담고 있었기 때문에 조 단위 결정론 Recall 불변이 agent 선택 안전성을 보장하지 않았다.
C10은 기각한다.

후속 C11 top-4 탐색판은 조별 복수 snippet을 보존해 R@5 `.6750`, R@10 `.7250`,
suff@10 `.7000`을 보였으나, 과거 raw를 다른 시간대에서 재사용한 적응적 실험이므로 상승
주장을 하지 않는다. cap=0 정리 전 판도 s30×1 R@5 `.6500`으로 비열등성을 입증하지
못했다. 독립 리뷰 3인은 공통으로 `C10 폐기 / C11은 정보보존형 구조화 challenger / 같은
실행의 randomized raw 대조 전 채택 금지`를 판정했다. 현재 코드는 raw rank·ID·score·preview를
한 번씩 보존하도록 정리했으며, 다음 clean frozen s30×2와 이후 full-agent s30×2가 남아 있다.

후속 최종 결과: clean frozen s30×2는 R@5 `.6542→.6833`(+.0292), suff@10
`.6667→.7333`(+.0667)로 승격 게이트를 통과했다. 그러나 실제 tag30+V9 meta10 검색 뒤
Terra가 read/재검색/submit을 선택하는 host-controlled live agent s30×2에서는 R@5
`.6583→.6292`(−.0292, qid-cluster CI `[−.1292,+.0625]`)로 악화했고, R@10은
`.7500→.7583`, suff@10은 `.7167→.7167`이었다. core R@5 `−.0833`, noncore
`+.0521`이며 `0037` hard regression이 발생했다. 검색 순서 rank-lock 재채점도 차이를
해소하지 못했다. 따라서 **C11은 production 채택 기각**이며, 표시 최적화가 `.95` Recall의
직접 상승 동력이 아니라는 결론을 확정한다.

## C12 context backtrack — 노출은 개선, agent R@5 승격 보류 (2026-08-21)

Host 실측의 첫 hybrid search는 tag BM25F 30+V9 meta 10이며 R@40 `.8583`이었다. 30개
qid 중 25개만 Gold를 완전히 노출했고, `0013·0166·0292`는 0, `0106·0509`는 부분
노출이었다. 모든 후속 search union도 baseline `.8750`이라 표시/선택만으로 `.95`에 도달할
수 없다.

`c12_context_backtrack`은 상품·질병 사전 없이 identity 후보 과다, 주계약-특약 관계,
상품/보험 문맥 표현, 복수 주제 열거에서만 identity-slot-relaxed BM25F 하위질의를 연다. 기존 seed
20과 내부 BM25F 순위는 보존하며 별도 LLM·embedding·reranker는 없다.

고정 s30 실제 첫 hybrid search, strict fractional gold, 오류 0:

| arm | R@1 | R@5 | R@10 | R@20 | R@40 |
|---|---:|---:|---:|---:|---:|
| C9 baseline | .1667 | .3833 | .5333 | .8417 | .8583 |
| C12 context backtrack | .1667 | .3833 | .5333 | .8667 | .8917 |

상위 10은 불변이고 `0292`만 R@40에서 신규 회복했다. R@20의 순변화 +.025는 `0292`
+1과 `0106` −.25의 합이다. raw seed 20 보존은 중복 조를 먼저 제거한 Top-20 정의까지
불변으로 만들지 않는다. 해당 qid Terra×2 스모크에서 C12는
새 Gold를 두 번 모두 제출 6위에 두어 R@10은 `1.0`이지만 R@5는 `0`이었다. baseline은
R@5/R@10 각 `.5`였다. strict Gold인 주계약 면책기간 표보다 질문을 직접 비교하는 특약
근거를 앞세운 선택으로, R@5 승격 조건을 만족하지 않았다. 30×2 agent 확대는 중단하고 이
arm은 결정론 장거리 recall challenger로만 유지한다.

평가 감사상 현재 strict 생성 규칙(gap30, exact-key OR, occurrence 미확장, required 구분 없음)은
정답기준 명세 v1 초안(gap200, required/supporting, 내용 동일 OR, suff@10 주지표) 및 회의의
유사·파생 a/a′ 인정과 완전히 같지 않다. 특히 `0509`는 명시 특약의 직접 근거를 회수해도 다른
특약 span 두 개가 별도 분모라 최대 `.5`에 머무는 구조다. 검색기 실험과 Gold 변경은 분리하되,
`.95` 판단 전에 불완전 노출 5문항을 사람 승인 Gold 버전으로 감사해야 한다.

Terra high, Sol medium, Sol xhigh의 2차 독립 원자료 리뷰도 C12 확대 중단과 Gold 분리 감사를
공통 권고했다. `0106`의 Gold 조는 raw 첫 20이 같아도 ordered-unique-jo 18→25위로 밀렸다.
다음 C12.1의 사전 게이트는 baseline unique-jo Top-20 완전 보존, `0106` R@20 복구,
`0292` R@40 유지, s30 R@1/5/10/20 hard loss 0이다. 사람 승인 Gold가 나오기 전 기존
strict 수치는 수정하지 않는다.

## C13~C23: 점수 가산/RRF의 한계와 Gold v3 (2026-08-21)

C13 axis/coord 가산, C17 identity 가산, C19 evidence-role 가산은 모두 상품명·질병명
hard-code나 후단 LLM reranker 없이 Semantic Tag 점수만 바꿨다. 그러나 full-agent paired
실험에서는 각각 R@5가 하락하거나, 0과 구분되지 않거나, 실제 feature가 발화하지 않은
문항의 모델 선택 차이가 상승분을 만들었다. 따라서 세 arm은 default 채택하지 않았다.

C22/C23은 여러 규칙 tag query를 RRF로 합치되 기존 상위 후보를 보호하는 portfolio였다.
C23의 Luna-medium, Gold-v3, s29×2 clean agent 결과는 다음과 같다.

| arm | R@1 | R@5 | R@10 | suff@5 | suff@10 |
|---|---:|---:|---:|---:|---:|
| C16 기준선 | .4052 | .7672 | .8190 | .7586 | .8103 |
| C23 safe axes | .4224 | .7845 | .8190 | .7759 | .8103 |

R@5 Δ+.0172의 qid-cluster 95% CI는 `[−.0517,+.0862]`이며 3승/2패/53무였다.
직접 감사 결과 29/29 질문의 첫 20 element와 첫-search Gold 노출이 양팔에서 같았고,
초기 구현은 `portfolio_prompt:false`를 실제로 적용하지 않아 phase/RRF 정보가 모델에
보였다. 해당 표시 누출은 이후 수정했지만, 이 실행은 검색 노출 향상이 아니라 tail 문맥에
따른 agent 선택 변동이므로 C23을 채택하지 않았다.

평가셋은 회의의 부분점수/파생 판본 원칙을 구현하되 자동 유사도 확장은 폐기하고,
원문 span을 독립 감사한 Gold v3로 고정했다. v3는 29문항, 2개 이상 required group은
부분점수를 유지하며, QA/Gold는 검색 tag 생성 입력으로 사용하지 않는다. 남은 appendix
scope 오염과 직접 조항 누락 의심은 별도 ledger에서 계속 감사한다.

## C26: 범용 fact tag + 규칙 ranker quota (2026-08-21)

C26은 단순 score weight 조정이 아니라 다음 검색 구조를 추가했다.

1. 원문 문장·표 행에서 `evidence_anchor`, `answer_values`, `benefit_aliases`,
   `fact_roles`, `parent_jo`, 명시적 표 참조를 규칙으로 생성한다.
2. legacy CLM tag rank와 fact BM25F rank가 서로 다른 JO를 일정 quota로 제안한다.
3. 기존 기준선의 첫 5개 unique JO prefix는 원 순서 그대로 보존한다.
4. agent에는 rank-only score만 보이고 내부 phase/portfolio metadata는 숨긴다.
5. Meta V9, Vector fallback, rule router, agent prompt/model은 양팔에서 고정한다.

builder는 element/tags/JO만 읽으며 QA·Gold·agent 결과를 읽지 않는다. 최초 U4에서 마지막
appendix table label이 먼 후속 table까지 이어지는 결함을 감사로 발견했다. 해당 s10×3
실행은 탐색 자료로만 봉인했다. 수정본은 heading 직후의 연속 table run만 연결하며,
32,046행 중 appendix context가 4,565→239행으로 줄었다. 또한 11만 문서 확장을 위해
table label/reference graph key를 `(document_key, table_label)`로 고정해 문서 간 `표2-1`
충돌을 차단했다.

수정 U4, Luna-medium, Gold-v3, 실패 원인 개발 s10×2 agent smoke는 R@5
`.475→.800`, R@10 `.550→.900`, suff@10 `.500→.900`이었다. 이 표본은 adaptive debug
set이라 일반 성능 주장이 아니라 full 실행 승격 게이트로만 사용했다.

full s29×2에서 protocol 오염 3개 qid block은 양팔을 같이 재실행해 exact key로 교체했다.
clean agent 결과는 다음과 같다.

| arm | R@1 | R@5 | R@10 | suff@5 | suff@10 | fatal/protocol | empty |
|---|---:|---:|---:|---:|---:|---:|---:|
| C26 고정 기준선 | .3793 | .8362 | .8362 | .8103 | .8103 | 0/0 | 1 |
| C26 fact ensemble | .4655 | **.8448** | **.9052** | .8276 | **.8966** | 0/0 | 2 |

paired R@5 Δ+.0086, qid-cluster 95% CI `[−.1121,+.1293]`, 4승/3패/22무다.
R@10 Δ+.0690, CI `[−.0086,+.1638]`; suff@10 Δ+.0862, CI `[0,+.1897]`이다.
새 근거 노출로 `0059·0185·0502`를 회복했지만 `0447` R@5 hard regression과
`0369·0450` 선택 손실이 남았다. 따라서 `.8448`은 현재 clean agent 최고 point estimate이나
C26을 default 채택하지 않는다. 다음 C27은 C26의 fact 회수를 유지하며 기준선 첫 10개
unique JO를 보호해 이 선택 손실을 줄이는 단일변인 실험이다.

## C26 namespace 수정·Gold v7·C29 본실험 (2026-08-21)

추가 감사에서 최초 U4의 표 참조 key가 `부표1`과 `별첨2 [표1]`을 같은
`table_label=표1`로 합치는 결함을 발견했다. 수정 U4는 다음 불변식을 적용한다.

- source document는 표시명이 아닌 소스 path hash로 식별한다.
- reference key는 `(document, exact contract, namespace, table number)`다.
- `부표`, `별첨 [표]`, bare `표`를 서로 다른 namespace로 보존한다.
- 참조 source/target ID는 provenance로만 남기고 검색 evidence 가중치는 0으로 fail-closed한다.

수정 태그는 32,046 elements, 7,612 JO에서 생성했다. `reference_target=62`,
`reference_edge=83`, `linked_identity=0`이며 발견된 cross-namespace 엣지는 0이다. 단,
현 artifact는 source document 1개이므로 신한 11만 문서 규모의 일반화를 실증한 것은
아니다.

Gold는 질문 문구를 바꾸지 않고 원문 span을 retrieval-blind 방식으로 감사했다.
최초 30문항 중 contract identity가 모호한 `0488`을 제외해 29문항을 평가했다.
Gold v7은 29문항을 유지하며, `0155`의 소액암 지급 근거에 네 판본의
지급표 JO를 동일 정답 OR로 추가했다. 지급 그룹과 `D06` 분류표 그룹은
서로 독립 required이므로 그룹 간 AND/부분점수를 유지한다. 질문·Gold·agent
제출은 태그 builder 입력으로 쓰지 않았다.

C28은 긴 표의 희귀 query bigram을 행 단위로 검색했지만 full s29×2에서
C26 `.9224` 대비 `.9138`로 악화해 기각했다. C29는 이 검색을 일반 code 질문에
넓게 열지 않고, 표면형이 `범주 X에 구성원 Y가 포함/해당`이며 X가 문서의
direct identity surface에 존재할 때만 발화한다. 발화 시 CLM/fact quota를
17/8에서 15/7로 줄이고, rare-row 3 JO와 direct-identity evidence-unit 3 JO를 추가한다.
비발화 질문은 C26과 첫 hybrid result 40개의 ID·순서가 같다.

Luna-medium, rule router, Meta V9, forced hybrid, Gold v7, s29×2에서 protocol 오류
3개 qid block을 양 arm 함께 재실행해 exact key로 교체한 clean agent 결과는 다음과 같다.

| arm | R@1 | R@5 | R@10 | suff@5 | suff@10 | fatal/protocol | empty |
|---|---:|---:|---:|---:|---:|---:|---:|
| C26 | .5948 | .9138 | .9828 | .8793 | .9828 | 0/0 | 0 |
| C29 | .6121 | **.9224** | .9483 | **.9138** | .9483 | 0/0 | 2 |

R@5 Δ+.0086의 qid bootstrap 95% CI는 `[−.0603,+.0776]`으로 0을 포함한다.
suff@5는 +.0345, CI `[−.0517,+.1379]`이다. 직접 표적 `0155`는 R@5 `.5→1.0`,
suff@5 `0→1.0`으로 회복했다. C29 비발화 문항의 승패는 동일 검색 목록에서
Luna가 다른 submit을 선택한 변동이며 C29 인과 효과로 세지 않는다. C29은
`.90` PR/whole-train 승격 기준은 통과했지만, `.95` 재현이나 통계적 우월을 입증한
것은 아니다.

전체 train 실행용 dataset은 347문항 scoped Gold에 reviewed v7의 고정 30문항
scope를 overlay한다. v7에서 제외된 `0488` 하나와 원 scoped Gold에서
`groups=[]`인 미매핑 65문항을 unscorable로 격리해 실제 채점 가능한 전체는
281문항이다. 29문항은 retrieval-blind completeness review, 나머지 252문항은
scoped Gold이지만 같은 수준의 전수 사람 감사를 받지 않았다. 두 층과 65개 격리
qid 전부를 overlay manifest에 고정한다.

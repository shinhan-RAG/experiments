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

## C29 evaluable train 281문항 agent 실험 (2026-08-21)

zero-group 65문항과 `0488`을 제외한 281문항을 C29 단일 arm,
Luna-medium, 1 rep, workers 8, max-actions 4로 실행했다. 최초 run의
인프라/도구 오류 2 cell만 동일 조건의 재실행 cell 전체로 교체했다.
중간에 보이지 않은 ID를 요청했지만 호스트가 회복해 최종 submit을 받은
protocol mistake는 성공할 때까지 재실행하지 않고 agent ITT 행동으로
보존했다.

| 대상 | n | R@1 | R@5 | R@10 | suff@5 | suff@10 |
|---|---:|---:|---:|---:|---:|---:|
| 전체 evaluable train | 281 | .3867 | **.6314** | .6874 | .6050 | .6584 |
| completeness reviewed | 29 | .7414 | **.9828** | .9828 | .9655 | .9655 |
| scoped, completeness 미검토 | 252 | .3459 | **.5909** | .6534 | .5635 | .6230 |

전체 R@5의 qid bootstrap 95% CI는 `[.5765,.6859]`, suff@5는
`[.5480,.6619]`이다. 최종 artifact의 fatal error는 0, 회복된 protocol mistake는
14 cell/25 events, empty submit은 7 cell이다. R@5 분포는 만점 170,
부분점수 16, 0점 95문항이다.

난이도 차이가 크다. `single_lookup` 172문항의 R@5/suff@5는 `.7500/.7500`,
`multi_evidence` 102문항은 `.4551/.3922`였다. comparison 4문항 R@5 `.25`,
document-global 1문항 `.0`, exhaustive-list 2문항 `.5`다. 따라서 29문항
`.9224` 실험은 전체 train 난이도를 대표하지 않으며, 281문항
`.6314`도 252문항의 Gold completeness 미검토를 검색기 실패와 분리하지
못한 개발 지표다.

최초 검색 40에서 Gold가 하나도 노출되지 않은 qid가 56개였고,
모든 agent 재검색의 union으로도 Gold 충분성을 만족하지 못한 qid가
57개였다. 이는 현 C29 후보·Gold 정의로 agent R@5 `.95`를 달성할 수
없음을 뜻한다. 다음 단계는 검색 결과를 보지 않은 전체 Gold completeness
감사와, 별도 holdout에서 multi-evidence/cross-reference 회수를 늘리는
범용 그래프 검색 가설을 분리해 검증하는 것이다.

## 전체 실패층 후속: C39~C42 (2026-08-21)

C39는 C29의 기존 하이브리드 40개를 그대로 둔 채 질문 표면형이 확인된
JO fact-card만 뒤에 붙였다. 깨끗한 mechanism 실행에서 `0351`이 두 반복 모두
`1→.5`, `.5→0`으로 악화했고 다른 선행 개선이 없어 조기 중단했다. 별도 후보를
추가하는 것만으로도 agent 선택 문맥이 흔들리므로 채택하지 않는다. 최초
s10×2 시도는 sandbox가 내부 Codex app-server 생성을 막아 양 arm 모두 전 행
실패했으며 성능 근거에서 제외한다.

C40은 QA·Gold·결과를 읽지 않은 U5 명시참조 graph의 1~2 hop target을 기존
결과 카드 아래에 중첩했다. top-level 40개 ID와 순서는 바꾸지 않았고,
`분류코드/분류표/포함` 질문에서만 발화했다. Luna-medium, 7문항×2회 agent
비교 결과는 다음과 같다.

| arm | R@1 | R@5 | suff@5 | fatal/protocol |
|---|---:|---:|---:|---:|
| C29 | .2143 | .6786 | .5714 | 0/5 |
| C40 nested 2-hop | .3571 | .6786 | .5714 | 0/0 |

0029의 상위 조항 아래에 실제 Gold `j07559`가 연결되는 등 graph 동작은
확인됐지만, 한 질문에 최대 14개의 target이 보였고 R@5 순이득은 없었다.
따라서 C40도 채택하지 않는다.

C41/C42는 특정 활동명을 사전에 두지 않고 `X로 인한 사고·상해·장해·골절이
보상되는가`라는 표면형을 `재해분류표/보장대상 재해/우발적인 외래 사고/
지급하지 않는 재해`로 정규화한다. C41은 정규화 후보를 별도 evidence로만
보여 0150에서 3회 중 1회만 회복했다. C42는 같은 규칙 검색의 1위 후보를
Semantic Tag 결과 선두에 융합했고, 0150 개발 smoke 3회 모두 Gold `j07544`를
R@5 안에 제출했다.

| arm | n×rep | R@1 | R@5 | suff@5 | fatal/protocol |
|---|---:|---:|---:|---:|---:|
| C29 | 1×3 | .000 | .000 | .000 | 0/0 |
| C42 scenario intent | 1×3 | .333 | 1.000 | 1.000 | 0/0 |

그러나 evaluable 281문항 전수 발화는 0150 한 건뿐이다. 따라서 현재 dataset에서
전체 R@5의 이론상 최대 상승은 `1/281≈.0036`이고, 별도 holdout 일반화 근거도 없다.
C42는 범용 mechanism prototype으로만 보관하며 전체 agent 재실행이나 default
승격은 하지 않는다. 다음 주된 상승축은 더 넓은 multi-evidence 실패층을 겨냥한
규칙 기반 claim 분해와 역할별 다중 검색이어야 한다.

## C43~C44 및 retrieval-blind Gold 전수 감사 (2026-08-21~22)

C43은 질문에서 서로 독립적인 claim 역할을 규칙으로 분해하고 역할별 후보를 별도
evidence group으로 제공했다. 상품명·질병명 사전, Gold, 비에이전트 LLM, reranker는
사용하지 않았다. 인프라 오류가 난 0217 양팔을 제외한 6개 paired Luna-medium smoke에서
C29→C43 agent R@5는 `.7500→.6667`, suff@5는 `.5000→.5000`이었다. 5개 tie와
0351 한 건의 손실이며, 0351에서는 질문에 특약 scope가 없어 역할별 지급 후보가 다른
특약으로 퍼졌다. broad role bundle은 채택하지 않는다.

C44는 C40의 반복·과다 참조 후보를 다음과 같이 줄였다.

- 상위 source JO에서 도달한 1~2 hop target을 전역 JO 단위로 중복 제거한다.
- target 원문이 질문의 구체적인 긴 표면형을 직접 포함할 때만 남긴다.
- 인용부호로 특정한 표/주제는 target에도 동일 표면형이 있어야 한다.
- `어떤 치료가 포함되는가`처럼 로컬 목록 질문은 appendix projection을 열지 않는다.
- 기존 hybrid 결과 40개와 Vector/Meta 채널은 변경하지 않는다.

초기 3문항×2회 Luna smoke에서 C29→C44 agent R@5는 `.4167→.6667`, suff@5는
`.3333→.5000`, 오류는 양팔 0이었다. 그러나 trace상 기능성 이득은 0029의 명시 참조
target `j07559`를 제출한 한 반복뿐이었다. 강화 후 현재 Gold v5 269문항에서 신규 target이
생기는 질문은 4개뿐이므로 전체 격차를 닫는 주동력이 될 수 없다. C44는 niche prototype으로
보류하며 full agent 실행으로 승격하지 않는다.

전체 281문항의 검색 실패와 Gold 결함을 분리하기 위해 검색 결과·arm·agent 제출을 볼 수 없는
retrieval-blind 감사 절차를 추가했다. 각 batch는 SHA 순서로 미감사 qid를 고정하고 Luna-medium
1차 감사, Sol-medium 독립 교차검토, JO/span/group validator, 명시적 adjudication resolution,
hash manifest 순으로 처리한다. batch 2와 3까지 반영한 Gold v5는 다음 상태다.

- evaluable: 269문항
- completeness-reviewed: 52문항
- 미감사: 217문항
- batch 2: 10개 중 pass 1, span fix 4, exclude 5
- batch 3: 20개 중 pass 8, span fix 5, exclude 7

기존 C29 full-agent 제출을 Gold v5의 reviewed 52문항에 동일하게 재채점한 agent 지표는
R@1 `.6058`, R@5 `.8654`, R@10 `.8750`, suff@5 `.8462`, suff@10 `.8654`다.
원 전체 281 R@5 `.6314`와의 격차는 미감사 Gold가 검색기 성능을 크게 교란한다는 증거지만,
52문항은 전체를 대표하지 않으므로 `.95` 달성 근거로 사용하지 않는다. 남은 217문항도 같은
절차로 감사한 뒤, 동결된 scoreable set에서 agent 실험을 다시 수행한다.

### Gold 감사 batch 4~6 진행 상태 (2026-08-22)

같은 retrieval-blind 절차로 batch 4~6을 추가 반영한 Gold v8은 251문항이며, 이 중
94문항은 질문·공식 답·공식 출처·원문 JO만으로 completeness 검토가 끝났다. 157문항은
아직 미감사다.

- batch 4: 20개 중 fix 12, exclude 8. Luna 1차 산출물은 qid 중복 3건·누락 1건으로
  구조적으로 무효였고, 그 사실을 manifest에 남긴 뒤 validator-clean Sol 결론만 선택했다.
- batch 5: 20개 중 fix 13, exclude 7. 0051은 경피적 수술 방식·88항·복강경/흉강경
  예외가, 0157은 수술분류표의 세 독립 구역이 기존 Gold에 완전하게 표현되지 않아 제외했다.
- batch 6: 20개 중 fix 17, exclude 3. 0225는 지급률과 인공관절 판정이 같은 JO에 있어
  중복 required group이 된 양 모델 산출물을 validator가 거부했다. 원 산출물을 보존하고
  최종 2-group 구조가 필요하다는 correction ledger를 남긴 뒤 exclude했다.

기존 C29 full-agent 제출을 reviewed 94문항에 동일하게 재채점한 agent 지표는 R@1 `.5798`,
R@5 `.8298`, R@10 `.8564`, suff@5 `.8085`, suff@10 `.8404`다. 이는 새 모델 실행이 아니라
동일 agent 제출의 Gold-only 재채점이며, 표본이 52→94로 넓어지자 R@5가 `.8654→.8298`로
낮아졌다. 따라서 reviewed 29/52의 `.93~.98`을 전체 일반화 성능으로 해석하지 않는다.

모든 C29/C44 실험은 Semantic Tag와 고정 Vector/Meta 검색을 결합한 하이브리드다. 개선
처치는 Semantic Tag 데이터·규칙·명시 참조 graph에만 적용하며 Vector/Meta 알고리즘,
임베딩, agent action schema는 양팔에서 고정한다. Gold 감사는 검색기 처치가 아니라 평가
오류 제거이며, 검색 성능과 별도 provenance로 기록한다.

## Gold batch 7~9, C45/C46 안전 게이트 (2026-08-22)

retrieval-blind 감사를 batch 7~9까지 확대했다. batch 7은 10문항 중 fix 8/exclude 2,
batch 8은 fix 9/exclude 1이었다. batch 9의 Luna 1차 결과는 10개 qid에 12개 decision을
내어 두 qid가 중복된 구조적 무효 산출물이었다. 이 산출물을 삭제하거나 고치지 않고
invalid source로 manifest에 남겼고, 공식 질문·답·출처와 원문 JO/span만 본 Sol-medium
교차검토의 validator-clean 결론만 명시적으로 선택했다. batch 9는 fix 6/exclude 4다.

batch 9 처리 중 `exclude` 문항의 기존 Gold 자체에 중복 required group이 있으면 validator가
격리 결정을 거부하는 결함을 발견했다. 제외 문항은 scoreable overlay에 출력되지 않으므로,
`exclude`에는 기존 final-group invariant 검사를 적용하지 않고 `pass/fix`에는 계속 강제하도록
validator와 회귀 테스트를 수정했다. 테스트는 6/6, adjudicator 테스트는 3/3 통과했다.

Gold v12는 SHA-256 `543f2a28d195a6a6fb6faf56389e850a07a4999cc52194435934cf904a6d4e28`,
qid 파일은 `90ab25310454f81ad175e29f46044978d8e0b38bb3a07d8c158182c08cbef33b`다.
총 244문항이 scoreable이고 117문항은 retrieval-blind completeness review를 마쳤으며,
127문항은 아직 같은 수준으로 검토되지 않았다. 기존 C29 whole-train의 **동일한 agent 제출**을
v12로 재채점한 지표는 다음과 같다. 이는 새 agent 실행이나 검색기 상승 실험이 아니다.

| 평가 범위 | n | R@1 | R@5 | R@10 | suff@5 | suff@10 |
|---|---:|---:|---:|---:|---:|---:|
| v12 전체 scoreable | 244 | .4344 | .6954 | .7445 | .6721 | .7172 |
| completeness-reviewed | 117 | .5641 | **.8291** | .8632 | .8034 | .8376 |

v10의 reviewed 111문항에서 같은 제출은 R@5 `.8108`이었다. v12의 `.8291` 상승은
q0373 같은 직접 지급사유 sibling 누락 및 batch 9 span/OR 오류를 고친 평가 변화이며,
검색 알고리즘 효과로 세지 않는다. 아직 미검토 127문항이 남아 있어 `.95` 일반화 주장을
할 수 없다.

C45는 질문에 명시된 대괄호 판본만 corpus-derived identity로 인정하는 strict bracket
규칙이다. 상품명·질병명 사전 없이 arbitrary bracket을 처리하고, Vector/Meta는 고정했다.
Gold v11로 재채점한 5문항×1회 clean hybrid-agent smoke에서 C29와 C45 모두
R@1 `.60`, R@5/R@10/suff@5/suff@10 `1.00`, fatal/protocol `0/0`이었다.
따라서 routing 수정은 보존하되 agent R@5 상승은 입증되지 않았다.

C46은 명시된 contract 안에서 여러 claim role 후보를 기존 결과 5개 뒤에 강제 삽입했다.
Luna-medium 10문항×2회 계획이었으나 clean pair에서 0415의 C45 R@5 `1→` C46 빈 제출
`0`, 0357의 C45 R@5 `1→` C46 R@5 `0`과 protocol error가 발생해 즉시 중단했다.
0389도 기계 점수는 `1→0`이었으나 C46 제출 `j02980`이 직접 지급사유로 보여 Gold
누락 가능성이 있으므로 효능 판정에서 분리했다. 중단 때문에 완전한 paired aggregate를
만들지 않으며, 인터럽트로 생긴 error cell은 성능 증거로 사용하지 않는다. 결론은
**후보 강제 삽입 방식 기각**이다. 다음 가설은 기존 top-level 카드 ID·순서를 완전히
보존하면서, 명시 참조와 구어체 의도를 검색 가능한 데이터/태그로 투영해야 한다.

## Gold batch 10~11, C47/C48 정확 참조영역 실험 (2026-08-22~23)

Gold batch 10은 retrieval-blind SHA 표본 10개를 감사해 fix 8/exclude 2로 반영했다.
Luna 1차 산출물은 q0444를 중복하고 q0440을 누락해 구조적으로 무효였으며, 원 산출물과
validator 오류를 보존한 뒤 모든 qid를 원문에서 다시 확인한 Sol-medium review만 명시적으로
선택했다. 이 결과의 Gold v13은 242문항, completeness-reviewed 125문항이었다.

C47 평가용 batch 11은 검색 결과를 보지 않고 미감사 문항 중 질문 문구만으로
`classification_membership` gate를 통과한 문항을 SHA 순서로 선택했다. 기존 개발 fixture
0029·0147은 사전에 제외했다. 후보가 7개뿐이었고, 독립 감사 결과 fix 2/exclude 5였다.
제외 사유는 검색 실패가 아니라 다음과 같은 Gold 불완전성이다.

- 0153: 뇌혈관 코드와 허혈심장 코드라는 두 required claim 중 한쪽 누락
- 0265: 코드 의미·변경 정책·청구서 기재의 분리와 원문에 없는 능동 주체 표현
- 0169: 암주요치료비와 다빈치수술의 별도 무효 claim 중 한쪽 누락
- 0071: 지급사유와 I63 코드표를 하나의 span으로 대체할 수 없음
- 0321: 심장질환 수술 정의와 두 수가코드 행이 별도 required claim

0097은 특정 류마티스 관절염 정의부터 M05·J99.0*·M06 행까지 span을 교정했고,
0289는 질문이 판본을 특정하지 않아 동일 소액암진단특약의 직접 지급사유·지급표 8개를
한 OR group으로 확장했다. Gold v14는 SHA-256
`dce3531d6a675bc32f201d52ee8478719dd2cb4e2fa601b852ee656a216f272b`, 총 237문항이며
completeness-reviewed 127, 미감사 110이다. 질문 문구와 task type은 바꾸지 않았다.

C47은 C45의 hybrid 결과 40개와 그 ID·순서를 그대로 두고, 질문이 코드·분류표·entity
membership을 명시할 때만 기존 결과의 source JO에서 U5 명시 참조 graph를 최대 2-hop
따라간다. target JO 전체 preview 대신 edge의 literal table label부터 최대 8 element,
1,600자까지의 bounded source region을 `reference_evidence`로 표시한다. graph edge가 없지만
같은 target JO가 이미 고정 hybrid 40 안에 있는 경우에는 0-hop exact region만 표시한다.
base hybrid 40의 candidate와 rank는 바꾸지 않지만, 선택·제출 가능한 auxiliary evidence를
별도 정렬해 agent에 표시한다. Vector/Meta V9, rule router, Gold scoring, agent prompt는
고정했다.

최초 batch11 2문항×4회 실행은 sandbox가 내부 Codex app-server 생성을 막아 16/16 cell이
실패했으므로 성능 근거에서 제외했다. 동일 block 전체를 정상 환경에서 재실행한 clean
Luna-medium 결과는 C45 R@1/R@5/suff@5 `.500/.750/.750`, C47 `1.000/1.000/1.000`, fatal/protocol
양팔 `0/0`이었다. 기능은 0097에서만 발화했고 C47은 4/4 성공, C45는 2/4 성공했다.
0289는 기능 미발화 상태로 양팔 4/4 성공했다. 따라서 C47의 직접 관측 이득은 0097 한
문항뿐이며 n=2를 일반 효과로 해석하지 않는다.

C47 trace 감사에서 0097의 정확한 표39 뒤에 같은 OCR element의 `표42`와 잘못 전파된
`대상포진 분류표` JO 제목이 함께 보이는 경계 결함을 발견했다. C48은 기존 C47 arm을
보존하면서 다음 두 fail-closed 규칙만 추가했다.

- 후속 element가 현재 label과 다른 table label을 하나라도 도입하면 그 element 전에 중단
- target JO 제목이 bounded region에 실제 존재하지 않으면 `참조표 {label}`로 대체

42개 회귀 테스트를 통과했다. 같은 batch11 2문항×4회 clean agent 결과는 C45
R@1/R@5/R@10/suff@5/suff@10 `.500/.500/.625/.500/.625`, C48는 다섯 지표 모두
`1.000`, fatal/protocol/empty
양팔 `0/0/0`이었다. 0097은 C45 0/4, C48 4/4였고 0289는 양팔 4/4였다. 모든 paired
first hybrid 40 ID·순서는 byte-for-byte 동일했다. 단 C48은 C47의 0097 trace에서 경계
결함을 발견한 뒤 설계했으므로, C48 관점의 fresh feature-active qid는 0개다. 이 실행은
적응적 mechanism 회귀 검사일 뿐 승격 확증이 아니다.

별도로 이미 개발·감사에 사용된 분류/포함 문항을 포함한 reviewed regression 7문항×3회에서
C45→C48 agent 지표는 R@1 `.4762→.6190`, R@5/R@10 `.7381→.7857`,
suff@5/suff@10 `.6190→.6667`이었다. R@5 qid-cluster bootstrap 95% CI는
`[0,.1429]`, 1승/0패/6무이며
유일한 R@5 이득은 기존 개발 fixture 0249 한 건의 3회 중 1회 제출이었다. hard regression은
없지만, 0274의 fractional R@1은 `.5→0`으로 악화했다. 모든 21 paired first hybrid
40 ID·순서는 같았다. C48은 평균 model call
`2.14→1.29`, elapsed `25.00s→15.64s`였지만 empty submit은 `1→2`였다. 이 7문항은
holdout 성능 주장이 아니라 이전 성공·실패층의 회귀 검사로만 사용한다.

결론은 C48의 정확 참조영역이 기존 base 후보 순위를 손상하지 않고 일부 개발 문항의 agent
선택을 개선했다는 제한적 증거만 얻었다. 전체 `.95`나 새 문서 일반화를 입증하지는 못했다.
현재 artifact는 32,046 elements/7,612 JO의 단일 source document이므로, 다음
승격 게이트는 source-document가 분리된 corpus와 사전 동결한 새 feature-active holdout에서
동일한 agent 비교를 수행하는 것이다.

## 정상 하이브리드 s29×2 확정 실행 — sonnet (2026-08-23)

Luna의 `c29c52_s29x2_hybridfixed` 재실행이 Codex 사용량 한도(8/27 해제)로 69/116 cell에서
중단됐다(완료 69 중 24 cell이 model_error). host runner에 `--resume`(정상 완료 cell 재사용,
model_error cell만 재실행)를 추가했고, 한도 해제 전 확정치를 위해 동일 조건을
`--provider claude --model sonnet`으로 전량 새로 실행했다. 조건: s29×2, Gold v14,
`--require-meta-first`, meta V9 sidecar, rule router, search-only. 116/116 cell에서
첫 hybrid search `fallback_status=ok:merge:10`을 확인했고 fatal error 0, empty submit 0이다.

| arm | R@1 | R@5 | R@10 | suff@5 | suff@10 | protocol | 비용 |
|---|---:|---:|---:|---:|---:|---:|---:|
| C29 | .6379 | **.9828** | .9914 | .9655 | .9828 | 2 | $2.46 |
| C52 | .6121 | .9741 | .9914 | .9655 | .9828 | 5 | $2.53 |

- paired R@5 Δ+.0086(C29−C52), 1승/0패/28무, BCa 95% CI `[0,+.0259]` — 우열 없음.
- C29 미만점은 `0351`(.5) 하나로, Luna full-train에서도 실패했던 multi-claim 문항이다.
- 해석 한계: 이 수치는 **Vector/Meta 결합이 실제 동작함을 검증한 첫 s29 확정치**이지만
  모델이 sonnet이라 Luna `.9224`(meta 미동작)와의 차이를 meta 수리 효과로 귀속할 수 없다.
  또한 s29는 retrieval-blind 감사를 마친 부분집합으로 전체 train을 대표하지 않는다
  (동일 arm의 Luna 전체 281 R@5 = .6314). Luna 한도 해제 후 `--resume`으로 원 실행을
  완주해 모델 축을 분리 보고한다.

## Gold 전수 감사 완료(batch 12~16)와 v19 전량 하이브리드 실측 (2026-08-23)

남은 미감사 110문항을 같은 retrieval-blind 절차(1차 감사 + 독립 교차검토 + validator +
명시적 adjudication + manifest)로 batch 12~16에서 소진했다. 1차는 Claude Opus, 교차검토는
Claude Sonnet 서브에이전트가 수행했고, 모델 산출물·resolution·ledger 는 전부 audits/ 에
보존했다. 결과: pass 16 / fix 71 / exclude 23 — 미감사 층의 85%가 결함이었다. 지배 유형은
표 span 의 표 제목·헤더행·급여금 명칭 유실, 판본 미특정 질문의 한쪽 판본 gold, 다중 claim
일부 미지지(→exclude), span 무관/과확장이다. 교차검토는 4건을 실질 정정했다(0277 pass→exclude,
0297 인코딩 위반→exclude, 0394 span 축소, 0478 OR member 보완).

**Gold v19 (동결)**: scoreable 213문항, 전 문항 retrieval-blind completeness reviewed,
미감사 0. exclude 누적 24건은 새 required AND group 이 필요한 유형이 대부분으로 수리 대장에
남긴다(적용기가 group 신설을 지원하면 복귀 가능).

v19 전 문항을 C29 하이브리드(meta V9 강제, rule router, search-only)로
sonnet-medium ×2 reps 실측했다(426 cell, fatal 0, empty 1, 비용 $25.1).

| 대상 | n | R@1 | R@5 | R@10 | suff@5 | suff@10 |
|---|---:|---:|---:|---:|---:|---:|
| **전체 (v19, 전 문항 감사)** | 213 | .5563 | **.8005** | .8521 | .7793 | .8333 |
| single_lookup | 169 | — | .8432 | — | .8432 | .8905 |
| multi_evidence | 42 | — | .6310 | — | .5357 | .6190 |

- R@5 qid-bootstrap 95% CI `[.7512, .8474]`. reps 불일치 8.9%. 문항 분포: 만점 157 ·
  부분 27 · 0점 29.
- 해석: (i) Luna 281 실행의 `.6314`와 이 `.8005`의 차이는 gold 감사(모집단·정답 정의 변경)와
  모델·meta 수리(sonnet, meta 강제)가 섞여 있어 단일 원인으로 귀속할 수 없다.
  (ii) 감사가 완료된 지금, 남은 실패는 채점 결함이 아니라 검색·제출 실패다 — 0점 29문항과
  multi_evidence(suff@5 .536)가 다음 개선의 명확한 표적이다. (iii) `.95` 대비 격차 −.15는
  주로 multi_evidence 층(전체 기여 −.073)과 single_lookup 잔여 실패(−.077)에 있다.

## v19 전량 실행 0점 29문항 실패 분해 (2026-08-23)

트레이스 전수 분해(스크립트 + 서브에이전트 정성 분석 2건, audits 절차와 동일하게 원문 대조):

| 단계 | 유형 | n | 내용 |
|---|---|---:|---|
| 검색기(미노출) | IDENT | 9 | 특약 라우팅 실패. 그중 3건은 **준용 교차참조**(질문의 특약과 정답이 있는 공통특약이 다름 — 분류코드표·구비서류가 소수 공통특약에 집중) |
| 검색기(미노출) | LEX | 2 | 질문 어휘와 gold 본문 어휘가 표면적으로 무관("해외"↔"국내 한정", "계약 유지"↔"예금보험 지급보장") |
| 검색기(미노출) | TABLE/기타 | 2 | 상품 전체 요약표 미인덱싱 1, 근중복 조번호 미스 1 |
| 에이전트(선택) | VERSION | 7 | 같은 특약의 다른 판본/유사 문언 이웃 제출 |
| 에이전트(선택) | BOILER | 4 | 여러 특약 반복 상용구의 다른 특약 사본 제출 |
| 에이전트(정렬) | MISJUDGE | 3+6 | gold 를 보고도 하위(6~10위) 배치 — D군 6건은 전부 gold 제출·순위만 실패(R@10 대부분 1.0) |
| 질문 자체 | AMBIG | 2 | 전역·요약형 질문의 스코프 해석 불일치 |

행동 관측: 실패 32세션 중 read 사용 4세션뿐 — 검색 1회 후 즉시 제출이 지배적.
결정론 노출 게이트(check_exposure.py, A군 13 기준선): 확실 MISS 9, 30~40위 꼬리 4(dense 꼬리 비결정 구간).

gold 개정 후보(감사와 별도, 사람 승인 대상): 0256(내용 동등 본문 조항 vs 부표 각주 위치), 0288(요약형 질문 OR 범위), 0231(통합 한도표 vs 판본별 조문) — a/a′ 동치의 "내용 동등" 확장 여부는 회의 기준 재확인 필요.

## 개선 arm 설계 (우선순위)

| arm | 측 | 표적(회수 상한) | 내용 | 게이트 |
|---|---|---|---|---|
| **R3 verify-submit** | 에이전트 규약 | VERSION+BOILER+MISJUDGE ≈ 20/29 (전체 +.07 상한) | 프롬프트 추가: 동일 문구가 여러 특약/판본에 반복되면 질문이 특정한 판본·범위와 정확히 일치하는지 read 로 재확인 후 선택, 표/분류/코드 질문은 표 후보 우선, 확신 높은 순으로 정렬 | 실패 16 + 통과 대조 14 의 mechanism set paired ×2 → 회귀 없으면 full 213×2 |
| **R1 준용·identity 별칭** | 검색기(규칙 태그) | IDENT 9 (전체 +.04 상한) | 문서에서 준용 문장·특약명 목록만으로 (i) 준용 교차참조 identity 매핑 (ii) 특약명 변형 별칭(상급종합병원 계열, 상품명↔특약명 구분) — QA/Gold 미참조 | check_exposure 9-MISS 에서 노출 개선 없으면 폐기 |
| R2 판본 facet | 검색 표시 | VERSION 일부 | 근중복 후보의 판본 라벨 명시(표시 변경은 C10/C11 전례상 위험 — R3 결과 후 판단) | 보류 |
| AMBIG·gold 개정 | 채점 기준 | 2~5 | 회의 승인 사항으로 상신 | — |

## R3 verify-submit — full 213×2 확정 (2026-08-23)

실패 분해에서 에이전트 측(VERSION·BOILER·MISJUDGE·정렬)이 0점의 절반 이상임을 확인하고,
prompt 3규칙(반복 상용구·판본은 read 로 원문 확인 후 선택 / 표·분류 질문은 표 후보 우선 /
확신 순 정렬)만 추가한 `c29_verify_submit_hybrid` arm 을 만들었다(검색·태그·예산 동일,
host runner 에 arm 별 prompt_addendum 지원 추가).

mechanism 30×2(실패 16 + 만점 대조 14): 실패층 R@5 .188→.375(6승/1패), 대조 회귀 0 → 승격.

full 213×2 paired(동일 qid·gold v19·sonnet, 기준선 = 직전 full run):

| 지표 | base | verify | Δ | sign | BCa 95% CI |
|---|---:|---:|---:|---:|---|
| R@5 | .8005 | **.8263** | +.0258 | 15/7 (p=.134) | **[+.0023, +.0516]** |
| suff@5 | .7793 | .8075 | +.0282 | 15/6 (p=.078) | **[+.0047, +.0540]** |
| R@10 | .8521 | .8697 | +.0176 | 15/9 | [−.0047, +.0423] |
| suff@10 | .8333 | .8545 | +.0211 | 15/9 | [−.0047, +.0469] |

single_lookup .843→.873, multi_evidence .631→.643. errors 0, empty 2, model calls 1.24,
비용 $22. hard regression(문항 평균 −.5 이상) 6건(0041·0177·0245·0259·0380 .5→0, 0366 1→.5)
— 개선 34건 대비 수용 범위이나 후속 실패층 분석에 포함한다.

판정: **R3 채택(신규 기본 arm = c29_verify_submit_hybrid).** R@5·suff@5 의 BCa CI 가 0 을
제외했다(부호검정은 비유의 — 효과가 소수 문항에 집중된 구조). 채택 후 현재 위치:
**전체 R@5 .8263 [.751 대비 상향], 목표 .95 잔여 격차 −.124** (multi_evidence .643 이 최대 병목 유지).

## R1/R1.1 identity 확장 + 참조 팔로우 (2026-08-23)

IDENT 실패 9건을 겨냥해 두 규칙 기능을 arm 옵션으로 추가했다(문서 도출만, fail-closed,
비발화 질문 첫 페이지 byte 보존, 회귀 테스트 62 OK):
- identity_expand: 특약명 집합 통계에서 공용 수식어를 도출해 별칭 변형 생성(soft, ≤12)
- reference_follow: 질문에 표·코드·서류 표면형이 있고 특약이 라우팅된 경우에만 U5 참조
  그래프의 target JO ≤5개를 하위 quota 로 추가. R1.1 에서 이 후보에 가시 라벨(note) 부착.

결정론 노출 게이트(A13): exposed 4/13 → **7/13**, 기존 노출 순위 회귀 0, 무해성 19/19 byte 동일.
표적 에이전트(10 qid ×2 arm ×3 reps, sonnet): 0278 0→.33~.67, 0029 0→.33(라벨 후),
0232 는 노출(25위)돼도 미선택, 대조 7건 전 rep 1.0 유지.

판정: **r1v(verify+R1) 를 기본 arm 으로 채택**(무손상·순증). 전체 기대 이득이 +.005 미만이라
단독 full 재실행은 하지 않고 다음 full 확정 실행에 포함해 반영한다. 남은 미노출 6건 중
4건은 질문에 특약 표면형 자체가 없는 유형(질의 의도 정규화 필요), 0130 은 gold 주석 문제
(질문 특약 자신의 동일 문언 제2-5조가 OR 에 없음 — gold 개정 후보 4호), 0086 은 참조
게이트 표면형 부재.

## 사이클 종결: r1v 최종 full 확정 (2026-08-24)

채택 스택 r1v(verify-submit + identity 별칭 + U5 참조 팔로우+라벨)로 v19 전량 213×2를
재실측했다(sonnet, meta 강제, 426 cell 오류 0, empty 2, $25.8).

| 실행 | R@1 | R@5 | R@10 | suff@5 | suff@10 |
|---|---:|---:|---:|---:|---:|
| 기준선 c29 (8/23) | .5563 | .8005 | .8521 | .7793 | .8333 |
| + R3 verify | .5587 | .8263 | .8697 | .8075 | .8545 |
| **+ R1 (최종 r1v)** | .5634 | **.8451** | .8744 | .8239 | .8615 |

- r1v R@5 qid-bootstrap 95% CI **[.7981, .8873]**. r3 대비 Δ+.0188(14/10, CI[−.006,+.047])은
  단독 유의는 아니나, 기준선 대비 누적 **+.0446** 은 R3 단계에서 CI 0 제외로 확정된 이득 위의
  순증이다. 0점 문항 29→22. multi_evidence .6845, single .8876.
- R4(다중근거 제출 다양화)는 mechanism 순효과 0(제로섬)으로 기각 — 이 층의 남은 지렛대는
  (i) 적용기 required AND-group 신설 op(exclude 24건 복귀) (ii) gold 개정 후보 4건
  (0256·0288·0231·0130) (iii) 검색기 표 후보 소프트 부스트다. (i)(ii)는 회의 승인 사항.
- 계보 주의: 이 절의 모든 수치는 (gold v19 · u3/u4 · sonnet-medium · meta V9 강제 ·
  search-only · 2 reps) 튜플이며, Luna·이전 gold 수치와 같은 표에 놓지 않는다.

## batch 17 (a/a′ 기준 정합화) → Gold v20 · R5 기각 (2026-08-24)

R5(table_note: 표 포함 후보 가시 라벨)는 mechanism ×3 에서 이득 0·미세 손실 2건(0249·0274)으로
기각했다 — C11 과 동일하게 표시 메타데이터 변경은 선택을 돕지 못한다.

batch 17은 회의의 **유사·파생 특약 a/a′ 동치 규칙**을 감사 기준으로 정식 반영한 표적 재감사다
(기준 갱신은 qids manifest 에 기록, 감사 자체는 retrieval-blind 유지, 대상 = 명세 v1 §6
실패 루프 + 과거 감사에서 "문언 차이로 제외"된 판본 전건). 1차(Opus)·교차검토(Sonnet) 17/17 일치,
pass 5 / fix 12 / exclude 0:
- 판본 OR 확장 8건('최초계약의' 유무류 형식 차이만; 삭감 구조 부재 등 실질 차이는 유지 제외)
- 0130 오배정 교정(정기특약 조 → 질문이 지목한 특약 자신의 동일 문언 조 4판본)
- 0256 내용 동등 본문 조문 OR, 0490 형제 조 OR, 0231·0288·0290 은 기존 gold 완결 확인(pass)

**Gold v20**(sha 8a723f0e…, 213문항 전 문항 reviewed 유지). OR 추가는 채점을 관대하게만 하므로
기존 r1v full 제출을 재실행 없이 v20 으로 재채점했다(결정론):

| gold | R@1 | R@5 | R@10 | suff@5 | suff@10 | 0점 |
|---|---:|---:|---:|---:|---:|---:|
| v19 | .5634 | .8451 | .8744 | .8239 | .8615 | 22 |
| **v20 (동일 제출 재채점)** | .5939 | **.8638** | .8838 | .8427 | .8709 | 18 |

R@5 CI95 [.8216, .9038]. single .9112 / multi .6845. 계보: 검색·제출은 v19 실행 그대로이고
gold 정의만 회의 기준으로 정합화된 것 — "성능 향상"이 아니라 **측정 정합화**로 기록한다.
현 위치 .8638, 목표 .95 잔여 −.086. 잔여 표적: multi_evidence .685(구조), 0점 18
(A 미노출 11 중심 — 질의 의도 정규화·참조 게이트 확장), exclude 24 복귀(AND-group op).

## R6 intent-role 보조질의 — 기각 (2026-08-24)

특약 미라우팅 ∧ ROLE_RULES 매치 시 역할 조제목 어휘(문서 도출) 보조질의를 tail quota 로 넣는
설계. 노출 게이트(9 MISS, gold v20): 신규 노출 0, 발화 2/9, 경계 순위 2건이 tail quota 에 밀려
순 −1(4/9→3/9). 무해성은 통과(미발화 20/20 byte 동일). 원인: (i) ROLE_RULES 는 문서 표면형
규칙이라 겨냥한 구어 의도문("지급되나요/받을 수 있나요")과 역상관 (ii) tail quota 가 40칸 안
32~39위를 점유해 경계 문항을 밀어냄. 코드는 비활성 옵션으로 보존(전 arm 기본 off), 채택하지
않는다. 남은 9 MISS 는 구어·전역 의도의 경성 꼬리로, 개별 도달 가능성 진단 후에만 재시도한다.

## R8 verify-gate — full 에서 기각 (2026-08-24)

mechanism 29×2(표적 +.06~.13, 대조 무손상, gate 28/58 발화)로 승격했으나 full 213×2 에서
R@5 .8638→.8568(Δ−.0070, 9/12), suff@5 −.0070, multi .685→.637, hard regression 9건,
비용 2배(gate 반려로 호출 1.38→2.08). mechanism 이득이 전수에서 역전 — 소표본 선별의
winner's curse 를 full 확정 게이트가 차단한 사례로 기록한다. **공식 arm 은 r1v(.8638) 유지.**
프롬프트·프로토콜 수준 선택 개입은 R3 외 전부 기각(R4·R5·R7·R8).

## effort-high — full 에서 미채택 (2026-08-24)

r1v arm 동일, sonnet reasoning-effort 만 medium→high. mechanism 29×2 는 +.072(8/2, 대조 무손상)
였으나 full 213×2 paired 는 R@5 .8638→.8697(Δ+.0059, 10/9, CI[−.015,+.029]) — 유의 없음,
R@1 +.019·R@10 +.015 도 CI 0 포함, hard regression 8건, 비용 +30%, 오류 2. mechanism→full
역전이 R8 에 이어 두 번째로 재현됨(소표본 선별 편향). **미채택 — 공식 arm 은 r1v·medium(.8638)
유지**, high 점추정 .8697 [.826,.910] 은 참고 병기.

### 현 위치와 정직한 상한 (2026-08-24)
- 공식: R@5 .8638 [.822,.904] (gold v20·213·sonnet-medium·2reps). 참고: high .8697, R@10 .8991.
- 선택·정렬 층 개입은 소진(R3 만 생존, R4·R5·R7·R8·effort-high 기각/미채택). 미노출 9건은
  규칙화 불가 확정(상한 .958). multi_evidence .685 는 제출 다양화·게이트로 안 움직였다.
- .90+ 로 가는 잔여 후보: (a) 에이전트 모델 상향(opus — 사용자 승인 필요, "sonnet 유지" 지시와 충돌)
  (b) 요약·전역형 질문의 gold OR 범위 재정의(회의 기준 해석 — 승인 필요)
  (c) exclude 69 복귀·multi 구조 정상화(분모 확장 계보 — 213 수치는 상승하지 않음).

## 에이전트 모델 축: opus 채택 (2026-08-24)

sonnet 계열 선택층 개입 소진 후, 사용자 승인 하에 모델 축을 시험했다. mechanism 29×2 에서
opus-medium 이 +.118(노출0점층 +.241, 대조 무손상)로 역대 최강 신호였고, full 213×2 paired 에서
**처음으로 mechanism 이득이 일반화**됐다:

| 지표 | sonnet-medium | opus-medium | Δ | sign | BCa CI |
|---|---:|---:|---:|---:|---|
| R@5 | .8638 | **.8936** | +.0297 | 23/7 p=.0052 | [−.0008,+.0587] |
| R@10 | .8838 | .9182 | +.0344 | 21/4 p=.0009 | [+.0082,+.0618] |
| suff@5 | .8427 | .8803 | +.0376 | 20/6 p=.0094 | [+.0023,+.0681] |
| suff@10 | .8709 | .9085 | +.0376 | 19/4 p=.0026 | [+.0070,+.0634] |

multi_evidence .685→**.806**, single .917, 0점 20→16, errors 0, calls 1.14, $43.4.
**공식 계보 갱신: (gold v20 · u3/u4 · opus-medium · r1v · meta 강제 · 2reps) R@5 .8936 [.854,.931].**
hard regression 6건(0029·0094·0126·0275·0311·0400)은 rep 변동·판본 선택 혼재 — 후속 분해 대상.

## Gold v21 (명세 v1 §3 동일문구 출현 확장) — R@5 .9030 (2026-08-24)

정답기준 명세 v1 §3("특약 지정 질문은 해당 scope 안 출현만, 일반 질문은 문서 전체 출현 인정")을
결정론 스크립트(apply_spec3_occurrence_expansion.py)로 적용했다 — 질문 문자열의 특약명 표면형으로
scope 를 판정(일반 89 / 지정 124), member 정규화 텍스트(앞 160자·40자 이상)의 정확 일치 출현을
OR member 로 추가(316개, cap 40). QA·검색 결과 미사용, 기존 member 무변경(관대화만).

opus full 제출 재채점(재실행 불필요):

| gold | R@1 | R@5 | R@10 | suff@5 | suff@10 | 0점 |
|---|---:|---:|---:|---:|---:|---:|
| v20 | .5880 | .8936 | .9182 | .8803 | .9085 | 16 |
| **v21** | .5951 | **.9030** | .9229 | .8897 | .9131 | 14 |

**공식: (gold v21 · u3/u4 · opus-medium · r1v · meta 강제 · 2reps) R@5 .9030 [.8646, .9370].**
사용자 기준(.90) 도달. 계보 주의: v20→v21 변화는 검색 개선이 아니라 명세 §3 채점 정합화.
잔여: 0점 14(경성 MISS 6 포함), multi .806, 부분점수 층 — .95 는 이 층들에서.

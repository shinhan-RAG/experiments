## 종합 결론

C49–C51을 “고정된 정상 Vector/Meta hybrid 위의 Semantic Tag·reference-evidence 개선”으로 채택할 근거는 아직 없습니다.

- C49: 안전성 개선 의도는 있으나 holdout agent 성능이 C48보다 낮고, provenance가 완전히 fail-closed가 아닙니다.
- C50: 한 개발성 질문에서는 작동하지만 실제 비교가 C45↔C50이라 C50 단독 효과가 아닙니다. 정규화 규칙도 보험 도메인에 강하게 하드코딩되어 있고 계약상태 질문 오탐 가능성이 남습니다.
- C51: 두 feature-active 질문에서 작동 원리는 trace로 확인되지만, catalog에 실제 cross-table/overlong region이 다수 있으며 두 질문 모두 이미 소비된 질문입니다.
- 세 실행 모두 첫 검색의 Meta fallback이 `ModuleNotFoundError`로 실패했습니다. 따라서 “정상 작동하는 고정 Vector/Meta lane 위에서의 개선”은 검증되지 않았습니다.

## 1. 검증된 사실

### Arm 격리와 실행 프로토콜

Arm 정의 자체는 다음처럼 차분되어 있습니다.

- C48→C49: `reference_evidence.stats`만 v1.1에서 v1.2로 변경.
- C49→C50: `intent_bundle`만 추가.
- C50→C51: stats를 v1.3으로 바꾸고 `catalog_title_lookup`, `catalog_max_variants=5`만 추가.

이 차분은 [arms.json](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/arms.json) 및 [test_experiment_arms.py:546](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/test_experiment_arms.py:546)에서 확인됩니다.

각 paired run 내부에서는 다음이 고정됐습니다.

- 모델: `gpt-5.6-luna`, reasoning `medium`
- runner, system prompt, action schema, scoring SHA
- Gold 및 qid 파일
- `elements_u3`, `tags_u4_fact_rules`, `elements_u3jo`
- Vector 구현 SHA와 `meta_view=V9`
- arm 순서: qid/rep별 balanced rotation

다만 C50 mechanism run은 C49가 아니라 C45와 비교했습니다. C45에는 reference graph/evidence가 없으므로 관측 차이는 “C49 계열 reference evidence + C50 colloquial intent”의 합성 효과입니다. C50 단독 격리가 아닙니다.

또한 모든 실행에서 첫 검색 Meta fallback이 `error:ModuleNotFoundError`였습니다.

- C48/C49: 20개 첫 검색
- C45/C50: 6개 첫 검색
- C50/C51: 12개 첫 검색

설정은 pair 내에서 같았지만 실제 Vector/Meta fallback lane은 기능하지 않았습니다.

### Agent 지표 재계산

`results.jsonl` 원자료에서 직접 재계산한 평균입니다.

| 비교 | arm | n | R@1 | R@5 | R@10 | RR@10 | suff@5 | suff@10 | protocol errors |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| 10문항×1 | C48 | 10 | 0.333 | 0.508 | 0.508 | 0.533 | 0.400 | 0.400 | 1 |
|  | C49 | 10 | 0.283 | 0.417 | 0.450 | 0.446 | 0.300 | 0.300 | 6 |
| 1문항×3 | C45 | 3 | 0 | 0 | 0 | 0 | 0 | 0 | 0 |
|  | C50 | 3 | 0.667 | 0.667 | 1.000 | 0.722 | 0.667 | 1.000 | 0 |
| 2문항×3 | C50 | 6 | 0 | 0.583 | 0.750 | 0.249 | 0.333 | 0.500 | 5 |
|  | C51 | 6 | 0.500 | 1.000 | 1.000 | 0.667 | 1.000 | 1.000 | 0 |

C49는 C48 대비 전반적으로 하락했습니다. 특히 q0242와 q0309에서 손실이 있었고 q0088만 R@5가 개선됐습니다.

C50의 결과는 한 질문 `v3-offline-0150`을 3회 반복한 것이며 Gold도 train-scoped reviewed Gold입니다. 일반화 증거가 아니라 mechanism/dev evidence입니다.

C51의 protocol-error 문제를 분리해도 mechanism 신호는 일부 남습니다. 문제가 된 q0310/r1 pair 전체를 제외하면:

- q0310의 error-free 2회에서 C50: R@1 0, 평균 R@5 0.5
- C51: R@1 1.0, R@5 1.0

따라서 C51의 모든 상승을 C50 protocol error 5건만으로 설명할 수는 없습니다. 하지만 표본은 여전히 두 질문뿐입니다.

### Builder의 직접 누수

[build_tags_u5_reference_graph.py:346](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/filesearch/build_tags_u5_reference_graph.py:346)의 직접 입력은 다음 세 가지뿐입니다.

- elements
- tags
- JO
- 출력 경로

코드에는 QA, Gold, qid, result, agent output을 여는 경로가 없습니다. `SEMTAG_QID`나 agent 산출물도 참조하지 않습니다. stats의 input SHA도 세 입력만 기록합니다.

반면 builder는 `build_tags_u4_fact`에서 `explicit_table_refs`, `table_keys`를 import합니다. 그 파일은 지정 검토 범위 밖이므로 해당 dependency와 upstream `tags_u4_fact_rules`까지 포함한 간접 누수는 완전히 인증할 수 없습니다.

또한 테스트에는 실제 mechanism 질문과 사실상 같은 문자열이 들어 있습니다.

- “스쿠버다이빙으로 인한 후유장해도 보상 돼?”
- “통합건강원 1~5종 수술분류표”

즉 실행 중 Gold를 읽은 흔적은 없지만 feature 설계·테스트 단계에서 평가 질문이 소비된 것은 분명합니다.

### v1.2 graph

확인된 긍정 요소:

- 해소 인덱스가 `document_key`를 포함합니다.
- 로컬 `부표`는 document+contract identity+table key로 좁힙니다.
- 2-hop은 첫 key가 `부표:`일 때만 허용됩니다.
- v1.2 ledger의 200개 2-hop 모두 이 방향을 지켰습니다.
- 동일 source/local key가 여러 최종 target으로 갈라지는 항목은 ledger에서 발견되지 않았습니다.
- 101개 ambiguity가 기록됐습니다: 별첨 78, 부표 23. 예를 들어 `별첨2 표3`은 10개 후보, 일부 `부표2-1`은 2개 후보라 연결되지 않았습니다.

그러나 “ambiguity는 항상 fail-closed”라는 stats 주장은 코드보다 강합니다. [build_tags_u5_reference_graph.py:276](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/filesearch/build_tags_u5_reference_graph.py:276)은 여러 후보 중 `refs`를 가진 후보가 하나뿐이면 그 후보를 선택합니다. 이는 ambiguity를 닫는 것이 아니라 heuristic으로 해소하는 것입니다.

추가 위험:

- `active_local`은 새 부표 heading이 나올 때만 바뀌고 다른 섹션에서 reset되지 않습니다.
- 뒤쪽의 별첨 참조가 오래된 active local에 귀속될 수 있습니다.
- 한 참조행에 로컬 key가 여러 개면 명시적 source를 버리고 stale `active_local`로 fallback할 수 있습니다.
- `toc_like`는 target 후보에서만 제외됩니다. 목차성 source가 대량의 direct edge를 만드는 것은 허용됩니다.

따라서 document scope는 확인되지만 provenance-safe 및 엄밀한 fail-closed는 확인되지 않았습니다.

### v1.3 table catalog

문서 scope는 구현과 현 artifact 모두 확인됩니다.

- 모든 7,612개 JO→document 매핑이 단 하나의 document를 가리킵니다.
- catalog 49개도 모두 같은 document입니다.
- JO가 문서를 가로지르면 예외를 발생시킵니다.

제목 중복은 현재 catalog의 동일 문서 내 normalized title 기준으로 0건입니다.

그러나 영역 종료 로직은 안전하지 않습니다. [build_tags_u5_reference_graph.py:164](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/filesearch/build_tags_u5_reference_graph.py:164)은 제목이 `분류표/지급기준표/장해분류표`로 끝나는 표만 anchor로 인정합니다. 영역은 “다음 인식된 anchor”까지 계속됩니다. 따라서 그 사이에 일반 표 heading이 있어도 종료되지 않습니다.

확인된 명백한 cross-table 예:

- `표44 통풍 분류표`: 60 elements, 5 JO. preview에 표57 중환자실 시설규격, 표58 응급환자, 표59 표적항암제 등이 포함됩니다.
- `표129 21대다빈도생활질병 분류표`: 56 elements. 표130, 표141 등 다른 표가 포함된 뒤 표143에서야 종료됩니다.
- `표85 급여 특정NGS...분류표`: 45 elements, 4 JO. 뒤의 표86 등 비-분류표 catalog가 섞입니다.

따라서 overlong/cross-table region이 실제 artifact에 존재합니다.

그 밖의 문제:

- split heading은 다음 element의 첫 줄을 제목으로 사용하지만 다음 element가 같은 문서인지 확인하지 않습니다.
- 같은 key의 연속 anchor는 title이 달라도 key만 같으면 collapse됩니다.
- 마지막 표는 문서 끝이 아니라 전체 elements 끝까지 잡은 뒤 doc filter합니다.
- catalog 자체에는 모든 JO variant가 저장되지만 C51 노출은 최대 5개입니다. 표102는 18개, 표120은 9개 variant이므로 “모든 submit-compatible JO 노출”이 아닙니다.
- submit 최대치는 10이므로 18-JO 표는 현재 프로토콜로 전부 제출할 수도 없습니다.

### C50 colloquial normalization

활동명 dictionary나 qid 분기는 없고 cause phrase도 확장 query에 복사하지 않습니다. 이 점은 일반적인 활동명에 대해서는 양호합니다.

그러나 결과 query는 다음 보험 전용 문구로 고정됩니다.

- 재해의 정의
- 재해분류표
- 보장대상이 되는 재해
- 우발적인 외래의 사고
- 보험금을 지급하지 않는 재해

따라서 generic normalization이라기보다 보험 사고 시나리오 전용 hardcoding입니다.

Fail-closed도 충분하지 않습니다. 테스트의 “계약 실효 상태에서 유지중 발생한 사고…”는 붙여 쓴 `유지중` 때문에 connector regex를 통과하지 않을 뿐입니다. 다음과 같은 표현은 코드상 오탐 가능합니다.

- “계약 실효 상태 중 사고가 나면 보험 돼?”
- “계약 해지로 인해 상해보험금 받을 수 있어?”

각각 `중` 또는 `로 인해`, harm=`사고/상해`, coverage ask를 만족해 accident coverage로 정규화될 수 있습니다. 이는 활동 사고가 아니라 계약상태·효력 질문입니다.

또 C50은 evidence metadata만 추가하는 것이 아닙니다. [agent_tools.py:1333](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/agent_tools.py:1333)에서 normalized query로 새 tag ranking을 만든 뒤 첫 JO로 prepend하여 40-result page를 바꿉니다. 이는 비-LLM deterministic retrieval fusion/reranking입니다.

### C51 동작과 trace

C51 catalog lookup은 다음을 수행합니다.

- base 결과에서 첫 번째 document를 선택
- normalized exact title이 질문에 포함될 때만 발화
- 같은 문서에서 최장 title match가 하나가 아니면 fail closed
- catalog items를 `reference_evidence.items` 앞에 배치

C51 코드 자체는 base `items`를 변경하지 않고 Semantic metadata에 catalog evidence를 추가합니다. 비-agent LLM reranker는 사용하지 않습니다. 다만 agent가 이 정렬된 후보를 보고 최종 제출 순위를 바꾸므로 시스템 관점에서는 deterministic evidence/candidate augmentation 후 LLM-agent reranking입니다. “아무런 fusion도 아니다”라고 하기는 어렵지만 base result reranker는 아닙니다.

원 trace에서 catalog JO 제출은 매 rep 모두 정확했습니다.

- q0309: catalog `[j07560,j07561,j07562,j07563]`; 3/3 rep 모두 네 JO 전부 제출.
- q0310: catalog `[j07558]`; 3/3 rep 모두 제출, 모두 rank 1.

q0309의 C50은:

- r0: j07560–62만 제출, j07563 누락
- r1: j07561–63 제출, j07560 누락
- r2: j07561–62 제출, j07560·j07563 누락

C51은 한 번의 search response에서 네 catalog JO를 명시적으로 보고 그대로 제출했습니다. q0309의 suff 개선은 trace상 catalog mechanism과 직접 연결됩니다.

q0310에서 C50은:

- error-free r0: j07558 누락
- protocol-error r1: j07558을 rank 8에 제출
- error-free r2: j07558을 rank 5에 제출

C51은 세 번 모두 j07558을 rank 1로 제출했습니다. protocol error 5건은 r1에서 한 번에 10개 element를 batch-read해 read budget을 초과하면서 생겼습니다. 관련 JO를 전혀 찾지 못한 실패는 아닙니다.

Base result의 실측 동일성은 완전하지 않았습니다.

- q0309: C50/C51의 첫 40개 returned가 3/3 동일.
- q0310: 상위 19개는 같지만 20위 이후 순서가 달랐습니다.

코드상 C51 catalog는 base ranking 계산 후 실행되므로 이 후반 차이는 catalog에 의한 변경이라기보다 동점/실행 순서 비결정성으로 추정됩니다. 그래도 “paired raw base 결과가 byte-for-byte 동일했다”는 주장은 사실이 아닙니다.

## 2. 추론

- C49의 의도는 C48의 Cartesian-product 오염 제거이고 구조적으로 더 합리적입니다. 하지만 현재 agent 결과는 안전성 교정이 성능 개선으로 이어졌음을 보여주지 않습니다.
- C51의 두 feature-active 질문에서는 exact-title catalog가 agent의 evidence 선택을 단순화한 것이 실제 이득 원인입니다.
- 반면 catalog 전체에 적용하면 표44·표129 같은 오염된 region이 동일한 방식으로 강하게 노출될 수 있습니다. 두 성공 질문만으로 전체 catalog 안전성을 추론할 수 없습니다.
- 현재 Meta lane이 전부 실패했기 때문에, 정상 Vector/Meta 결과가 들어왔을 때 selected document와 catalog projection의 상호작용은 알려져 있지 않습니다.
- C51의 `allowed_documents[0]` 선택은 여러 문서 corpus에서 “유일한 문서”를 요구하지 않습니다. 다문서 환경에서는 base 첫 결과 하나가 catalog namespace를 사실상 결정합니다.

## 3. 한계와 일반화 상태

- 현재 v1.3 artifact의 실질 문서 수: 1개.
- C49 평가 질문: 10개, 그중 q0309/q0310은 이후 C51 mechanism 평가에서 재사용.
- C50 평가: q0150 한 개×3회이며 테스트 코드에 해당 스쿠버다이빙 질문이 그대로 존재.
- C51 평가: q0309/q0310 두 개×3회이며 q0309의 정확한 문구가 테스트 코드에 존재하고 두 qid 모두 이전 C48/C49 실행에서 이미 소비됨.
- 따라서 C49/C50/C51 어느 것에도 fresh feature-active holdout은 없습니다.
- builder의 직접 입력 누수는 부정할 수 있지만 import dependency와 upstream tag 생성까지 포함한 간접 누수는 지정 범위만으로 인증할 수 없습니다.
- 비결정적 agent 표본 수가 매우 작고 C49는 rep 1뿐입니다.

## 4. 판정

### C49 — 거부

현재 holdout agent 결과가 C48보다 낮고 provenance가 완전한 fail-closed가 아닙니다.

최소 다음 조치:

1. ambiguity의 “유일 redirector 선택” 예외 제거.
2. 새 section/heading에서 `active_local` reset.
3. 복수 local ref 행은 무조건 거부.
4. TOC source에서 alias propagation 금지.
5. 모든 2-hop에 `(doc, contract identity, local key, explicit line, global key)` provenance 저장.
6. 수정 후 기존 10문항과 겹치지 않는 다문서 fresh feature-active holdout 실시.

### C50 — mechanism-only

q0150에서 mechanism은 확인됐지만 C45↔C50 비교라 단독 효과가 아니며 질문도 개발에 소비됐습니다.

Fresh holdout 전 안전 게이트:

1. C49↔C50만 비교.
2. 계약 해지·실효·부활·면책·보장개시·유지 상태 문맥은 fail closed.
3. cause가 사람의 외부 활동/사건이고 harm가 그 이후 발생한 경우만 허용.
4. 활동명은 정규화 query에 복사하지 않는 현 규칙 유지.
5. 보험 전용 canonical terms임을 명시하고 “generic” 주장 철회.
6. 최소 다수의 unseen activity positive와 contract-state/진단/지급 negative 세트 통과.
7. 정상 작동하는 동일 Meta lane에서 paired 평가.

### C51 — mechanism-only

q0309/q0310의 성공 인과는 trace로 확인되지만 현재 catalog artifact는 전체 배포에 안전하지 않습니다.

Fresh holdout 전 안전 게이트:

1. 모든 literal table heading이 region을 종료하도록 변경. 제목 suffix가 `분류표`가 아니어도 boundary로 사용.
2. split heading의 다음 줄/element가 반드시 같은 document·연속 source 위치인지 확인.
3. repeated key는 normalized title까지 동일할 때만 collapse.
4. duplicate key 또는 duplicate normalized title은 document 내 fail closed.
5. region 내부에서 다른 literal table key가 발견되면 reject.
6. 전체 variant 수가 submit 한도 이하일 때만 발화하고, 발화 시 모든 JO를 노출. 임의 `max_variants=5` 절단 금지.
7. selected document가 하나로 유일하지 않으면 fail closed.
8. 표44·표85·표129 등 현재 cross-table region을 0건으로 만드는 corpus audit.
9. 이전 q0150/q0309/q0310 및 테스트 문자열과 겹치지 않는 다문서 fresh exact-title holdout.
10. paired raw base JO 순위 동일성, 정상 Meta 출력 동일성, agent protocol-error 동률을 실행 게이트로 기록.

이 게이트를 통과하기 전에는 C49–C51 어느 것도 `adopt` 판정을 받을 수 없습니다.
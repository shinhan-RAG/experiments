## 최종 판정

**C48 승격은 보류**가 타당하다.

- base hybrid 40의 ID·순서 보존과 C48의 경계/제목 정정은 확인됐다.
- 그러나 C48의 유일한 fresh feature-active qid로 제시된 `v3-offline-0097`은 C47 trace를 보고 C48을 설계한 직접 개발 사례다. **C48 기준 fresh feature-active qid는 0개**다.
- 7×3 회귀의 qid-level R@5 이득은 개발 fixture `v3-offline-0249` 하나뿐이며, 3회 중 1회만 성공했다.
- C48 empty submit은 `1→2`, `v3-offline-0274`의 평균 R@1은 `.5→0`으로 퇴행했다.
- 따라서 전체 성능 `.95`, 새 문서, 11만 문서 corpus 일반화는 주장할 수 없다.

아래 수치는 저장된 metric 필드를 단순 인용하지 않고, `submitted` 순서와 Gold의 group/JO를 대조해 다시 계산했다.

---

## 1. 실행 유효성 및 독립 재집계

### 최초 C47 run

**사실**

- 16개 cell 전부 첫 search는 성공했다. 각각 40개 결과와 `tag+fallback:meta` channel이 존재한다.
- 이후 model call이 모두 다음 오류로 실패했다.

  `failed to initialize in-process app-server client: Operation not permitted`

- 각 cell의 `errors=2`는 “model error 1 + submit하지 못함 1”의 합이다. 총계는 arm당 16 fatal-count, 전체 32다.
- 영향받은 cell은 C45 8/8, C47 8/8, 합계 16/16이다.
- protocol error는 0이다.
- empty submit도 0이지만, 이는 성공이 아니다. runner가 `did_submit == true && submitted == []`만 empty로 세기 때문에 submit 자체가 없었던 fatal cell은 empty가 아니다. 이 정의는 [host_agent_runner.py](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/host_agent_runner.py:294)에 있다.

**판정**

최초 run은 agent 성능 자료로는 전부 무효다. 다만 search layer의 결과 보존 여부를 확인하는 trace 자료로는 사용할 수 있다.

### 유효 run 재집계

| 실험 | arm | R@1 | R@5 | R@10 | suff@5 | suff@10 | fatal | protocol | empty |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| C47 retry, 2×4 | C45 | 4/8=.500 | 6/8=.750 | 6/8=.750 | 6/8=.750 | 6/8=.750 | 0 | 0 | 0 |
|  | C47 | 8/8=1.000 | 8/8=1.000 | 8/8=1.000 | 8/8=1.000 | 8/8=1.000 | 0 | 0 | 0 |
| C48 fresh-labelled, 2×4 | C45 | 4/8=.500 | 4/8=.500 | 5/8=.625 | 4/8=.500 | 5/8=.625 | 0 | 0 | 0 |
|  | C48 | 8/8=1.000 | 8/8=1.000 | 8/8=1.000 | 8/8=1.000 | 8/8=1.000 | 0 | 0 | 0 |
| reviewed regression, 7×3 | C45 | 10/21=.4762 | 15.5/21=.7381 | 15.5/21=.7381 | 13/21=.6190 | 13/21=.6190 | 0 | 0 | 1 |
|  | C48 | 13/21=.6190 | 16.5/21=.7857 | 16.5/21=.7857 | 14/21=.6667 | 14/21=.6667 | 0 | 0 | 2 |

분수형 R은 multi-group 문항에서 한 group만 맞춘 `.5`를 포함한다.

근거: [C47 retry results](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/out/host_agent/gpt56luna_medium_c47_batch11_reviewed2x4_goldv14_retry_20260823/results.jsonl), [C48 fresh results](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/out/host_agent/gpt56luna_medium_c48_batch11_reviewed2x4_goldv14_20260823/results.jsonl), [7×3 results](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/out/host_agent/gpt56luna_medium_c48_reviewed_reference7x3_goldv14_20260823/results.jsonl).

---

## 2. first hybrid 40·channel·통제 변수

**사실**

paired base result 배열 비교 결과:

- C47 retry: 8/8 pair 완전 동일
- C48 batch11: 8/8 pair 완전 동일
- C48 reviewed regression: 21/21 pair 완전 동일

모든 최초 search는 결과 수 40, channel은 `tag+fallback:meta`였다. 직접 `msearch`는 `search_only=true`여서 허용되지 않고, 첫 search에서 Meta hybrid가 하위 최대 10칸에 병합된다. 구현은 [agent_tools.py](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/agent_tools.py:1018)에 있다.

fresh 두 qid의 실제 고정 순서는 다음과 같다.

- `v3-offline-0097`:

  `e17406,e17492,e17393,e17395,e17493,e17397,e17404,e17394,e17396,e17402,e17421,e17360,e17378,e17392,e00073,e00184,e00191,e17355,e17357,e17358,e17359,e17361,e17362,e17363,e17482,e31476,e17453,e17463,e17391,e17416,e17469,e17365,e17364,e17366,e17367,e17368,e17369,e17370,e17371,e17372`

- `v3-offline-0289`:

  `e00295,e00190,e00338,e00085,e05248,e00064,e00175,e00189,e00078,e03095,e03172,e03260,e03467,e05154,e05247,e05324,e05367,e05471,e05537,e00137,e05458,e03380,e03375,e05242,e03168,e06329,e29659,e10684,e00205,e00163,e00165,e00177,e00181,e00192,e00218,e00270,e00364,e00370,e00396,e00402`

Manifest 통제도 확인됐다.

- Gold SHA: `dce353...f272b`로 전 run 동일
- fresh qid SHA: `915e11...9c1d` 동일
- elements/tags/JO SHA 동일
- reference graph SHA 동일
- model: `gpt-5.6-luna`
- reasoning: `medium`
- system prompt SHA: `6a1b74...d5b`
- runner SHA: `999b89...14d2`

**차이/불확실성**

- C47과 C48의 seed는 각각 `47011`, `48011`로 다르다. 이는 job 순서에 영향을 주지만 first result에는 영향을 주지 않은 것으로 trace에서 확인된다.
- `agent_tools.py` SHA는 C47 `dd64fd...`, C48 `368b8...`로 다르다. 과거 C47 소스 snapshot은 허용 산출물에 보존되지 않았다.
- 따라서 manifest에 내장된 arm 설정상 두 flag만 다르다는 것은 확인되지만, **역사적 구현 파일 전체가 오직 그 두 변경뿐이었다는 것은 완전 재현할 수 없다.**
- Meta의 `hybrid_search.py`, dense index/model 및 실행환경 fingerprint는 manifest에 없다. 실제 결과 동일성은 trace로 확인되지만 외부 재현성은 불완전하다.

---

## 3. C48 변경, reranker·Gold·hardcoding 감사

### C48 변경

현재 [arms.json](/Users/ralph/Desktop/ᄉᆫ한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/arms.json:924)에서 C47과 C48의 설정 차이는 다음 두 값뿐이다.

- `stop_on_mixed_labels=true`
- `sanitize_region_title=true`

0097 trace에서도 직접 확인된다.

- C47: `e31475~e31482`, 뒤쪽 `표42` 및 잘못된 `# 대상포진 분류표` 포함
- C48: `e31475~e31479`에서 중단, 제목 `참조표 39`

경계 중단은 [agent_tools.py](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/agent_tools.py:214), 제목 정정은 같은 파일 [339행](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/agent_tools.py:339)에 있다.

### reranker

**사실**

- base hybrid 40을 재정렬하는 LLM/non-LLM reranker는 C48에서 추가되지 않았다.
- 하지만 “비LLM reranker가 전혀 없다”는 넓은 의미에서는 틀리다. auxiliary `reference_evidence`는 provenance, query surface 수, source rank, hop, JO를 이용해 결정론적으로 정렬하고 최대 2개를 선택한다. [candidate_order](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/agent_tools.py:274)와 region 내부 정렬이 이에 해당한다.
- 이것은 C47부터 존재하던 auxiliary selector이며 C48가 새로 추가한 것은 아니다.
- `visible_ids()`가 응답 전체를 재귀 탐색하므로 auxiliary ID도 submit 가능하다. [host_agent_runner.py](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/host_agent_runner.py:46).

실제 `v3-offline-0083` C48 trace의 `j07603`은 `base_result_rank=null`이다. 즉 base 40은 그대로여도 **새로운 prompt-visible, submit-compatible auxiliary candidate가 생긴 사례**가 있다.

### Gold/qid/상품·질병 hardcoding

**사실**

- runtime 코드에 특정 qid 문자열은 없다.
- C45/C47/C48은 `router=rule`이며 qid별 qtags를 사용하지 않는다.
- Gold group은 submit 후 scoring에만 사용되고 agent prompt에는 들어가지 않는다. [scoring 호출](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/host_agent_runner.py:297).
- runtime에 상품명·질병명별 사전이나 분기문은 발견되지 않았다.

**과적합 위험**

- C48 테스트는 `표39/표42`, 류마티스, 대상포진이라는 0097의 실제 결함을 그대로 fixture화했다. [test_experiment_arms.py](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/vector_search/noah/0819/test_experiment_arms.py:405).
- 제자리암 direct-region 테스트도 0249와 사실상 동일한 개발 사례다.
- 따라서 literal hardcoding은 아니지만, C48 효능 검증 표본이 개발 fixture와 겹치는 적응적 과적합은 존재한다.

---

## 4. qid별 기능 발화와 승패 원인

### batch11

- `v3-offline-0097`: feature-active. Gold는 `j07564`.
  - C47 retry: C47은 `j07564`를 auxiliary로 보여 4/4 모두 rank 1 제출.
  - C45는 rep0에서 `e31476→j07564` rank 3, rep3에서 `j07564` rank 3으로 2/4 R@5 성공; 나머지 2회 누락.
  - C48 run: C48 4/4 성공. C45는 R@5 0/4, rep2에서만 `j07564`가 rank 6이라 R@10 성공.
  - C45가 같은 qid에서 run별로 2/4와 0/4를 보인 것은 agent 비결정성을 직접 보여준다.

- `v3-offline-0289`: gate는 enabled지만 auxiliary item은 빈 배열이므로 feature-inactive. Gold는 8개 동등 판본 OR group이며 양팔 모두 4/4 성공.

### 7×3 regression

기능 발화 qid는 5개다.

- `0083`: `j07559,j07603` 표시. 지표 변화 없음. `j07603`은 base 40 밖 새 auxiliary candidate.
- `0092`: `j07564,j07556` 표시.
  - rep0 C48 승리: 두 Gold group 모두 제출.
  - rep1 C45 승리: C45는 `j04045`와 `j07564`를 모두 제출했지만 C48은 `j07564`만 제출.
  - qid 평균 R@5/suff는 상쇄되어 동률.
- `0097`: C48가 `j07564`를 첫 제출해 R@1 `0→1`이 3/3 발생. R@5는 양팔 모두 1이라 개선 없음.
- `0249`: `j07557,j07544` 표시.
  - C48 rep0·rep1은 명시적으로 `submit []`.
  - rep2만 `j07557`을 read 후 제출해 성공.
  - 이것이 전체 regression에서 유일한 qid-level R@5/suff 승리다.
- `0274`: `j07563` 표시했지만 agent가 제출하지 않았다.
  - R@5는 양팔 `.5`, suff는 모두 0.
  - R@1은 C45 `.5→` C48 `0`으로 3/3 퇴행했다.

기능 미발화 qid는 `0155`, `0289`이며 지표 변화가 없다.

“hard regression 없음”은 analyzer가 qid 평균 `1→0`만 hard regression으로 정의하기 때문에 형식상 맞다. 그러나 `0274`의 fractional R@1 `.5→0` 퇴행을 숨기는 표현이므로 안전성 서술로는 불충분하다.

---

## 5. 주장할 수 없는 것

**사실**

- C47 관점의 fresh feature-active qid: 1개(`0097`)
- C48 관점의 fresh feature-active qid: **0개**
  - C48은 0097의 C47 trace 결함을 보고 설계·fixture화한 뒤 같은 qid로 평가했다.
- regression feature-active: 5개지만 모두 reviewed/development 회귀층이다.
- 독립 qid 수는 fresh-labelled 2, regression 7에 불과하다. rep는 qid 독립 표본 수를 늘리지 않는다.
- regression empty submit은 C45 1, C48 2다.
- 모델 출력은 비결정적이며 동일 C45/0097의 R@5가 두 run에서 2/4와 0/4로 달라졌다.

**불가능한 주장**

- 전체 R@5 또는 suff `.95` 달성
- 미감사 전체 query 분포에 대한 개선
- 제품/질병 전반의 범용 효과
- 새로운 약관 문서 일반화
- 11만 문서 corpus 일반화
- 속도 개선의 구조적 인과성: model-call 감소와 elapsed 감소는 관측됐지만 표본·비결정성·empty 차이 때문에 일반화 불가
- “candidate를 전혀 추가하지 않는다”: base result에는 맞지만 agent-visible auxiliary 기준으로는 틀림

문서가 말하는 `32,046 elements/7,612 JO의 단일 source document` 수치는 허용 파일 중 [STRUCTURED_SEARCH_EXPERIMENT.md](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/filesearch/STRUCTURED_SEARCH_EXPERIMENT.md:724)에만 있다. 실제 element/JO 원자료가 허용 목록에 없으므로 수량 자체는 독립 재계수하지 못했다. 단일 shared corpus bundle을 사용했다는 점은 manifest로 확인된다.

---

## 6. 다음 최소 실험

C48 코드는 버리지 않되 승격을 보류하고 다음을 수행해야 한다.

1. C45/C48 코드, prompt, model, Meta index와 모든 SHA를 사전 동결한다.
2. `0097`, `0249` 및 테스트에 사용한 모든 질병/표 경계 사례를 제외한다.
3. 최소 3개 이상의 완전히 미사용 source document에서 mixed-label/title-mismatch가 실제 발화하는 새 qid를 retrieval-blind하게 모집한다.
4. 최소 20개 fresh feature-active qid × 3회 paired run으로 mechanism 채택 여부를 먼저 본다.
5. primary metric은 qid-clustered R@5·suff@5, safety는 fatal/protocol/empty 및 R@1 regression으로 사전 등록한다.
6. `.95`를 통계적으로 주장하려면 representative independent qid가 필요하다. 모두 성공한다고 가정해도 성공확률 ≥.95의 단측 95% 하한을 얻으려면 약 59개의 독립 qid가 필요하다. rep를 59개로 세면 안 된다.
7. 11만 문서 일반화는 문서 단위 holdout과 다문서 corpus에서 별도 검증해야 한다.

---

## 7. 문서 C47/C48 절의 오류·과장·누락

[해당 절](/Users/ralph/Desktop/신한라이프/sementic_tag/experiments/tos-skeleton/hybrid-enrich_v2/filesearch/STRUCTURED_SEARCH_EXPERIMENT.md:664)에 다음 정정이 필요하다.

1. **사실 오류:** “C48 feature-active fresh qid가 하나”  
   → C48은 0097 trace로 설계됐으므로 C48 기준 fresh는 0개다.

2. **과장:** “새 candidate나 rank를 만들지 않는다”  
   → base 40의 rank는 보존하지만 auxiliary candidate를 정렬·선택하고 submit 가능하게 한다. `0083/j07603`은 실제 `base_result_rank=null`이다.

3. **과장:** “범용 mechanism”  
   → fresh 효능 표본 0, regression R@5 승리는 개발 fixture 0249 하나의 1/3 성공뿐이다.

4. **불완전한 안전성 표현:** “hard regression 없음”  
   → analyzer 정의상 맞지만 `0274`의 R@1 `.5→0` 및 empty `1→2`를 함께 밝혀야 한다.

5. **지표 누락:**  
   - C48 batch11 C45 R@10/suff@10은 `.625/.625`, C48은 `1/1`.
   - regression R@10은 `.7381→.7857`, suff@10은 `.6190→.6667`.
   - C47 retry R@1은 `.5→1`.

6. **재현성 누락:** historical C47 `agent_tools.py` snapshot, Codex CLI/version, model sampling 설정, Python/runtime 환경, Meta hybrid 코드/index/model SHA가 없다.

7. **검증 불충분:** “42개 회귀 테스트 통과”  
   → 테스트 파일에 `test_*` 메서드가 정확히 42개 있는 것은 확인했으나, 실행 로그·환경·결과 artifact가 없어 통과 여부는 허용 파일만으로 재검증할 수 없다.

결론적으로 C48은 **base retrieval 보존형 UI/evidence 경계 수정으로는 유망하지만, agent 성능 개선안으로 채택할 근거는 아직 부족하다.**
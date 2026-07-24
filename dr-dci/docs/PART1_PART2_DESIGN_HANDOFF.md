# Part 1·2 설계 인수인계

기준일: 2026-07-23  
기준 코드: `feature_ralph`; 실행 시점의 정확한 commit은 결과 manifest의 `git_commit`을 사용한다.

## 1. 이 문서의 목적

이 문서는 Peter의 DR-DCI 실험을 새 Agentic RAG나 DocNav 아키텍처로 재설계하기 위한 문서가 아니다. 설계 담당자는 이미 존재하는 Part 1·2를 아래의 좁은 질문으로 재검증할 수 있도록 실험 계약과 다음 결정을 검토한다.

1. Part 1에서 taxonomy soft score boost가 검색 워크스페이스의 gold coverage를 높이는가.
2. Part 2에서 corpus 규모 증가에 따른 검색기·query rewrite·multi-pull 영향을 분리할 수 있는가.

현재는 외부 모델, 임베딩 endpoint, H200 인스턴스를 실행할 승인이나 재현에 필요한 입력이 완비되지 않았다. 따라서 이 문서의 상태는 **설계·하네스 구현 완료, 실험 실행 대기**다.

## 2. 범위와 불변 조건

### 범위

- Part 1: `baseline` 대 `taxonomy_only` 한 쌍만 비교한다.
- Part 2: 공통 dense retrieval-only scale probe를 먼저 검증하고, 이후 동일 두 agent arm을 20K·50K·110K에서 비교한다.
- 기존 pull backend 비교와 Part 3·4·5 결과는 보존하고 확장하지 않는다.

### 범위 밖

- DocNav, Deep Connect, Docurator, Airflow, 저장소 구조를 변경하지 않는다.
- tags, prefix, metadata를 새 treatment에 추가하지 않는다.
- graph/agent framework를 추가하지 않는다.
- gold/qrel을 검색 입력, taxonomy 생성, 필터 생성에 사용하지 않는다.
- 공개 TREC-COVID 결과를 신한라이프 운영 성능으로 해석하지 않는다.

## 3. 확인된 현재 사실

| 항목 | 확인 결과 | 설계상 의미 |
|---|---|---|
| Git | 현재 작업 브랜치는 `feature_ralph` | 새 작업 브랜치를 만들지 않는다. ahead/behind 값은 변하므로 실행 manifest와 `git status --short --branch`로 확인한다. |
| 통합 PR | PR #2는 open, `feature_ralph_noah_integration` → `dev`, head `e83d044` | 해당 PR을 병합하거나 기준 브랜치를 바꾸지 않는다. |
| Part 1 기존 arm | `taxonomy_only`와 `stack_tax`의 treatment 설정이 동일 | 두 arm을 독립적인 비교로 사용하지 않는다. |
| Part 1 기존 작동점 | taxonomy schema prompt, score boost, workspace taxonomy 탐색이 함께 바뀌었다 | 기존 향상 수치를 taxonomy 단일 효과로 귀속할 수 없다. |
| TREC-COVID | corpus 171,332, queries 50, qrels 66,336, positive-gold documents 17,537 | 작은 표본과 다수 gold 구조를 명시한다. |
| Scale subset | 20K·50K·110K가 nested이며 모든 positive-gold document를 포함 | corpus 규모가 아니라 distractor 수만 바꾸는 controlled scale 조건이다. |
| 원시 결과 | Part 1~4와 기존 scale probe의 raw result JSON이 로컬에 없음 | 문서의 기존 숫자는 보존하되 재현·paired 검정 완료로 표현하지 않는다. |
| 재현 provenance | HF source ID/split은 알지만 immutable revision이 기록되지 않음 | 새 model-backed 실행은 revision을 입력하기 전 차단한다. |
| taxonomy artifact | 20K·50K·110K artifact가 로컬에 없음 | taxonomy treatment는 artifact가 없으면 fail-loud로 중단해야 한다. |
| 신한 PDF | 지정된 PDF 경로가 로컬에 없음 | 다른 PPTX 등으로 추정 대체하지 않는다. |

### 3.1 MIRACL-ko preparation track (separate from TREC focused execution)

MIRACL-ko M0 data preparation and its standalone lexical plumbing smoke are complete as Korean **passage** controlled-distractor-scaling work; they neither replace TREC-COVID nor establish a Shinhan physical-document result. The fixed 20K/50K/110K fixtures preserve judged passages and scale only distractors.

The MIRACL taxonomy artifact contract and synthetic validation harness are implemented in `docs/MIRACL_KO_TAXONOMY_ARTIFACT_CONTRACT_20260723.md`, but an actual 110K taxonomy body has **not** been generated. It is deliberately not registered in `run_experiment.py`, the TREC configuration, the focused Part 1/2 result contract, or an Agent path. A future real artifact must be generated once from 110K passage title/text only, then filter-projected to 50K/20K with shared mappings invariant. It must pass the standalone hash/provenance/coverage audit before a separately instructed MIRACL consumer adapter or passage-specific effect criterion is considered.

## 4. 실제 실행 경로

| 경로 | 목적 | 현재 설계 판단 |
|---|---|---|
| `run_part1()` | Peter의 historical 9-arm stacking | 기록 보존용이다. taxonomy 인과 비교로 쓰지 않는다. |
| `run_part1(..., focused=True)` | baseline 대 taxonomy-only | Part 1의 유일한 신규 비교다. |
| `run_part2()` | historical `stack_all` DR-DCI 대 Hybrid | 복합 처치이므로 Part 1 taxonomy 효과와 합산 해석하지 않는다. |
| `run_part2(..., focused=True)` | 두 arm × 20K·50K·110K | Part 1 screening 이후에만 실행한다. |
| `run_part2_scale_probe()` | original query 1회 dense retrieval × 3 scale | LLM 없이 distractor 증가에 따른 retrieval degradation을 먼저 측정한다. |

## 5. 확정된 Part 1 실험 계약

### 단일 가설

**H1: taxonomy와 일치하는 문서에 대한 soft score boost가 baseline보다 질의별 workspace document gold recall을 높인다.**

| 조건 | Control | Treatment |
|---|---|---|
| taxonomy category schema prompt | 동일하게 제공 | 동일하게 제공 |
| workspace `find()` taxonomy 탐색 | 비활성화 | 비활성화 |
| taxonomy document artifact | score에 사용하지 않음 | score boost에만 사용 |
| 그 외 모델·seed·top-k·turn·workspace cap | 동일 | 동일 |

따라서 treatment의 유일한 변경점은 retriever score의 soft boost다. taxonomy는 hard filter가 아니며, gold는 taxonomy 생성이나 query/pull 입력에 전달되지 않는다.

### 평가와 판정

- TREC agent workspace 주 지표: 질의별 **workspace document gold recall**의 `treatment - control`, paired bootstrap 95% CI
- TREC agent 운영 지표: pull 수, latency, token, workspace 문서 수, 실패·제외 질의 수
- 답변 지표: 유효한 judge 분모에서의 accuracy와 judge error 수
- retrieval-only 공통 dense probe: document Recall@5/20, document Hit@5/10, nDCG@10, P@20, latency
- 신한 전이 평가: chunk Recall@5/20·chunk Hit@5/10은 DocNav chunk/qrel이 제공된 뒤 별도 계약으로 구현한다. TREC corpus-id 문서 지표와 혼용하지 않는다.
- 작동점 telemetry: 각 실제 pull의 boost eligible/positive-score document 수, dense-stage 전후 rank, Top-K 진입·이탈, target cosine의 최소·최대·음수 비율, bounded pull trace
- 최소 실질 효과 크기: macro workspace document gold recall **0.01** (`minimum_practical_effect_version: v1`). 이는 TREC·논문·신한 합의 기준이 아닌 **잠정 내부 screening 기준**이다. config의 크기·버전·`minimum_practical_effect_status: approved`를 모두 명시하기 전에는 focused model-backed 실행을 차단한다. 승인 뒤 CI 하한이 0.01보다 클 때만 긍정 신호로 분류하고, CI 상한이 −0.01보다 작으면 부정 신호로 분류한다. 그 밖은 불확실이다. 기준의 정의나 수치가 바뀌면 version을 올린다.

한 실행의 CI는 질의 간 차이만 반영한다. 긍정 신호가 있어도 비용 승인 전에는 반복 횟수나 새 treatment를 늘리지 않는다.

## 6. 확정된 Part 2 실험 계약

### 6.1 공통 dense retrieval-only probe 우선

동일 query·dense model·retrieval parameter에서 nested 20K/50K/110K corpus만 바꾼다. 이 probe는 taxonomy arm이 없는 공통 baseline이다. 주 지표는 document graded nDCG@10과 document P@20이고, document Recall@5/20·document Hit@5/10·latency는 보조로 기록한다. 주 scale 비교는 **110K−20K**로 고정하고, 20K−50K·50K−110K는 탐색 결과로 분류한다.

이 probe는 검색단 성능 저하를 국소화할 뿐, agent 성능 저하의 원인을 확정하지 않는다. 기존 H200 수치가 문서에 있더라도 raw JSON과 manifest를 회수해 validator를 통과하기 전에는 미검증 historical observation이다.

### 6.2 arm별 agent 단계

Part 1의 두 arm을 각 scale에 적용한다. 이 단계는 arm별 **dynamic multi-pull agent**와 **single-pull static workspace**만 비교한다. 공통 dense probe를 taxonomy arm의 retrieval-only 결과로 표현하지 않는다.

주 scale 비교는 각 arm의 **110K−20K workspace document gold recall** paired delta다. 20K−50K·50K−110K와 scale별 taxonomy−baseline은 탐색 비교다. `dynamic multi-pull - single-pull`도 **pull 횟수의 순수 ablation이 아닌 exploratory interface ablation**이다. single-pull은 별도 LLM 실행이며 system prompt와 첫 pull query가 달라질 수 있다. 따라서 두 arm의 차이를 multi-pull 단독 효과로 부르지 않는다. dynamic 실행 내부에서는 첫 실제 pull 직후의 document gold recall과 최종 workspace recall의 차이를 별도로 기록한다. 이 값도 기술적 진단이지 LLM·prompt를 통제한 인과 효과가 아니다. query rewrite의 영향은 `pull_traces`와 공통 original-query dense probe를 나란히 보되, 이 둘로 taxonomy 인과 효과를 계산하지 않는다. Part 1 screening과 Part 2는 같은 50 query를 쓰므로 Part 2를 독립 재현으로 표현하지 않는다.

공유 문서의 taxonomy 값은 scale마다 같아야 한다. 110K 정본을 만든 뒤 50K·20K를 필터링해 파생하는 방식을 우선하며, 별도 생성 artifact를 쓰면 공유 문서 값의 hash/동일성 검사를 통과해야 한다.

## 7. 이미 구현된 보호 장치

- 선택된 duplicate arm을 blocker로 처리한다.
- 요청된 augmentation이 없으면 baseline으로 무음 강등하지 않고 즉시 실패한다.
- model-backed Part 1·2 실행 전 corpus/query/qrels의 immutable source revision을 검사한다.
- Part 1 focused arm은 prompt schema를 고정하고 workspace taxonomy 탐색을 양쪽에서 끈다.
- single-pull은 실제 첫 pull만 실행하고 attempted pull과 실제 pull을 분리해 기록한다.
- taxonomy boost는 양수 cosine에만 적용한다. dense-stage 전후 rank, Top-K 진입·이탈, score 분포와 bounded pull trace를 보존한다.
- 순위 계측은 `argpartition`으로 뽑은 제한된 candidate 집합에서만 수행한다. baseline 또는 점수가 실제로 바뀌지 않은 pull에는 전후 순위 정렬·rank map을 만들지 않는다. agent raw row에는 전체 latency, rank-telemetry 추가 시간, 그리고 이를 뺀 latency를 함께 기록한다.
- 결과 manifest는 dataset/subset counts, 원본·subset SHA-256, config hash, 코드 commit, 분석 paired-bootstrap seed의 용도, 실제 agent/judge temperature·max tokens·generation seed, model/instruction/control, 실행 환경을 보존한다.
- Part 1 validator는 승인 파일의 raw baseline/taxonomy row에서 manifest의 bootstrap seed와 고정된 **10,000회**로 `compare_result_rows()`를 재실행하고, 기록된 mean/CI·paired 수·판정을 모두 대조한다. 10,000 이외의 반복 횟수는 재계산 전에 차단하므로 저장된 `positive_practical_signal` 문자열이나 과도한 iteration 값만으로는 통과하지 못한다.
- `experiment_contract_sha256`은 runner, retriever, agent, workspace, judge, comparison/판정 코드, retrieval의 BM25/cache/fusion/init, judge prompt, dataset별 taxonomy schema prompt의 파일 hash로 계산한다. mutable approval path가 있는 experiment YAML 자체는 제외하며, Part 2는 Part 1과 현재의 contract hash가 같아야 한다.
- manifest는 실제 `numpy`·`requests`·`PyYAML` 버전을 기록한다. Part 2 gate는 이 세 버전도 Part 1 실행 환경과 정확히 대조한다.
- scale probe와 focused Part 1·2 모두 raw per-query rows, provenance, metric, paired CI와 필수 telemetry 계약이 빠지면 저장 전 실패한다.
- focused Part 2는 승인된 Part 1 result의 path/SHA-256, raw-row 재계산 validator, positive practical signal, focused arm 구성, data·subset·taxonomy artifact hash, model/retrieval control, minimum-effect 크기·버전, execution contract hash를 모두 대조한 뒤에만 query를 로드한다.
- H200 전송은 git pull이 아닌 code/config/manifest bundle만 사용하며 data, key, cache, result를 넣지 않는다.

관련 구현은 `run_experiment.py`, `src/agent/dci_agent.py`, `src/agent/retriever.py`, `src/eval/part12_contracts.py`, `src/eval/scale_probe_contract.py`, `src/eval/part12_result_contract.py`에 있다.

## 8. 설계 담당자가 유지할 해석 원칙

- 기존 Part 1 수치의 상승은 prompt·boost·workspace 탐색이 혼재돼 있어 taxonomy 단일 효과가 아니다.
- historical scale 수치는 삭제하지 않되 raw result 부재 상태에서는 재현된 증거가 아니다.
- TREC-COVID qrel과 현재 코드는 `corpus-id`, 즉 문서 ID 단위다. `workspace_docs`도 문서 ID이므로 모든 현재 TREC endpoint를 document 단위로 표기한다.
- TREC-COVID는 질의당 gold가 많아 document Hit@5/10이 포화될 수 있다. Hit를 주 결론으로 쓰지 않는다.
- 20K에서 gold를 전부 포함한 scale 구성은 운영 corpus의 무작위 확장을 뜻하지 않는다. 검색기 순도 희석을 통제한 probe일 뿐이다.
- DR-DCI의 controlled distractor scaling 원칙은 차용하되 Peter 구현을 논문의 공식 재현으로 표현하지 않는다.
- 최종 근거는 원문 section과 좌표이며, taxonomy·관계·트리는 navigation 보조 신호다.

## 9. 실행 전 필요한 외부 입력

### 신규 TREC focused 실행의 필수 입력

1. `BeIR/trec-covid` corpus/query 및 `BeIR/trec-covid-qrels` test의 immutable revision, subset, split, row count가 든 acquisition manifest
2. Peter가 사용한 20K taxonomy artifact와 생성 model, prompt, source revision, artifact hash. Part 2에는 110K 정본과 20K/50K 파생 규칙도 필요하다.
3. H200/외부 모델 실행에 대한 명시적 승인
4. `0.01` 최소 실질 효과 기준의 명시적 승인(`minimum_practical_effect_status: approved`) 또는 승인된 대체 기준
5. Part 1의 승인된 focused 결과 파일 경로·SHA-256. Part 2는 이 결과가 raw-row 재계산 validator를 통과하고 `positive_practical_signal`이며, 현재 data revision·20K taxonomy artifact·model/retrieval controls·minimum-effect 크기/버전·execution contract hash·`numpy`/`requests`/`PyYAML` 실제 버전과 일치할 때만 실행된다.

### 과거 결과 재해석에만 필요한 입력

- Part 1~4 및 2026-07-22 scale probe의 원시 result JSON과 당시 코드 commit/config

### 신한 전이 검증에만 필요한 입력

- 지정된 신한 PDF 파일 또는 정확한 접근 경로, 그리고 DocNav chunk/qrel 계약

## 10. 재현과 인수 기준

무비용 검증:

```bash
cd /Users/donggyu/Documents/논문/PageIndex/experiments/dr-dci
PYTHONPATH=. python -m unittest discover -s tests -v
PYTHONPATH=. python scripts/audit_part12.py --step baseline --size 20000
PYTHONPATH=. python scripts/audit_part12.py --step taxonomy_only --size 20000
bash scripts/package_part12_h200_bundle.sh /private/tmp/dr-dci-part12.tar.gz
```

현재 taxonomy audit이 `blocked`로 끝나는 것은 정상이다. 누락 artifact 경로를 명시해야 하며, 그 상태에서 모델을 호출해서는 안 된다.

외부 입력과 승인이 갖춰진 뒤에만 다음 순서로 진행한다.

1. acquisition manifest와 taxonomy artifact를 추가하고 preflight를 통과시킨다.
2. retrieval-only scale probe를 실행하고 raw result validator를 통과시킨다.
3. 승인된 최소 실질 효과 기준을 config에 기록한 뒤 baseline 대 taxonomy-only Part 1을 한 번 실행한다.
4. paired 결과와 telemetry를 검토해 H1의 방향만 판정한다.
5. Part 1 결과를 명시 승인하고 경로·SHA-256을 `part2_scaling.approved_part1_result`에 기록한다. Part 2는 raw row 재계산으로 positive signal을 확인하고 data/artifact/control·criterion version·execution contract 정합성을 검증한 뒤에만 시작한다.

```yaml
parts:
  part2_scaling:
    approved_part1_result:
      status: "approved"
      path: "/approved-results/part1_taxonomy_focused.json"
      sha256: "<64-character SHA-256>"
```

Part 1 결과 파일의 byte가 바뀌거나 현재 input/control이 달라지면 이 gate는 실패한다. 결과 파일이나 고객 데이터를 코드 bundle에 포함하지 않는다.

## 11. 설계 검토에서 확인할 결정

- H1의 최소 실질 효과 크기 0.01 macro workspace document gold recall을 승인할지. 현재는 잠정 내부 기준이며, 승인 또는 대체 기준을 config와 근거 commit으로 남긴다.
- Part 1 screening의 최소 유효 질의 수와 실패 query 처리 규칙을 실행 전에 확정할지
- Part 1 양성 신호에 필요한 반복 실행 수와 비용 상한을 별도 승인 항목으로 둘지
- raw historical result가 회수되지 않을 경우, 기존 수치를 참고 부록으로만 남길지

이 네 결정 외에 새로운 retrieval·agent 기법을 추가하는 제안은 Part 1·2 범위 밖으로 보류한다.

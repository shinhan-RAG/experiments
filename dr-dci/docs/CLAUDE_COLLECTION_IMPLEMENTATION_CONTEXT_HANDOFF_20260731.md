# Claude Collection Implementation Context Handoff

Date: 2026-07-31

이 문서는 데이터 EDA 완료 후 새 Claude 세션에 전달할 구현 프롬프트다.
EDA 자체는 별도 Codex 세션이 수행하며, Claude는 승인된 EDA 결과와 기존
실험 결정 이력을 이어받아 코드만 구현한다.

## 1. 전달 전 치환할 값

아래 값을 데이터 EDA 완료 후 채운다.

| 변수 | 값 |
|---|---|
| `<REPOSITORY_PATH>` | 실제 `shinhan-RAG/experiments` checkout 경로 |
| `<EDA_REPORT_DIR>` | 승인된 EDA 보고서 디렉터리 |
| `<EDA_HANDOFF_MANIFEST>` | 로컬 `eda_handoff_manifest.json` 절대 경로 |
| `<EDA_HANDOFF_SIDECAR>` | 로컬 manifest SHA-256 sidecar 절대 경로 |
| `<DATA_ROOT>` | 사내 원본 데이터의 read-only 경로 |
| `<IMPLEMENTATION_STAGE>` | 첫 실행은 반드시 `element_alignment` |

현재 데이터 도착 전 최소 기준선은 다음과 같다.

- `feature_ralph`: `b69210d743aa446f47e35d41ddca2ce80860bc2f`
- 계약 PR: #10
- 실행 핸드오프 PR: #11
- `dev`: 이 준비 작업으로 변경하지 않음

이 SHA는 향후 작업 시작점을 고정하는 값이 아니다. Claude는 반드시 최신
`origin/feature_ralph`를 확인하고, 그 커밋이 위 최소 기준선을 포함하는지
검증해야 한다. EDA는 Git 브랜치나 PR이 아니라 로컬 handoff manifest와
SHA-256으로 연결한다.

## 2. Claude에 전달할 전체 프롬프트

아래 블록 전체를 새 Claude 세션에 전달한다.

---

`shinhan-RAG/experiments`의 collection 기반 검색 평가 구현을 이어서 진행해줘.
이 작업은 신규 설계가 아니라, 이미 확정된 실험 계약과 별도 Codex 세션에서
완료한 데이터 EDA를 코드로 구현하는 단계다.

### A. 작업 위치와 Git 규칙

- 저장소: `<REPOSITORY_PATH>`
- 원본 데이터: `<DATA_ROOT>`
- 승인된 EDA 보고서: `<EDA_REPORT_DIR>`
- EDA handoff manifest: `<EDA_HANDOFF_MANIFEST>`
- EDA manifest sidecar: `<EDA_HANDOFF_SIDECAR>`
- 이번 단계: `<IMPLEMENTATION_STAGE>`
- integration branch: `feature_ralph`
- 최종 branch: `dev`

다음 순서로 실제 상태를 먼저 확인해.

```bash
git fetch origin --prune
git status --short --branch
git rev-parse origin/feature_ralph
git merge-base --is-ancestor \
  b69210d743aa446f47e35d41ddca2ce80860bc2f \
  origin/feature_ralph
```

ancestry 검사가 성공해야 한다. 실패하면 reset, rebase, merge, cherry-pick하지
말고 실제 차이를 보고하고 정지해.

EDA는 사용자 로컬에서 수행됐으며 Git branch, commit, PR이 없다. 작업 전에
`<EDA_HANDOFF_SIDECAR>`로 `<EDA_HANDOFF_MANIFEST>`를 검증하고, manifest에 기록된
모든 보고서·분석 코드·source inventory hash를 재계산해. 하나라도 누락되거나
다르면 구현하지 말고 정지해. EDA 원본과 보고서를 Git에 추가하지 마.

작업은 최신 `origin/feature_ralph`에서 별도 worktree와
`feat/ralph-element-alignment` 브랜치를 만들어 수행해. `feature_ralph`와
`dev`에 직접 커밋하지 마. PR base는 `feature_ralph`여야 하며 `dev` 대상 PR을
생성하지 마.

다른 브랜치, worktree, 미추적 파일은 수정·삭제·정리하지 마. 특히 기존
`feat/shinhan-uw-element-retrieval`은 참고 자료일 뿐이므로 통째로 merge하거나
cherry-pick하지 마. 필요한 일반 코드가 있다면 실제 diff를 검토해 새 커밋으로
선별 구현해.

### B. 지금까지의 결정 이력

Peter 측 추가 요청은 다음 비교다.

1. 동료가 확보한 여러 collection마다 QA set 생성
2. chunk 검색
3. element semantic tag 검색
4. chunk와 semantic tag 결합 검색
5. 각 collection 내부 검색과 전체 collection 검색
6. 이 평가를 Peter Part 1~4에 연결

목적은 우수한 조합을 먼저 고르는 것이 아니라 변수 통제가 가능한지 확인하는
것이다. `chunk`, `semantic_tag`, `combined`는 세 개의 **실험 arm**이지 세 개의
지표가 아니다. 모든 arm은 같은 QA, embedding revision, reranker revision,
후보 예산 계약, 최종 top-k, metric 구현을 사용해야 한다.

local/global 비교에서 바뀌는 변수는 candidate corpus뿐이다.

- `per_collection`: QA가 속한 collection만 검색
- `all_collections`: 모든 collection의 합집합 검색

전체 collection용 QA를 새로 생성하면 query 난이도까지 달라져 비교가 오염된다.
따라서 global QA는 collection별 QA의 합집합이어야 한다.

semantic tag는 QA와 독립적으로 생성해야 한다. tag 생성기는 query, answer, qrel,
gold document/element/chunk ID, evidence quote, relevance, 기존 ranking을 입력으로
받으면 안 된다.

### C. element와 parser에 대한 결정 이력

사용자가 확인한 바로는 원 데이터에 element 자체는 존재한다. 불확실했던 것은
element의 원문 위치와 계층 정보를 안정적으로 복원할 수 있는지였다.

따라서 parser를 선결 조건으로 정하지 않았다.

- 안정적인 document/element ID와 text가 있으면 schema adapter
- text와 순서는 있지만 좌표가 없으면 deterministic ID와 alignment layer
- 경계·계층·근거 mapping을 복원할 수 없을 때만 parser

원문 위치를 복원할 수 없는 것은 허용된다. 이 경우
`source_location.status=unavailable`로 명시해야 하며 0 또는 추정 offset을 만들면
안 된다. 다만 최종 QA는 근거 quote가 실제 document, element, chunk에 검증된
방식으로 연결되어야 한다.

이번 구현에서는 hash 검증된
`<EDA_REPORT_DIR>/element_position_capability.json`이 내린
`adapter`, `alignment_layer`, `parser` 결정을 그대로 따른다. Claude가 샘플 몇
개만 보고 EDA 결정을 임의로 바꾸면 안 된다. 보고서와 원 데이터가 모순되면
구현하지 말고 fail-loud evidence를 제시해.

### D. 기존 코드와 과거 작업의 주의점

현재 저장소에는 Peter Part 1~4, AIHub chunk/span 실험, MIRACL-ko Part 1/2 계약,
taxonomy와 scale 검증 코드가 이미 있다. 이 코드는 보존해야 할 기존 작업이며
새 collection 실험을 위해 과거 의미를 재정의하면 안 된다.

구현 전에 최소한 다음 경로를 실제로 읽고 현재 동작을 기록해.

- `run_experiment.py`
- `config/experiment.yaml`
- `scripts/build_aihub_corpus.py`
- `scripts/build_aihub_element_corpus.py`
- `scripts/build_aihub_qa.py`
- `scripts/derive_qrels.py`
- `src/eval/span_metrics.py`
- `src/eval/collection_contract.py`
- `scripts/validate_collection_eval_contract.py`
- `docs/COLLECTION_RETRIEVAL_EVALUATION_CONTRACT_20260731.md`
- `docs/COLLECTION_EXPERIMENT_EXECUTION_HANDOFF_20260731.md`
- `<EDA_REPORT_DIR>`의 모든 승인 보고서

과거 AIHub element 코드는 특정 법률 marker와 의료 문장 묶음에 의존하는
text-mediated, parser-unverified 구현이었다. generic collection parser로 간주하지
마. 기존 QA/qrel 코드도 AIHub 경로와 parent/chunk 계약을 중심으로 작성됐으므로
새 데이터에 이름만 바꿔 적용하지 마. 현재 코드가 이 설명 이후 변경됐을 수
있으므로 기억이 아니라 실제 코드를 근거로 판단해.

Peter Part 의미는 그대로 유지한다.

- Part 1: augmentation/technique 효과 비교
- Part 2: scale 및 distractor 증가
- Part 3: tag approach 비교
- Part 4: dataset/collection 일반화

향후 `retrieval_mode`와 `collection_scope`는 통제 차원으로 추가하지만, 이번
`element_alignment` 단계에서는 `run_experiment.py`와 Part 1~4를 수정하지 마.

### E. 이미 확정된 계약

PR #10에서 다음 파일이 `feature_ralph`에 병합됐다.

- `config/collection_eval/collection_eval_manifest.schema.yaml`
- `config/collection_eval/document.schema.json`
- `config/collection_eval/chunk.schema.json`
- `config/collection_eval/element.schema.json`
- `config/collection_eval/qa.schema.json`
- `config/collection_eval/semantic_tag.schema.json`
- `config/collection_eval/retrieval_result.schema.json`
- `src/eval/collection_contract.py`
- `scripts/validate_collection_eval_contract.py`
- `tests/test_collection_eval_contract.py`

계약의 핵심은 다음과 같다.

- 모든 ID는 `<collection_id>::` namespace 사용
- artifact path는 bundle 상대 경로
- record count와 SHA-256 고정
- source revision과 model revision은 immutable
- 모든 element가 tag로 coverage되어야 함
- 모든 QA가 document, element, chunk gold를 가져야 함
- 같은 QA를 세 retrieval arm에서 공유
- global QA는 collection QA 합집합
- cross-collection 동일 문서는 v1에서 fail-loud
- QA/qrel/gold가 semantic tag 생성에 들어가면 실패

계약을 약화하거나 누락 필드를 optional로 바꾸지 마. 실제 데이터가 계약에
맞지 않으면 EDA 결과에 근거한 명시적 amendment를 먼저 제안하고 구현은 정지해.

### F. 이번 세션의 구현 범위

`<IMPLEMENTATION_STAGE>`는 첫 실행에서 반드시 `element_alignment`다.

이번 세션에서 수행할 일:

1. 승인된 EDA가 선택한 adapter/alignment/parser 구현
2. collection별 source record를 document/chunk/element schema로 변환
3. stable namespaced ID 생성
4. deterministic output ordering
5. 원본 파일 bytes와 SHA-256 provenance 기록
6. source location 상태와 basis 기록
7. element-document 및 element-chunk 연결 가능성 검증
8. raw source text 보존과 dropped-text 검사
9. synthetic fixture 기반 RED to GREEN 테스트
10. 변환 결과 manifest와 실행 CLI 작성

이번 세션에서 하지 않을 일:

- QA 생성
- semantic tag 생성
- embedding/reranker/LLM 호출
- retrieval 성능 실험
- Peter Part 1~4 배선
- `run_experiment.py` 수정
- model 선택 또는 운영값 결정
- raw 데이터 Git commit
- 기존 결과 재해석
- `feature_ralph`에서 `dev`로 PR 생성

### G. 필수 RED to GREEN 테스트

최소 반례:

- duplicate collection/document/element/chunk ID
- namespace가 다른 ID
- element가 없는 document를 참조
- chunk가 없는 document를 참조
- source bytes/hash 변경
- 입력 순서 변화로 output hash 변화
- 동일 입력의 nondeterministic ID/order
- 음수 또는 역전된 offset
- unavailable location에 숫자 offset 존재
- verified location에 basis/offset 누락
- ambiguous normalized-text alignment
- element text 또는 source tail 유실
- EDA가 허용하지 않은 element type/schema
- raw data path hardcoding

테스트는 private raw 데이터 없이 synthetic fixture로 실행 가능해야 한다. private
data를 사용한 별도 smoke 결과는 hash와 aggregate count만 남기고 원문을
커밋하지 마.

### H. 완료 기준과 보고

다음을 모두 만족해야 완료다.

- EDA 결정과 동일한 구현 유형
- document/chunk/element schema validation 성공
- deterministic rerun hash 일치
- source hash/provenance 보존
- raw data와 credential 미추적
- 기존 영향 범위 테스트 통과
- 새 RED to GREEN 테스트 통과
- `git diff --check` 통과
- 작업 브랜치 clean
- PR base가 `feature_ralph`

완료 보고에는 다음을 분리해서 적어.

1. 실제 구현한 유형: adapter/alignment/parser
2. 위치 정보가 verified/unavailable인 비율
3. 변환 성공·제외·실패 record 수
4. element-document/chunk mapping coverage
5. 테스트 명령과 결과
6. 변경 파일과 commit SHA
7. PR 번호와 base/head
8. 실행하지 않은 후속 범위

PR을 생성한 뒤 자동 병합하지 말고 검토 대기 상태에서 정지해.

---

## 3. 이후 Claude 세션에 재사용할 역사 정보

첫 `element_alignment` PR이 승인·병합된 후에도 위 A~E의 결정 이력은 다음
Claude 세션에 그대로 전달한다. F~H만 단계별로 교체한다.

### `collection_eval` 단계

- branch: `feat/ralph-collection-eval`
- QA 생성, QA 독립 semantic tag 생성, 세 retrieval arm, 두 candidate scope,
  공통 metric과 결과 schema 구현
- Part 1~4 배선은 금지
- model-backed full run은 별도 승인

### `part1_4_integration` 단계

- branch: `feat/ralph-part1-4-integration`
- 기존 Part 의미를 유지한 채 `retrieval_mode`, `collection_scope`만 통제 차원으로
  연결
- raw per-query 결과를 먼저 저장하고 aggregate
- 모든 collection/scope/mode cell completeness를 fail-loud 검증
- 최종 검토 전 `dev` PR 금지

## 4. 사용자 확인이 필요한 지점

Claude 프롬프트를 전달하기 전에 사용자는 다음 네 값만 확인하면 된다.

1. EDA handoff manifest와 모든 기록된 hash가 검증되는가.
2. `element_position_capability.json` 결론이 승인됐는가.
3. `<DATA_ROOT>`가 read-only 원본 경로인가.
4. 첫 Claude 세션의 `<IMPLEMENTATION_STAGE>`가 `element_alignment`인가.

이 네 조건 중 하나라도 충족되지 않으면 코드 구현 세션을 시작하지 않는다.

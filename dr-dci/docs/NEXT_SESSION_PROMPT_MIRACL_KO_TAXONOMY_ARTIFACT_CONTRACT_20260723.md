# MIRACL-ko Taxonomy Artifact Contract 작업 지시서

MIRACL-ko taxonomy artifact 생성 계약 설계를 진행해줘.

## 1. 시작 상태

먼저 실제 Git 상태를 확인한다.

- 저장소: `/Users/donggyu/Documents/논문/PageIndex/experiments/dr-dci`
- 작업 브랜치: `feature_ralph`
- 기대 HEAD: `8e12cd1`
- 기대 상태: `origin/feature_ralph` 대비 25커밋 ahead
- 사용자 미추적 파일:
  `docs/AIHUB_MIRACL_DATASET_SELECTION_HANDOFF_20260723.md`
  `docs/NEXT_SESSION_PROMPT_MIRACL_KO_TAXONOMY_ARTIFACT_CONTRACT_20260723.md`

상태가 다르면 기대 커밋을 강제로 적용하지 말고 실제 상태와 차이를 먼저
보고한다.

사용자 미추적 파일은 삭제하거나 커밋하지 않는다. 이 지시서도 작업 산출물로
수정하지 않는다. clean checkout 검증이 필요하면 별도 clean worktree를 사용한다.

## 2. 현재 확정된 상태

MIRACL-ko Gate M0와 standalone lexical plumbing smoke는 완료됐다.

- 데이터 단위: passage
- fixture: 결정론적 20K/50K/110K
- 전체 judged passage와 positive passage 보존
- lexical backend: Anserini 2.1.1 / Lucene CJKAnalyzer
- 기존 raw 결과와 baseline은 변경하지 않는다.
- lexical 검색을 재실행하지 않는다.
- 기존 수치는 controlled distractor scaling 결과로만 해석한다.
- taxonomy, Agent, LLM, focused Part 1/2는 아직 실행하지 않았다.

정본 문서:

- `docs/MIRACL_KO_PRETEST_20260723.md`
- `docs/MIRACL_KO_LEXICAL_SMOKE_20260723.md`
- `docs/MIRACL_KO_LEXICAL_SMOKE_REVALIDATION_20260723.md`
- `docs/PART1_PART2_DESIGN_HANDOFF.md`
- `docs/PART1_PART2_FOCUSED_EXPERIMENT.md`

근거:

- MIRACL 공식 저장소 및 논문
- DR-DCI: <https://arxiv.org/html/2606.14885v1>
- 현재 저장소의 실제 taxonomy consumer와 score-boost 구현

DR-DCI가 taxonomy soft boost의 효과를 입증한 논문이라고 표현하지 않는다.
이번 taxonomy 효과는 이후 단일변수 A/B로 별도 검증해야 한다.

## 3. 이번 작업 목표

이번 작업은 taxonomy를 생성하거나 성능 실험을 실행하는 작업이 아니다.

목표는 MIRACL-ko taxonomy artifact가 다음 실험에서 누출 없이,
결정론적으로, 재현 가능하게 사용되도록 입력, 출력, provenance, 검증
계약을 코드와 하네스로 만드는 것이다.

새 retrieval 기법을 추가하거나 실험 아이디어를 확장하지 않는다.

## 4. Gate 0: 현행 코드 대조

수정 전에 다음을 코드로 확인해 문서에 기록한다.

- taxonomy artifact를 읽는 실제 코드 경로
- passage와 taxonomy label의 연결 키
- query taxonomy를 만드는 경로
- taxonomy soft boost가 적용되는 정확한 위치
- 누락 artifact가 fail-loud인지
- 기존 TREC taxonomy artifact schema
- 20K/50K/110K에서 taxonomy가 어떻게 재사용되는지
- `taxonomy_only`와 control 사이에서 실제로 달라지는 변수

기억이나 과거 보고가 아니라 코드 위치를 근거로 작성한다.

## 5. 데이터 누출 금지 계약

taxonomy artifact 생성 입력으로 허용하는 정보:

- `corpus_id`
- `title`
- `text`
- MIRACL 고정 revision
- subset/data manifest와 SHA-256

`corpus_id`는 원문 passage와 taxonomy mapping을 연결하는 불투명한 키로만
허용한다. label의 의미 특징을 만들기 위해 ID 문자열, 접두어, article 번호,
namespace를 tokenization, embedding, clustering 또는 label naming 입력으로
사용하면 안 된다. 결정론적 출력 정렬과 충돌 해소에 ID가 필요하면 그 용도와
영향을 manifest에 별도로 기록한다.

다음 정보는 생성, 명명, 분류 입력으로 사용하면 안 된다.

- query
- qrel
- relevance
- positive/negative 판정
- answer
- gold/evidence ID
- gold passage 목록
- 평가 결과
- 기존 검색 순위

여기서 금지하는 ID는 relevance를 알려 주는 gold/evidence 식별자다.
원문 연결을 위한 불투명 `corpus_id`까지 금지한다는 뜻은 아니다.

금지 필드를 artifact에서 제거하는 수준에 그치지 않는다. generator API와
실행 manifest에서 해당 경로를 입력받지 못하도록 막는다.

query taxonomy 분류가 이후 필요할 경우에도 query text만 입력으로 허용하며
qrel과 gold는 금지한다.

## 6. Scale 계약

scale마다 taxonomy를 독립적으로 다시 생성하면 taxonomy 자체가 달라져
scale 효과와 taxonomy 변화가 혼재할 수 있다. 기본 계약은 다음 하나로
고정한다.

- 고정된 110K fixture의 `title`과 `text`만으로 taxonomy를 한 번 생성한다.
- 20K와 50K artifact는 110K mapping을 `corpus_id` 집합으로 filter projection한다.
- 하위 scale을 위해 taxonomy를 재학습하거나 label을 재명명하거나 passage를
  재배정하지 않는다.
- 공통 passage의 taxonomy ID, label, score는 20K/50K/110K에서 불변이어야 한다.
- projection 결과의 행 수와 ID 집합은 해당 scale manifest와 정확히 일치해야 한다.

이는 controlled distractor scaling에서 taxonomy 처치를 고정하기 위한
전이적(transductive) 전처리다. 따라서 20K 결과를 "20K corpus만으로 학습한
taxonomy"로 표현하면 안 된다. 110K 밖의 전체 MIRACL corpus 정보는 사용하지
않으며, 독립 scale별 재학습은 이번 실험 계약에서 금지한다.

## 7. Artifact 계약

최소한 다음을 정의한다.

- `schema_version`
- `taxonomy_artifact_id`
- dataset/revision
- `retrieval_unit=passage`
- generator type과 version
- generator code SHA-256
- model/tokenizer 또는 알고리즘 버전
- prompt/template가 있다면 해당 hash
- seed와 전체 parameter
- 입력 corpus/subset manifest SHA-256
- label taxonomy와 stable label ID
- `passage_id`에서 label ID 또는 score로의 연결
- 생성 시각
- artifact SHA-256
- 허용 입력 필드와 금지 입력 필드
- unknown/unassigned 처리
- 중복 passage, 중복 label, 고아 passage 처리
- 110K 생성 및 20K/50K filter projection 방식

label 이름이 자유형 LLM 출력이라면 동일 입력 재생성 결정론을 주장하지
않는다. 생성 모델, prompt, 응답 artifact와 비용 승인 여부를 명확히
기록한다.

## 8. 품질 사전검사

성능 실험 전에 artifact 자체에 대해 다음을 검사한다.

- 전체 passage coverage
- unknown/unassigned 비율
- label 수와 분포
- 최대/최소 label 크기
- 심한 불균형
- 중복/충돌 mapping
- 존재하지 않는 passage ID
- 공통 passage의 scale 간 label 불변성
- 20K/50K mapping이 110K mapping의 정확한 부분집합인지
- projection 결과의 passage ID 집합이 각 scale manifest와 정확히 같은지
- 동일 입력 재생성 시 artifact hash 결정론
- taxonomy label에 query, qrel, gold 정보가 유입되지 않았는지
- `corpus_id`가 semantic feature나 label naming 입력으로 유입되지 않았는지
- taxonomy consumer가 누락 artifact를 baseline으로 무음 강등하지 않는지

이 수치는 taxonomy 검색 성능으로 표현하지 않는다.

## 9. RED -> GREEN 하네스

구현 전에 실패 반례를 먼저 작성한다.

필수 반례:

- query 또는 qrel 경로가 generator 입력에 들어오면 실패
- relevance 또는 gold 필드가 들어오면 실패
- corpus revision 또는 SHA가 다르면 실패
- 누락 taxonomy artifact가 baseline으로 강등되지 않고 실패
- passage ID 중복 또는 고아 mapping 실패
- 공통 passage의 scale 간 label 변경 실패
- 20K/50K projection에 110K 밖 passage가 있거나 scale passage가 누락되면 실패
- semantic generator 입력에 `corpus_id` 또는 gold/evidence ID가 들어오면 실패
- seed, generator version, parameter 누락 실패
- artifact 내용 변조 시 SHA 불일치 실패
- 동일 입력에서 결정론적 generator 결과가 달라지면 실패
- control과 treatment의 차이가 taxonomy score boost 외에 생기면 실패

실제 taxonomy 생성 모델이나 외부 API는 호출하지 않는다. synthetic fixture
또는 mock artifact로 계약을 검증한다.

## 10. 산출물

- taxonomy artifact 계약 문서
- schema/validator 구현
- manifest 및 provenance 계약
- synthetic example artifact
- audit 또는 validation CLI
- RED -> GREEN 단위 테스트
- 이후 실제 artifact 생성에 필요한 재현 명령
- `PART1_PART2_DESIGN_HANDOFF.md`의 관련 상태 갱신

Git에는 schema, validator, manifest, synthetic fixture와 재현 명령만 추적한다.
향후 생성할 실제 110K taxonomy artifact 본문은 크기와 재생성 가능성을 고려해
Git 비추적 경로에 두고, 추적 manifest에 경로 규약, byte size, SHA-256,
generator/provenance와 20K/50K projection hash를 기록한다. 실제 artifact가
없거나 manifest hash와 다르면 consumer와 실행 preflight는 fail-loud 해야 한다.

권장 문서명:

`docs/MIRACL_KO_TAXONOMY_ARTIFACT_CONTRACT_20260723.md`

실제 파일명은 저장소 관례를 우선한다.

## 11. 금지 범위

- Anserini 재실행
- taxonomy 실제 생성
- embedding 생성
- 외부 LLM/API 호출
- taxonomy boost 성능 A/B
- Agentic RAG 실행
- focused Part 1/2 실행
- 기존 raw/result 파일 수정
- 0.01 기준 승인
- push, PR, tag 생성
- 사용자 미추적 파일 변경
- 관련 없는 리팩터링

## 12. 완료 보고

다음을 구분해 보고한다.

- taxonomy 계약 코드 구현 완료 여부
- synthetic/mock 하네스 검증 여부
- 실제 MIRACL taxonomy artifact 생성 여부
- taxonomy 성능 A/B 실행 여부
- Part 1/2 실행 여부

전체 테스트, 변경 파일, 커밋, 작업 트리 상태를 보고한다. 실제 artifact와
성능 실험은 미실행 상태로 남아야 한다.

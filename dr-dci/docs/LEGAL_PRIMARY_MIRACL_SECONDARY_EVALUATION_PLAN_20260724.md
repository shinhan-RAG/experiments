# AIHub 법률 primary · MIRACL-ko secondary 평가 계획

기록일: 2026-07-24  
상태: **사전 평가 위계 기록 — 실행 미승인**

## 1. 변경 시점과 사전성

이 문서는 MIRACL-ko taxonomy artifact 생성, taxonomy A/B, focused Part 1·2와 AIHub 법률 primary 결과가 모두 **미실행**인 상태에서 기록한다. 따라서 결과를 본 뒤 유리한 데이터셋으로 성공 기준을 바꾼 결정이 아니다.

변경 대상은 최종 성공 판정과 결과 해석의 위계뿐이다. 기존 MIRACL-ko fixture, taxonomy generation contract, standalone audit, focused Part 1·2 gate와 하네스는 이 문서로 수정하지 않는다.

## 2. 평가 역할

| 데이터셋 | 역할 | 검색·평가 단위 | 허용 해석 |
|---|---|---|---|
| AIHub 한국어 법률 | **primary confirmatory target-domain evaluation** | parent document | 법률 질의에서 source-parent gold 회수가 개선되는지의 최종 주 판정 |
| MIRACL-ko | method/development benchmark 및 secondary robustness evidence | passage | 일반 도메인 controlled distractor scaling에서 방법·강건성의 보조 근거 |

두 데이터셋의 점수·질의·raw row·CI를 평균·합산하지 않는다. 한쪽의 높은 수치로 다른 쪽의 실패를 덮거나, 결과가 유리한 데이터셋만 선택해 성공을 선언하지 않는다.

## 3. AIHub 법률 primary 가설과 해석 경계

주 가설은 다음이다.

> 고정된 taxonomy soft boost가 AIHub 한국어 법률 질의에서 원천 gold 판례의 parent-document 회수 성능을 baseline보다 개선하는가.

현재 AIHub qrel은 법률 전반의 relevance judgment가 아니라 source-parent 연결이다. 그러므로 이 primary 평가는 “전반적인 법률 관련성 향상”을 주장하지 않는다. 결과는 고정된 법률 query 집합에서 source-parent gold를 회수하는 범위로만 해석한다.

## 4. 최종 판정 위계

| AIHub 법률 primary | MIRACL-ko secondary | 최종 해석 |
|---|---|---|
| 성공 | 성공 | 법률 target-domain 효과와 제한적 일반-domain 강건성을 함께 지지 |
| 성공 | 실패 | 법률 데이터 한정 효과로 기록 |
| 실패 | 성공 | 목표 도메인 가설은 실패이며 MIRACL 보조 결과만 유지 |
| 실패 | 실패 | 가설 기각 |

MIRACL-ko 성공은 AIHub 법률 primary 실패를 구제하지 않는다.

## 5. 기존 MIRACL 계획의 유지 범위

- MIRACL-ko 20K/50K/110K fixture, taxonomy generation execution contract, standalone artifact audit, taxonomy consumer/A-B 절차, focused Part 1→Part 2 gate는 현재 기술 계약을 유지한다.
- MIRACL-ko 결과는 AIHub 법률 primary의 성공 판정값으로 사용하지 않는다.
- MIRACL 내부 Part 1 결과가 positive practical signal일 때만 Part 2를 허용하는 기존 gate도 유지한다.
- MIRACL-ko 결과의 단위는 passage이며, AIHub 법률 parent-document 결과와 절대 점수를 직접 비교하지 않는다.

## 6. Gate L0 — AIHub 법률 primary 실행 전 선행 조건

대상 경로:

```text
/Users/donggyu/Downloads/data/dr-dci_정답셋/aihub-full
```

Gate L0는 현재 **미완료**다. 다음을 결과 확인이나 model-backed 실행 전에 완료·승인해야 한다.

1. 원본 corpus를 재생성하고 immutable source·byte hash manifest를 고정한다.
2. 110K parent-document subset과 subset hash를 고정한다.
3. query/qrel referential integrity를 검증한다.
4. query leakage 148건의 처리 규칙을 결과 보기 전에 확정한다.
5. mismatch 84건, duplicate content 315건, duplicate ID 5건, PII 가능성 89건을 재검증하고 처리·제외 여부를 기록한다.
6. taxonomy 입력에서 query, qrel, gold, 판례 연결 ID, 정답 label 필드를 제외한다.
7. 라이선스와 내부 사용 범위를 확인한다.
8. exclusion 목록과 최종 query 집합 SHA-256을 결과 보기 전에 고정한다.

L0가 끝날 때까지 법률 corpus/qrel은 결과를 확인하지 않는 held-out 상태로 유지한다.

## 7. 법률 primary 설정 동결

MIRACL 개발·방법 검증 이후 법률 primary에서 다음 값을 임의로 재튜닝하지 않는다.

- taxonomy generator 알고리즘과 prompt
- boost 계수
- retrieval backend/model
- Top-K와 탐색 budget
- bootstrap seed와 iteration 수
- 성공 판정 기준

법률 corpus에는 별도 taxonomy artifact를 생성한다. 다만 taxonomy generator 알고리즘과 실행 제어는 위 동결값을 그대로 사용한다. 법률 primary metric과 minimum practical effect 기준은 L0 및 primary protocol 승인 전에 별도로 승인하며, MIRACL passage 기준을 자동 승계하지 않는다.

## 8. 결과·provenance 분리

- MIRACL-ko: passage 단위 metric, raw rows, manifest, CI, provenance를 별도 보존한다.
- AIHub 법률: parent-document 단위 metric, raw rows, manifest, CI, provenance를 별도 보존한다.
- 데이터셋 간 절대 점수, latency, 최소 실질 효과 기준을 직접 비교하거나 합산하지 않는다.
- 법률 primary 결과는 source-parent qrel 계약의 범위로만 보고한다.

## 9. 승인된 순서

1. 현재 MIRACL-ko 계획을 기존 계약 안에서 계속 수행한다.
2. AIHub 법률 corpus/qrel은 held-out 상태로 유지한다.
3. MIRACL taxonomy contract와 개발 설정을 동결한다.
4. AIHub Gate L0와 legal primary protocol을 별도 승인한다.
5. 승인된 법률 primary를 한 번 실행한다.
6. 이 문서의 판정 위계에 따라 결과를 해석한다.

## 10. 현재 비결과와 다음 입력

- AIHub 법률 primary protocol, taxonomy artifact, A/B, Agent, Part 1·2: **미실행**.
- MIRACL taxonomy artifact, A/B, Agent, focused Part 1·2: **미실행**.
- 이 문서는 법률 primary 성공·일반화·운영 성능을 주장하지 않는다.

다음 외부 입력은 Gate L0의 immutable corpus/qrel/exclusion 계약, 법률 primary metric·minimum practical effect 승인, 라이선스·내부 사용 범위 확인이다.

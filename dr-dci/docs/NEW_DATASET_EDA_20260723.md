# Peter Part 1·2 신규 데이터 후보 EDA — 2026-07-23

## 결론

두 후보 모두 현재 상태에서는 Peter Part 1·2의 TREC-COVID 대체 **검색 평가셋**이 아니다.

| 후보 | 파일 무결성 확인 | EDA 완료 | 검색 평가 가능 | 현재 판정 |
| --- | --- | --- | --- | --- |
| 법률·규정/상황별 판례 | 외부 ZIP은 통과했으나, 필수 nested label ZIP 1개가 비정상이라 데이터셋 전체는 미확인 | 아니오 — 스키마 참고만 수행 | 아니오 — 명시 qrel/gold 없음 | **부적합(현재); 원본 교체 후 재판정** |
| 의료·법률 전문 서적 말뭉치 | 예 — 외부 ZIP과 nested ZIP 8개 모두 통과 | 예 — 원문 TXT 113,862건 및 label JSON 전체 | 아니오 — query/qrel/gold 없음 | **조건부 적합(corpus 후보); 평가셋으로는 부적합** |

가장 먼저 확인해야 하는 query와 gold 연결은 두 후보 모두 충족하지 않는다. 따라서 이 결과는 corpus/스키마 적합성 EDA이며, 새 질문·qrel·taxonomy를 만들거나 Part 1·2를 실행한 결과가 아니다.

기계 판독 결과는 [EDA summary JSON](/Users/donggyu/Documents/논문/PageIndex/experiments/dr-dci/docs/NEW_DATASET_EDA_20260723.json), 원본 및 안전하게 가상 추출한 nested ZIP의 해시는 [hash manifest](/Users/donggyu/Documents/논문/PageIndex/experiments/dr-dci/docs/NEW_DATASET_HASH_MANIFEST_20260723.json)에 있다. 원본 데이터나 원문 텍스트는 저장소에 추가하지 않았다.

## Gate 0 — 파일 무결성

직접 재검사 결과 두 **외부** ZIP은 모두 `PK\x03\x04` local-header와 EOCD signature를 가지고 Python `zipfile`의 중앙 디렉터리·전체 CRC 검사를 통과했다. 사전의 “EOCD 없음” 관찰은 외부 두 파일에는 재현되지 않았고, 실제 EOCD/중앙 디렉터리 오류는 아래 법률 ZIP의 내부 구성요소에서 확인됐다. `zip -FF` 등 복구 명령은 사용하지 않았다.

| 항목 | 법률·규정/판례 | 의료·법률 전문서적 |
| --- | ---:| ---:|
| 원본 크기 | 533,538,608 B | 723,086,416 B |
| 원본 SHA-256 | `c343c29715d8d586b81f6f7a9ab2a35cedbc9f2b38a76b3c5c19cea296f683d8` | `697f32d7901982520dfb8ea03c0a92b8509f9670c6b66af070191ecaee94031b` |
| 외부 중앙 디렉터리 entry / regular entry | 65 / 55 | 15 / 8 |
| 안전하게 읽힌 outer regular entry | 55 / 55 | 8 / 8 |
| nested ZIP CRC·중앙 디렉터리 통과 | 54 / 55 | 8 / 8 |
| 전체 archive test | 통과 | 통과 |

법률 데이터의 실패 대상은 다음과 같다.

`3.개방데이터/1.데이터/Training/02.라벨링데이터/TL_01.민사.zip`

- 수신된 nested 파일 SHA-256: `9726f1fc8f987e238c2d6b6bfd339fb8a373611ff0045bb202406b34b3e214fc`
- local header signature는 있으나 parse 가능한 중앙 디렉터리가 없다.
- `zipfile`은 `File is not a zip file`, `unzip`/`zipinfo`는 EOCD를 찾을 수 없다고 판정했다.
- 이 파일은 부분 복구하지 않았고, 해당 파일이나 나머지 54개 구성요소에서 나온 수치를 법률 데이터셋 전체 통계로 사용하지 않았다.

필요한 외부 입력은 `TL_01.민사.zip`의 중앙 디렉터리를 포함한 재다운로드본, 또는 이를 포함하는 제공처의 완전한 대체 원본이다. 교체본은 새 SHA-256으로 다시 Gate 0부터 검사해야 한다.

## Gate 1 — 인벤토리와 스키마

### 법률·규정/상황별 판례: 스키마 참고만

전체 EDA를 중지했으므로 총 문서 수, 중복률, 길이 분포, null/파싱 실패율을 보고하지 않는다. 단, 정상 구성요소에서 확인한 스키마는 재다운로드본 재검사 시 사용할 참고 정보다.

| 레코드 | 형식·주요 필드 | 해석 |
| --- | --- | --- |
| 원천 판례 JSON | `판례일련번호`, `사건번호`, `사건명`, `판시사항`, `판결요지`, `참조조문`, `참조판례`, `판례내용`, `사건종류명` | 장문 판결문 및 조문/판례 참조 구조 |
| 라벨 JSON | `info`, `jdgmn`, `jdgmnInfo`, `Summary`, `keyword_tagg`, `Reference_info`, `Class_info` | `info.caseNoID`, 질문/답변형 `jdgmnInfo`, 요약, 키워드, 분류 후보 |
| Other QA JSON | `id`, `title`, `question`, `answer`, `commentary`, `keyword`, `reference_rules`, `reference_court_case` | 질의 같은 필드는 있으나 qrel 파일은 아님 |

후보 taxonomy/semantic tag 필드는 `Class_info.class_name`, `Class_info.instance_name`, `keyword_tagg.keyword`, `Reference_info.reference_rules`, `사건종류명`이다. 이들은 qrel 생성 또는 같은 케이스의 gold 증거에 쓰면 안 된다.

### 의료·법률 전문서적: 전체 EDA

외부 ZIP은 Training/Validation과 원천 TXT/라벨 JSON으로 나뉜다. TXT는 `utf-8-sig`로 모두 파싱됐다. source TXT의 파일명 stem은 `book_id`와 113,862/113,862건 일치한다.

| domain·split | 원천 문서 수 | 고유 ID | 빈 본문 | 본문 문자 p50 / p95 |
| --- | ---:| ---:| ---:| ---:|
| 법률 Training | 63,704 | 63,704 | 0 | 2,359 / 5,873 |
| 법률 Validation | 7,963 | 7,963 | 0 | 2,358 / 5,603 |
| 의료 Training | 37,507 | 37,507 | 0 | 3,903 / 8,543 |
| 의료 Validation | 4,688 | 4,688 | 0 | 3,888 / 8,547 |
| 전체 | 113,862 | 113,862 | 0 | 2,597 / 8,100 |

전체 원천 TXT의 평균 길이는 3,551자(휴리스틱 토큰 평균 824), p99는 13,057자, 최댓값은 133,165자다. ID 중복은 0건, 공백 정규화 본문 hash 기준 내용 중복은 10건(0.0088%)이다. 줄바꿈 빈 문단 기준 문단 수는 모두 1이므로, 장문이더라도 제공 원문에는 별도 목차/section 경계가 노출되어 있지 않다.

법률은 71,667건, 의료는 42,195건이다. 법률 상위 범주는 민법일반(11,590), 헌법(10,938), 형법/형사소송법등(10,527), 상사법등(10,014), 행정법(8,582)이다. 의료는 내과학(6,237), 재활의학/물리치료학/작업치료학(5,190), 방사선과학(3,452), 약학/약리학(3,450) 등이 중심이다. 두 도메인은 자동 병합하지 않았다.

라벨 JSON의 schema는 모든 113,862건에서 동일했다.

| 필드 | 타입 | 사용상 의미 |
| --- | --- | --- |
| `book_id` | string | 원천 TXT ID. query ID가 아님 |
| `category` | string | 제공자 분류; taxonomy 후보 |
| `keyword` | array | 제공자 키워드; semantic tag 후보 |
| `NE` | array | `entity`, `type`, `begin`, `end`가 있는 entity span |
| `text` | string | 라벨측 본문. 원천 TXT와 함께 색인 금지 |
| `word_segment`, `popularity`, `publication_ymd` | number/number/string | 제공자 메타데이터 |

라벨의 `text`와 원천 TXT가 공백 정규화 hash까지 일치한 것은 101,024/113,862건(88.72%)이다. ID는 전부 일치한다. 따라서 label JSON을 source corpus와 함께 색인하면 강한 중복·누출 위험이 있다. `NE.type`에는 법률 조문·당사자·사건·판결과 의료 질환·약물·시술·기관 등 유형이 모두 존재한다. 이는 tag 효과의 **후보 신호**일 뿐, 검색 효과를 뜻하지 않는다.

정규식 선별에서는 원천 문서 중 전화번호형 4,034건, 이메일형 23건, 주민번호형 16건이 탐지됐다. 이는 개인정보 존재의 확정이나 비식별화 판정이 아니라, 사용 전에 별도 privacy/legal review가 필요하다는 경보다. 두 데이터셋 모두 archive 내 license/README를 찾지 못했으므로 제공처의 라이선스와 허용 이용 범위가 별도 필요하다.

## Gate 2 — 검색 평가 가능성

| 항목 | 법률·규정/판례 | 의료·법률 전문서적 |
| --- | --- | --- |
| query 제공 | 질문형 필드 관찰. 단, 전체 archive 무결성 미충족 | 없음 (`question` 필드 0건) |
| 명시 qrel/gold 연결 | 없음 | 없음 |
| gold 단위·복수 gold·등급 | 계약 없음 | 계약 없음 |
| 후보 ID 연결 | `info.caseNoID`와 source `사건번호`의 이름상 연결 후보만 존재 | `book_id`는 source ID이나 qrel이 아님 |
| leakage 위험 | 같은 케이스의 라벨 요약/답변/질문을 evidence로 색인할 경우 높음 | label `text`를 source TXT와 함께 색인하면 높음 |
| 현재 판정 | 검색 평가 불가 | corpus 후보만 가능 |

법률 데이터의 case number 일치는 qrel이 아니다. 이를 자동으로 gold로 승격하면 query/answer/summary와 동일 사건의 label 정보를 통해 정답을 입력으로 되넣을 수 있다. 전문서적은 label의 `category`/`keyword`/`NE`가 있어도 query와 gold가 없으므로 taxonomy boost A/B의 recall을 계산할 분모 자체가 없다.

## Gate 3 — Peter Part 1·2 적합성

| 기준 | 법률·규정/판례 | 의료 전문서적 | 법률 전문서적 |
| --- | --- | --- | --- |
| taxonomy soft boost 단일변수 A/B | 재다운로드·독립 qrel 후에만 검토 | 독립 qrel 후에만 검토 | 독립 qrel 후에만 검토 |
| semantic tag/구조 후보 | 강함: 판결문 section·참조·분류 | 있음: category/keyword/NE, 긴 TXT | 있음: category/keyword/NE, 긴 TXT |
| 20K/50K/110K 독립 nested scale | 현 상태 판정 불가 | 20K만 가능 (42,195건) | 20K·50K 가능 (71,667건) |
| 110K scale | 손상 해결 후 cardinality 확인 필요 | 단독 불가 | 단독 불가 |
| 장문·정확 문자열·법률 참조 | 구조상 유망하나 평가 불가 | 장문은 있으나 구조 경계 약함 | 장문·법률 텍스트 후보는 있으나 구조 경계 약함 |
| 신한 보험 전이 | 법률/판례 관점에서 가장 가깝지만 보험 약관 자체는 아님 | 건강보험 주제에는 간접적 | 규정·분쟁 검색에는 간접적 |

전문서적 두 domain을 합치면 113,862건으로 110K는 만들 수 있다. 그러나 의료와 법률의 query 분포·label 분포·긴 문서 비율이 섞여 scale 효과와 도메인 효과를 분리할 수 없으므로, 이번 판정에서는 합치지 않는다. 110K 단일-domain scale을 주장하려면 같은 domain의 추가 immutable corpus와 gold 보존 설계가 필요하다.

기존 TREC-COVID보다 나아지는 후보 특성은 한국어, 법률/의료 전문 용어, 긴 문서, 제공 taxonomy/NE metadata다. 새 편향은 제공자 label 본문 중복, label을 gold로 오인할 위험, domain 혼합, 그리고 검색 qrel 부재다. 그러므로 “TREC-COVID보다 더 좋은 Peter 평가셋”이라는 결론은 아직 낼 수 없다.

## 누락 계약과 최소 다음 데이터 계약

다음은 이번 EDA에서 만들지 않은 외부 입력이다.

1. 법률 ZIP의 완전한 immutable replacement와 제공처 revision/라이선스.
2. corpus source의 명시: source TXT/원천 판례만 evidence로 쓰고 label JSON은 metadata join으로만 쓸지 여부.
3. 고정된 `queries.jsonl`: `query_id`, `text`, query source/split, 작성·검수 provenance.
4. 고정된 `qrels.jsonl`: `query_id`, `corpus_id`, relevance grade. 복수 gold, gold 단위(문서/section/chunk), 제외 질의도 명시.
5. split과 leakage 차단: 같은 문서·사건·label text가 corpus/query/gold의 서로 다른 역할에 중복되지 않는 규칙과 검증 hash.
6. taxonomy artifact: `corpus_id`별 tag와 provenance, coverage, version/hash. qrel 생성과 독립이어야 한다.
7. scale manifest: 20K/50K/110K nested `corpus_id` 목록, 각각의 gold 보존율 및 corpus/query/qrel/taxonomy hash.

최소 계약은 아래 네 immutable 파일과 하나의 manifest다. 이는 제안일 뿐 이번 작업에서 생성한 평가 데이터는 아니다.

```text
corpus.jsonl   {"corpus_id", "title?", "text", "domain", "source_revision"}
queries.jsonl  {"query_id", "text", "split", "query_provenance"}
qrels.jsonl    {"query_id", "corpus_id", "relevance", "gold_unit"}
taxonomy.jsonl {"corpus_id", "tags", "artifact_revision", "provenance"}
manifest.json  archive/data SHA-256, counts, splits, exclusion/leakage policy,
               scale subset IDs, gold-preservation report, license/privacy approval
```

이 계약이 준비되면 Part 1의 첫 비교는 동일 corpus/query/qrel에서 `baseline` 대 `taxonomy_only` 하나로 제한한다. label 본문, gold, query를 taxonomy 입력에 재사용하지 않는지 audit한 후에만 Part 2의 retrieval-only scale 비교를 검토한다.

## 재현

다음 명령은 네트워크·API·LLM·임베딩 호출 없이 원본을 읽어 두 JSON 산출물을 다시 만든다. 파일명에 공백과 한글이 있으므로 인용부호를 유지한다.

```bash
cd /Users/donggyu/Documents/논문/PageIndex/experiments/dr-dci
python scripts/new_dataset_eda.py \
  --law-zip '/Users/donggyu/Documents/논문/신한라이프/115.법률-규정 텍스트 분석 데이터_고도화_상황에 따른 판례 데이터.zip' \
  --books-zip '/Users/donggyu/Documents/논문/신한라이프/154.의료, 법률 전문 서적 말뭉치.zip' \
  --output-dir docs --stamp 20260723
python -m unittest tests.test_new_dataset_eda -v
```

`new_dataset_eda.py`는 원본 archive를 수정하지 않고, legacy path를 filesystem에 풀지 않기 위해 nested member를 안전한 임시 파일로만 검사한다. 법률 데이터에 비정상 nested ZIP이 있으면 최종 JSON도 dataset-level numeric EDA를 `not_performed`로 기록한다.

## 이번 작업의 실행 상태

- 파일 무결성: 전문서적은 확인 완료. 법률·판례는 외부 ZIP만 확인 완료이며 데이터셋 전체는 차단 상태.
- EDA: 전문서적은 완료. 법률·판례는 스키마 참고만 완료.
- 검색 평가 가능성: 두 후보 모두 아직 불가.
- 미실행: taxonomy 생성, 질문/qrel 합성, embedding, reranking, agent, 외부 API/LLM 호출, Part 1·2 실행.

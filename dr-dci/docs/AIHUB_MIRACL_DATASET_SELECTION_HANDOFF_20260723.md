# Peter 실험 데이터 선택 검토: MIRACL-ko와 AI Hub 법률 데이터

검토일: 2026-07-23  
대상 저장소: `experiments/dr-dci`  
신규 데이터 경로: `/Users/donggyu/Downloads/data`  
현재 MIRACL-ko 경로: `data/miracl-ko`

## 1. 목적

Peter의 기존 Part 1·2를 넓히지 않고 다음 두 질문에 사용할 데이터를 객관적으로 선택한다.

1. Part 1: taxonomy soft score boost가 검색 workspace의 gold coverage를 높이는가.
2. Part 2: 동일 질의와 gold를 유지한 채 20K→50K→110K로 distractor를 늘렸을 때 성능이 어떻게 변하는가.

선택 기준은 데이터 확인 전에 사용하던 다음 기준을 그대로 적용했다.

- corpus, query, qrel의 독립성과 ID 정합성
- 사람이 판정한 positive·negative relevance의 범위
- multi-gold와 Agentic multi-pull 평가 가능성
- 20K/50K/110K controlled distractor scaling 가능성
- taxonomy treatment의 label leakage 위험
- raw→normalized→subset provenance와 재현성
- 한국어 법률·보험 업무와의 도메인 거리
- 이용정책과 외부 실행 환경 반출 가능성

## 2. 최종 판정

**현재 Part 1·2의 주 평가 데이터는 MIRACL-ko를 유지한다.**

AI Hub 데이터는 폐기하지 않는다. 아래 역할에 한정한 **법률 도메인 secondary transfer benchmark** 후보로 유지한다.

- 한국어 법률 문서 검색 전이
- parent document와 chunk retrieval 집계 계약 검증
- 법률·의료 문서를 distractor로 사용한 별도 stress condition
- 전문가 작성 법률 QA의 answer/judge 평가

AI Hub 데이터를 MIRACL-ko 대신 주 평가에 바로 투입하거나, 두 결과를 같은 의미로 평균내지 않는다.

## 3. 독립 실측 비교

### 3.1 검색 정답 품질

| 항목 | MIRACL-ko | AI Hub `aihub-full` |
|---|---:|---:|
| query | 1,081 | 6,599 |
| qrel | 15,824 | 6,599 |
| positive judgment | 2,520 | 6,599 |
| negative judgment | 13,304 | 0 |
| multi-positive query | 550 | 0 |
| query당 gold | 1~12 | 정확히 1 |
| 중복 query text | 0 | 34개 그룹, 100행 |
| 서로 다른 gold로 연결된 중복 query 그룹 | 0 | 34 |

MIRACL-ko의 query와 qrel은 원어민 평가자가 후보 passage별로 판정한 공식 IR 평가 자료다.

AI Hub 원본의 `jdgmnInfo.question/answer`는 사람이 만든 유효한 Q&A 라벨이다. 그러나 현재 `aihub-full/qrels.jsonl`은 AI Hub가 제공한 corpus-wide relevance judgment가 아니다. 로컬 생성기가 각 질문을 **그 질문이 부착된 자기 판례 하나**에 자동 연결한 qrel이다.

근거 코드:

- 질문 추출: `/Users/donggyu/Downloads/data/_build_scripts/src/data/aihub.py:124`
- 자기 판례 qrel 생성: `/Users/donggyu/Downloads/data/_build_scripts/src/data/aihub.py:419`

따라서 AI Hub의 6,599 qrel은 “이 판례는 관련 있음”은 지지하지만, “다른 판례는 관련 없음” 또는 “관련 판례를 대부분 찾았다”는 것을 보증하지 않는다.

### 3.2 query-gold lexical shortcut

동일 정규화로 query 문자열이 자기 gold 본문에 포함되는지 독립 계산했다.

| 조건 | MIRACL-ko: positive passage 중 하나 | AI Hub: 자기 gold 판례 |
|---|---:|---:|
| query 전문 일치 | 0.0% | 0.0% |
| query 첫 60자 일치 | 2.8% | 50.8% |
| query 첫 30자 일치 | 6.5% | 72.9% |

AI Hub 수치는 오류나 부정행위를 뜻하지 않는다. 판례별 Q&A가 원 판례를 읽고 만들어졌기 때문에 생기는 구조적 특성이다. 다만 검색기가 일반적인 관련성을 학습한 것이 아니라 긴 lexical prefix로 자기 판례를 찾는 shortcut이 성능을 지배할 수 있다.

AI Hub 결과를 사용한다면 질의별 overlap stratum을 저장하고, overlap이 높은 질의와 낮은 질의를 분리해 보고해야 한다.

### 3.3 Agentic multi-pull 적합성

MIRACL-ko는 1,081개 중 550개 query가 둘 이상의 positive passage를 갖는다. 첫 pull 이후 추가 탐색으로 새 gold를 확보했는지 측정할 수 있다.

`aihub-full`과 `ruling-anon`은 모든 query가 single-gold다. 첫 gold를 찾은 뒤 workspace recall이 즉시 1이 되므로 multi-pull의 추가 탐색 가치를 충분히 검증하지 못한다.

`legal-qa`는 19,660개 전문가 작성 질의를 제공하지만 현재 판례 corpus에 qrel이 연결된 query는 3,686개(18.75%)뿐이다. multi-gold query도 6개뿐이다.

### 3.4 Scale fixture

MIRACL-ko v2 fixture는 다음 계약을 이미 검증한다.

- 전체 judged passage 12,601개를 20K부터 고정
- relevance는 mandatory membership에만 사용
- 저장 순서는 `SHA256("miracl-ko-scale-v2\0" + corpus_id)`로 relevance와 독립
- 20K⊂50K⊂110K 집합 포함 관계
- 공유 passage 내용과 상대 순서 불변
- raw, normalized, subset SHA-256과 revision lock 보존

AI Hub 생성기는 현재 gold parent를 목록 앞에 배치한 뒤 distractor를 shuffle한다.

근거 코드:

- `/Users/donggyu/Downloads/data/_build_scripts/src/data/aihub.py:234`

membership 관점에서는 모든 gold 보존에 유리하지만, 파일 순서가 relevance를 노출한다. 색인 순서, 동점 tie-break 또는 일부 로더가 입력 prefix에 영향을 받으면 실험이 오염될 수 있다.

또한 전달된 `dr-dci_정답셋/aihub-full`에는 manifest가 참조하는 `20k_parent_ids.json`, `50k_parent_ids.json`, `110k_parent_ids.json`, corpus가 포함돼 있지 않다. 재생성기는 README와 달리 `ijson` 외부 의존성을 요구하며 현재 기본 Python과 DocNav venv 모두에 설치돼 있지 않다.

### 3.5 문서 단위

| 데이터 | retrieval unit | 현재 실험 의미 |
|---|---|---|
| MIRACL-ko | passage | 한국어 passage retrieval mechanism screening |
| AI Hub `aihub-full` | qrel은 parent, corpus는 3,000자 chunk | parent document retrieval 평가 가능, chunk→parent 집계 규칙 필요 |
| AI Hub 전문서적 | source record | distractor·domain corpus, 독립 query/qrel 없음 |
| `law_longdoc`, `synth_longdoc` | 장문 corpus | 구조·성능 smoke 전용, qrel 없음 |

AI Hub manifest 기준 170,000 parent에서 299,606 chunk가 나온다. 평균 약 1.76 chunk/parent이므로 신한의 3~4천 페이지 문서나 DocNav tree 탐색을 대표하지 않는다.

두 데이터 모두 실제 보험약관 장문 성능을 직접 증명하지 않는다.

## 4. 실험별 데이터 선택

| 실험 | 선택 | 이유 |
|---|---|---|
| Part 1 taxonomy soft boost | MIRACL-ko | positive·negative와 multi-gold가 있어 rank/coverage 변화를 판정 가능 |
| Part 2 retrieval-only scale probe | MIRACL-ko | v2 nested fixture와 relevance 독립 순서가 이미 고정됨 |
| Part 2 Agentic multi-pull | MIRACL-ko | 550개 multi-positive query로 workspace expansion 측정 가능 |
| 한국어 법률 domain transfer | AI Hub `aihub-full` 보완본 | 법률 corpus와 parent-document 단위가 업무에 더 가까움 |
| 자유서술 answer/judge | AI Hub `legal-qa` | 전문가 작성 answer·commentary가 있음 |
| 장문 구조·메모리 smoke | `law_longdoc`/별도 신한 fixture | qrel 없는 성능·구조 검증으로만 사용 |

## 5. 지표 해석

### MIRACL-ko

허용:

- Recall@k
- Hit@k
- nDCG@k
- Precision@k
- workspace gold recall
- first-pull 대비 final workspace gold recall

### 현재 AI Hub qrel

조건부 허용:

- 알려진 자기 gold에 대한 Recall@k
- MRR 또는 known-gold rank
- parent Hit@k
- scale별 동일 known-gold rank 변화

주 결론에 사용 금지:

- unjudged 문서를 모두 비관련으로 간주한 Precision@k
- qrel completeness를 전제로 한 nDCG/MAP
- single-gold 결과를 multi-hop 또는 multi-evidence 성능으로 표현

AI Hub를 주 평가 수준으로 승격하려면 여러 retrieval run의 상위 후보를 pooling하고 사람이 positive/negative를 판정해야 한다.

## 6. AI Hub secondary benchmark 승격 게이트

### A0. 획득·권리·provenance

- AI Hub dataset ID와 다운로드 승인 주체 기록
- raw 파일 상대 경로·bytes·SHA-256 manifest 생성
- 원본 수정 금지, normalized와 subset은 별도 경로
- 사내 H200 위치와 국외 반출 여부 확인
- AI Hub 출처 표기와 제3자 공유 제한 확인

### A1. query/qrel 정합

- 동일 정규화 query text를 하나의 canonical query로 병합
- 동일 query에 연결된 gold parent를 합집합 처리
- 중복 query 34개 그룹의 분리·병합 결과를 manifest에 기록
- orphan qid/parent, conflicting content, duplicate ID 차단
- `ruling-anon`의 gold 본문 query 전문 잔존 질의는 제외 또는 별도 leakage stratum

### A2. scale-v2

- 모든 known-gold parent를 20K부터 고정
- membership 선택 후 전체 parent를 relevance 독립 SHA-256 rank로 직렬화
- 20K⊂50K⊂110K와 공유 parent 내용 불변 검증
- subset별 parent와 chunk SHA-256 기록
- qrel·query는 모든 scale에서 동일

### A3. taxonomy leakage 차단

taxonomy 생성 입력 허용:

- 원천 판례·서적의 검색 대상 title/text

taxonomy 생성·query 분류 입력 금지:

- `jdgmnInfo`
- `answer`
- `Summary`
- `keyword_tagg`
- `Class_info`
- qrel
- reference answer/commentary

AI Hub native class를 treatment boost에 사용하면 query와 class가 같은 label object에서 생성돼 순환 평가가 될 수 있다.

### A4. parent/chunk 평가 계약

- retriever가 chunk를 반환할 때 parent score 집계 방식을 실행 전에 고정
- `max`, RRF 등 여러 집계를 결과를 본 뒤 선택하지 않음
- 같은 parent의 여러 chunk가 workspace budget을 중복 소비하는지 명시
- citation 평가는 원 chunk, 검색 gold 평가는 parent로 분리

### A5. qrel 보강

주 평가 승격 조건:

- baseline dense, lexical, hybrid, taxonomy treatment의 pooled top candidate 생성
- query별 후보를 법률 검토자 또는 승인된 판정자가 relevance 판정
- judged negative와 추가 positive 보존
- 판정자·가이드·불일치 처리·판정 revision 기록

이 작업 전에는 AI Hub 결과를 secondary known-gold retrieval 결과로만 표현한다.

## 7. Codex 작업 지시

현재 MIRACL-ko Gate M0의 `suitable` 판정을 유지한다. AI Hub 발견을 이유로 MIRACL fixture나 현재 focused Part 1·2 계약을 교체하지 않는다.

현재 우선 작업은 기존 합의대로 승인된 한국어 lexical backend의 standalone plumbing smoke다. AI Hub ingestion이나 모델 실험을 이 작업에 섞지 않는다.

AI Hub 준비는 사용자가 별도로 착수 지시할 때 독립된 **data suitability/preparation 작업 단위**로 수행한다.

착수 시 순서는 다음으로 제한한다.

1. A0 acquisition/hash/rights manifest
2. A1 canonical query/qrel audit
3. A2 relevance-independent scale-v2 fixture
4. A3 leakage validator
5. A4 parent/chunk metric adapter
6. model 호출 없는 retrieval plumbing smoke

taxonomy 생성, embedding, Agent, LLM, Part 1·2 본 실행은 위 단계와 별도 승인 전 수행하지 않는다.

원본 AI Hub 파일, 질문·답변, corpus text는 Git에 커밋하지 않는다. 코드·manifest·통계·hash만 저장한다.

## 8. 근거

공식·논문:

- MIRACL TACL 2023: https://aclanthology.org/2023.tacl-1.63/
- MIRACL 공식 저장소: https://github.com/project-miracl/miracl
- AI Hub 상황에 따른 판례 데이터: https://aihub.or.kr/aihubdata/data/view.do?dataSetSn=71723
- AI Hub 의료·법률 전문 서적 말뭉치: https://www.aihub.or.kr/aihubdata/data/view.do?dataSetSn=71487
- AI Hub 이용정책: https://aihub.or.kr/intrcn/guid/usagepolicy.do?currMenu=151&topMenu=105
- NIST relevance judgment와 pooling: https://trec.nist.gov/data/reljudge_eng.html
- DR-DCI arXiv: https://arxiv.org/abs/2606.14885

로컬 검토 자료:

- `docs/MIRACL_KO_PRETEST_20260723.md`
- `/Users/donggyu/Downloads/data/dr-dci_정답셋/README.md`
- `/Users/donggyu/Downloads/data/dr-dci_정답셋/*/manifest.json`
- `/Users/donggyu/Downloads/data/_build_scripts/src/data/aihub.py`
- `/Users/donggyu/Downloads/data/_build_scripts/src/data/local_corpora.py`

## 9. 표현 규율

- MIRACL 결과: “한국어 passage retrieval mechanism screening”
- AI Hub 결과: “한국어 법률 secondary transfer / known-gold retrieval”
- 신한 결과: 별도 내부 query/qrel이 있을 때만 “보험약관·신한 환경 검증”
- 공개 데이터 20K/50K/110K 결과를 11만 개 실제 신한 문서 운영 성능으로 표현하지 않는다.
- 코드 구현, fixture 준비, 모델 실험, 신한 운영 검증을 서로 같은 완료 상태로 표현하지 않는다.

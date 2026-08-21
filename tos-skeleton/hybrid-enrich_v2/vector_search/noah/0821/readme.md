# noah/0821 — Semantic Tag 하이브리드 검색 실험 준비

## 목적과 고정 조건

목표는 Claude Sonnet 에이전트의 조(jo) 단위 fractional `R@5`를 공식 test 분모에서 0.90 이상으로 올리는 것이다. 이 디렉터리는 실험 준비와 실행 도구만 제공하며 Claude 유료 런 결과를 포함하지 않는다.

변인은 `search`(Semantic Tag 검색)뿐이다. `msearch`는 어제와 동일하게 `vector_search/hybrid_search.py`의 `ChunkHybridSearch(view="V9")`를 호출하며, V9 BM25와 BGE-m3-ko dense 결과를 RRF `k=60`으로 합친다. 메타데이터·청크·임베딩 파일은 재생성하거나 수정하지 않는다.

## 고도화한 태그 검색 방식

기존 태그 검색은 질문에서 contract/role/subject/qualifier/schema 슬롯과 어휘를 추출하고, 일치한 조건 수로 CLM 순위를 만들었다. 0821 검색기는 이 CLM을 버리지 않고 다음 태그 전용 채널을 추가한다.

1. **CLM 구조 채널**: 기존 `SlotSearch`의 슬롯 및 어휘 매칭을 그대로 사용한다.
2. **태그 BM25 채널**: 각 element의 특약명, 대상어, 역할 코드와 한글 의미, 조건, 참조, 조 제목·표 헤더·행 키, 태그 검색문, 원문 앞 800자를 하나의 태그 검색 view로 만든다. 한국어 단어와 문자 bigram으로 sparse 검색한다.
3. **RRF 융합**: CLM과 BM25의 top-200을 `weight/(60+rank)`로 합친다. 같은 조에 속한 여러 element는 가장 높은 대표 element 하나만 남겨 한 페이지를 중복 조가 점유하지 않게 한다.

시맨틱 태그는 별도 임베딩하지 않는다. 따라서 태그 자체에 대한 dense 벡터검색은 없으며, 의미 확장은 역할 코드의 한글 표현·태그 필드 통합·한국어 bigram으로 처리한다. dense 벡터검색은 고정된 V9 메타데이터 `msearch`에서만 수행된다.

태그 view에는 QA 질문이나 gold 정답을 넣지 않는다. test 질문용 사전 qtags도 생성하지 않는다. 기존 `qtags_haiku.jsonl`은 train qid만 포함하며 test에서는 규칙 라우터와 Claude가 명시한 슬롯만 사용된다.

에이전트가 보는 명령 형식은 유지된다.

```powershell
python agent_tools.py search --q "보험료 안 내도 되는 조건" --role premium_waiver
python agent_tools.py msearch --q "보험료 안 내도 되는 조건"
python agent_tools.py read --id e12345
python agent_tools.py submit --ids e12345,c01234
```

## Arm과 선택 규칙

`arms.json`은 다음 arm을 사전 등록한다.

| arm | CLM | BM25 | 조 중복 제거 |
|---|---:|---:|---:|
| `baseline_fs_slot` | 1 | 0 | 꺼짐 |
| `tag_sparse_no_collapse` | 1 | 1 | 꺼짐 |
| `tag_sparse_equal` | 1 | 1 | 켜짐 |
| `tag_sparse_struct` | 2 | 1 | 켜짐 |
| `tag_sparse_lexical` | 1 | 2 | 켜짐 |

`eval_det.py`가 train337의 jo R@5가 가장 높은 arm을 `out/det/selected_arm.txt`에 기록한다. 동률이면 suff@10, R@10, 동일 가중치의 단순성 순으로 결정한다. test 결과를 본 뒤 arm이나 가중치를 바꾸지 않는다.

### 준비 시점 결정론 선발 결과

2026-08-21 train337 전수에서 다음 결과를 확인했다. 이는 Claude 에이전틱 결과가 아니라 유료 런 전에 태그 검색기만 비교한 결정론 수치다.

| arm | R@5 | R@10 | suff@5 | suff@10 |
|---|---:|---:|---:|---:|
| `baseline_fs_slot` | 0.4526 | 0.5742 | 0.4036 | 0.5193 |
| `tag_sparse_equal` | 0.5742 | **0.6721** | 0.5223 | **0.6142** |
| **`tag_sparse_lexical` (선택)** | **0.5868** | 0.6610 | **0.5341** | 0.6053 |

주지표 R@5 기준 선택 arm은 `tag_sparse_lexical`이며 baseline 대비 `+13.42%p`다. R@10과 suff@10은 equal arm이 조금 높지만 사전 등록한 선택 규칙에 따라 R@5 최고 arm을 사용한다.

## 데이터셋과 채점 분모

### 튜닝

- `hybrid-enrich_v2/out/noah/gold_v4_train_full.jsonl`
- v4 감사 후 유효한 337문항
- arm 선택과 모든 파라미터 결정은 이 데이터에서만 수행한다.

### 공식 기본 test

- 사용자 지정 원본: `hybrid-enrich_v2/out/gold_mapped_noah_v3_149_test.jsonl`
- 현재 채점 파일: `hybrid-enrich_v2/out/noah/gold_v4_test_full.jsonl`
- 원본 149 중 v4 인용을 현재 250212 코퍼스에 정상 매핑한 90문항이 공식 기본 분모다.
- 주지표는 n=90 jo 단위 fractional R@5이다.
- `c3_partial` 27문항을 제외한 n=63 R@5와 R@10/suff@5/suff@10은 보조로 함께 출력한다.

### 외부 spanmap을 받는 경우

로컬 저장소와 Git 이력을 조사했지만 유효한 `원문 span 매핑 파일.jsonl`은 없다. 확인한 대체 후보는 train-v2 span overlap이 다음과 같아 사용할 수 없다.

- `hybrid-enrich/out/elements.jsonl`: 0.7%
- `hybrid-enrich/out/elements_repaired_v1.jsonl`: 0.7%
- `filesearch/out/elements_u2.jsonl`: 0.0%

외부 동료가 원본 spanmap을 전달하면 아래 명령으로 검증·변환한다.

```powershell
python prepare_gold.py --spanmap "C:\path\원문 span 매핑 파일.jsonl"
```

다음 조건을 모두 만족해야 `out/gold_spans_lsh_test_v3_138.jsonl`을 공식 분모로 승격한다.

- train-v2 유효 문항 재구성률과 group 수 일치율 각각 95% 이상
- train-v2 span overlap 99% 이상
- raw test gold 보유 문항 partial 매핑 0개
- empty gold 정확히 11개, 유효 분모 정확히 138개
- 모든 gold group이 현 jo 인덱스에서 도달 가능하고 최대 제출 10개로 충족 가능

spanmap이 없으면 `prepare_gold.py`가 test90을 공식 분모로 고정한 `out/gold_manifest.json`을 만든다.

## 준비 절차

PowerShell에서 이 디렉터리로 이동한 뒤 실행한다. 태그 인덱스 생성에는 임베딩 서버가 필요하지 않다.

```powershell
.\prepare_experiment.ps1
```

이 명령은 다음을 순서대로 수행한다.

1. gold manifest와 V9 파일 SHA-256 manifest 생성
2. 단위 테스트
3. 32,366개 태그의 BM25 sparse index 생성(태그 임베딩 없음)
4. train337의 5개 arm 결정론 평가 및 후보 자동 선택

V9 실제 검색 결과까지 어제 코드와 대조하려면 임베딩 엔드포인트가 켜진 상태에서 실행한다.

```powershell
python verify_frozen_meta.py --n 5
```

0820과 0821의 같은 다섯 질의 top-40 `(chunk_id, jo)`가 하나라도 다르면 실패한다.

## Claude 실행 절차

Claude CLI는 여기서 자동 실행하지 않는다. 준비가 끝난 뒤 사용자가 다음 순서로 실행한다. `-ClaudeBin`에는 로컬 native 실행 파일 전체 경로를 줄 수 있다.

```powershell
# 1) train 5문항 smoke: baseline과 선택 후보 각각 1회
.\run_claude.ps1 -Stage smoke -ClaudeBin "claude"

# 2) train60 스크리닝
.\run_claude.ps1 -Stage train60 -ClaudeBin "claude"

# 3) 필요 시 train337 전수 확인
.\run_claude.ps1 -Stage trainfull -ClaudeBin "claude"

# 4) 사전 확정된 baseline/후보를 공식 test에서 각각 딱 1회
.\run_claude.ps1 -Stage test -ClaudeBin "claude"
```

각 단계는 resume 모드이며 결과는 `out/agent/0821_<stage>_{baseline,candidate}`에 쌓인다. A/B 결과는 `out/<stage>_comparison.json`에 기록된다.

공식 성공 조건은 candidate의 전체 공식 분모 R@5가 `0.9000` 이상인 것이다. paired wins/losses와 sign-test를 함께 기록하지만, 목표 미달 시 분모를 사후 변경하거나 여러 실행 중 최고값만 고르지 않는다.

## 파일 안내

- `tag_hybrid.py`, `build_tag_index.py`: 태그 view, BM25 index, CLM+RRF 검색기
- `agent_tools.py`, `agent_runner.py`: Claude용 검색 도구와 Sonnet 실행·채점 러너
- `eval_det.py`, `compare_runs.py`: train arm 선택과 paired 결과 비교
- `prepare_gold.py`, `verify_frozen_meta.py`: test90/test138 분모 및 고정 메타 검증
- `prepare_experiment.ps1`, `run_claude.ps1`: Windows 실행 진입점
- `test_tag_hybrid.py`: sparse/view/token 단위 테스트

대형 원본 데이터와 V9 임베딩은 이 디렉터리로 복사하지 않고 기존 경로를 읽기 전용으로 참조한다.

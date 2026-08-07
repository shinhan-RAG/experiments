# `tos-skeleton/results` — 실험 결과 등록 규약

검색 실험 결과를 **한 곳에 같은 형식으로** 모은다.
서로 다른 사람이 낸 숫자를 **나란히 놓고 비교할 수 있게** 하는 것이 목적이다.

---

## 1. 폴더 구조

```
results/
├── README.md                     이 파일
├── elements/                     element(청크) 단위 검색 결과
│   ├── _schema/
│   │   └── element_result.schema.json
│   └── YYYY-MM-DD/
│       └── <config_id>.json
└── metadata/                     메타데이터·태그·스키마 관련 결과
    ├── _schema/
    │   └── metadata_result.schema.json
    └── YYYY-MM-DD/
        └── <config_id>.json
```

| 폴더 | 무엇을 넣는가 |
|---|---|
| **`elements/`** | 문서를 자른 단위로 검색한 결과. 분할·검색 채널·재순위 등 |
| **`metadata/`** | 태그·스키마·필드 설계 결과. 메타데이터가 검색에 미친 영향 |

**날짜는 실행일**(`YYYY-MM-DD`)이다. 같은 날 여러 결과면 `config_id`로 구분한다.

---

## 2. `config_id` 규칙

```
<채널>_<스키마>_<분할>_<모집단>
```

소문자·숫자·하이픈만. 예:

```
bm25_none_p0_train338.json
bm25_newstv3_p0_confirmed210.json
dense-bgem3ko_none_p0_train338.json
rrf-bm25-bgem3ko_newstv3_p0_train338.json
```

**같은 `config_id`를 다른 날짜 폴더에 다시 올려도 된다.** 재현·재측정 기록이 된다.

---

## 3. 필수 필드 — 왜 이렇게 많은가

이 프로젝트는 **결론이 9번 뒤집혔다.** 매번 수치는 재현됐고 해석이 무너졌다.
원인은 늘 같았다 — **대조군 부재 · 비교 조건 불균형 · 모집단 차이 · 재현 앵커 없음.**

아래 필드는 그 재발을 막기 위한 것이다. **하나라도 빠지면 나중에 그 숫자를 쓸 수 없다.**

### 3-1. 재현 앵커 (`repro`)

| 필드 | 뜻 |
|---|---|
| `doc_sha256` | 대상 문서의 SHA-256 |
| `gold_sha` | 정답셋의 SHA-256 |
| `code_commit` | 실행한 커밋 해시. **출력 JSON에서 복사할 것** (손으로 적으면 틀린다) |
| `two_pass_sha` | 같은 코드를 2회 실행해 얻은 산출물 SHA. **일치해야 결정적** |

**`two_pass_sha`가 없으면 그 결과는 재현 검증되지 않은 것이다.**

### 3-2. 모집단 (`population`)

| 필드 | 뜻 |
|---|---|
| `name` | `train350` · `train338` · `confirmed210` · `review128` 등 |
| `n` | 문항 수 |
| `note` | 제외·필터 사유 |

**전체 평균만 적지 마라.** 층이 다르면 다른 측정 체제다.
(예: 확정 210은 인용검증 96.7%, 검수필요 128은 0%)

### 3-3. 구성 (`config`)

| 필드 | 뜻 |
|---|---|
| `channel` | `bm25` · `dense` · `rrf` · `rerank` 등 |
| `channel_params` | `{k1, b}` · `{model, revision, max_len, batch, dtype}` · `{rrf_k}` |
| `schema` | 태그 스키마 이름과 필드 목록 |
| `split` | 분할 방식과 element 수 |

**`channel_params`에 하이퍼파라미터를 전부 적어라.** 나중에 "무엇이 달랐는가"를 못 찾는다.

### 3-4. 지표 (`metrics`)

`R@1` `R@5` `R@10` `R@20` `R@50` `R@100` · `MRR@10` · `L_star_median`
측정 안 한 것은 `null`. **0으로 채우지 마라.**

### 3-5. **널 대조 (`nulls`) — 가장 중요**

| 널 | 무엇을 가리는가 |
|---|---|
| `constant_dummy` | "뭔가 붙였다"만으로 오르는 몫 |
| `shuffled_tag` | 태그가 **맞는 자리**에 있어야 효과가 나는가 |
| `query_shuffle` | 검색기가 **질문을 실제로 읽는가** |
| `random_vector` | dense가 **의미를 담고 있는가** |

각 널은 `{value, seeds, min, max, note}`.

> **널을 넘지 못한 값은 개선이 아니다.**
> 측정 안 했으면 `null`로 두고 `limitations`에 적어라. **빼지 마라.**

### 3-6. 한계 (`limitations`)

문자열 배열. 아래는 **매번 적어야 하는 것**이다.

- 절단·표본·시드 수 제약
- 다중비교 (칸이 여럿이면 최고값은 부풀려져 있다)
- 평가셋 알려진 결함 (아래 §5)

---

## 4. 최소 예시

`elements/2026-08-07/bm25_none_p0_train338.json`

```json
{
  "config_id": "bm25_none_p0_train338",
  "date": "2026-08-07",
  "author": "donggyu",
  "experiment": "exp23",
  "repro": {
    "doc_sha256": "279f134b…",
    "gold_sha": "b3dc7162…",
    "code_commit": "137ab4c…",
    "two_pass_sha": "24a45956…"
  },
  "population": { "name": "train338", "n": 338, "note": "gold 매핑 성공분" },
  "config": {
    "channel": "bm25",
    "channel_params": { "tokenizer": "char-bigram", "k1": 1.2, "b": 0.75 },
    "schema": { "name": "none", "fields": [] },
    "split": { "name": "p0", "n_elements": 6852, "median_chars": 309 }
  },
  "metrics": {
    "R@1": 0.148, "R@5": 0.396, "R@10": 0.550,
    "R@20": 0.657, "R@50": 0.757, "R@100": 0.858,
    "MRR@10": 0.257, "L_star_median": 314
  },
  "nulls": {
    "query_shuffle": { "value": 0.018, "seeds": 200, "min": 0.011, "max": 0.026 },
    "constant_dummy": null,
    "shuffled_tag": null,
    "random_vector": null
  },
  "limitations": [
    "탐색적 측정 — 확증 검정 없음",
    "평가셋 gold 오류율 추정 11% (Wilson 95% CI 4.5~26.0%)"
  ]
}
```

---

## 5. 알려진 평가셋 결함 — `limitations`에 해당하면 적어라

| 결함 | 규모 |
|---|---|
| gold 오류율 | 추정 **11%** (Wilson 95% CI 4.5~26.0%) |
| 질문 품질 | 재주석 70건 중 **54%가 `부분/불가`** (CI 42.7~65.4%) |
| **평가셋 순환성** | 확정 210 중 **55건(26%)의 gold가 BM25로 매핑**됨. **그 55건은 어떤 검색 구성을 써도 `.655~.673`으로 동일** — 검색기 비교에 기여하지 않으면서 전체 수준만 올린다 |
| 층 이질성 | 확정 210(인용검증 96.7%) vs 검수필요 128(0%) — **합쳐서 하나의 평균으로 보고하면 안 됨** |
| test 오염 | test 150 중 **57%가 과거 라운드에 노출**. 청정 65건 |

---

## 6. 올리기 전 점검

- [ ] `repro` 4개 필드가 전부 채워졌는가 (`two_pass_sha` 포함)
- [ ] `population.name`·`n`이 명시됐는가
- [ ] `channel_params`에 하이퍼파라미터가 전부 있는가
- [ ] **널이 하나 이상 측정됐는가.** 안 됐으면 `limitations`에 적었는가
- [ ] 측정 안 한 지표를 `0`이 아니라 `null`로 뒀는가
- [ ] `limitations`에 다중비교·표본 제약을 적었는가

---

## 7. 하지 말 것

- **널 없는 결과를 "개선"으로 기록하는 것**
- 모집단을 안 적고 지표만 올리는 것
- 하이퍼파라미터를 탐색해 최고값만 올리는 것 (탐색했으면 격자 전체를 적어라)
- 출처를 재현할 수 없는 수치를 올리는 것
- **원문·정답셋 원본을 커밋하는 것** — 경로와 SHA만 적는다

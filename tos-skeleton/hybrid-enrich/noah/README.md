# noah — 에이전틱 하이브리드 RRF 검색 실험 코드

2026-08-14 실험(`docs/0814/`)에 실제로 사용된 코드만 모았다.

- 코드는 이 폴더(`noah/`), **데이터는 상위 폴더**(`hybrid-enrich/out/`, `hybrid-enrich/noah_qaset/`)에 있다.
- 각 스크립트는 `HERE`(= `noah/`)와 `BASE`(= `hybrid-enrich/`)를 구분해서 쓴다. `OUT = BASE / "out"`.
- 실행은 이 폴더 안에서 한다: `cd noah && python <script>.py`
- 검색기가 읽는 자료는 전부 `hybrid-enrich/` 내부다. 폴더 밖 참조 없음.

## 파이프라인 순서

| 단계 | 스크립트 | 출력 |
|------|----------|------|
| 1 | `build_unified_gold.py --split {train,test}` | `out/gold_{train,test}.jsonl` (173 / 67, `coord_space:"raw"`) |
| 2 | `build_elements_psection.py` | `out/elements_psection.jsonl` (5,408, 중앙값 336자) |
| 3 | `gen_meta_codex.py --schema v9` | `out/llm_meta_v9.jsonl` (5,797) |
| 4 | `gen_element_tags_v2.py` | `out/element_tags_v2.jsonl` (5,408, 8슬롯) |
| 5 | `build_views_local.py` | `out/view_{BASE,V6,V9}.jsonl` |
| 6 | `embed_local.py --views BASE,V9 --batch 64` | `out/vec_{VIEW}.npy` + `_ids.json` |
| 7 | `run_agentic_rrf.py --gold out/gold_train.jsonl --arms ...` | `out/results_*.jsonl`, `out/sessions_*/` |
| 8 | `eval_agentic_rrf.py --results ... --gold ...` | R@K, MRR, McNemar, 부트스트랩 CI |

## 검색기

| 파일 | 역할 |
|------|------|
| `tools_rrf.py` | 에이전트 도구 CLI. `hybrid` / `slot` / `slotand` / `grep` / `read`. 희소(BM25)+밀집(bge-m3) RRF k=60 융합 |
| `clm_filesearch.py` | CLM 엘리먼트 태그 검색 — 슬롯 union ∪ 렉시컬. 등록 실험 우승 구성 |
| `slot_filesearch.py` | AND 교집합 엘리먼트 검색 — 비교용. `V9_SLOT`/`BASE_SLOT` arm 에서만 노출 |
| `retrieval_prompt_rrf.md` | 시스템 프롬프트 템플릿. `{{TOOLS}}`/`{{STEP3}}` 를 arm 별로 치환 |

## arm

| arm | 채널 | slot(CLM) | slotand(AND) |
|-----|------|-----------|--------------|
| `BASE` | bm25/BASE + dense/BASE | ✗ | ✗ |
| `BASE_CLM` | bm25/BASE + dense/BASE | ✓ | ✗ |
| `V9` | bm25/V9 + dense/V9 | ✗ | ✗ |
| `V9_CLM` | bm25/V9 + dense/V9 | ✓ | ✗ |
| `V9_SLOT` / `BASE_SLOT` | 위와 동일 | ✗ | ✓ |
| `V6` / `V6V9` / `V6V9_CLM` | v6 포함 | — | ✗ |

## 반드시 지킬 것

1. **좌표계.** `gold_spans` 와 `char_start/char_end` 는 둘 다 원본 문서(2,757,704자) raw 좌표여야 한다. 공백 정규화 좌표를 섞으면 최대 56,667자 어긋나고 모든 recall 수치가 무의미해진다. 러너·평가기가 `coord_space != "raw"` 면 중단한다.
2. **도구 게이팅은 두 곳에서.** 프롬프트(`build_system_prompt`)와 CLI(`tools_rrf.py` main) 양쪽에서 막아야 한다. 한쪽만 막으면 arm 구분이 성립하지 않는다 — 2026-08-14 실험에서 실제로 이 결함으로 태그 축이 통째로 무효가 됐다.
3. **사전 점검을 건너뛰지 말 것.** `run_agentic_rrf.py` 는 뷰·벡터·덴스 커버리지(95% 이상)·ollama·CLM 을 실행 전에 검사한다. ollama 가 죽으면 밀집 채널이 조용히 빠지고 하이브리드 실험이 BM25 단독 실험이 된다.
4. **반복 실행을 넣을 것.** n=173 에서 동일 구성 반복쌍이 R@5 +6.35pp(McNemar p=0.027)까지 벌어졌다. 단발 비교로 6pp 미만 차이를 주장하면 위양성이다.
5. **인코딩.** 모든 `open()` 에 `encoding="utf-8"`. 이 머신 기본값은 cp949 라 한국어가 깨진다.

## 재실행

```bash
cd noah

# 실패 세션만 다시
python run_agentic_rrf.py --gold out/gold_train.jsonl \
  --arms BASE,BASE_CLM,V9,V9_CLM --retry-failed \
  --out out/results_0814_train.jsonl --session-root out/sessions_exp0814

# 도구 0회 세션까지 포함
python run_agentic_rrf.py ... --retry-zero-calls
```

# filesearch

Semantic Tag 기반 파일서치. 문서 250212판, element 우주 u2(항 단위 32,366) 색인, 조 단위(u2jo 7,599) 채점.

## 파이프라인
```
build_universe.py --doc <250212.md> --clean --name elements_u2.jsonl   # element
build_jo_universe.py --elements out/elements_u2.jsonl --out out/elements_u2jo.jsonl
tag_rules.py            # NEW 8슬롯 태그 (규칙)
tag_old.py              # OLD 스키마 태그 (대조)
map_gold_spans.py --doc … --qa <train csv>          # 임시 gold (진단용)
map_lsh_gold.py                                      # lsh gold → span. strict(주지표)/lenient(부지표) 두 파일
qtag_llm.py --model haiku                            # 질의 태그 (질문+특약사전만 입력, 캐시)
eval_det.py --arms "clm:lex=count:w=contract=2;and"  # 결정론 선별
agent_runner.py / agent_runner_oai.py                # 에이전트 실측 (claude / OpenAI 호환)
stats.py --a A.jsonl --b B.jsonl --key R@5           # paired 검정
```

## 검색기 (clm_search.py)
- 후보 = 슬롯 합집합 ∪ 어휘 채널. 점수 = 맞은 슬롯 수(가중 가능) + 질문 토큰 매치 수(`lex=count`) 또는 유무(`lex=binary`). 동점은 문서순.
- 라우터 = 규칙(특약명·역할·대상어·값) + LLM qtags + 에이전트 지정 슬롯. contract 소프트 부스트.
- `mode=and` 는 이전 AND 검색기 대조용.
- BM25, 임베딩, reranker, 학습 정렬은 쓰지 않는다. 메타 채널은 `agent_tools.py msearch` 로 `../vector_search`를 호출한다(고정).

## gold / 채점
- gold: lsh QA셋 gold element 텍스트를 원문에 앵커링해 char span. 인접 span(gap≤30) 병합, 같은 텍스트 group은 OR. lenient는 정답 특약 안 동일문구 출현을 OR로 최대 5개 추가.
- 채점(scoring.py): Recall@K = Top-K가 덮은 group / 전체 group (OR은 1개면 충족). Success@K, sufficient@K, MRR@10.
- 제출 id는 e*(항), j*(조), c*(청크) 모두 받는다. units.py 가 조 단위로 사상한다.

## 에이전트 도구 (agent_tools.py)
search / msearch / read / submit. 예산: 검색 20회(search+msearch), read 8회, page 40, preview 160자, submit 10개. 호출과 반환 후보를 세션 로그로 남긴다.
arm 예: `{"lex":"count","w":{"contract":2},"router":"llm","qtags":"out/qtags_haiku.jsonl","meta":1,"meta_view":"V9"}`

## 결과 (train, 조 채점, 결정론 = 선별용)
| gold | arm | R@5 | R@10 | R@40 | suff@10 |
|---|---|---|---|---|---|
| lsh strict (334) | CLM + qtags + contract w2 | .466 | .578 | .731 | .542 |
| lsh strict (334) | AND | .323 | .394 | .460 | .371 |
| 임시 (338) | CLM 규칙 라우터 | .364 | .465 | .653 | — |
| 임시 (338) | 정답 특약 scope 오라클 | .646 | .746 | .846 | — |

에이전트 파일럿(60문항×2, 임시 gold): sonnet 규칙 .558 / qtags .543, Qwen3-32B(text) 규칙 .324 / qtags .337 / +메타 .374.
보고 수치는 에이전트 실측만 쓴다.

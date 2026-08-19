# noah/0819 — Semantic Tag 검색 툴 고도화 실험

0819 회의(`docs/0819/SemanticTag_Metadata_검색회의.md`) 기반. 평가 프로세스와 메타데이터
검색기(msearch, V9 hybrid)는 **고정**, Semantic Tag 검색기(search)만 변경하며 비교한다.
설계 근거는 0818 실험 결과(`docs/0818/`) — 상세는 계획서/설계 문서 참조.

## 도구 (filesearch 대비 변경점)

신규 서브커맨드 없음 — 기존 search/msearch/read/submit 4개의 내부 확장. 예산 동일
(search+msearch 20회, read 8회, submit ≤10). 신규 arm 플래그가 전부 꺼지면 원본과 동일 동작.

| 플래그 | 내용 |
|---|---|
| `alias:1` | `filesearch/aliases.json` 양방향 질의 확장 — **어휘(lex) 채널 한정**(태그 채널 강화는 단조 손해). `calls.jsonl`에 `alias_expanded` 로깅 |
| `scope_boost:3.0` | `search --scope "<특약>[/<관>[/<조>]]"` = {대}→{중}→{소} **soft descent**(매치 시 가산 부스트, 필터 아님). `--q` 없이 `--scope`만 주면 계층 하위 목록(browse). SEARCH_CAP에 포함 |
| `fallback:{"min_n":5,"tau":τ}` | 태그 결과 0건→msearch로 **대체**, `<min_n`건 또는 top score`<τ`→후미 **병합**. 같은 호출 1회로 계산, `channel: tag+fallback:meta`. 회의의 "백트래킹"을 에이전트 판단 없이 자동 보장 |
| `fallback_prompt:1` | 자동 폴백 없이 프롬프트 유도만(자동 vs 유도 대조군) |
| `ref_expand:1` | read에 참조 조 미리보기 `refs`(≤4, 예산 미차감), search 상위 10건에 `ref_jo`. 근거 산재(실패의 65%) 겨냥. 그래프는 `build_refs.py` 사전계산 |

## arms.json

`base_c3` = filesearch `C_new_clm_meta` 동일 사본(기준선 재현 게이트). `c4_alias / c4_scope /
c4_fb / c4_fbp / c4_ref` = 변인 1개씩. `c4_all` = 결합.

## 재현

```bash
# 0) 데이터 (filesearch out 재생성 — git에는 stats만 있음)
cd ../../filesearch
python build_universe.py --doc "../vector_search/doc/판매약관_….md" --clean --name elements_u2.jsonl
python build_jo_universe.py --elements out/elements_u2.jsonl --out out/elements_u2jo.jsonl
python tag_rules.py

# 1) 참조 그래프
cd ../vector_search/noah/0819
python build_refs.py

# 2) 결정론 사전검증 + τ 튜닝 (LLM 0회, gold=train 348→329)
python eval_det.py --arms "clm:lex=count:w=contract=2;clm:lex=count:w=contract=2:alias=1"
python tune_tau.py --ranks out/det_ranks_clm_lex-count_w-contract-2.jsonl   # τ → arms.json c4_fb/c4_all 반영

# 3) Agent 실측 (KT Cloud Qwen3-32B, vLLM :8080 / 임베딩 :8101)
export EMBED_ENDPOINT=http://localhost:8101/v1
python agent_runner_oai.py --run base_1rep --arm base_c3 --protocol text --no-think --reps 1
python agent_runner_oai.py --run c4all_1rep --arm c4_all --protocol text --no-think --reps 1

# 4) 최종 확증: base vs 최고 arm ×3 reps → filesearch/stats.py 페어드 검정
# 5) test 149 (최종 1회): cowork "원문 span 매핑 파일.jsonl" 필요
python build_gold_spans_test.py --spanmap <원문_span_매핑_파일.jsonl> --validate   # train 대조 게이트
python build_gold_spans_test.py --spanmap <...>                                    # out/gold_spans_lsh_test.jsonl
python agent_runner_oai.py --run final_test --arm <최고arm> --gold out/gold_spans_lsh_test.jsonl --reps 1
```

Windows 로컬 실행 시 `PYTHONUTF8=1` 필요.

## 채점 — 회의 규칙과의 대응

`filesearch/scoring.py` 를 그대로 사용한다. 회의(0819)의 채점 규칙이 이미 구현되어 있다:

| 회의 규칙 | 구현 |
|---|---|
| 복수 gold 비율 부분점수 (4개 중 3개 = 75점) | `R@K = 덮은 evidence group 수 / 전체 group 수` (fractional) |
| 유사/파생 특약 a·a′ 는 a만 찾아도 정답 | gold group의 `members` OR — 멤버 1개 겹치면 그 group 충족 (`overlaps()`) |

주지표 suff@5/suff@10(모든 group AND 충족), 부지표 R@5/R@10/MRR@10.

## 주의

- `hybrid-enrich/out/elements.jsonl` 로 test gold span 을 만들지 말 것 — element_id 가 같아도
  cowork 우주와 다르다(train 재구성 검증에서 span 겹침 0.5%).
- 단일 런 R@5 차이 ~6pp 이하는 주장 금지(rep 분산 전례). 최종 판단은 3 reps + paired 검정.
- τ 는 train 에서 튜닝하므로 test 는 최종 1회만 사용(과적합 방지). `tune_tau.py` 민감도 곡선 기록.

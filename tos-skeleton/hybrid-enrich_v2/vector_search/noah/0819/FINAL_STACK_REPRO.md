# 최종 채택 스택 재현 — r1v + Gold v22 (train 213, R@5 .9147)

최종 조합은 arm `r1v_verify_identity_reference_hybrid`(arms.json) + `gold_train_scoped_u3_reviewed_overlay_v22.jsonl`이다.
검색기는 rule router + Semantic Tag/BM25F + CLM quota 앙상블(결정론)이며, 비에이전트 LLM·reranker를 쓰지 않는다.
Meta V9 hybrid 채널(`vector_search/hybrid_search.py`)은 고정 변수로 항상 결합된다(require_meta_first).
에이전트는 opus, reasoning effort medium, host-controlled search/read/submit, max-actions 5, ITT reps 2.

판정 이력·기각 목록·gold 감사 체인은 `filesearch/STRUCTURED_SEARCH_EXPERIMENT.md`(대장)에 있다.
과거 C29/Luna 재현 절차는 이 파일의 git 이력(구 `C29_REPRO.md`) 참조.

## 필수 환경

- Python 3.9+
- 로컬에 인증된 `claude` CLI (agent model opus, reasoning effort medium)
- Meta 질의 embedding: `sentence-transformers`, `torch`, `numpy` 환경 (모델 `dragonkue/BGE-m3-ko`) 또는 OpenAI-compatible `EMBED_ENDPOINT`
- Meta 문서·V9 view·embedding matrix는 `vector_search/out/` 아래 Git 추적 파일 사용. 질의 embedding만 런타임 생성.

## 고정 데이터 무결성 (전부 Git 추적)

```text
filesearch/out/elements_u3.jsonl                       # element 우주 u3 (32,046)
  9d13e8d61781a4d39593795b0a30b33d41da7e1134335f04056e3d35540f5a49
filesearch/out/elements_u3jo.jsonl                     # 조 단위 채점 우주 u3jo (7,612)
  44d856004c473e186880cbe749c70d701d1ae25f528c6128cc575a004e8d980c
filesearch/out/tags_u4_fact_rules.jsonl                # U4 fact 태그 (규칙)
  19df8e21d9f9287394cf9d02924d5d0e2c84be7e9bf3ae9fca4f02aaacd18104
filesearch/out/tags_u5_reference_graph_v14_stats.json  # U5 참조 그래프 edge_ledger (reference_follow 입력)
  8b76b9eb15f89afe4b33c3dd00b74e2ad9ff3afc02cd66e65032d38db1e110db
filesearch/out/gold_train_scoped_u3_reviewed_overlay_v22.jsonl   # Gold v22 (213문항)
  f75805c85ed4b02f5834cd9d17a4ff203695ffccf4b7653c1544d737f6addd19
filesearch/out/qids_train_scoped_u3_reviewed_overlay_v22.json    # qid 목록 (213)
  ac610496980d4385bd41226bd84588857cf3283f0b82ec31b297216d17ae6bd4
vector_search/out/view_V9.jsonl                        # Meta V9 view
  582602982443acf6294317c58867db882f1e71c271a95ff189fa6f775a5b0b85
vector_search/out/emb/chunk_V9.npy                     # Meta V9 문서 embedding
  a741535632e28f276d174bbada641a5e5bc2b364470f93ca11463e75af545090
```

macOS `shasum -a 256 <file>`, Linux `sha256sum <file>`. SHA가 다르면 결과 비교를 진행하지 않는다.
u3/u4는 `filesearch/build_universe.py → stabilize_universe_ids.py → build_jo_universe.py → build_tags_u4_fact.py`,
U5 stats는 `build_tags_u5_reference_graph.py`로 원문 md에서 재생성 가능(전부 규칙, 결정론).

## train 213 x 2 재현

```bash
cd vector_search/noah/0819
export SEMTAG_META_PYTHON=/path/to/python-with-sentence-transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 host_agent_runner.py \
  --run r1v_full213x2_goldv22 \
  --arms r1v_verify_identity_reference_hybrid \
  --qids ../../../filesearch/out/qids_train_scoped_u3_reviewed_overlay_v22.json \
  --gold ../../../filesearch/out/gold_train_scoped_u3_reviewed_overlay_v22.jsonl \
  --reps 2 --workers 4 \
  --model opus --reasoning-effort medium \
  --timeout 240 --max-actions 5 --seed 20260850
```

공식 run(`out/host_agent/opus_r1v_full213x2_goldv20_20260824/`, gold v22 재채점)의 agent ITT 지표:

| arm | R@1 | R@5 | R@10 | suff@5 |
|---|---:|---:|---:|---:|
| r1v (최종) | .6025 | **.9147** [.878,.946] | .9300 | .9014 |

모델 비결정성 때문에 단일 재실행 수치가 동일할 필요는 없다. manifest의 Gold/data/code SHA,
agent model, reasoning effort, worker 수, seed, 오류 처리 규칙이 일치해야 비교 가능하다.
기존 run 결과의 gold 재채점만 필요하면 `rescore_agent_results.py`(에이전트 재실행 없음).

## Gold만 재사용하는 팀원용

- Gold: `gold_train_scoped_u3_reviewed_overlay_v22.jsonl` — 1행 1문항, `groups`=[AND 그룹], 그룹 안 `members`=[OR], member는 char span(`c0`,`c1`)+`jo`.
- 채점: `filesearch/scoring.py`의 fractional evidence-group Recall@K + `filesearch/units.py`(제출 id e*/j*/c* → 조 사상). 어느 검색기든 제출 상위 K id 리스트만 있으면 채점된다.
- qid/질문 문구는 lsh train QA셋 기준(test149는 별도 봉인 — 이 저장소에서 열람·사용 금지 정책 유지).

## 오류 처리 규칙

- `errors>0`인 인프라/도구 실패 cell은 지표 확정 전 동일 조건으로 재실행하고, 동일 key cell 전체를 교체한다(성공 항목만 선택 병합 금지).
- `protocol_errors>0`이어도 호스트가 회복해 최종 submit을 받은 cell은 agent ITT 행동으로 보존, 별도 건수 보고.
- 정상 도구 실행 후 빈 submit은 유효한 agent 실패로 보존.
- 최종 results 및 manifest에 원 run·retry SHA와 교체 key를 기록한다.

## 해석 한계

- 213문항은 train(evaluable) 감사 완료분이다. test149 최종 확증은 개봉 사전등록 후 별도 진행.
- corpus는 32,046 elements / 7,612 JO / 원문 1개. 11만 문서 일반화는 corpus-split 실험으로 별도 검증 필요.

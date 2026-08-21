# C29 Semantic Tag agent 실험 재현

이 문서의 대상은 PR #69에 고정된 C29 검색기와 Gold v7이다. C29는 rule router와 결정론적 Semantic Tag/BM25F 점수만 사용하며, 비에이전트 LLM과 reranker를 사용하지 않는다. Meta V9 hybrid 채널은 실험 고정 변수다.

## 필수 환경

- Python 3.9+
- 로컬에 인증된 `codex` CLI
- agent model `gpt-5.6-luna`, reasoning effort `medium`
- Meta 질의 embedding을 생성할 `sentence-transformers`, `torch`, `numpy` 환경 또는 OpenAI-compatible `EMBED_ENDPOINT`
- 로컬 embedding을 쓸 때 모델 `dragonkue/BGE-m3-ko`

Meta 문서·V9 view·embedding matrix는 `vector_search/out/` 아래 Git 추적 파일을 사용한다. 질의 embedding은 같은 모델로 런타임에 생성한다.

## 고정 데이터 무결성

```text
filesearch/out/elements_u3.jsonl
  9d13e8d61781a4d39593795b0a30b33d41da7e1134335f04056e3d35540f5a49
filesearch/out/elements_u3jo.jsonl
  44d856004c473e186880cbe749c70d701d1ae25f528c6128cc575a004e8d980c
filesearch/out/tags_u4_fact_rules.jsonl
  19df8e21d9f9287394cf9d02924d5d0e2c84be7e9bf3ae9fca4f02aaacd18104
filesearch/out/gold_c17_reviewed_final_s29_v7.jsonl
  53835dd9fddf960f63a7013c9699bdf1b23e8a045cac6773502328879ad59ada
filesearch/out/gold_train_scoped_u3_reviewed_overlay_v1.jsonl
  d8038733c364f6441dd3d8bfc9bd3ac5bccb1b640b8458f344a96291be0936af
filesearch/out/gold_train_scoped_u3_reviewed_overlay_v1_qids.json
  63d35e9f132392a659c34fd004bb60e550bcbbba38c766dd6acc074fe78752ea
```

macOS에서는 `shasum -a 256 <file>`, Linux에서는 `sha256sum <file>`로 검증한다. SHA가 다르면 결과 비교를 진행하지 않는다.

## 29문항 x 2 재현

```bash
cd vector_search/noah/0819
export SEMTAG_META_PYTHON=/path/to/python-with-sentence-transformers
export HF_HUB_OFFLINE=1
export TRANSFORMERS_OFFLINE=1

python3 host_agent_runner.py \
  --run c29_repro_s29x2 \
  --arms c26_scope_fact_tag_ensemble_hybrid c29_membership_dual_evidence_hybrid \
  --qids qids_c17_reviewed_final_s29_v3.json \
  --gold ../../../filesearch/out/gold_c17_reviewed_final_s29_v7.jsonl \
  --reps 2 --workers 4 \
  --model gpt-5.6-luna --reasoning-effort medium \
  --timeout 180 --max-actions 4 --seed 20260848
```

공유 clean 결과의 agent 지표:

| arm | R@1 | R@5 | R@10 | suff@5 | suff@10 |
|---|---:|---:|---:|---:|---:|
| C26 | .5948 | .9138 | .9828 | .8793 | .9828 |
| C29 | .6121 | .9224 | .9483 | .9138 | .9483 |

모델 비결정성 때문에 단일 재실행 수치가 동일할 필요는 없다. 다만 manifest의 Gold/data/code SHA, agent model, reasoning effort, worker 수, seed와 오류 처리 규칙은 일치해야 한다.

## 281문항 evaluable train 재현

원 347문항에서 `groups=[]` 65문항과 scope 불명 q0488 1문항을 제외한다. zero-group은 검색 실패가 아니라 Gold 미구성으로 인한 unscorable 데이터다.

```bash
python3 host_agent_runner.py \
  --run c29_repro_evaluable_train281x1 \
  --arms c29_membership_dual_evidence_hybrid \
  --qids ../../../filesearch/out/gold_train_scoped_u3_reviewed_overlay_v1_qids.json \
  --gold ../../../filesearch/out/gold_train_scoped_u3_reviewed_overlay_v1.jsonl \
  --reps 1 --workers 8 \
  --model gpt-5.6-luna --reasoning-effort medium \
  --timeout 180 --max-actions 4 --seed 20260849
```

## 오류 처리 규칙

- `errors>0`인 인프라/도구 실패 cell은 지표 확정 전 동일 조건으로 재실행한다.
- 재실행 결과는 동일 key cell 전체를 교체한다. 성공한 항목만 선택적으로 합치지 않는다.
- `protocol_errors>0`이어도 호스트가 회복하여 최종 submit을 받은 cell은 agent ITT 행동으로 보존하고 별도 건수로 보고한다. 정상 성공할 때까지 재실행하지 않는다.
- 정상 도구 실행 후 agent가 빈 submit을 낸 경우는 유효한 agent 실패로 보존한다.
- 최종 `results_clean.jsonl` 및 manifest에 원 run·retry SHA와 교체 key를 기록한다.

## 해석 한계

- 29문항은 모두 retrieval-blind completeness 검토를 거친 개발셋으로, 전체 train 난이도를 대표하지 않는다.
- 281문항 중 252문항은 scope mapping만 되었고 completeness human review를 거치지 않았다.
- 현재 corpus 검증은 32,046 elements / 7,612 JO / source document 1개이다. 11만 문서 일반화는 다문서 corpus-split 실험으로 별도 검증해야 한다.

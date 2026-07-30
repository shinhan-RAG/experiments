# shinhan-uw KT Cloud 실행 런북

대상: 신한라이프 언더라이팅 파싱본 7문서 → dr-dci **Part 1 · 3 · 4**
브랜치: `feat/shinhan-uw-element-retrieval`
정제 근거: `dr-dci/docs/SHINHAN_UW_DATA_CLEANING_KO.md`

기존 `KT_RUNBOOK_20260730.md`의 하드 규칙을 **그대로 상속**한다.

> - `nohup` 없이 파트를 실행하지 않는다
> - 두 파트를 동시에 실행하지 않는다 (임베딩 shim이 단일 스레드다)
> - **PID 993785(`:8101` shim)를 kill 하지 않는다** — base 환경에서는 다시 못 띄운다
>   (`Qwen2Config has no attribute rope_theta`). `embed` conda 환경 전용이다
> - 파트마다 **즉시** `results/`+`logs/`를 아카이브한다. `save_results`는 파트 완료
>   시에만 호출되므로(`run_experiment.py:1186-1216`) 중간 크래시는 완료된 arm까지 전부 유실된다
> - 아카이브는 **`dr-dci/` 아래에서** 만든다 — 홈 디렉터리는 Jupyter `root_dir` 밖이라 404
> - 임베딩 백엔드를 바꾸면 `cache/embeddings/*.npz`를 **삭제하지 말고 이동**한다
>   (`_cache_key`가 엔드포인트를 해싱하지 않아 그대로 `[Cache HIT]`이 된다, `retriever.py:67-76`)

환경: `work@main1`, workdir `~/source/embed_exp/experiments/dr-dci`,
A100-SXM4-80GB ×2 (공유 계정 — 남의 프로세스를 죽이지 않는다),
접근은 **Jupyter 터미널 + 파일 브라우저뿐**(ssh/scp 없음).
conda root는 `/home/work/source/miniconda3`.

---

## 로컬에서 이미 끝난 것 (재실행 불필요)

| 산출물 | 건수 | 생성 |
|---|---|---|
| `data/raw/shinhan-uw/corpus.jsonl` | 3,365청크 (gold 563 + distractor 2,802) | `build_shinhan_uw_corpus.py` |
| `data/raw/shinhan-uw/{queries,qrels,qa_meta}.jsonl` | 질의 50 · gold 50 · span 60(verbatim 60) | `build_shinhan_uw_qa.py` |
| `data/reference_answers/shinhan-uw.json` | 50 | `build_shinhan_reference_answers.py --dataset shinhan-uw` |
| `data/subsets/shinhan-uw/4k.json` | 3,365 (`full_corpus: true`) | 코퍼스 빌더 |
| `data/metadata/shinhan-uw-parser_4k.json` | 3,365 (결정론적) | 코퍼스 빌더 |
| `data/tags/shinhan-uw/approach_p/4k.json` | 3,365 (결정론적) | 코퍼스 빌더 |
| 감사: `corpus_stats` · `audit_{duplicates,long_chunks,element_type}` · `manifest` · `qa_manifest` | — | 코퍼스/QA 빌더 |

서버에서 만들어야 하는 것은 **LLM 증강**뿐이다:
`taxonomy` · `prefix` · `metadata`(LLM) · `tags` A/B/C.

**업로드할 tarball은 `shinhan_uw_data.tar.gz`(2.9MB) 하나다.**
`part4_datasets.tar.gz`·`aug_part*.tar.gz`는 **법률 실험용**이며 이번 작업과 무관하다
(법률 Part 4를 재개할 때만 필요하다).

### 파트별 선행 조건

| 파트 | 증강 필요 | `:8100` Qwen3-8B | `:8002` 리랭커 | 지금 실행 가능? |
|---|---|---|---|---|
| **Part 4** (DR-DCI vs Hybrid) | ✗ (`augment: false`) | 불요 | **필요** | **예** — `:8101`+`:8002`만 있으면 된다 |
| Part 1 (9 arm 적층) | ✓ | 필요 | 불요 | 증강 생성 후 |
| Part 3 (태그 A/B/C) | ✓ | 필요 | 불요 | 증강 생성 후 |

Part 1의 `baseline`·`parser_meta_only` 두 arm은 결정론적 산출물만 쓰므로 증강 없이도
로드되지만, 한 파트는 arm 전체를 한 번에 돌리므로 Part 1 자체는 증강이 있어야 시작된다.

---

## 0. 환경 점검 — 먼저 할 일

GPU 여유 · 로컬 모델 가중치 · conda 환경 · 서비스 · 데이터를 한 번에 확인한다.
읽기 전용이라 아무것도 바꾸지 않는다.

```bash
cd ~/source/embed_exp/experiments/dr-dci
bash scripts/check_kt_env.sh
```

종료 코드로 판정한다.

| exit | 뜻 |
|---|---|
| **0** | Part 1 / 3 / 4 전부 실행 가능 → 바로 preflight(4단계) |
| **1** | **Part 4 만 지금 실행 가능** (`augment: false` 라 증강 불요). Part 1/3 은 `:8100` 기동 후 증강 생성 필요 |
| **2** | 실행 불가 — 임베딩(`:8101`) 또는 데이터가 없다 |

수동으로 볼 때는:

```bash
nvidia-smi                                                    # 여유 VRAM
nvidia-smi --query-compute-apps=pid,used_memory --format=csv   # 점유 프로세스
ls -d ~/.cache/huggingface/hub/models--Qwen--Qwen3-8B          # 로컬 가중치 존재 여부
du -sh ~/.cache/huggingface/hub/models--*                      # 모델별 캐시 크기
curl -s localhost:8101/v1/models; curl -s localhost:8002/v1/models; curl -s localhost:8100/v1/models
```

7/29 기준 A100 2장이 거의 만석이었다(78.9/81.9 GiB, 76.6/81.9 GiB). Qwen3-8B는
bf16 ~16GB + KV가 필요해 **여유 24GB 이상**이 안전하다.
**안 들어가면 co-tenant 작업이 끝날 때까지 기다린다. 남의 프로세스를 죽이지 않는다.**

## 1. 증강 생성 LLM 기동 (`:8100` Qwen3-8B)

`scripts/utils.py:10-11`이 이 주소·모델명으로 **하드코딩**돼 있고, 법률 실험의
증강도 같은 모델로 만들었으므로 조건 일치를 위해 바꾸지 않는다.

```bash
cd ~/source/embed_exp/experiments/dr-dci
CUDA_VISIBLE_DEVICES=<여유GPU> nohup \
  /home/work/source/miniconda3/envs/infopt-vllm/bin/vllm serve Qwen/Qwen3-8B \
  --port 8100 --gpu-memory-utilization <여유에 맞춰> --max-model-len 8192 \
  > logs/qwen3_8100.log 2>&1 &

curl -s localhost:8100/v1/models        # 200 + 모델명 확인까지 기다린다
```

## 2. 데이터 업로드

`.gitignore`가 `data/`를 제외하므로 tarball을 Jupyter 파일 브라우저로 올린다.

```bash
cd ~/source/embed_exp/experiments/dr-dci
tar xzf shinhan_uw_data.tar.gz -C data/
git checkout feat/shinhan-uw-element-retrieval && git pull

# 무결성 확인
wc -l data/raw/shinhan-uw/corpus.jsonl              # 3365
wc -l data/raw/shinhan-uw/queries.jsonl             # 50
python -c "import json;print(len(json.load(open('data/reference_answers/shinhan-uw.json'))))"   # 50
```

## 3. LLM 증강 생성 (`:8100` 필요, 대상 3,365청크)

```bash
cd ~/source/embed_exp/experiments/dr-dci
PYTHONPATH=. python scripts/build_taxonomy.py shinhan-uw --size=4000
PYTHONPATH=. python scripts/build_prefix.py   shinhan-uw --size=4000
PYTHONPATH=. python scripts/build_metadata.py shinhan-uw --size=4000
PYTHONPATH=. python scripts/build_tags.py     shinhan-uw --size=4000 --approaches A      # Part 1
PYTHONPATH=. python scripts/build_tags.py     shinhan-uw --size=4000 --approaches B,C    # Part 3
```

스키마는 이 코퍼스 전용으로 새로 썼다 — `config/taxonomy_schemas/shinhan-uw.yaml`
(L1: `IssueLimit`/`SimplifiedIssue`/`MedicalUW`/`HealthExam`/`OperationNotice`/`Other`),
`config/metadata_schemas/shinhan-uw.yaml`. 기존 `shinhan.yaml`의 L1(약관/상품/보험금심사/
고객상담/LICO)은 이 7문서가 전부 언더라이팅 문서라 한 칸으로 뭉쳐 분류축이 죽는다.

## 4. Preflight — exit 0이어야 다음으로 간다

`--step` 기본값이 `baseline` + `taxonomy_only` **둘뿐**이므로 9개 arm을 전부 지정한다.

```bash
PYTHONPATH=. python scripts/audit_part12.py --config config/experiment_shinhan_uw.yaml \
  --step baseline --step taxonomy_only --step tags_only --step prefix_only \
  --step metadata_only --step parser_meta_only --step stack_tax_tags \
  --step stack_tax_tags_prefix --step stack_all \
  --output results/preflight_shinhan_uw.json
echo "exit=$?"     # 0 = ready, 2 = blocked
```

기대: `status: ready`, `blockers: []`, `duplicate_arms: []`, 그리고
`augmentations.requirements`가 다음과 같아야 한다.

```
taxonomy: ["enabled"]   prefix: ["enabled"]
tags:     ["A","P"]      metadata: ["", "parser"]
```

`metadata` 요구가 `["", "parser"]` 두 개로 나오는지 반드시 확인한다 —
`""`는 LLM 생성분(`shinhan-uw_4k.json`), `"parser"`는 결정론적
(`shinhan-uw-parser_4k.json`)이다. 하나만 나오면 `parser_meta_only` arm이
라벨만 남고 처치 없이 돌아간다.

## 5. 서비스 확인

```bash
curl -s localhost:8101/v1/models     # 임베딩 gte-Qwen2-1.5B-instruct (PID 993785, kill 금지)
curl -s localhost:8002/v1/models     # 리랭커 bge-reranker-v2-m3
```

**`:8002`가 죽어 있으면 Part 4의 hybrid arm이 `Connection refused`로 전멸하고
`check_arm_failure_rate`가 파트를 abort시킨다** — 7/29 법률 Part 2가 정확히 이 이유로
죽었고 완료된 arm까지 전부 유실됐다. 없으면 먼저 띄운다:

```bash
CUDA_VISIBLE_DEVICES=<GPU> nohup \
  /home/work/source/miniconda3/envs/infopt-vllm/bin/vllm serve BAAI/bge-reranker-v2-m3 \
  --port 8002 --runner pooling --gpu-memory-utilization 0.15 --max-model-len 8192 \
  > logs/reranker_8002.log 2>&1 &
```

## 6. 파트 실행 — 하나씩, 끝날 때마다 아카이브

```bash
cd ~/source/embed_exp/experiments/dr-dci
mkdir -p logs

# Part 1 — 9 arm × 50질의. 증강 적층 효과 (핵심 파트)
nohup python run_experiment.py --part 1 --config config/experiment_shinhan_uw.yaml \
  > logs/uw_p1.log 2>&1 &
# 완료 후:
tar czf results_uw_p1_$(date +%Y%m%d_%H%M).tar.gz results/ logs/uw_p1.log

# Part 3 — 태그 스킴 A/B/C
nohup python run_experiment.py --part 3 --config config/experiment_shinhan_uw.yaml \
  > logs/uw_p3.log 2>&1 &
tar czf results_uw_p3_$(date +%Y%m%d_%H%M).tar.gz results/ logs/uw_p3.log

# Part 4 — DR-DCI vs Hybrid (augment 없음, 리랭커 필요)
nohup python run_experiment.py --part 4 --config config/experiment_shinhan_uw.yaml \
  > logs/uw_p4.log 2>&1 &
tar czf results_uw_p4_$(date +%Y%m%d_%H%M).tar.gz results/ logs/uw_p4.log
```

임베딩은 `prefix` on/off 두 가지 인덱스가 필요하므로 3,365청크 × 2회 색인이
캐시에 쌓인다. 청크가 작아(p95 1,391자) 법률 50k 색인과는 비교가 안 되게 빠르다.

## 7. 결과 검증

```bash
PYTHONPATH=. python scripts/verify_results.py results/
```

**Part 1이 끝나면 가장 먼저 볼 것:**

1. **9개 arm의 `avg_gold_recall`이 전부 같은가?** 같으면 증강이 적용되지 않았다는 뜻이다
   (`verify_results.py` check 3). `manifest.preflight`와 arm별
   `avg_taxonomy_filtered_pulls`를 함께 확인한다.
2. **`parser_meta_only` vs `tags_only` / `metadata_only`.** 이 실험의 고유한 질문이다 —
   LLM 증강이 파서가 공짜로 주는 사실보다 실제로 나은가. 파서 arm이 대등하면
   증강 비용의 정당성이 사라진다.
3. **judge 분모.** `judged_n`이 arm마다 다르면 accuracy를 그대로 비교하지 말고
   공통 질의 집합으로 다시 맞춘다(7/29 신한 Part 4에서 DR-DCI가 답변 생성 실패 3건으로
   `judged 46` vs hybrid `judged 49`가 나왔다).

## 8. 보고 시 지켜야 할 것

`docs/SHINHAN_UW_DATA_CLEANING_KO.md`의 "남는 한계" 10개 항목을 그대로 붙인다. 요약:

- `density@k`·`span_f1@k` **해석 금지** (분모가 top-k 총 길이, 이론상 최대 ≈0.002).
  `coverage@k`만 해석한다
- `avg_efficiency` 해석 금지 (`gold_recall / pull_count` → 1-pull hybrid가 구조적으로 유리)
- 리랭커 비대칭 명시 (hybrid만 리랭킹)
- **법률 Part 및 기존 `shinhan` 결과와 절대 점수 비교 금지**
- 후보 풀 3,365청크 → top-20이 0.6%다. 높은 recall은 **천장 효과를 먼저 의심**
- distractor 도메인이 완전히 같지 않다. recall이 잘 안 떨어지면
  "규모에 강건"이 아니라 **"distractor가 쉬웠다"**를 먼저 의심
- 질의 50 × gold 1 ⇒ 분해능 2%p. "효과 없음"과 "판정 불가"를 구분
- gold는 gpt-4o-mini 단일 생성, 사람 검수 없음

## Part 2를 하지 않는 이유

gold 563 + distractor 2,802 = 3,365청크로는 규모 확장 실험의 최소 조건(20K청크)을
충족하지 못한다. 같은 데이터를 복제해 크기만 늘리는 것은 규모 실험이 아니다.
규모 곡선은 `aihub-full`(`experiment_legal.yaml`) 또는
`shinhan-mixed`(`build_shinhan_mixed.py`) 결과를 인용한다.
`experiment_shinhan_uw.yaml`에 `part2_scaling` 블록이 없으므로 `audit_part12`는
Part 1 subset만 검사한다(`src/eval/part12_contracts.py:262`).

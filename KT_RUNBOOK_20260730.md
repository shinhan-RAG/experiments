# KT Cloud 실행 런북 — 2026-07-30

주피터 터미널에서 **위에서 아래로** 붙여넣는다. 각 블록의 판정 기준을 확인하고 넘어간다.

- 서버: `work@main1` · `~/source/embed_exp/experiments/dr-dci`
- 오늘의 변경: 임베딩 shim을 **batch 128 / :8102 / GPU1**로 새로 띄우고, 캐시를 비켜둔 뒤
  법률 Part 1→2→3→4를 **전부 새 백엔드로 재실행**한다.

> **절대 금지 3개**
> 1. `kill 993785` (기존 :8101 shim) — `base` 환경에서 재기동 불가
> 2. `nohup` 없이 파트 실행
> 3. 파트 2개 동시 실행 — shim이 단일 스레드다

---

## A0. 상태 확인 (5분)

```bash
cd ~/source/embed_exp/experiments/dr-dci
ps -ef | grep run_experiment | grep -v grep
```

```bash
ls -la results/part1_stacking/ results/part2_scaling/ results/part3_tags/ results/part4_generalization/ 2>&1
```

```bash
tail -5 logs/legal_part2.log
date; cat /proc/1157759/io | grep rchar
```

`rchar` 값을 메모한다 (A2에서 기존 속도 환산에 쓴다).

### A0-b. Part 1·3 결과가 "제대로" 있는지 판정

파일 존재만으로는 부족하다. 내용을 본다.

```bash
python3 - <<'PY'
import json, glob
REQ1 = {"baseline","taxonomy_only","tags_only","prefix_only","metadata_only",
        "stack_tax_tags","stack_tax_tags_prefix","stack_all"}
for pat, label, req in [("results/part1_stacking/*.json", "part1", REQ1),
                        ("results/part3_tags/*.json",     "part3", None)]:
    for p in sorted(glob.glob(pat)):
        d = json.load(open(p))
        s = d.get("summary") or {}
        print("=" * 70)
        print(f"{label}  {p}")
        print(f"  arms({len(s)}): {sorted(s)}")
        recalls = []
        for k, v in s.items():
            m = v.get("metrics", v) if isinstance(v, dict) else {}
            r = m.get("avg_gold_recall")
            recalls.append(r)
            print(f"    {k:28s} recall={r} acc={m.get('accuracy')} judged={m.get('judged_n')}")
        if req:
            missing = req - set(s)
            print("  [FAIL] 누락 arm:", sorted(missing)) if missing else print("  [OK] arm 8개 완비")
        uniq = {r for r in recalls if r is not None}
        if len(recalls) > 1 and len(uniq) <= 1:
            print("  [FAIL] 모든 arm의 recall이 동일 → 증강이 적용되지 않았다 (subset null 붕괴 의심)")
        elif uniq:
            print("  [OK] arm 간 recall이 서로 다름")
        if all(m is None for m in [ (v.get("metrics",v) if isinstance(v,dict) else {}).get("accuracy") for v in s.values() ]):
            print("  [WARN] accuracy 전부 None → judge 미실행 (reference_answers 확인)")
PY
```

**판정**

| 결과 | 뜻 | 조치 |
|---|---|---|
| `[OK] arm 8개 완비` + arm 간 recall 다름 | 배치8 기준선 확보 | A5 아카이브에 **반드시 포함**해서 내려받는다 |
| `[FAIL] 누락 arm` | 미완주 | 어차피 재실행하므로 그대로 진행 |
| `[FAIL] recall 동일` | 증강 미적용 버그 | 재실행 후 A6 검증에서 다시 확인 |
| 디렉터리 자체가 없음 | 실행된 적 없음 | 그대로 진행 |

> 오늘 전부 재실행하므로 이 단계의 목적은 **"새 결과와 비교할 배치8 기준선을 확보"**하는 것이다.
> 뭐가 나와도 멈추지 않는다.

---

## A1. 배치 128 shim을 GPU1 · :8102에 띄운다

서버의 `serve_embed_shim_gpu1.py`는 batch 64로 만들어져 있다. 128로 고친다.

```bash
cd /home/work/source/embed_exp
cp serve_embed_shim_gpu1.py serve_embed_shim_gpu1.py.bak64
sed -i 's/batch_size=64/batch_size=128/' serve_embed_shim_gpu1.py
grep -n 'batch_size\|8102' serve_embed_shim_gpu1.py
```

→ `batch_size=128` 과 `8102` 두 줄이 보여야 한다.

```bash
CUDA_VISIBLE_DEVICES=1 nohup /home/work/source/miniconda3/envs/embed/bin/python \
  serve_embed_shim_gpu1.py > ~/embed_shim2.log 2>&1 &
echo "SHIM2_PID=$!"
```

**`embed` 환경 절대 필수.** `base`로 띄우면 `Qwen2Config has no attribute rope_theta`로 죽는다.

```bash
sleep 90; tail -5 ~/embed_shim2.log
curl -sf -m 60 http://localhost:8102/v1/models && echo " <- 8102 OK"
```

**통과 기준**: 로그에 `device: cuda:0` + `max_seq_length: 8192` + `serving on :8102`.
(`CUDA_VISIBLE_DEVICES=1`이라 물리 GPU1이 `cuda:0`으로 보이는 것이 정상)

---

## A2. 실제 법률 청크 256개로 속도 게이트

짧은 문장으로 재면 의미가 없다.

```bash
cd ~/source/embed_exp/experiments/dr-dci
python3 - <<'PY'
import json
docs=[]
with open('data/aihub/full/corpus.jsonl') as f:
    for i,l in enumerate(f):
        if i>=256: break
        d=json.loads(l); docs.append((d.get('title','')+' '+d.get('text',''))[:4096])
json.dump({'model':'Alibaba-NLP/gte-Qwen2-1.5B-instruct','input':docs},
          open('/tmp/bench.json','w'))
print('배치:',len(docs),'| 평균 길이:',sum(len(x) for x in docs)//len(docs))
PY
```

```bash
echo '=== 8102 (batch 128, GPU1) ==='
time curl -s -m 600 -H 'Content-Type: application/json' \
  -d @/tmp/bench.json http://localhost:8102/v1/embeddings -o /dev/null
```

**8101은 재지 말 것** — 아직 도는 Part 2를 방해한다. 기존 속도는 아래로 환산한다.

```bash
date; cat /proc/1157759/io | grep rchar
```

> A0 값과 비교. **증가량 ÷ 8MB ≈ 완료 배치 수** (응답 1건 ≈ 8MB).

### 게이트

| 8102 결과 | 판정 |
|---|---|
| **256청크 ≤ 60초** (≥4.3청크/s) | **통과 → A3.** 오늘 총 75만~100만 청크가 2~6시간권 |
| 60~150초 | 진행은 가능하나 하루 완주가 빡빡하다. A3로 가되 Part 4 `legal-qa`(299,606청크)를 마지막에 두고, 시간이 모자라면 다음 날로 넘긴다 |
| **> 150초** | **멈추고 보고.** batch 64(`.bak64`)로 되돌려 OOM/스와핑 여부를 먼저 확인한다 |

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```
GPU1 메모리가 160GiB 근처까지 차 있으면 배치를 64로 낮춘다.

---

## A3. 전환 — 캐시를 반드시 비켜둔다

`_cache_key`에 백엔드 정보가 없다(`src/agent/retriever.py:67-76`).
**안 비우면 배치8 임베딩을 `[Cache HIT]`으로 조용히 재사용한다.**

```bash
cd ~/source/embed_exp/experiments/dr-dci
kill 1157759
sleep 5; ps -ef | grep run_experiment | grep -v grep
```
→ 아무것도 안 나와야 한다.

```bash
mkdir -p cache/embeddings_batch8_backup
mv cache/embeddings/*.npz cache/embeddings_batch8_backup/
echo "남은 캐시: $(ls cache/embeddings/ 2>/dev/null | wc -l)  (0이어야 함)"
echo "백업된 캐시: $(ls cache/embeddings_batch8_backup/ | wc -l)"
```

> **삭제가 아니라 이동이다.** 되돌리려면
> `mv cache/embeddings_batch8_backup/*.npz cache/embeddings/`

```bash
sed 's|localhost:8101|localhost:8102|' config/experiment_legal.yaml > config/experiment_legal_b128.yaml
diff config/experiment_legal.yaml config/experiment_legal_b128.yaml
```
→ **차이가 embedding url 1줄뿐**이어야 한다.

**:8101(993785)은 죽이지 않고 그대로 둔다.** 되돌릴 여지를 남긴다.

---

## A4. 순차 실행 — Part 1 → 3 → 2 → 4

`save_results`는 파트가 완전히 끝날 때만 호출된다(`run_experiment.py:1216`).
중간에 죽으면 그 파트는 통째로 날아간다. **하나씩, 앞 파트 종료를 확인한 뒤 다음.**

먼저 편의 변수와 사전 점검 함수:

```bash
cd ~/source/embed_exp/experiments/dr-dci
CFG=config/experiment_legal_b128.yaml
pre() { curl -sf -m 60 http://localhost:8102/v1/models >/dev/null \
     && curl -sf -m 60 http://localhost:8002/v1/models >/dev/null \
     && echo "엔드포인트 정상" || { echo "미기동 — 실행 금지"; return 1; }; }
runpart() { pre && { nohup python3 run_experiment.py --part "$1" --config "$CFG" \
     > "logs/legal_part$1_b128_$(date +%Y%m%d_%H%M%S).log" 2>&1 & echo "PART$1_PID=$!"; }; }
```

### Part 1 (20k · arm 8개) — 가장 먼저

새 백엔드가 증강 경로까지 정상인지 가장 빨리 드러난다.

```bash
runpart 1
```
```bash
sleep 60; tail -20 $(ls -t logs/legal_part1_b128_*.log | head -1)
```
→ **`[Cache MISS] Embedding ... documents`가 보여야 한다.** `[Cache HIT]`이면 A3를 덜 했다.

진행 확인:
```bash
tail -5 $(ls -t logs/legal_part1_b128_*.log | head -1); ps -ef | grep run_experiment | grep -v grep
```

### Part 3 (20k · approach A/B/C) — Part 1 종료 후

같은 20k · prefix off 텍스트셋이라 Part 1 baseline 캐시를 재사용해 거의 즉시 끝난다.

```bash
runpart 3
```

### Part 2 (20k / 50k / 110k) — 가장 긴 파트

```bash
runpart 2
```

### Part 4 (ruling-anon 53,822 + legal-qa 299,606 + longdoc 5,642) — 마지막

최대 단일 항목이라 실패 시 손실이 가장 크다. **Part 2 완주 + 아카이브 회수 후**에 시작한다.

```bash
runpart 4
```

---

## A5. 파트가 끝날 때마다 즉시 회수

**어제 Part 2가 죽으면서 완료된 arm 결과까지 전부 날아갔다. 파트별로 그때그때 내린다.**

```bash
cd ~/source/embed_exp/experiments/dr-dci
TS=$(date +%Y%m%d_%H%M%S)
tar czf results_${TS}.tar.gz results/ logs/*.log
ls -lh results_${TS}.tar.gz
```

- 수 MB면 정상. 수백 MB면 `cache/`나 `data/`가 섞인 것이다.
- 아카이브는 **반드시 `dr-dci/` 아래**에 만든다 (홈은 주피터 root_dir 밖이라 404).
- 주피터 파일 브라우저 다운로드가 유일한 회수 경로다.
- **A0-b에서 확인한 배치8 결과도 이 아카이브에 함께 들어간다** — 로컬 비교용으로 꼭 내린다.

로컬에서 압축을 풀고:
```
python scripts/verify_results.py <풀어놓은 results 경로>
python scripts/verify_results.py <새 results> --baseline <배치8 results>
```

---

## B. 신한 Part 1·3·4 — 법률이 GPU를 놓아준 뒤

**Part 2는 실행하지 않는다** (2,802청크 < 최소 20K).

### B1. Qwen3-8B 기동 → 증강 4종 생성

```bash
pkill -f "vllm serve.*Qwen3-8B"
CUDA_VISIBLE_DEVICES=1 nohup /home/work/source/miniconda3/envs/infopt-vllm/bin/vllm \
  serve Qwen/Qwen3-8B --port 8100 --gpu-memory-utilization 0.4 > ~/vllm_qwen.log 2>&1 &
until curl -sf http://localhost:8100/v1/models >/dev/null; do sleep 10; echo -n .; done; echo " up"
```

```bash
cd ~/source/embed_exp/experiments/dr-dci
python3 scripts/build_taxonomy.py shinhan --size=3000
python3 scripts/build_prefix.py   shinhan --size=3000
python3 scripts/build_metadata.py shinhan --size=3000
python3 scripts/build_tags.py     shinhan --size=3000 --approaches A,B,C
```

`subset: 3000` → `size_key = "3k"`이므로 경로가 정확히 다음이어야 한다
(`run_experiment.py:206-229`):

```bash
ls -la data/taxonomy/shinhan_3k.json data/prefix/shinhan_3k.json \
       data/metadata/shinhan_3k.json data/tags/shinhan/approach_*/3k.json
```
→ **6개 파일**이 다 있어야 한다.

### B2. 프리플라이트 — 기본 `--step`이 2개뿐이라 8개를 모두 지정

```bash
PYTHONPATH=. python3 scripts/audit_part12.py --config config/experiment_shinhan.yaml --size 3000 \
  --step baseline --step taxonomy_only --step tags_only --step prefix_only \
  --step metadata_only --step stack_tax_tags --step stack_tax_tags_prefix --step stack_all
echo "exit=$?"
```
**exit=0이 아니면 Part 1을 걸지 않는다.**

### B3. 임베딩 백엔드를 :8102로 맞춘다

법률을 배치 128로 재실행했으므로 신한도 같은 백엔드를 써야 두 도메인을 나란히 놓을 수 있다.

```bash
sed 's|localhost:8101|localhost:8102|' config/experiment_shinhan.yaml > config/experiment_shinhan_b128.yaml
diff config/experiment_shinhan.yaml config/experiment_shinhan_b128.yaml
```

`run_shinhan_all.sh`의 `CFG` 한 줄을 환경변수로 덮어쓸 수 있게 바꾼다 (로컬 커밋에 포함됨):

```bash
grep -n 'CFG=' run_shinhan_all.sh      # CFG="${CFG:-config/experiment_shinhan.yaml}" 여야 함
```

### B4. 실행

```bash
cd ~/source/embed_exp/experiments/dr-dci
curl -sf -m 60 http://localhost:8102/v1/models >/dev/null \
  && curl -sf -m 60 http://localhost:8002/v1/models >/dev/null \
  && CFG=config/experiment_shinhan_b128.yaml nohup ./run_shinhan_all.sh 1 3 4 \
       > logs/shinhan_all_$(date +%Y%m%d_%H%M%S).log 2>&1 & echo "PID=$!"
```

2,802청크라 법률에 비해 매우 빠르다. 끝나면 A5로 회수한다.

---

## C. (선택) 신한 혼합 코퍼스 Part 2 — 신한이 끝난 뒤

신한 코퍼스 2,802청크로는 Part 2 최소 조건(20K)을 못 채운다. **같은 보험 도메인의 법률
문서를 distractor 로만** 섞어 20k/50k/110k 를 만든다. 질의·gold 는 신한 50건 고정이므로
청크 단위 평가가 그대로 유지된다.

빌더는 **GPU·LLM 을 쓰지 않고 50초면 끝난다.** corpus 가 633MB 라 업로드하지 말고
서버에서 직접 생성한다 (법률 원본이 이미 서버에 있다).

```bash
cd ~/source/embed_exp/experiments/dr-dci
git pull                                   # build_shinhan_mixed.py + config 받기
python3 scripts/build_shinhan_mixed.py
```

**통과 기준** (출력에 그대로 찍힌다):

```
풀 합계: T1 53,178청크  T2 122,563청크
  20k: 신한 2,802 + distractor 17,197 = 19,999청크  (parent T1 7,103 / T2 0)
  50k: 신한 2,802 + distractor 47,198 = 50,000청크  (parent T1 19,640 / T2 0)
 110k: 신한 2,802 + distractor 107,198 = 110,000청크  (parent T1 22,184 / T2 27,573)
  [OK] 20k/50k/110k: gold 50 포함, 하위 티어 중첩 유지
```

20k·50k 가 **T2 0** 이어야 한다(순수 보험 티어). T2 가 섞이면 필터가 어긋난 것이다.

```bash
sed 's|localhost:8101|localhost:8102|' config/experiment_shinhan_mixed.yaml \
  > config/experiment_shinhan_mixed_b128.yaml
curl -sf -m 60 http://localhost:8102/v1/models >/dev/null \
  && curl -sf -m 60 http://localhost:8002/v1/models >/dev/null \
  && nohup python3 run_experiment.py --part 2 --config config/experiment_shinhan_mixed_b128.yaml \
       > logs/shinhan_mixed_part2_$(date +%Y%m%d_%H%M%S).log 2>&1 & echo "PID=$!"
```

임베딩 총량은 20k+50k+110k = 약 180,000청크다. 법률 Part 2 와 비슷한 규모이므로
**법률·신한이 모두 끝난 뒤에만** 시작한다.

### 결과 해석의 1순위 점검

**20k 에서 hybrid recall 이 순정 3k(2,802청크) 대비 거의 떨어지지 않으면
"규모에 강건하다"가 아니라 "distractor 가 너무 쉬웠다"를 먼저 의심할 것.**
법률 판례 문체는 신한 실무문서(약관·표·수식)와 확연히 달라 dense 검색이 쉽게 걸러낸다.
비교 기준은 `results/part1_stacking` 의 신한 baseline arm recall 이다.

키워드 필터의 정밀도는 `data/raw/shinhan-mixed/audit_tier_sample.json` 의 T1 표본 50건을
눈으로 확인하고 `reviewed` 를 갱신한다.

---

## 결과 해석 시 잊지 말 것

1. **`density`·`span_f1` 보고 제외** — 분모가 top-k 전체 길이라 이론 최대치가 약 0.002다
   (`src/eval/span_metrics.py:75`). `coverage`만 해석한다.
2. **`efficiency` 해석 금지** — `gold_recall / pull_count` 정의라 1회 pull하는 hybrid가 구조적으로 이긴다.
3. **법률 hybrid @ 20k의 recall 1.0을 그대로 믿지 말 것.** 50k·110k에서도 1.0이 유지되면
   판시사항=질의 / 본문=gold 구성의 어휘 중복 산물이라는 강한 증거다.
4. **신한 accuracy는 분모를 병기한다.** DR-DCI `judged 46`은 qid 24(answer 공백) 1건 +
   답변 생성 실패 3건이다. 같은 49건 기준이면 DR-DCI는 24/49 = **0.490**.
5. **신한과 법률의 절대 점수를 직접 비교하지 말 것.** 신한은 청크 단위, 법률은 parent 단위 평가다.

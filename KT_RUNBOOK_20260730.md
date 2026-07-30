# KT Cloud 실행 런북 — 2026-07-30

주피터 터미널에서 **위에서 아래로** 붙여넣는다. 각 블록의 판정 기준을 확인하고 넘어간다.

- 서버: `work@main1` · `~/source/embed_exp/experiments/dr-dci`
- **현재 상태: 법률 Part 1·3 완료(배치8), Part 2·4 미완.**
- 오늘의 변경: 임베딩 shim을 **batch 128 / :8102 / GPU1**로 새로 띄우고 **미완인 Part 2·4를
  끝낸다.** Part 1·3 재실행 여부는 A2-b 동등성 테스트 결과가 결정한다.

> **절대 금지 3개**
> 1. `kill 993785` (기존 :8101 shim) — `base` 환경에서 재기동 불가
> 2. `nohup` 없이 파트 실행
> 3. 파트 2개 동시 실행 — shim이 단일 스레드다

### 전체 흐름

```
A0  상태 확인 · Part 1·3 결과 검증 후 즉시 회수
A1  batch 128 shim → :8102
A2  속도 게이트 (256청크 8초 이내여야 하룻밤 완주)
A2-b 동등성 테스트 ← 오늘의 갈림길
      ├─ EQUIVALENT → A3-보존 : 캐시 살림. Part 2·4 만. Part 1·3 재실행 없음
      └─ DIFFERENT  → A3-초기화: 캐시 비킴. Part 2·4·1·3 전부
A4  순차 실행 (Part 2 → 4 → [1 → 3])
A5  파트마다 즉시 회수
B   신한 Part 1·3·4
C   (선택) 신한 혼합 코퍼스 Part 2
```

**총 작업량은 최대 1,404,553청크다.** 하룻밤에 끝내려면 256청크를 8초 이내로 처리해야 하는데
A100 한 장의 이론적 상한에 가깝다. **오늘 완주는 낙관적으로도 어렵고, 야간 실행을 전제로 한다.**
그래서 A5(파트별 즉시 회수)가 이 런북에서 가장 중요한 단계다.

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
| `[OK] arm 8개 완비` + arm 간 recall 다름 | **Part 1·3 이 유효하다** | A2-b 가 EQUIVALENT 면 그대로 최종 결과로 쓴다 |
| `[FAIL] 누락 arm` | 미완주 | A2-b 결과와 무관하게 **Part 1 재실행 필요** |
| `[FAIL] recall 동일` | 증강 미적용 버그 | config 의 `subset` 이 non-null 인지 확인 후 재실행 |

**이 결과를 지금 바로 내려받는다.** Part 1·3 이 오늘 유일하게 이미 확보된 결과이고,
EQUIVALENT 분기에서는 이것이 최종 결과가 된다.

```bash
cd ~/source/embed_exp/experiments/dr-dci
tar czf legal_part13_batch8_$(date +%Y%m%d_%H%M%S).tar.gz \
  results/part1_stacking/ results/part3_tags/ logs/legal_part1*.log logs/legal_part3*.log
ls -lh legal_part13_batch8_*.tar.gz
```

주피터 파일 브라우저로 받은 뒤 로컬에서:
```
python scripts/verify_results.py <풀어놓은 results 경로>
```

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

## A2. 속도 게이트 — 실제 법률 청크 256개

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
time curl -s -m 900 -H 'Content-Type: application/json' \
  -d @/tmp/bench.json http://localhost:8102/v1/embeddings -o /dev/null
```

### 실제 필요 작업량 — 1,404,553청크

`src/hybrid/pipeline.py:48` 이 `namespace="hybrid"` 로 **별도 캐시**를 쓴다. hybrid arm 이
있는 파트는 같은 코퍼스를 두 번 임베딩한다.

| 파트 | 청크 | 비고 |
|---|---|---|
| Part 2 | 648,136 | 20k 76,554 / 50k 181,202 / 110k 390,380 (각 dr-dci+hybrid) |
| Part 4 | 718,140 | ruling-anon 107,644 / **legal-qa 599,212** / longdoc 11,284 |
| Part 1 | 38,277 | **Part 2 뒤에 돌리면** prefix-on 이 캐시 HIT → prefix-off 만 신규 |
| Part 3 | 0 | Part 1 캐시 재사용 |
| **합계** | **1,404,553** | |

### 게이트 판정표

| 256청크 소요 | 초당 | 1.4M 완주 | 판정 |
|---|---|---|---|
| 60초 | 4.3 | **91시간** | 불가 |
| 30초 | 8.5 | 46시간 | 불가 |
| 15초 | 17 | 23시간 | 이틀 |
| **8초** | 32 | **12시간** | 야간 포함 하루 |
| 6초 | 43 | 9시간 | 하루 안 |

**하룻밤에 끝내려면 256청크를 8초 이내로 처리해야 한다.** A100 한 장에서 1.5B 모델 ·
평균 1,500토큰 입력의 이론적 상한에 가까운 수치다. **배치만 올려서 오늘 완주는
낙관적으로도 어렵다** — A3 의 캐시 보존이 통하는지가 실제 관건이다.

30초를 넘으면 멈추고 보고할 것. batch 64(`serve_embed_shim_gpu1.py.bak64`)로 되돌려
OOM/스와핑을 먼저 확인한다.

```bash
nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu --format=csv
```

---

## A2-b. ⭐ 동등성 테스트 — 배치8 캐시를 살릴 수 있는지

`_cache_key` 에 백엔드 정보가 없다는 점(§2-D)이 **여기서는 이득이 된다.** 배치 8과 128의
임베딩이 사실상 같다면 config 만 :8102 로 바꿔도 기존 `.npz` 를 그대로 `[Cache HIT]` 으로
읽는다 → 이미 끝난 것을 다시 만들지 않고, **Part 1·3 재실행도 불필요해진다.**

원칙적으로 `sentence-transformers` 의 `batch_size` 는 임베딩을 바꾸지 않는다. 다만 배치 안
패딩 길이가 달라져 fp16 누적 순서가 바뀌므로 미세한 수치 차이는 생길 수 있다. **가정하지 말고
측정한다.**

32청크만 쓴다. 배치 8이면 4배치, 배치 128이면 1배치 — 배치 경계 차이를 정확히 겨냥하면서
:8101 을 쓰는 Part 2 의 진행을 1분 남짓만 방해한다.

```bash
cd ~/source/embed_exp/experiments/dr-dci
python3 - <<'PY'
import json
docs=[]
with open('data/aihub/full/corpus.jsonl') as f:
    for i,l in enumerate(f):
        if i>=32: break
        d=json.loads(l); docs.append((d.get('title','')+' '+d.get('text',''))[:4096])
json.dump({'model':'Alibaba-NLP/gte-Qwen2-1.5B-instruct','input':docs},
          open('/tmp/eq.json','w'))
print('동등성 테스트 입력:',len(docs),'청크')
PY
```

```bash
time curl -s -m 900 -d @/tmp/eq.json -H 'Content-Type: application/json' \
  http://localhost:8101/v1/embeddings -o /tmp/eq_8101.json
time curl -s -m 900 -d @/tmp/eq.json -H 'Content-Type: application/json' \
  http://localhost:8102/v1/embeddings -o /tmp/eq_8102.json
```

```bash
python3 - <<'PY'
import json
import numpy as np
a = np.array([d["embedding"] for d in json.load(open('/tmp/eq_8101.json'))["data"]])
b = np.array([d["embedding"] for d in json.load(open('/tmp/eq_8102.json'))["data"]])
print("shape", a.shape, b.shape)
an = a / np.linalg.norm(a, axis=1, keepdims=True)
bn = b / np.linalg.norm(b, axis=1, keepdims=True)
cos = (an * bn).sum(1)
print(f"cosine  min={cos.min():.6f}  mean={cos.mean():.6f}")
print(f"max abs diff = {np.abs(a-b).max():.3e}")
# 검색 순위가 실제로 바뀌는지 — 이게 최종 판정 기준이다
sa, sb = an @ an.T, bn @ bn.T
ra, rb = np.argsort(-sa, 1)[:, :10], np.argsort(-sb, 1)[:, :10]
same = float(np.mean([len(set(x) & set(y)) / 10 for x, y in zip(ra, rb)]))
print(f"top-10 이웃 일치율 = {same:.4f}")
verdict = "EQUIVALENT" if cos.min() > 0.9999 and same > 0.99 else "DIFFERENT"
print(f"\n>>> {verdict}")
PY
```

### 판정

| 결과 | 뜻 | 다음 |
|---|---|---|
| **EQUIVALENT** (cosine min > 0.9999 · top-10 일치 > 0.99) | 배치 크기가 임베딩을 바꾸지 않는다 | **A3-보존** 으로 |
| **DIFFERENT** | 배치가 결과를 바꾼다 | **A3-초기화** 로. 파트 간 조건 통일을 위해 전면 재생산이 불가피하다 |

> cosine 이 0.9999~0.999 사이면서 top-10 일치가 1.0 이면 실무상 EQUIVALENT 로 봐도 된다.
> 판정 근거를 보고서에 그대로 옮겨 적을 것.

---

## A3-보존. EQUIVALENT 인 경우 — 캐시를 그대로 둔다

```bash
cd ~/source/embed_exp/experiments/dr-dci
kill 1157759
sleep 5; ps -ef | grep run_experiment | grep -v grep    # 없어야 함
```

**캐시는 손대지 않는다.** 다만 지금 도는 50k 는 통째로 날아간다 — `np.savez` 가 해당 인덱스의
임베딩을 **전부** 끝낸 뒤에만 호출되므로(`src/agent/retriever.py:122-127`) 중간 진행은 저장되지
않는다. 어제 하루 종일 돌린 50k 는 회수할 수 없다.

```bash
ls cache/embeddings/*.npz | wc -l      # 보존되는 캐시 개수. 0이면 뭔가 잘못됐다
du -sh cache/embeddings/
```

살아 있을 것으로 기대되는 것 (약 114,831청크 상당):
- 20k dr-dci prefix-off (Part 1)
- 20k dr-dci prefix-on (Part 1 · Part 2 stack_all 공용)
- 20k hybrid (Part 2)

```bash
sed 's|localhost:8101|localhost:8102|' config/experiment_legal.yaml > config/experiment_legal_b128.yaml
diff config/experiment_legal.yaml config/experiment_legal_b128.yaml
```
→ 차이가 embedding url 1줄뿐이어야 한다.

**Part 1·3 은 재실행하지 않는다.** 임베딩이 동등하므로 어제 결과가 그대로 유효하다.
남은 작업은 **Part 2 의 50k·110k + Part 4 전체 ≈ 1,289,722청크.**

→ **A4 로. 실행할 파트는 2, 4 두 개다.**

---

## A3-초기화. DIFFERENT 인 경우 — 캐시를 비켜둔다

```bash
cd ~/source/embed_exp/experiments/dr-dci
kill 1157759
sleep 5; ps -ef | grep run_experiment | grep -v grep
```
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

→ **A4 로. 실행할 파트는 2, 4, 1, 3 네 개다** (1·3 도 새 백엔드로 다시).

---

**두 분기 공통: :8101(993785)은 죽이지 않고 그대로 둔다.** 되돌릴 여지를 남긴다.

---

## A4. 순차 실행 — Part 2 → 4 (→ DIFFERENT 면 1 → 3)

**순서를 바꿨다.** 미완인 2·4 를 먼저 끝낸다. 그리고 Part 1 은 **Part 2 뒤에 돌려야** 싸다 —
Part 2 의 `stack_all` arm 이 prefix=ON 이라(`run_experiment.py:795`) 20k prefix-on 캐시를
만들어 주고, Part 1 은 prefix-off 38,277청크만 새로 하면 된다. Part 3 은 Part 1 캐시를 그대로
쓴다. 1·3 재실행의 한계비용이 **전체의 2.7%** 로 떨어진다.

`save_results` 는 파트가 완전히 끝날 때만 호출된다(`run_experiment.py:1216`).
중간에 죽으면 그 파트는 통째로 날아간다. **하나씩, 앞 파트 종료를 확인한 뒤 다음.**

```bash
cd ~/source/embed_exp/experiments/dr-dci
CFG=config/experiment_legal_b128.yaml
pre() { curl -sf -m 60 http://localhost:8102/v1/models >/dev/null \
     && curl -sf -m 60 http://localhost:8002/v1/models >/dev/null \
     && echo "엔드포인트 정상" || { echo "미기동 — 실행 금지"; return 1; }; }
runpart() { pre && { nohup python3 run_experiment.py --part "$1" --config "$CFG" \
     > "logs/legal_part$1_b128_$(date +%Y%m%d_%H%M%S).log" 2>&1 & echo "PART$1_PID=$!"; }; }
```

### Part 2 (20k / 50k / 110k) — 먼저

```bash
runpart 2
```
```bash
sleep 120; tail -30 $(ls -t logs/legal_part2_b128_*.log | head -1)
```

**A3-보존 분기라면 20k 구간에서 `[Cache HIT]` 이 떠야 한다.** `[Cache MISS]` 면 캐시 키가
안 맞는 것이니(모델명·subset·prefix 확인) 멈추고 원인을 찾는다. 헛되게 20k 를 다시 만들 이유가 없다.
**A3-초기화 분기라면 전부 `[Cache MISS]` 가 정상이다.**

진행 확인:
```bash
tail -5 $(ls -t logs/legal_part2_b128_*.log | head -1); ps -ef | grep run_experiment | grep -v grep
```

Part 2 가 끝나면 **즉시 A5 로 회수한다.** 어제 여기서 다 날렸다.

### Part 4 — 가장 큰 파트

`ruling-anon` 53,822 + **`legal-qa` 299,606**(subset null, 전체 코퍼스) + `longdoc` 5,642.
dr-dci·hybrid 각각이므로 718,140청크다. **legal-qa 만 전체의 43%** 다.

```bash
runpart 4
```

밤을 넘길 가능성이 높다. `nohup` 이라 세션이 끊겨도 살아 있다.

### Part 1 → Part 3 — **A3-초기화(DIFFERENT) 분기에서만**

```bash
runpart 1      # Part 2 뒤라 prefix-on 은 캐시 HIT. prefix-off 38,277 만 신규
```
```bash
runpart 3      # Part 1 캐시 재사용 → 임베딩 거의 0
```

EQUIVALENT 분기에서는 **돌리지 않는다.** 어제 결과가 그대로 유효하다.

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
- **Part 2 가 끝나는 즉시 한 번, Part 4 가 끝나면 또 한 번.** Part 4 는 밤을 넘길 수 있으니
  Part 2 결과를 먼저 확보해 두는 것이 핵심이다.

로컬에서 압축을 풀고:
```
python scripts/verify_results.py <풀어놓은 results 경로>
python scripts/verify_results.py <새 results> --baseline <A0-b 에서 받은 배치8 results>
```

`--baseline` 비교는 **DIFFERENT 분기에서만 의미가 있다**(1·3 을 재실행했을 때).
EQUIVALENT 분기에서는 1·3 이 그대로 최종 결과이므로 비교 대상이 없다.

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

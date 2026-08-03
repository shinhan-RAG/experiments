"""질의 패러프레이즈 변형 (§6-8): Qwen(:8100)으로 질의 재표현.

질의-gold 어휘 결합을 약화시켜 dense 미붕괴·hybrid 천장의 원인을 검증한다.
고유명사·수치·전문용어는 보존하도록 프롬프트로 제약.

출력은 원 질의 파일을 덮어쓰지 않고 별도 variant 파일로 쓴다:
  queries.jsonl          → queries_paraphrase.jsonl
  agent_queries_50.jsonl → agent_queries_50_paraphrase.jsonl (있으면 이쪽 우선)
러너에서 --query-variant paraphrase 로 선택한다. 원질의/패러프레이즈 두 세트를
같은 코퍼스·qrels 위에서 병행 비교하기 위한 구조 — 덮어쓰기 방식은 결과 JSON에
어느 질의본이 쓰였는지 남지 않아 재현 불가능했다.

  python scripts/paraphrase_queries.py paper-mixed
  python scripts/paraphrase_queries.py paper-mixed --restore   # 구버전 덮어쓰기 복구

--restore: 구버전 스크립트가 queries.jsonl을 덮어쓴 데이터셋에서
queries.jsonl.orig를 원위치로 복원한다(.orig가 없으면 no-op).
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from utils import load_jsonl, run_batch_llm, strip_thinking

args = [a for a in sys.argv[1:] if not a.startswith("--")]
flags = {a for a in sys.argv[1:] if a.startswith("--")}
ds = args[0] if args else "paper-mixed"
d = Path("data/raw") / ds

if "--restore" in flags:
    qpath = d / "queries.jsonl"
    orig_path = d / "queries.jsonl.orig"
    if orig_path.exists():
        qpath.write_bytes(orig_path.read_bytes())
        print(f"restored: {qpath} <- {orig_path}")
    else:
        print(f"no backup to restore: {orig_path} (nothing to do)")
    sys.exit(0)

# 소스 선택은 run_experiment.load_queries와 같은 우선순위(층화 표본 우선)
src = d / "agent_queries_50.jsonl"
if not src.exists():
    src = d / "queries.jsonl"
# 구버전이 이미 덮어쓴 데이터셋이면 .orig가 진짜 원본이다
orig_backup = d / "queries.jsonl.orig"
if src.name == "queries.jsonl" and orig_backup.exists():
    print(f"[notice] {orig_backup.name} 발견 — 원본으로 간주하고 이 파일에서 생성"
          f" (queries.jsonl은 구버전 덮어쓰기 결과일 수 있음, --restore로 복구 가능)")
    src = orig_backup
    out_path = d / "queries_paraphrase.jsonl"
else:
    out_path = d / f"{src.stem}_paraphrase.jsonl"

queries = load_jsonl(src)

system = (
    "You rewrite Korean search queries. Rewrite the given question so it asks "
    "exactly the same thing, but with a different sentence structure and "
    "different general vocabulary (synonyms). Rules: keep all proper nouns, "
    "titles of works, numbers, and domain-specific technical terms unchanged; "
    "do not add or remove any information; the result must remain a natural "
    "Korean question. Output ONLY the rewritten question, nothing else."
)
prompts = [{"id": str(q["_id"]), "text": q.get("title") or q["text"]}
           for q in queries]
raw = run_batch_llm(prompts, system, max_tokens=200)

out, fail = [], 0
for q, r in zip(queries, raw):
    new = " ".join(strip_thinking(r).split()) if r else ""
    if not new:
        fail += 1
        new = q["text"]
    nq = dict(q)
    nq["text"] = new
    if "title" in nq:
        nq["title"] = new
    out.append(nq)

with open(out_path, "w", encoding="utf-8") as f:
    for q in out:
        f.write(json.dumps(q, ensure_ascii=False) + "\n")
with open(d / "queries_paraphrase_map.jsonl", "w", encoding="utf-8") as f:
    for q, nq in zip(queries, out):
        f.write(json.dumps({"_id": q["_id"], "orig": q.get("title") or q["text"],
                            "para": nq["text"]}, ensure_ascii=False) + "\n")

for q, nq in zip(queries, out):
    print(f"[{q['_id']}] {q.get('title') or q['text']}")
    print(f"  -> {nq['text']}")
print(f"total={len(out)} llm_fail_kept_original={fail}")
print(f"written: {out_path} (source: {src})")
print(f"사용: python run_experiment.py ... --query-variant paraphrase")

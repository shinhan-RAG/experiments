"""질의 패러프레이즈 변형 (§6-8): Qwen(:8100)으로 질의 재표현.

질의-gold 어휘 결합을 약화시켜 dense 미붕괴·hybrid 천장의 원인을 검증한다.
원본은 queries.jsonl.orig로 백업, 항상 원본에서 재생성(멱등).
고유명사·수치·전문용어는 보존하도록 프롬프트로 제약.

  python scripts/paraphrase_queries.py paper-mixed
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, "scripts")
from utils import load_jsonl, run_batch_llm, strip_thinking

ds = sys.argv[1] if len(sys.argv) > 1 else "paper-mixed"
d = Path("data/raw") / ds
qpath = d / "queries.jsonl"
orig_path = d / "queries.jsonl.orig"
if not orig_path.exists():
    orig_path.write_bytes(qpath.read_bytes())
queries = load_jsonl(orig_path)

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

with open(qpath, "w", encoding="utf-8") as f:
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
print(f"written: {qpath} (backup: {orig_path})")

"""법률 데이터셋 reference answer 생성 (LLM-as-Judge accuracy용).

- legal-qa: 이미 gold answer(answers.jsonl)가 있으므로 LLM 없이 그대로 변환한다.
- aihub-full: gold precedent parent 청크 텍스트를 근거로 vLLM(Qwen3-8B)로 생성한다
  (로컬 vLLM @ 8100 필요). retrieval 지표(Recall/Efficiency)는 reference answer가
  없어도 계산되므로 aihub 생성은 선택 사항(accuracy만 추가로 얻는 용도).

출력: data/reference_answers/<dataset>.json
      [{"query_id": ..., "reference_answer": ...}, ...]
"""
import json
import sys
from pathlib import Path

from utils import dataset_dir, run_batch_llm, strip_thinking

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "reference_answers"

AIHUB_SYSTEM_PROMPT = """당신은 한국 법률 전문가입니다. 질의와 관련 판례 발췌문이 주어지면
근거에 기반한 간결한 참조 답변을 작성하세요.

요건:
- 제공된 판례 내용에만 근거할 것
- 쟁점과 결론을 구체적으로 명시할 것
- 2~4문장, 간결하지만 완결되게
- 문서 간 상충이 있으면 양쪽을 모두 언급할 것"""

AIHUB_USER_TEMPLATE = """질의: {query}

관련 판례(gold):
{docs_text}

위 판례에 근거한 참조 답변을 작성하세요."""


def _load_jsonl(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f]


def build_legal_qa() -> None:
    """legal-qa: answers.jsonl → reference_answers/legal-qa.json (LLM 불필요)."""
    ds_dir = dataset_dir(DATA_DIR, "legal-qa")
    answers = _load_jsonl(ds_dir / "answers.jsonl")
    out = [
        {"query_id": str(a["_id"]), "reference_answer": a.get("answer", "").strip()}
        for a in answers
        if a.get("answer", "").strip()
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "legal-qa.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  legal-qa: {len(out)} reference answers -> {out_path}")


def build_aihub_full() -> None:
    """aihub-full: gold parent 청크 텍스트로 vLLM 생성 (agent_queries_50 대상)."""
    from collections import defaultdict

    ds_dir = dataset_dir(DATA_DIR, "aihub-full")
    query_file = ds_dir / "agent_queries_50.jsonl"
    if not query_file.exists():
        query_file = ds_dir / "queries.jsonl"
    queries = _load_jsonl(query_file)
    qrels = _load_jsonl(ds_dir / "qrels.jsonl")
    gold_by_q = defaultdict(list)
    for row in qrels:
        if row.get("score", 0) >= 1:
            gold_by_q[str(row["query-id"])].append(str(row["corpus-id"]))

    # gold parent → 첫 청크 텍스트 매핑 (근거 텍스트 확보)
    parent_text: dict[str, str] = {}
    wanted = {cid for cids in gold_by_q.values() for cid in cids}
    with open(ds_dir / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            doc = json.loads(line)
            pid = doc.get("parent_id", doc["_id"])
            if pid in wanted and pid not in parent_text:
                parent_text[pid] = doc.get("text", "")[:2000]

    prompts, qids = [], []
    for q in queries:
        qid = str(q["_id"])
        docs = [parent_text.get(cid, "") for cid in gold_by_q.get(qid, [])]
        docs_text = "\n\n".join(d for d in docs if d)[:6000]
        if not docs_text:
            continue
        prompts.append({"id": qid, "text": AIHUB_USER_TEMPLATE.format(
            query=q.get("text", ""), docs_text=docs_text)})
        qids.append(qid)

    raw = run_batch_llm(prompts, AIHUB_SYSTEM_PROMPT, max_tokens=400)
    out = [
        {"query_id": qid, "reference_answer": strip_thinking(r)}
        for qid, r in zip(qids, raw) if r
    ]
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "aihub-full.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"  aihub-full: {len(out)} reference answers -> {out_path}")


if __name__ == "__main__":
    datasets = [a for a in sys.argv[1:] if not a.startswith("-")] or ["legal-qa"]
    for ds in datasets:
        if ds == "legal-qa":
            build_legal_qa()
        elif ds == "aihub-full":
            build_aihub_full()  # 로컬 vLLM(8100) 필요
        else:
            print(f"  (skip) unsupported dataset: {ds}")

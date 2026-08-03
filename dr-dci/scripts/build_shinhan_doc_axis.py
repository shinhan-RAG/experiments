"""문서(파일) 단위 메타데이터 축 라우팅 아티팩트 생성 — 축 입도 실험 (종합보고서 §7).

신한 파일명(doc)이 결정론 메타데이터라는 점을 이용해, 라우팅 축을
기존 5분류(L1)에서 문서 28개 단위로 세분화한 변형 taxonomy·라우팅 맵을 만든다.
기존 taxonomy_routed/partitioned 백엔드를 코드 수정 없이 재사용하기 위해
같은 파일 포맷으로 생성하고, KT에서 swap(cp)해서 probe를 돌린다.

출력 (기존 아티팩트를 덮지 않도록 별도 확장자):
  data/taxonomy/shinhan-mixed_{size}.docaxis.json
  data/routing/shinhan-mixed_{size}.docaxis.oracle.json     (로컬 생성 가능, acc=1.0)
  data/routing/shinhan-mixed_{size}.docaxis.llm.json        (--llm, KT에서 Qwen :8100 필요)

사용:
  python scripts/build_shinhan_doc_axis.py          # taxonomy 3종 + oracle 맵 3종
  python scripts/build_shinhan_doc_axis.py --llm    # LLM 맵 3종 (질의→문서 28지선다)
"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl, run_batch_llm, parse_llm_content  # noqa: E402

DATA = Path(__file__).parent.parent / "data"
RAW = DATA / "raw" / "shinhan-mixed"
SIZES = ["20k", "50k", "110k"]


def load_doc_by_chunk():
    doc_by_chunk = {}
    with open(RAW / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            doc_by_chunk[str(d["_id"])] = d.get("doc") or ""
    return doc_by_chunk


def build_taxonomy_and_oracle():
    doc_by_chunk = load_doc_by_chunk()
    metas = load_jsonl(RAW / "qa_meta.jsonl")
    gold_doc_by_q = {str(m["qid"]): m["doc"] for m in metas}

    for size in SIZES:
        base = json.load(open(DATA / "taxonomy" / f"shinhan-mixed_{size}.json",
                              encoding="utf-8"))
        tax = {}
        for cid, v in base.items():
            if isinstance(v, dict) and v.get("L1") == "Legal":
                tax[cid] = v  # Legal distractor는 기존 유지
            else:
                doc = doc_by_chunk.get(cid, "")
                tax[cid] = {"L1": doc, "L2": v.get("L1") if isinstance(v, dict) else None,
                            "L3": v.get("L3") if isinstance(v, dict) else None}
        out = DATA / "taxonomy" / f"shinhan-mixed_{size}.docaxis.json"
        json.dump(tax, open(out, "w", encoding="utf-8"), ensure_ascii=False)

        shinhan_docs = sorted({v["L1"] for v in tax.values() if v["L1"] != "Legal"})
        part_sizes = Counter(v["L1"] for v in tax.values() if v["L1"] != "Legal")
        l1_values = shinhan_docs + ["Legal"]

        routing = {qid: {"L1": doc, "L1_top2": [doc], "confidence": 1.0}
                   for qid, doc in gold_doc_by_q.items()}
        miss = sum(1 for doc in gold_doc_by_q.values() if doc not in part_sizes)
        rmap = {"dataset": "shinhan-mixed", "size_key": size, "l1_values": l1_values,
                "routing": routing,
                "diagnostics": {"oracle": True, "axis": "doc",
                                "routing_accuracy": 1.0, "routing_accuracy_top2": 1.0,
                                "evaluated_queries": len(routing),
                                "gold_docs_missing_in_subset": miss,
                                "n_docs": len(shinhan_docs),
                                "partition_size_min_med_max": [
                                    min(part_sizes.values()),
                                    sorted(part_sizes.values())[len(part_sizes) // 2],
                                    max(part_sizes.values())],
                                "confidence_mean": 1.0}}
        rout = DATA / "routing" / f"shinhan-mixed_{size}.docaxis.oracle.json"
        json.dump(rmap, open(rout, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
        print(f"{size}: docs={len(shinhan_docs)} partition min/med/max="
              f"{rmap['diagnostics']['partition_size_min_med_max']} "
              f"queries={len(routing)} gold-doc-missing={miss}")


def build_llm_maps():
    """질의→문서 28지선다 LLM 라우팅 (Qwen guided enum). 진단: gold doc 대조 정확도."""
    metas = load_jsonl(RAW / "qa_meta.jsonl")
    gold_doc_by_q = {str(m["qid"]): m["doc"] for m in metas}
    queries = load_jsonl(RAW / "queries.jsonl")

    # 문서 목록은 20k oracle 산출물에서 (세 규모 동일: gold 문서는 전 규모 포함)
    orc = json.load(open(DATA / "routing" / "shinhan-mixed_20k.docaxis.oracle.json",
                         encoding="utf-8"))
    doc_list = [d for d in orc["l1_values"] if d != "Legal"]

    system = f"""You are a document router for Shinhan Life insurance documents.
Given a Korean query, pick which document it should be answered from.
Documents (filenames are descriptive: category__doc-type_product_version):
{json.dumps(doc_list, ensure_ascii=False, indent=0)}

Respond in JSON:
{{"L1": "<most likely document>",
 "L1_second": "<second most likely (repeat L1 if none)>",
 "confidence": <0.0-1.0>}}"""
    guided = {"type": "object",
              "properties": {"L1": {"type": "string", "enum": doc_list},
                             "L1_second": {"type": "string", "enum": doc_list},
                             "confidence": {"type": "number", "minimum": 0, "maximum": 1}},
              "required": ["L1", "L1_second", "confidence"]}
    prompts = [{"id": str(q["_id"]), "text": f"Query: {q['text']}"} for q in queries]
    raw = run_batch_llm(prompts, system, max_tokens=400, guided_json=guided)

    routing, fallback = {}, 0
    for q, r in zip(queries, raw):
        l1, l1s, conf = None, None, None
        if r:
            try:
                p = json.loads(parse_llm_content(r))
                l1, l1s, conf = p.get("L1"), p.get("L1_second"), p.get("confidence")
            except (json.JSONDecodeError, TypeError, AttributeError):
                pass
        if l1 not in doc_list:
            l1 = doc_list[0]
            fallback += 1
        if l1s not in doc_list:
            l1s = l1
        if not isinstance(conf, (int, float)) or not 0 <= conf <= 1:
            conf = 1.0
        top2 = [l1] if l1s == l1 else [l1, l1s]
        routing[str(q["_id"])] = {"L1": l1, "L1_top2": top2,
                                  "confidence": round(float(conf), 3)}

    n = len(routing)
    acc = sum(1 for qid, r in routing.items() if r["L1"] == gold_doc_by_q.get(qid)) / n
    acc2 = sum(1 for qid, r in routing.items()
               if gold_doc_by_q.get(qid) in r["L1_top2"]) / n
    diag = {"oracle": False, "axis": "doc",
            "routing_accuracy": round(acc, 4), "routing_accuracy_top2": round(acc2, 4),
            "evaluated_queries": n, "llm_fallback": fallback,
            "routed_distribution": dict(Counter(
                r["L1"].split("__", 1)[0] for r in routing.values())),
            "confidence_mean": round(
                sum(r["confidence"] for r in routing.values()) / n, 3)}
    for size in SIZES:
        orc = json.load(open(DATA / "routing" / f"shinhan-mixed_{size}.docaxis.oracle.json",
                             encoding="utf-8"))
        out = DATA / "routing" / f"shinhan-mixed_{size}.docaxis.llm.json"
        json.dump({"dataset": "shinhan-mixed", "size_key": size,
                   "l1_values": orc["l1_values"], "routing": routing,
                   "diagnostics": diag},
                  open(out, "w", encoding="utf-8"), ensure_ascii=False, indent=2)
    print(f"LLM doc-routing: acc={acc:.2f} top2={acc2:.2f} fallback={fallback} "
          f"conf_mean={diag['confidence_mean']} prefix-dist={diag['routed_distribution']}")


if __name__ == "__main__":
    if "--llm" in sys.argv:
        build_llm_maps()
    else:
        build_taxonomy_and_oracle()

#!/usr/bin/env python3
"""Claude-facing search/read/submit CLI for the 0823 tag-hybrid experiment.

Two modes, selected by SEMTAG_ARM["mode"]:
- agentic: 0821 behavior (search/msearch/read/submit) with arm-configurable caps.
- host_retrieve: the host pre-runs tag+meta search into $SEMTAG_SESSION/candidates.json;
  search/msearch are disabled and the agent uses `candidates` instead.

`msearch` deliberately retains the V9 ChunkHybridSearch(hybrid) implementation.
Only `search` changes across arms.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
VS = ROOT / "vector_search"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(VS))
sys.path.insert(0, str(HERE))

from tag_hybrid import TagHybridSearch  # noqa: E402
from units import Units  # noqa: E402

PAGE = 40
PREVIEW = 360


def _query_tokens(q: str) -> list[str]:
    return [x.lower() for x in re.findall(r"[가-힣A-Za-z0-9%]+", q or "") if len(x) > 1]


def snippet(text: str, qtoks: list[str], chars: int = PREVIEW) -> str:
    clean = " ".join((text or "").split())
    positions = [clean.lower().find(t) for t in qtoks if clean.lower().find(t) >= 0]
    start = max(0, min(positions) - 80) if positions else 0
    return clean[start : start + chars]


def split_contract(value: str) -> tuple[str, str]:
    match = re.search(r"\(무배당[^)]*\)", value or "")
    variant = match.group(0) if match else ""
    return re.sub(r"\(무배당[^)]*\)", "", value or "").strip(), variant


def resolve_arm(value: str) -> dict:
    if value.strip().startswith("{"):
        return json.loads(value)
    arms = json.loads((HERE / "arms.json").read_text(encoding="utf-8"))
    if value not in arms:
        raise SystemExit(f"arms.json에 없는 arm: {value}")
    return arms[value]


def frozen_qtag_slots(arm: dict, qid: str) -> dict[str, list[str]]:
    """Return train-only frozen qtags; unseen test qids correctly fall back to rule/agent slots."""
    rel = arm.get("qtags")
    if arm.get("router") not in ("llm", "union") or not rel:
        return {}
    path = FS / rel
    if not path.exists():
        return {}
    for line in path.open(encoding="utf-8"):
        row = json.loads(line)
        if row.get("qid") == qid:
            return {k: row.get(k) or [] for k in ("contract", "role", "subject", "qualifier", "schema")}
    return {}


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    search = sub.add_parser("search")
    search.add_argument("--q", required=True)
    search.add_argument("--page", type=int, default=1)
    for field in ("contract", "role", "subject", "qualifier", "schema"):
        search.add_argument(f"--{field}", default="", help="쉼표 구분 선택 슬롯")
    meta = sub.add_parser("msearch")
    meta.add_argument("--q", required=True)
    meta.add_argument("--page", type=int, default=1)
    meta.add_argument("--strategy", default="hybrid", choices=("hybrid", "bm25", "dense"))
    cand = sub.add_parser("candidates")
    cand.add_argument("--page", type=int, default=1)
    read = sub.add_parser("read")
    read.add_argument("--id", required=True)
    submit = sub.add_parser("submit")
    submit.add_argument("--ids", required=True)
    return ap


def main():
    a = parser().parse_args()
    session = Path(os.environ["SEMTAG_SESSION"])
    session.mkdir(parents=True, exist_ok=True)
    qid = os.environ.get("SEMTAG_QID", "")
    arm = json.loads(os.environ.get("SEMTAG_ARM", "{}"))
    mode = arm.get("mode", "agentic")
    search_cap = int(arm.get("search_cap", 20))
    read_cap = int(arm.get("read_cap", 25 if mode == "host_retrieve" else 8))
    submit_max = int(arm.get("submit_max", 10))
    calls_path = session / "calls.jsonl"
    calls = [json.loads(line) for line in calls_path.open(encoding="utf-8")] if calls_path.exists() else []
    n_search = sum(c["cmd"] in ("search", "msearch") for c in calls)
    n_read = sum(c["cmd"] == "read" for c in calls)

    def log(record: dict):
        record.update({"t": time.time(), "cmd": a.cmd})
        with calls_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    def output(record: dict):
        print(json.dumps(record, ensure_ascii=False))

    elements = FS / "out" / arm.get("elements", "elements_u2.jsonl")
    tags = FS / "out" / arm.get("tags", "tags_u2_rules.jsonl")
    jo = FS / "out" / arm.get("jo", "elements_u2jo.jsonl")
    units = Units(str(jo))

    if mode == "host_retrieve" and a.cmd in ("search", "msearch"):
        output({"error": "이 모드에서는 search/msearch를 사용할 수 없습니다. candidates 명령으로 후보를 확인하십시오."})
        return

    if a.cmd == "candidates":
        cand_path = session / "candidates.json"
        if not cand_path.exists():
            output({"error": "candidates.json이 없습니다."})
            return
        payload = json.loads(cand_path.read_text(encoding="utf-8"))
        all_items = payload.get("candidates", [])
        items = all_items[PAGE * (a.page - 1) : PAGE * a.page]
        log({"qid": qid, "page": a.page, "returned": [(x.get("id"), x.get("score")) for x in items]})
        output({
            "query": payload.get("query", ""),
            "total_candidates": payload.get("total", len(all_items)),
            "page": a.page,
            "page_size": PAGE,
            "results": items,
        })
        return

    def meta_items(query: str, top_k: int, strategy: str = "hybrid") -> list[dict]:
        # Frozen 0820 path: V9 BM25+dense RRF.  No tag score enters this function.
        from hybrid_search import ChunkHybridSearch

        hs = ChunkHybridSearch(view=arm.get("meta_view", "V9"))
        results = hs.search(query, strategy=strategy, top_k=top_k)
        qtoks = _query_tokens(query)
        items = []
        for result in results:
            unit = units.jo_of_span(result["char_start"], result["char_end"]) if result.get("char_start") is not None else None
            base, variant = split_contract((unit or {}).get("contract_scope", ""))
            items.append({
                "id": result["id"],
                "jo": unit["element_id"] if unit else "",
                "contract": base,
                "variant": variant,
                "preview": snippet(result.get("preview") or "", qtoks),
            })
        return items

    if a.cmd in ("search", "msearch") and n_search >= search_cap:
        output({"error": f"search 예산 초과({search_cap}회). submit 하십시오."})
        return

    if a.cmd == "search":
        explicit = frozen_qtag_slots(arm, qid)
        for field in ("contract", "role", "subject", "qualifier", "schema"):
            manual = [x.strip() for x in getattr(a, field).split(",") if x.strip()]
            explicit[field] = list(dict.fromkeys(list(explicit.get(field, [])) + manual))
        engine = TagHybridSearch(elements, tags, jo, HERE / "out" / "tag_index")
        rows, slots, qtoks = engine.search(
            a.q,
            explicit=explicit,
            weights=arm.get("tag_weights"),
            channel_top_k=int(arm.get("channel_top_k", 200)),
            limit=400,
            collapse_jo=bool(arm.get("collapse_jo", True)),
        )
        page_rows = rows[PAGE * (a.page - 1) : PAGE * a.page]
        items = []
        for row in page_rows:
            element = row["element"]
            unit = units.jid.get(row["jo"])
            base, variant = split_contract(element.get("contract_scope", ""))
            items.append({
                "id": element["element_id"],
                "jo": row["jo"],
                "score": round(row["score"], 7),
                "contract": base,
                "variant": variant,
                "jo_title": ((unit or {}).get("title") or "")[:60],
                "preview": snippet(element.get("text", ""), qtoks),
                "ranks": row["provenance"],
            })
        log({
            "qid": qid,
            "q": a.q,
            "slots": slots,
            "weights": arm.get("tag_weights"),
            "collapse_jo": bool(arm.get("collapse_jo", True)),
            "total": len(rows),
            "page": a.page,
            "returned": [(x["id"], x["score"]) for x in items],
        })
        output({
            "query": a.q,
            "channel": "tag:rrf" if sum(bool(v) for v in (arm.get("tag_weights") or {}).values()) > 1 else "tag:clm",
            "slots_used": slots,
            "total_candidates": len(rows),
            "page": a.page,
            "page_size": PAGE,
            "search_calls_left": search_cap - n_search - 1,
            "results": items,
        })
        return

    if a.cmd == "msearch":
        if not arm.get("meta"):
            output({"error": "이 arm에서는 msearch를 사용할 수 없습니다."})
            return
        all_items = meta_items(a.q, PAGE * a.page, a.strategy)
        items = all_items[PAGE * (a.page - 1) : PAGE * a.page]
        log({"qid": qid, "q": a.q, "strategy": a.strategy, "page": a.page, "returned": [(x["id"], 0) for x in items]})
        output({
            "query": a.q,
            "channel": f"meta:{a.strategy}",
            "page": a.page,
            "page_size": PAGE,
            "search_calls_left": search_cap - n_search - 1,
            "results": items,
        })
        return

    if a.cmd == "read":
        if n_read >= read_cap:
            output({"error": f"read 예산 초과({read_cap}회)."})
            return
        resolved = units.resolve([a.id.strip()])
        if not resolved:
            output({"error": "unknown id"})
            return
        unit = resolved[0]
        text = (unit.get("text") or "")[:4000]
        log({"qid": qid, "id": a.id.strip(), "jo": unit["element_id"], "chars": len(text)})
        output({
            "id": a.id.strip(),
            "jo": unit["element_id"],
            "contract": unit.get("contract_scope", ""),
            "members": (unit.get("members") or [])[:60],
            "text": text,
            "read_calls_left": read_cap - n_read - 1,
        })
        return

    ids = [x.strip() for x in a.ids.split(",") if x.strip()][:submit_max]
    # Preserve first occurrence of each scoring jo; this only removes score-neutral duplicates.
    kept, seen = [], set()
    for item_id in ids:
        resolved = units.resolve([item_id])
        key = resolved[0]["element_id"] if resolved else item_id
        if key not in seen:
            kept.append(item_id)
            seen.add(key)
    log({"qid": qid, "ids": kept, "deduped": len(ids) - len(kept)})
    (session / "submit.json").write_text(
        json.dumps({"qid": qid, "ranked": kept, "n_search": n_search, "n_read": n_read}, ensure_ascii=False),
        encoding="utf-8",
    )
    output({"ok": True, "submitted": kept, "note": "세션 종료. 더 이상 도구를 호출하지 마십시오."})


if __name__ == "__main__":
    main()

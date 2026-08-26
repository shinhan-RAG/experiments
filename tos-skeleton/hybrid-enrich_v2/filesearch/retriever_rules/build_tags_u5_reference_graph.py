#!/usr/bin/env python3
"""U4 fact tags에 문서 내부의 명시적 표 참조 graph를 추가한다.

QA·Gold·검색 결과를 읽지 않는다. 문서/계약/표 namespace와 원문 ``참조``
표현만으로 최대 2-hop edge를 만들며, target이 모호하면 연결하지 않는다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from build_tags_u4_fact import explicit_table_refs, table_keys


HERE = Path(__file__).resolve().parent.parent  # filesearch 루트 (retriever_rules 하위로 이동)
VERSION = "semtag-u5-reference-graph-1.4"
PRECISE_VERSION = "semtag-u5-reference-graph-1.2"
LEGACY_VERSION = "semtag-u5-reference-graph-1.1"
QUOTED_RE = re.compile(r'["“「『](.{2,80}?)["”」』]')


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def ordered_unique(values, limit=40):
    output = []
    for value in values:
        value = str(value).strip()
        if value and value not in output:
            output.append(value)
        if len(output) >= limit:
            break
    return output


def declared_and_referenced_keys(text: str) -> tuple[list[str], list[str]]:
    declared, referenced = [], []
    for line in str(text).splitlines():
        keys = table_keys(line)
        starts_as_heading = bool(re.match(
            r"^\s*#*\s*[<\[]?\s*(?:부표|별첨|표)\s*\d", line))
        if "참조" in line:
            referenced.extend(explicit_table_refs(line))
        elif keys and not starts_as_heading:
            # 문장 중간의 <부표N>은 표 선언이 아니라 해당 표를 가리키는 링크다.
            referenced.extend(keys)
        else:
            declared.extend(keys)
    return ordered_unique(declared), ordered_unique(referenced)


def explicit_redirect_details(text: str) -> dict[str, dict[str, str]]:
    """Pair each local schedule with its own explicitly referenced appendix.

    A single JO can contain many ``부표`` sections.  Flattening all local and
    global references and then joining them creates a Cartesian product (for
    example, every disease schedule points at every appendix table in the JO).
    This parser keeps line order and associates a global reference only with the
    nearest active local schedule.  More than one target for the same local
    schedule is ambiguous and therefore fails closed.
    """
    candidates: dict[str, list[dict[str, str]]] = defaultdict(list)
    active_local = ""
    for line in str(text).splitlines():
        line_keys = table_keys(line)
        starts_as_heading = bool(re.match(
            r"^\s*#*\s*[<\[]?\s*부표\s*\d", line))
        heading_locals = [key for key in line_keys if key.startswith("부표:")]
        if starts_as_heading:
            active_local = heading_locals[0] if len(heading_locals) == 1 else ""
        elif ("참조" not in line and
              re.match(r"^\s*(?:#{1,6}\s+|제\s*\d+(?:-\d+)*\s*조|"
                       r"[<\[]?\s*(?:별첨|표)\s*\d)", line)):
            # A new legal/article/table section ends the previous local table.
            # Without this reset, a distant appendix reference can be attached
            # to a stale local schedule.
            active_local = ""

        if "참조" not in line:
            continue
        refs = explicit_table_refs(line)
        local_refs = [key for key in refs if key.startswith("부표:")]
        global_refs = [key for key in refs if key.startswith("별첨:")]
        if len(local_refs) > 1:
            continue
        source_local = local_refs[0] if len(local_refs) == 1 else active_local
        if source_local and len(global_refs) == 1:
            target = global_refs[0]
            detail = {"target": target, "line": " ".join(line.split())}
            if detail not in candidates[source_local]:
                candidates[source_local].append(detail)

    output = {}
    for local, details in candidates.items():
        targets = ordered_unique(detail["target"] for detail in details)
        if len(targets) == 1:
            output[local] = {"target": targets[0],
                             "line": next(detail["line"] for detail in details
                                          if detail["target"] == targets[0])}
    return output


def explicit_redirect_pairs(text: str) -> dict[str, str]:
    return {local: detail["target"]
            for local, detail in explicit_redirect_details(text).items()}


def reference_terms(text: str) -> list[str]:
    terms = QUOTED_RE.findall(str(text))
    terms.extend(re.findall(
        r"([가-힣A-Za-z0-9·()\[\]-]{2,50}(?:분류표|지급기준표|장해분류표))",
        str(text)))
    return ordered_unique(terms, 24)


def normalized_table_title(value: str) -> str:
    return re.sub(r"[^0-9A-Za-z가-힣]", "", str(value)).lower()


def _table_heading(line: str, following: str, active_appendix: str) -> dict | None:
    """Parse every literal table heading, including non-classification tables.

    All headings are region boundaries.  Query projection remains exact-title
    only, but an unrelated table can no longer leak into the preceding region.
    """
    explicit = re.match(
        r"^\s*\|?\s*\[?\s*별첨\s*(\d+)\s*\]?\s*표\s*"
        r"(\d+(?:-\d+)?)\s*[.|]?\s*(.*?)\s*$", line, re.I)
    if explicit:
        appendix_no, table_no, title = explicit.groups()
    else:
        generic = re.match(
            r"^\s*\|?\s*표\s*(\d+(?:-\d+)?)\s*[.|]?\s*(.*?)\s*$",
            line, re.I)
        if not generic or not active_appendix:
            return None
        table_no, title = generic.groups()
        appendix_no = active_appendix
    title = re.sub(r"\s+", " ", title or following).strip(" |:.-#\t")
    return {"appendix_no": appendix_no, "table_no": table_no,
            "key": f"별첨:{appendix_no}:표:{table_no}", "title": title,
            "title_surface": normalized_table_title(title)}


def build_table_catalog(elements: list[dict], tags: list[dict],
                        member_to_jo: dict[str, str]) -> tuple[list[dict], dict[str, str]]:
    """Build document-scoped table regions from source order, without QA input.

    Converted appendices repeat a heading at page boundaries and often split one
    legal table across several JO units.  Consecutive repetitions remain one
    region; a different table heading closes it.  Titles that are not unique in
    the same source document fail closed at query time via the recorded surface.
    """
    headings = []
    active_appendix: dict[str, str] = {}
    element_docs = {}
    for index, (element, tag) in enumerate(zip(elements, tags)):
        doc = str(tag.get("document_key", ""))
        element_docs[element["element_id"]] = doc
        element_text = str(element.get("text", ""))
        lines = element_text.splitlines()
        source_is_toc = toc_like_reference_list(element_text)
        for line_no, line in enumerate(lines):
            if line_no + 1 < len(lines):
                following = lines[line_no + 1]
            elif index + 1 < len(elements):
                next_lines = str(elements[index + 1].get("text", "")).splitlines()
                following = next_lines[0] if next_lines else ""
            else:
                following = ""
            heading = _table_heading(line, following, active_appendix.get(doc, ""))
            if not heading:
                continue
            active_appendix[doc] = heading["appendix_no"]
            headings.append({"doc": doc, "index": index, "line": line_no,
                             "toc_like": source_is_toc, **heading})

    by_doc = defaultdict(list)
    for anchor in headings:
        by_doc[anchor["doc"]].append(anchor)
    catalog = []
    for doc, rows in by_doc.items():
        ordered = sorted(rows, key=lambda x: (x["index"], x["line"]))
        # Only consecutive page-repeat headings may collapse. A key/title that
        # reappears after another table is ambiguous and is excluded entirely.
        blocks = []
        for row in ordered:
            signature = (row["key"], row["title_surface"])
            if blocks and blocks[-1]["signature"] == signature:
                blocks[-1]["repeats"].append(row)
            else:
                blocks.append({"signature": signature, "row": row, "repeats": [row]})
        signature_count = defaultdict(int)
        title_count = defaultdict(int)
        key_count = defaultdict(int)
        for block in blocks:
            if block["row"]["toc_like"]:
                continue
            signature_count[block["signature"]] += 1
            key_count[block["row"]["key"]] += 1
            if block["row"]["title_surface"]:
                title_count[block["row"]["title_surface"]] += 1

        for offset, block in enumerate(blocks):
            row = block["row"]
            if (row["toc_like"] or not row["title_surface"] or
                    len(row["title_surface"]) < 4 or
                    signature_count[block["signature"]] != 1 or
                    key_count[row["key"]] != 1 or
                    title_count[row["title_surface"]] != 1):
                continue
            next_row = blocks[offset + 1]["row"] if offset + 1 < len(blocks) else None
            end_index = next_row["index"] if next_row else len(elements) - 1
            region_slices = []
            for element_index in range(row["index"], end_index + 1):
                element = elements[element_index]
                if element_docs.get(element["element_id"]) != doc:
                    continue
                lines = str(element.get("text", "")).splitlines()
                start_line = row["line"] if element_index == row["index"] else 0
                stop_line = (next_row["line"] if next_row and
                             element_index == next_row["index"] else len(lines))
                if stop_line <= start_line:
                    continue
                text_slice = "\n".join(lines[start_line:stop_line]).strip()
                if text_slice:
                    region_slices.append((element, text_slice))
            region = [element for element, _ in region_slices]
            if not region or len(region) > 120:
                continue
            variants = []
            for jo_id in ordered_unique(
                    member_to_jo.get(element["element_id"], "") for element in region):
                jo_slices = [(element, text_slice) for element, text_slice in region_slices
                             if member_to_jo.get(element["element_id"]) == jo_id]
                preview = "\n".join(text_slice for _, text_slice in jo_slices)
                variants.append({
                    "jo": jo_id,
                    "element_ids": [element["element_id"] for element, _ in jo_slices],
                    "preview": preview[:1600],
                })
            combined = "\n".join(variant["preview"] for variant in variants)
            semantic_lines = []
            for line in combined.splitlines():
                stripped = line.strip(" |:.-#\t_")
                if not stripped or _table_heading(line, "", row["appendix_no"]):
                    continue
                semantic_lines.append(stripped)
            if variants and len(normalized_table_title(" ".join(semantic_lines))) >= 4:
                catalog.append({
                    **row, "start_element": region[0]["element_id"],
                    "end_element": region[-1]["element_id"],
                    "variants": variants,
                })

    jo_documents = {}
    for element_id, jo_id in member_to_jo.items():
        doc = element_docs.get(element_id, "")
        if jo_id and doc:
            previous = jo_documents.setdefault(jo_id, doc)
            if previous != doc:
                raise ValueError(f"JO crosses source documents: {jo_id}")
    return catalog, jo_documents


def audit_table_catalog(catalog: list[dict]) -> dict:
    key_counts, title_counts = defaultdict(int), defaultdict(int)
    cross_heading = []
    for row in catalog:
        key_counts[(row["doc"], row["key"])] += 1
        title_counts[(row["doc"], row["title_surface"])] += 1
        for variant in row["variants"]:
            active_appendix = row["key"].split(":")[1]
            lines = str(variant.get("preview", "")).splitlines()
            for line_no, line in enumerate(lines):
                following = lines[line_no + 1] if line_no + 1 < len(lines) else ""
                heading = _table_heading(line, following, active_appendix)
                if heading:
                    active_appendix = heading["appendix_no"]
                    if heading["key"] != row["key"]:
                        cross_heading.append({"catalog_key": row["key"],
                                              "found_key": heading["key"],
                                              "jo": variant["jo"]})
    audit = {
        "duplicate_key_count": sum(count > 1 for count in key_counts.values()),
        "duplicate_title_count": sum(count > 1 for count in title_counts.values()),
        "toc_source_count": sum(bool(row.get("toc_like")) for row in catalog),
        "cross_table_heading_count": len(cross_heading),
        "cross_table_heading_examples": cross_heading[:20],
    }
    if any(audit[key] for key in ("duplicate_key_count", "duplicate_title_count",
                                  "toc_source_count", "cross_table_heading_count")):
        raise ValueError(f"unsafe table catalog: {audit}")
    return audit


def toc_like_reference_list(text: str) -> bool:
    """표 제목과 페이지 번호만 나열한 목차 JO를 실제 표 target에서 제외한다."""
    key_lines = [line for line in str(text).splitlines() if table_keys(line)]
    if len(key_lines) < 3:
        return False
    page_lines = [line for line in key_lines if re.search(
        r"(?:\|\s*[\d,]+\s*\|?\s*$|\s[\d,]{3,}\s*$)", line)]
    return len(page_lines) / len(key_lines) >= 0.7


def build_reference_aliases(elements: list[dict], tags: list[dict], jo_rows: list[dict],
                            precise_redirects: bool = True):
    element_by_id = {row["element_id"]: row for row in elements}
    tag_by_id = {row["element_id"]: row for row in tags}
    records = {}
    local_index, global_index = defaultdict(set), defaultdict(set)

    for jo in jo_rows:
        member_tags = [tag_by_id[mid] for mid in jo.get("members", []) if mid in tag_by_id]
        doc_keys = ordered_unique(tag.get("document_key", "") for tag in member_tags)
        if len(doc_keys) > 1:
            raise ValueError(f"JO crosses documents: {jo['element_id']} {doc_keys}")
        identities = ordered_unique(
            tag.get("contract_key", "") or element_by_id[mid].get("contract_scope", "")
            for mid, tag in ((mid, tag_by_id[mid]) for mid in jo.get("members", [])
                             if mid in tag_by_id))
        declared, refs = declared_and_referenced_keys(jo.get("text", ""))
        record = {
            "jo": jo["element_id"], "doc": doc_keys[0] if doc_keys else "",
            "identities": identities, "declared": declared, "refs": refs,
            "terms": reference_terms(jo.get("text", "")),
            "redirects": explicit_redirect_pairs(jo.get("text", "")),
            "redirect_details": explicit_redirect_details(jo.get("text", "")),
            "toc_like": toc_like_reference_list(jo.get("text", "")),
        }
        records[jo["element_id"]] = record
        for key in declared:
            global_index[(record["doc"], key)].add(record["jo"])
            for identity in identities:
                local_index[(record["doc"], identity, key)].add(record["jo"])

    ambiguous = []

    def resolve(source: dict, key: str):
        candidates = set()
        if key.startswith("부표:"):
            for identity in source["identities"]:
                candidates.update(local_index.get((source["doc"], identity, key), ()))
        elif key.startswith("별첨:"):
            candidates.update(global_index.get((source["doc"], key), ()))
        else:
            # namespace 없는 표 번호는 문서 전체에서 유일할 때만 허용한다.
            candidates.update(global_index.get((source["doc"], key), ()))
        candidates.discard(source["jo"])
        candidates = {candidate for candidate in candidates
                      if not records[candidate]["toc_like"]}
        if len(candidates) > 1 and not precise_redirects:
            # Historical v1.1 compatibility only. New graphs fail closed.
            redirectors = {candidate for candidate in candidates
                           if records[candidate]["refs"]}
            if len(redirectors) == 1:
                candidates = redirectors
        if len(candidates) == 1:
            return next(iter(candidates))
        if candidates:
            ambiguous.append({"source": source["jo"], "key": key,
                              "candidates": sorted(candidates)})
        return None

    direct_edges = []
    for source in records.values():
        if precise_redirects and source["toc_like"]:
            continue
        for key in source["refs"]:
            target = resolve(source, key)
            if target:
                direct_edges.append({"source": source["jo"], "target": target,
                                     "key": key})

    outgoing = defaultdict(list)
    for edge in direct_edges:
        outgoing[edge["source"]].append(edge)

    paths = []
    for source in records.values():
        for first in outgoing.get(source["jo"], ()):
            path = {"source": source["jo"], "target": first["target"],
                    "keys": [first["key"]], "hops": 1}
            if precise_redirects:
                path.update(document=source["doc"],
                            source_contracts=source["identities"])
            paths.append(path)
            bridge = records[first["target"]]
            # 허용하는 2-hop은 계약 조항의 로컬 부표가 공용 별첨표로
            # redirect하는 한 방향뿐이다. 공용 별첨 JO 안의 다른 표 선언을
            # 다시 따라가면 거대한 appendix JO 경계 때문에 무관 표로 샌다.
            if not first["key"].startswith("부표:"):
                continue
            if precise_redirects:
                redirect_keys = [bridge["redirects"].get(first["key"])]
            else:
                # Reproduce the historical C47/C48 artifact exactly.  New arms
                # must never use this Cartesian-product compatibility branch.
                redirect_keys = [key for key in bridge["refs"]
                                 if key.startswith("별첨:")]
            for key in (key for key in redirect_keys if key):
                final = resolve(bridge, key)
                if final and final not in {source["jo"], first["target"]}:
                    detail = bridge.get("redirect_details", {}).get(first["key"], {})
                    path = {"source": source["jo"], "bridge": first["target"],
                            "target": final, "keys": [first["key"], key],
                            "hops": 2}
                    if precise_redirects:
                        path.update(document=source["doc"],
                                    source_contracts=source["identities"],
                                    redirect_line=detail.get("line", ""))
                    paths.append(path)

    aliases, terms, provenance = defaultdict(list), defaultdict(list), defaultdict(list)
    for path in paths:
        source = records[path["source"]]
        target = path["target"]
        aliases[target].extend(source["identities"])
        terms[target].extend(source["terms"])
        provenance[target].append(path)
    for jo_id in aliases:
        aliases[jo_id] = ordered_unique(aliases[jo_id], 80)
        terms[jo_id] = ordered_unique(terms[jo_id], 80)

    return aliases, terms, provenance, {
        "direct_edges": direct_edges,
        "paths": paths,
        "ambiguous": ambiguous,
        "record_count": len(records),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", default=str(HERE / "out/elements_u3.jsonl"))
    parser.add_argument("--tags", default=str(HERE / "out/tags_u4_fact_rules.jsonl"))
    parser.add_argument("--jo", default=str(HERE / "out/elements_u3jo.jsonl"))
    parser.add_argument("--out", default=str(HERE / "out/tags_u5_reference_graph.jsonl"))
    parser.add_argument("--legacy-cross-product", action="store_true",
                        help="rebuild historical C47/C48 v1.1 only")
    parser.add_argument("--precise-redirects-only", action="store_true",
                        help="rebuild historical precise graph v1.2 without table catalog")
    args = parser.parse_args()
    element_path, tag_path, jo_path, output_path = map(
        Path, (args.elements, args.tags, args.jo, args.out))
    elements, tags, jo_rows = map(load_jsonl, (element_path, tag_path, jo_path))
    if [row["element_id"] for row in elements] != [row["element_id"] for row in tags]:
        raise ValueError("element/tag id order mismatch")

    member_to_jo = {member: jo["element_id"] for jo in jo_rows
                    for member in jo.get("members", [])}
    if args.legacy_cross_product and args.precise_redirects_only:
        raise ValueError("legacy and precise-only modes are mutually exclusive")
    precise_redirects = not args.legacy_cross_product
    include_catalog = precise_redirects and not args.precise_redirects_only
    graph_version = (LEGACY_VERSION if args.legacy_cross_product else
                     PRECISE_VERSION if args.precise_redirects_only else VERSION)
    aliases, terms, provenance, graph = build_reference_aliases(
        elements, tags, jo_rows, precise_redirects=precise_redirects)
    table_catalog, jo_documents = (([], {}) if not include_catalog else
                                   build_table_catalog(elements, tags, member_to_jo))
    table_catalog_audit = ({} if not include_catalog else
                           audit_table_catalog(table_catalog))
    output = []
    for source in tags:
        tag = copy.deepcopy(source)
        jo_id = member_to_jo[tag["element_id"]]
        tag["reference_graph_version"] = graph_version
        if aliases.get(jo_id):
            tag["reference_scope_aliases"] = aliases[jo_id]
        if terms.get(jo_id):
            tag["reference_query_terms"] = terms[jo_id]
        if provenance.get(jo_id):
            tag["reference_graph_paths"] = provenance[jo_id]
        output.append(tag)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "version": graph_version,
        "n": len(output),
        "inputs": {"elements": sha256(element_path), "tags": sha256(tag_path),
                   "jo": sha256(jo_path)},
        "output_sha256": sha256(output_path),
        "record_count": graph["record_count"],
        "direct_edge_count": len(graph["direct_edges"]),
        "one_hop_path_count": sum(path["hops"] == 1 for path in graph["paths"]),
        "two_hop_path_count": sum(path["hops"] == 2 for path in graph["paths"]),
        "ambiguous_reference_count": len(graph["ambiguous"]),
        "target_jo_count": len(aliases),
        "tagged_element_count": sum(bool(row.get("reference_scope_aliases")) for row in output),
        **({"table_catalog_count": len(table_catalog),
            "table_catalog": table_catalog,
            "table_catalog_audit": table_catalog_audit,
            "jo_documents": jo_documents} if include_catalog else {}),
        "invariants": {"qa_or_gold_read": False, "element_id_order_preserved": True,
                       "max_hops": 2, "ambiguous_edges_fail_closed": True,
                       "two_hop_direction": "local_schedule_to_global_appendix",
                       **({"two_hop_redirect_pairing":
                           "nearest_explicit_local_section"}
                          if precise_redirects else {})},
        "edge_ledger": graph["paths"],
        "ambiguous_ledger": graph["ambiguous"],
    }
    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n",
                          encoding="utf-8")
    print(json.dumps({key: value for key, value in stats.items()
                      if key not in {"edge_ledger", "ambiguous_ledger",
                                     "table_catalog", "jo_documents"}},
                     ensure_ascii=False))


if __name__ == "__main__":
    main()

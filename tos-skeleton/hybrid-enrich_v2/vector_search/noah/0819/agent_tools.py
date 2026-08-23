#!/usr/bin/env python3
"""에이전트 도구 CLI (search / msearch / read / submit) — 0819 고도화판.

filesearch/agent_tools.py 사본에 4개 확장을 얹었다. 신규 arm 플래그가 전부 꺼져 있으면
원본과 동일하게 동작한다(기준선 재현 게이트).
  alias:1            aliases.json 질의 확장(lex 채널 한정)
  scope_boost:3.0    search --scope soft-descent 가산 부스트 + browse 모드
  fallback:{...}     태그 상위 구조 축이 빈약할 때 msearch 자동 폴백
  ref_expand:1       read 에 참조 조 미리보기, search 상위 10건에 ref_jo 첨부
데이터는 전부 filesearch/out/ 을 참조한다(복사 금지). pickle 캐시만 이 디렉터리 out/ 에 둔다.
"""
import argparse, hashlib, json, os, pickle, re, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
VS = HERE.parents[1]
sys.path.insert(0, str(FS))
sys.path.insert(0, str(HERE))
import enhance
PAGE, PREVIEW, SEARCH_CAP, READ_CAP, SUBMIT_MAX = 40, 160, 20, 8, 10
GENERIC_AXES = ("container", "identity", "topic", "function", "locator", "constraint",
                "relation", "structure", "evidence")
FACT_CARD_GENERIC_SURFACES = {
    "지급", "급여금", "보험금", "특약", "보험", "보장", "조건", "기준", "요건",
    "한도", "연간", "포함", "해당", "여부", "종류", "무엇", "얼마", "어떻게",
    "대상", "대상이", "되는", "되나요", "알려", "알려줘", "인가요",
}

REFERENCE_QUERY_RE = re.compile(
    r"분류\s*코드|질병\s*코드|수가\s*코드|(?<![A-Za-z])코드|분류표|부표|별첨|"
    r"(?:포함|해당)\s*(?:되|되는지|되나|되나요)"
)


# R1 검색기 개선(기본 off) — 규칙×규칙. 상품명·질병명 하드코딩 사전 없음.
# (a) identity_expand : 특약명 목록(문서 도출)에서 공용 수식어를 제거한 별칭 변형을 만들어
#                       라우터가 놓친 특약을 soft 후보로만 추가한다(하드 필터 없음).
# (b) reference_follow: 라우터가 특약 X를 잡고 질문이 분류표·코드·서류 표면형을 가질 때만
#                       U5 명시 참조그래프에서 X의 조가 가리키는 target JO 를 소수 후보로
#                       하위 quota 에 얹는다(상위 보존, 발화 안 하면 결과 불변).
REFERENCE_FOLLOW_RE = re.compile(
    r"분류\s*코드|질병\s*코드|수가\s*코드|(?<![A-Za-z])코드|분류표|부표|별첨|별표|"
    r"구비\s*서류|제출\s*서류|증명\s*서류|서류|(?<![가-힣])표(?![가-힣적준])")

_SHARED_SEGMENT_CACHE = {}


def reference_follow_enabled(raw_query, gate="table_code_or_document"):
    """참조 팔로우 발화 조건 — 표/코드/서류 표면형이 질문에 있을 때만."""
    if not gate:
        return True
    if gate != "table_code_or_document":
        raise ValueError(f"unknown reference follow gate: {gate}")
    return bool(REFERENCE_FOLLOW_RE.search(str(raw_query)))


def identity_core(contract):
    """특약명 → 대괄호 수식어([기본]·[한도형]…)와 판본 접미를 뺀 정규화 핵심명."""
    from textmatch import contract_core
    return contract_core(re.sub(r"\[.*?\]", "", str(contract or "")))


def contract_shared_segments(cores, min_len=3, max_len=8, min_contracts=2):
    """문서에서 도출한 공용 수식어 후보.

    서로 다른 특약명 core 에 함께 나타나는 부분열만 남긴다. 사전이 아니라 특약명
    집합의 통계이므로 상품명·질병명 하드코딩이 아니다.
    """
    import collections as _c
    key = (tuple(sorted(set(cores))), min_len, max_len, min_contracts)
    if key in _SHARED_SEGMENT_CACHE:
        return _SHARED_SEGMENT_CACHE[key]
    df = _c.Counter()
    for core in set(cores):
        seen = set()
        for n in range(min_len, max_len + 1):
            for i in range(len(core) - n + 1):
                seen.add(core[i:i + n])
        df.update(seen)
    shared = {seg for seg, n in df.items() if n >= min_contracts}
    _SHARED_SEGMENT_CACHE[key] = shared
    return shared


def identity_variants(core, shared, min_variant_len=4, max_variants=6):
    """core 에서 공용 수식어 1개를 뺀 별칭 변형(긴 수식어부터)."""
    output = []
    for seg in sorted(shared, key=lambda s: (-len(s), s)):
        if seg not in core:
            continue
        variant = core.replace(seg, "", 1)
        if len(variant) >= min_variant_len and variant != core and variant not in output:
            output.append(variant)
        if len(output) >= max_variants:
            break
    return output


def identity_expansion_candidates(router, raw_query, slots, conf, config=None):
    """라우터가 놓친 특약을 별칭 변형으로 회수한다.

    라우터가 이미 완전 일치(exact)로 특약을 잡았으면 변형도 완전 일치만 인정해
    과발화를 막고, 부분 일치/미부여였을 때만 라우터와 동일한 LCS 규칙을 쓴다.
    여러 특약이 동시에 걸리면 전부 soft 후보로 넣는다(하드 필터 금지).
    """
    from textmatch import compact
    config = config if isinstance(config, dict) else {}
    cq = compact(raw_query)
    if not cq:
        return [], {}
    routed = list(slots.get("contract", []))
    routed_set = set(routed)
    require_exact = bool(routed) and "contract" not in (conf or {})
    min_variant_len = int(config.get("min_variant_len", 4))
    cores = {c: identity_core(c) for c in router.contracts}
    shared = contract_shared_segments(
        [core for core in cores.values() if core],
        min_len=int(config.get("segment_min_len", 3)),
        max_len=int(config.get("segment_max_len", 8)),
        min_contracts=int(config.get("segment_min_contracts", 2)))
    hits = {}
    for contract, core in cores.items():
        if contract in routed_set or len(core) < min_variant_len:
            continue
        best = (0.0, 0, "")
        for variant in identity_variants(
                core, shared, min_variant_len=min_variant_len,
                max_variants=int(config.get("max_variants", 6))):
            if require_exact and variant in shared:
                # 라우터가 이미 완전 일치로 특약을 확정했으면, 여러 특약 core 에 공통으로
                # 나타나는 비변별 별칭(예: 어미만 남은 조각)으로는 다른 특약을 끌어오지 않는다.
                continue
            if variant in cq:
                score = 1.0
            elif require_exact:
                continue
            else:
                overlap = router._lcs(variant, cq)
                ratio = overlap / len(variant)
                score = ratio if (overlap >= 6 or (overlap >= 4 and ratio >= 0.6)) else 0.0
            if score > 0 and (score, len(variant)) > (best[0], best[1]):
                best = (score, len(variant), variant)
        if best[0] > 0:
            hits[contract] = best
    if not hits:
        return [], {}
    top = max((row[0], row[1]) for row in hits.values())
    selected = [c for c, row in hits.items() if (row[0], row[1]) == top]
    selected.sort(key=lambda c: (-len(cores[c]), c))
    selected = selected[:int(config.get("limit", 12))]
    audit = {"mode": "document_derived_identity_alias",
             "require_exact": require_exact,
             "routed_before": routed,
             "added": selected,
             "via": {c: hits[c][2] for c in selected},
             "score": round(top[0], 4)}
    return selected, audit


def reference_follow_ranking(search, slots, stats, limit=5, max_hops=2):
    """라우터가 잡은 특약 X 의 조가 가리키는 참조 target JO 를 후보 ranking 으로."""
    import collections as _c
    from textmatch import compact
    routed = list(slots.get("contract", []))
    if not routed:
        return [], {}
    routed_keys = {compact(c) for c in routed if compact(c)}
    counts = _c.Counter()
    meta = {}
    for edge in stats.get("edge_ledger", []):
        hops = int(edge.get("hops", 0))
        if not 0 < hops <= max_hops:
            continue
        sources = edge.get("source_contracts") or []
        if not any(compact(c) in routed_keys for c in sources):
            continue
        target = edge.get("target", "")
        if not target:
            continue
        counts[target] += 1
        row = meta.setdefault(target, {"hops": hops, "keys": [], "sources": []})
        row["hops"] = min(row["hops"], hops)
        for key in (edge.get("keys") or []):
            if key not in row["keys"]:
                row["keys"].append(key)
        if edge.get("source") and edge["source"] not in row["sources"]:
            row["sources"].append(edge["source"])
    if not counts:
        return [], {}
    order = sorted(counts, key=lambda t: (meta[t]["hops"], -counts[t], t))[:limit]
    jo_index = {unit["element_id"]: unit for unit in search._jo}
    ranked = []
    for rank, target in enumerate(order, start=1):
        unit = jo_index.get(target)
        if not unit:
            continue
        for member in unit.get("members", []):
            index = search._eidx.get(member)
            if index is None:
                continue
            ranked.append((search.E[index], round(1.0 / rank, 8)))
    audit = {"mode": "u5_explicit_reference_targets_in_tail_quota",
             "routed_contracts": routed[:12],
             "targets": [{"jo": t, "hops": meta[t]["hops"], "edges": counts[t],
                          "via": meta[t]["keys"][:4],
                          "source_jo": meta[t]["sources"][:3]} for t in order],
             "candidate_elements": len(ranked)}
    return ranked, audit


def reference_bundle_enabled(raw_query, gate="membership_or_code"):
    """Only expose explicit reference paths for questions that need them."""
    if not gate:
        return True
    if gate != "membership_or_code":
        raise ValueError(f"unknown reference bundle gate: {gate}")
    return bool(REFERENCE_QUERY_RE.search(str(raw_query)))


def reference_projection_enabled(raw_query, gate="classification_membership"):
    """Stricter graph gate: classification/code or entity-membership questions.

    A question such as "which treatments are included" asks for a local list,
    not a disease appendix.  Explicit code/table wording remains authoritative;
    otherwise immediate treatment/action membership surfaces fail closed.
    """
    if not gate:
        return True
    if gate != "classification_membership":
        raise ValueError(f"unknown reference projection gate: {gate}")
    text = str(raw_query)
    if re.search(r"분류\s*코드|질병\s*코드|수가\s*코드|(?<![A-Za-z])코드|분류표|부표|별첨", text):
        return True
    if not re.search(r"(?:포함|해당)\s*(?:되|되는지|되나|되나요)", text):
        return False
    return not bool(re.search(
        r"(?:치료|수술|입원|통원|특약)\s*(?:이|가|은|는|도|에|에는)?\s*"
        r"(?:포함|해당)", text))


def explicit_reference_anchor(raw_query):
    """Return a quoted table/topic anchor that a projected target must contain."""
    if not re.search(r"분류표|분류\s*코드|질병\s*코드|수가\s*코드|(?<![A-Za-z])코드", str(raw_query)):
        return ""
    quoted = re.findall(r"['\"“”‘’]([^'\"“”‘’]{5,100})['\"“”‘’]", str(raw_query))
    return max(quoted, key=len) if quoted else ""


def reference_bundles_for_items(items, paths, jo_index, tokens, snip,
                                max_source_rank=12, max_targets_per_source=2,
                                max_hops=2):
    """Attach graph targets under displayed source JOs without re-ranking."""
    outgoing = {}
    for path in paths:
        if int(path.get("hops", 0)) > max_hops:
            continue
        outgoing.setdefault(path.get("source", ""), []).append(path)
    linked = []
    for source_rank, item in enumerate(items[:max_source_rank], start=1):
        source_jo = item.get("jo") or ""
        seen_targets, evidence = set(), []
        ordered_paths = sorted(outgoing.get(source_jo, ()),
                               key=lambda row: (row.get("hops", 99), row.get("target", "")))
        for path in ordered_paths:
            target_id = path.get("target", "")
            target = jo_index.get(target_id)
            if not target or target_id in seen_targets:
                continue
            seen_targets.add(target_id)
            evidence.append({
                "id": target_id, "jo": target_id,
                "hops": int(path.get("hops", 0)),
                "via": list(path.get("keys") or []),
                "bridge": path.get("bridge", ""),
                "contract": (target.get("contract_scope") or "")[:40],
                "jo_title": (target.get("title") or "")[:40],
                "preview": snip(target.get("text") or "", tokens),
            })
            if len(evidence) >= max_targets_per_source:
                break
        if evidence:
            item["linked_evidence"] = evidence
            linked.append({"source_rank": source_rank, "source_jo": source_jo,
                           "targets": [row["jo"] for row in evidence]})
    return linked


def reference_projection_for_items(items, paths, jo_index, tokens, raw_query="",
                                   max_source_rank=12, limit=2, max_hops=2,
                                   min_token_matches=2, strong_token_chars=5):
    """Return a small globally deduplicated set of query-supported graph targets.

    C40 attached the same appendix below several source JOs and also exposed
    unrelated targets from a broad source.  This projection keeps the base page
    unchanged, deduplicates by target JO, and fails closed unless target text
    itself contains concrete surfaces from the question.
    """
    outgoing = {}
    for path in paths:
        hops = int(path.get("hops", 0))
        if not 0 < hops <= max_hops:
            continue
        outgoing.setdefault(path.get("source", ""), []).append(path)
    existing_jo = {item.get("jo") for item in items if item.get("jo")}
    anchor = explicit_reference_anchor(raw_query)
    compact_anchor = re.sub(r"[^가-힣a-zA-Z0-9]", "", anchor).lower()
    candidates = {}
    for source_rank, item in enumerate(items[:max_source_rank], start=1):
        source_jo = item.get("jo") or ""
        for path in outgoing.get(source_jo, ()):
            target_id = path.get("target", "")
            target = jo_index.get(target_id)
            if not target or target_id in existing_jo:
                continue
            support = fact_card_query_support(
                tokens, target.get("text", ""), min_matches=min_token_matches,
                strong_token_chars=strong_token_chars)
            compact_target = re.sub(
                r"[^가-힣a-zA-Z0-9]", "", target.get("text", "")).lower()
            if (not support["allowed"] or not support["strong"]
                    or (compact_anchor and compact_anchor not in compact_target)):
                continue
            row = candidates.setdefault(target_id, {
                "jo": target_id, "source_ranks": [], "source_jos": [],
                "hops": int(path.get("hops", 0)), "paths": [], "support": support,
                "explicit_anchor": anchor,
            })
            row["source_ranks"].append(source_rank)
            row["source_jos"].append(source_jo)
            row["hops"] = min(row["hops"], int(path.get("hops", 0)))
            row["paths"].append({
                "source": source_jo, "bridge": path.get("bridge", ""),
                "keys": list(path.get("keys") or []), "hops": int(path.get("hops", 0)),
            })

    def order(row):
        support = row["support"]
        return (-len(support["distinctive"]), -len(support["matched"]),
                -max((len(token) for token in support["strong"]), default=0),
                -len(set(row["source_jos"])), min(row["source_ranks"]),
                row["hops"], row["jo"])

    output = sorted(candidates.values(), key=order)[:limit]
    for row in output:
        row["source_ranks"] = sorted(set(row["source_ranks"]))
        row["source_jos"] = list(dict.fromkeys(row["source_jos"]))
    return output


def reference_evidence_enabled(raw_query, gate="reference_or_scenario"):
    """Open precise linked evidence only for explicit reference or scenario intent."""
    if not gate:
        return True
    if gate != "reference_or_scenario":
        raise ValueError(f"unknown reference evidence gate: {gate}")
    return (reference_projection_enabled(raw_query)
            or scenario_intent_query(raw_query) is not None)


def _compact_reference_key(key):
    """Return the literal table label carried by a graph edge (for example 12-1)."""
    match = re.search(r"(?:^|:)표:([^:]+)$", str(key))
    return re.sub(r"[^0-9A-Za-z가-힣]", "", match.group(1)).lower() if match else ""


def _member_reference_labels(text):
    labels = re.findall(r"(?:부표|별첨\s*\d*\s*\[?\s*표|(?<![가-힣])표)\s*([0-9]+(?:[-의][0-9]+)*)",
                        str(text), flags=re.IGNORECASE)
    return {re.sub(r"[^0-9A-Za-z가-힣]", "", label).lower() for label in labels}


def table_catalog_evidence(raw_query, catalog, allowed_documents, max_variants=5):
    """Expose every JO in one exact-title table region from the top source doc.

    This is a deterministic data projection, not a reranker.  It fires only when
    a catalog title of at least four normalized characters is literally present
    in the question.  The fixed hybrid result selects the source document; if
    that document contains duplicate matching regions, the lookup fails closed.
    """
    query_surface = re.sub(r"[^0-9A-Za-z가-힣]", "", str(raw_query)).lower()
    allowed = [doc for doc in dict.fromkeys(allowed_documents) if doc]
    if not allowed:
        return []
    matches = [row for row in catalog
               if row.get("doc") in allowed
               and len(str(row.get("title_surface", ""))) >= 4
               and str(row.get("title_surface", "")) in query_surface]
    if not matches:
        return []
    longest = max(len(str(row.get("title_surface", ""))) for row in matches)
    matches = [row for row in matches
               if len(str(row.get("title_surface", ""))) == longest]
    if len(matches) != 1:
        return []
    match = matches[0]
    selected_doc = str(match.get("doc", ""))
    variants = list(match.get("variants", ()))
    variant_limit = max(1, int(max_variants))
    if not variants or len(variants) > variant_limit:
        return []
    output = []
    for rank, variant in enumerate(variants, start=1):
        jo_id = str(variant.get("jo", ""))
        if not jo_id:
            continue
        output.append({
            "id": jo_id, "jo": jo_id,
            "provenance": "exact_table_title_catalog",
            "source_document": selected_doc,
            "catalog_key": match.get("key", ""),
            "catalog_title": match.get("title", ""),
            "catalog_variant_rank": rank,
            "region_member_ids": list(variant.get("element_ids") or []),
            "preview": str(variant.get("preview", ""))[:1600],
        })
    return output


def _bounded_reference_regions(members, required_label="", max_members=8,
                               max_chars=1600, stop_on_mixed_labels=False):
    """Yield table-local member windows while tolerating OCR spacer paragraphs.

    Converted insurance appendices often split a table label, heading, OCR noise,
    definition, and the actual code row across five or more elements.  A fixed
    three-element window loses the row.  The bound below is structural: begin at
    a literal table label, stop before a different table label, and cap both
    member count and characters.  It therefore expands the *source region*, not
    the retrieval candidate set or rank.
    """
    seen = set()
    for index, member in enumerate(members):
        labels = _member_reference_labels(member.get("text", ""))
        if not labels or (required_label and required_label not in labels):
            continue
        region, chars = [], 0
        for offset, row in enumerate(members[index:index + max(1, int(max_members))]):
            row_labels = _member_reference_labels(row.get("text", ""))
            if offset and row_labels:
                if required_label not in row_labels:
                    break
                # OCR conversion sometimes places the repeated current label
                # and the next table label in one spacer element ("표39 ...
                # 표42").  The legacy C47 window included that spacer and the
                # following unrelated table.  A strict arm stops before a row
                # that introduces any different label.
                if (stop_on_mixed_labels and required_label
                        and any(label != required_label for label in row_labels)):
                    break
            text = str(row.get("text", ""))
            if region and chars + len(text) > max_chars:
                break
            region.append(row)
            chars += len(text)
        member_ids = tuple(row.get("element_id", "") for row in region)
        if member_ids and member_ids not in seen:
            seen.add(member_ids)
            yield {
                "label": required_label or sorted(labels)[0],
                "members": region,
                "text": "\n".join(row.get("text", "") for row in region),
            }


def reference_evidence_for_items(items, paths, jo_index, element_index, tokens,
                                 raw_query="", normalized_tokens=(),
                                 max_source_rank=40, limit=2, max_hops=2,
                                 min_token_matches=2, strong_token_chars=5,
                                 region_members=8, stop_on_mixed_labels=False,
                                 sanitize_region_title=False):
    """Project exact table regions without changing the base hybrid result order.

    U5 resolves legal references at JO level.  A target JO can contain several
    appendix tables, so displaying its generic JO preview still hides the row the
    question needs.  This function locates the graph edge's literal table label
    inside the target JO and scores only the heading plus adjacent source members.
    The returned ID remains the original submit-compatible target JO; the concise
    region is presentation/provenance, not a new rank or synthetic Gold unit.
    """
    outgoing = {}
    for path in paths:
        hops = int(path.get("hops", 0))
        if 0 < hops <= max_hops:
            outgoing.setdefault(path.get("source", ""), []).append(path)
    existing_jo_rank = {}
    for rank, item in enumerate(items, start=1):
        jo_id = item.get("jo")
        if jo_id and jo_id not in existing_jo_rank:
            existing_jo_rank[jo_id] = rank
    query_tokens = list(dict.fromkeys(
        str(token) for token in (*tokens, *normalized_tokens) if str(token).strip()))
    explicit_ref = bool(re.search(
        r"분류\s*코드|질병\s*코드|수가\s*코드|(?<![A-Za-z])코드|분류표|부표|별첨",
        str(raw_query)))
    candidates = {}

    def candidate_order(row):
        # A source-derived legal reference is stronger provenance than a
        # lexical explanation of an already retrieved table.  Direct regions
        # remain a fail-safe only when no graph-backed region fills the budget.
        return (row.get("provenance") != "reference_graph",
                -len(row["distinctive_query_surfaces"]),
                -len(row["matched_query_surfaces"]), row["source_rank"],
                row["hops"], row["jo"])

    def add_candidate(target_id, source_rank, source_jo, hops, keys, bridge,
                      provenance, required_label=""):
        target = jo_index.get(target_id)
        if not target:
            return
        members = [element_index[mid] for mid in target.get("members", [])
                   if mid in element_index]
        regions = []
        for region in _bounded_reference_regions(
                members, required_label=required_label,
                max_members=region_members,
                stop_on_mixed_labels=stop_on_mixed_labels):
            support = fact_card_query_support(
                query_tokens, region["text"], min_matches=min_token_matches,
                strong_token_chars=strong_token_chars)
            has_code = bool(re.search(r"(?<![A-Za-z])[A-Z]\d{2}(?:\.\d+)?(?!\d)",
                                      region["text"]))
            # Four-syllable Korean disease names (대상포진, 제자리암, 뇌출혈 …)
            # are already distinctive when the same bounded region contains an
            # actual classification code.  Keep the stricter five-character
            # threshold for non-code/scenario paths.
            code_subject = (explicit_ref and has_code and any(
                len(surface) >= 4 for surface in support["distinctive"]))
            if not (support["allowed"] or code_subject):
                continue
            if not (support["strong"] or code_subject):
                continue
            if explicit_ref and re.search(r"코드|분류표", str(raw_query)) and not has_code:
                continue
            if provenance == "existing_hybrid_region":
                direct_generic = {"간편", "보장대상", "질병분류", "분류코드", "코드",
                                  "질병", "분류", "무엇", "무엇인가요"}
                subject_surfaces = [normalize_surface_token(token)
                                    for token in query_tokens]
                subject_surfaces = [surface for surface in subject_surfaces
                                    if len(surface) >= 3 and surface not in direct_generic]
                # Exact Korean/Latin token boundaries reject near-but-different
                # rows such as 대상포진수막염(B02.1) for a 대상포진(B02) query.
                if not any(re.search(
                        rf"(?<![가-힣A-Za-z0-9]){re.escape(surface)}"
                        rf"(?![가-힣A-Za-z0-9])", region["text"], re.IGNORECASE)
                           for surface in subject_surfaces):
                    continue
            regions.append({**region, "support": support, "has_code": has_code})
        if not regions:
            return
        best = sorted(
            regions,
            key=lambda row: (-len(row["support"]["distinctive"]),
                             -len(row["support"]["matched"]),
                             -max((len(token) for token in row["support"]["strong"]),
                                  default=0),
                             not row["has_code"],
                             [member["element_id"] for member in row["members"]]),
        )[0]
        target_title = (target.get("title") or "")[:80]
        if sanitize_region_title and target_title:
            compact_title = re.sub(r"[^0-9A-Za-z가-힣]", "", target_title).lower()
            compact_region = re.sub(r"[^0-9A-Za-z가-힣]", "", best["text"]).lower()
            if compact_title and compact_title not in compact_region:
                target_title = (f"참조표 {required_label}" if required_label
                                else "참조 근거 영역")
        candidate = {
            "id": target_id, "jo": target_id,
            "source_rank": source_rank, "source_jo": source_jo,
            "hops": int(hops), "via": list(keys), "bridge": bridge,
            "provenance": provenance,
            "contract": (target.get("contract_scope") or "")[:80],
            "jo_title": target_title,
            "base_result_rank": existing_jo_rank.get(target_id),
            "region_member_ids": [row["element_id"] for row in best["members"]],
            "preview": best["text"][:1200],
            "matched_query_surfaces": best["support"]["matched"],
            "distinctive_query_surfaces": best["support"]["distinctive"],
        }
        previous = candidates.get(target_id)
        if previous is None or candidate_order(candidate) < candidate_order(previous):
            candidates[target_id] = candidate

    for source_rank, item in enumerate(items[:max_source_rank], start=1):
        source_jo = item.get("jo") or ""
        for path in outgoing.get(source_jo, ()):
            keys = list(path.get("keys") or [])
            final_label = _compact_reference_key(keys[-1] if keys else "")
            if not final_label:
                continue
            add_candidate(
                path.get("target", ""), source_rank, source_jo,
                int(path.get("hops", 0)), keys, path.get("bridge", ""),
                "reference_graph", required_label=final_label)

    # Some appendix rows already occur deep in the fixed hybrid page, but no
    # source clause on that page exposes the graph edge.  For explicit code/table
    # questions only, show the exact table-local region of those existing JOs.
    # This is a 0-hop explanation of an existing candidate: no ID is introduced
    # and the 40-result order remains byte-for-byte unchanged.
    if explicit_ref:
        for target_id, source_rank in existing_jo_rank.items():
            if source_rank > max_source_rank:
                continue
            add_candidate(
                target_id, source_rank, target_id, 0, [],
                "existing_hybrid_result", "existing_hybrid_region")
    return sorted(
        candidates.values(), key=candidate_order)[:limit]


SCENARIO_COVERAGE_RE = re.compile(
    r"(?P<cause>[가-힣A-Za-z0-9·\s]{2,50}?)(?:으로|로|때문에|중에|하다가)\s*"
    r"(?:인한|생긴|발생한)?\s*(?P<harm>[가-힣A-Za-z0-9·\s]{2,30}?)"
    r"(?:도|이|가|은|는)?\s*(?:보상|보장)\s*(?:돼|되|가능|받)"
)
SCENARIO_CAUSE_LINK_RE = re.compile(
    r"(?:으로\s*인해|로\s*인해|때문에|(?<=\s)중에(?=\s)|"
    r"(?<=\s)중(?=\s)|하다가)"
)
SCENARIO_HARM_RE = re.compile(
    r"후유장해|장해|상해|골절|재해|사고|다(?:치|쳐|셨)|사망"
)
SCENARIO_COVERAGE_ASK_RE = re.compile(
    r"보상|보장|보험금.{0,18}(?:받|나오|지급)|보험\s*(?:돼|되|가능)"
)


def scenario_intent_query(raw_query):
    """Normalize a concrete accident scenario into a generic insurance intent.

    The cause phrase is deliberately not copied into the expansion: unseen
    activities (diving, hiking, sports, work) share the same accident/coverage
    ontology, so no activity dictionary or evaluation-QID rule is needed.
    """
    text = str(raw_query)
    match = SCENARIO_COVERAGE_RE.search(text)
    harm = re.sub(r"\s+", "", match.group("harm")) if match else ""
    if not match:
        connector = SCENARIO_CAUSE_LINK_RE.search(text)
        tail = text[connector.end():] if connector else ""
        harm_match = SCENARIO_HARM_RE.search(tail)
        if not connector or not harm_match or not SCENARIO_COVERAGE_ASK_RE.search(tail):
            return None
        harm = harm_match.group(0)
    # Fail closed: a generic payment/diagnosis sentence that happens to contain
    # "...으로 ... 보장" must not receive an unrelated scenario candidate.
    if not SCENARIO_HARM_RE.search(harm):
        return None
    if "후유장해" in harm:
        canonical_harm = "후유장해"
    elif "장해" in harm:
        canonical_harm = "장해"
    elif "골절" in harm:
        canonical_harm = "골절"
    elif "사망" in harm:
        canonical_harm = "재해사망"
    elif re.search(r"상해|다(?:치|쳐|셨)", harm):
        canonical_harm = "상해"
    else:
        canonical_harm = "재해"
    terms = [canonical_harm, "재해의 정의", "재해분류표", "보장대상이 되는 재해",
             "우발적인 외래의 사고", "보험금을 지급하지 않는 재해"]
    intent = "accident_coverage"
    return {"intent": intent, "harm": canonical_harm, "query": " ".join(terms)}


def requested_claim_roles(raw_query, slots, limit=3):
    """Return two or more independently requested evidence roles."""
    from structured_search import requested_evidence_roles
    raw_roles = requested_evidence_roles(
        raw_query, slots, limit=12, expanded=True)
    if ("definition" in raw_roles and not re.search(
            r"정의|뜻|의미|이라\s*함은|라\s*함은|무엇을\s*말", raw_query)):
        raw_roles.remove("definition")
    families = {"payment_trigger": "eligibility",
                "code_reference": "classification"}
    if "criteria_rule" in raw_roles:
        if "payment_trigger" in raw_roles:
            families["criteria_rule"] = "eligibility"
        elif "code_reference" in raw_roles:
            families["criteria_rule"] = "classification"
        else:
            families["criteria_rule"] = "eligibility"
    if "definition" in raw_roles and "code_reference" in raw_roles:
        families["definition"] = "classification"
    preferred = {
        "eligibility": ("payment_trigger", "criteria_rule"),
        "classification": ("code_reference", "definition"),
    }
    grouped = {}
    for role in raw_roles:
        grouped.setdefault(families.get(role, role), []).append(role)
    roles = []
    for family, members in grouped.items():
        order = preferred.get(family, tuple(members))
        roles.append(next(role for role in order if role in members))
    return roles[:limit] if len(roles) >= 2 else []


def explicit_contract_scopes(raw_query, slots):
    """Return routed contracts whose document-derived identity is written in the query.

    This is deliberately stricter than Router partial/reverse-index inference.  The helper
    is used only by the opt-in claim-role arm so a role query cannot drift into another
    rider merely because it shares generic payment/definition tags.
    """
    from structured_search import explicit_identity_name
    query = re.sub(r"[^가-힣a-zA-Z0-9]", "", str(raw_query)).lower()
    output = []
    for contract in slots.get("contract", []):
        identity = explicit_identity_name(contract).lower()
        if identity and "특약" in identity and identity in query:
            output.append(contract)
    return list(dict.fromkeys(output))


def role_bundle_items(search, raw_query, base_slots, roles, weights, profile,
                      per_role=2, allowed_contracts=None):
    """Retrieve submit-compatible JO evidence independently for each claim role."""
    from patterns import ROLE_KO
    bundles = []
    for role in roles:
        role_query = f"{raw_query} {ROLE_KO.get(role, role)}"
        routed, role_tokens = search.router.route(role_query)
        routed.pop("_conf", None)
        # Preserve explicit identity/subject from the original route, then add
        # one exact evidence role.  No product/disease/QID dictionaries enter.
        role_slots = {key: list(value) for key, value in base_slots.items()
                      if isinstance(value, list) and value}
        for key in ("contract", "identity", "subject", "variant"):
            if routed.get(key):
                role_slots[key] = list(dict.fromkeys(
                    role_slots.get(key, []) + list(routed[key])))
        role_slots["function"] = [role]
        _, lexical = search.match_table(role_slots, role_tokens)
        ranked = search.rank_structured(
            role_slots, role_tokens, role_query, weights=weights,
            profile=profile, limit=200, lexical_counts=lexical)
        seen_jo, candidates = set(), []
        for element, score in ranked:
            if (allowed_contracts is not None
                    and element.get("contract_scope", "") not in allowed_contracts):
                continue
            jo = search._jo[search._m2j[element["element_id"]]]
            jo_id = jo["element_id"]
            if jo_id in seen_jo:
                continue
            seen_jo.add(jo_id)
            candidates.append((element, jo, score, role_tokens))
            if len(candidates) >= per_role:
                break
        bundles.append({"role": role, "label": ROLE_KO.get(role, role),
                        "candidates": candidates})
    return bundles


def inject_claim_role_items(items, groups, after=5, limit=3):
    """Promote one source-derived JO per requested role after a protected prefix.

    Existing candidates are moved rather than duplicated.  Round-robin role order keeps a
    multi-claim question from spending every slot on the first role.
    """
    after = max(0, min(int(after), len(items)))
    limit = max(0, int(limit))
    prefix = list(items[:after])
    protected = {item.get("jo") or item.get("id") for item in prefix}
    promoted, promoted_ids = [], set()
    depth = 0
    while len(promoted) < limit:
        added = False
        for group in groups:
            candidates = group.get("items", [])
            if depth >= len(candidates):
                continue
            candidate = dict(candidates[depth])
            jo_id = candidate.get("jo") or candidate.get("id")
            if not jo_id or jo_id in protected or jo_id in promoted_ids:
                continue
            candidate.setdefault("src", "semantic_claim_role")
            promoted.append(candidate)
            promoted_ids.add(jo_id)
            added = True
            if len(promoted) >= limit:
                break
        if not added and all(depth + 1 >= len(group.get("items", [])) for group in groups):
            break
        depth += 1
    tail = [item for item in items[after:]
            if (item.get("jo") or item.get("id")) not in promoted_ids]
    return (prefix + promoted + tail)[:len(items)], promoted


def normalize_surface_token(token):
    value = re.sub(r"[^가-힣a-zA-Z0-9.]", "", str(token)).lower()
    for suffix in ("으로", "에서", "까지", "부터", "처럼", "이라", "라고", "인가",
                   "에는", "에게", "보다", "의", "이", "가", "은", "는", "을", "를", "도", "에", "로"):
        if len(value) >= len(suffix) + 2 and value.endswith(suffix):
            value = value[:-len(suffix)]
            break
    return value


def fact_card_query_support(tokens, text, min_matches=2, strong_token_chars=5):
    """Fail closed unless a card contains concrete query surfaces.

    Generic intent words and a rider name alone are insufficient.  One long
    non-identity term (e.g. 사고증명서) may stand alone; otherwise two surfaces
    must co-occur in the source-derived card.
    """
    haystack = re.sub(r"[^가-힣a-zA-Z0-9.]", "", str(text)).lower()
    normalized = list(dict.fromkeys(filter(None, map(normalize_surface_token, tokens))))
    matched = [token for token in normalized if token in haystack]
    distinctive = [token for token in matched
                   if token not in FACT_CARD_GENERIC_SURFACES and "특약" not in token]
    strong = [token for token in distinctive if len(token) >= strong_token_chars]
    allowed = bool(distinctive) and (len(matched) >= min_matches or bool(strong))
    return {"allowed": allowed, "matched": matched, "distinctive": distinctive,
            "strong": strong}


def dependency_fingerprint(paths):
    digest = hashlib.sha256()
    entries = []
    for path in paths:
        path = Path(path).resolve()
        content_sha = hashlib.sha256(path.read_bytes()).hexdigest()
        digest.update(str(path).encode()); digest.update(b"\0"); digest.update(content_sha.encode())
        entries.append({"path": str(path), "sha256": content_sha})
    return digest.hexdigest(), entries


def compact_items_by_jo(items):
    """현재 표시 순서를 보존하고 같은 조(채점 단위)의 중복 element만 접는다."""
    compacted, seen = [], set()
    for item in items:
        key = item.get("jo") or item.get("id")
        if key in seen:
            continue
        seen.add(key)
        compacted.append(item)
    return compacted


def group_items_by_jo(items, max_evidence=0):
    """같은 조의 후보 수는 줄이되 서로 다른 근거 snippet과 원 ID는 보존한다."""
    grouped, by_key = [], {}
    for raw_rank, item in enumerate(items, start=1):
        key = item.get("jo") or item.get("id")
        if key not in by_key:
            group = {"id": key, "jo": item.get("jo") or ""}
            for field in ("contract", "jo_title", "title", "variant"):
                if item.get(field):
                    group[field] = item[field]
            group["evidence_variants"] = []
            by_key[key] = group
            grouped.append(group)
        group = by_key[key]
        if not max_evidence or len(group["evidence_variants"]) < max_evidence:
            evidence = {"raw_rank": raw_rank, "id": item.get("id"),
                        "score": item.get("score", 0), "preview": item.get("preview") or ""}
            for field in ("src", "phase", "tag", "variant"):
                if item.get(field):
                    evidence[field] = item[field]
            group["evidence_variants"].append(evidence)
    return grouped


def preserve_baseline_jo_prefix(baseline, challenger, member_to_jo, n_jo):
    """Keep the complete baseline raw prefix needed to expose its first N unique jo."""
    if not n_jo:
        return challenger
    seen_jo = set()
    prefix_end = 0
    for prefix_end, (element, _) in enumerate(baseline, start=1):
        jo = member_to_jo.get(element["element_id"], element["element_id"])
        seen_jo.add(jo if isinstance(jo, str) else jo.get("element_id", element["element_id"]))
        if len(seen_jo) >= n_jo:
            break
    prefix = baseline[:prefix_end]
    seen_ids = {element["element_id"] for element, _ in prefix}
    output = list(prefix)
    for item in list(challenger) + list(baseline[prefix_end:]):
        if item[0]["element_id"] not in seen_ids:
            seen_ids.add(item[0]["element_id"]); output.append(item)
    return output


def reciprocal_rank_fuse(rankings, limit=400, rrf_k=60):
    """서로 다른 규칙 기반 tag ranker의 순위만 융합한다(LLM/reranker 없음)."""
    fused, objects, first_seen = {}, {}, {}
    serial = 0
    for ranked in rankings:
        for rank, (element, _) in enumerate(ranked, start=1):
            eid = element["element_id"]
            objects[eid] = element
            if eid not in first_seen:
                first_seen[eid] = serial; serial += 1
            fused[eid] = fused.get(eid, 0.0) + 1.0 / (max(1, rrf_k) + rank)
    order = sorted(fused, key=lambda eid: (-fused[eid], first_seen[eid], eid))
    return [(objects[eid], fused[eid]) for eid in order[:limit]]


def unique_jo_quota(rankings, member_to_jo, quotas, limit=400):
    """각 규칙 ranker가 찾은 서로 다른 JO를 고정 quota로 후보화한다."""
    output, seen_ids, seen_jo = [], set(), set()

    def push(element, score):
        eid = element["element_id"]
        jo = member_to_jo.get(eid, eid)
        if eid in seen_ids or jo in seen_jo or len(output) >= limit:
            return False
        seen_ids.add(eid); seen_jo.add(jo); output.append((element, score))
        return True

    for ranked, quota in zip(rankings, quotas):
        added = 0
        for element, score in ranked:
            if push(element, score):
                added += 1
            if added >= quota:
                break
    for ranked in rankings:
        for element, score in ranked:
            push(element, score)
            if len(output) >= limit:
                return output
    return output


def original_question(current_query):
    """세션 원 질문을 반환한다. 파일이 없는 수동 호출은 현재 질의를 쓴다."""
    question_file = Path(os.environ.get("SEMTAG_SESSION", ".")) / "question.json"
    if question_file.exists():
        return json.load(open(question_file, encoding="utf-8"))["question"]
    return os.environ.get("SEMTAG_QUESTION", current_query)


def evidence_unit_gate(raw_query, mode):
    """긴 표의 희귀행 후보는 범주 구성원/코드 질문에만 추가한다."""
    if not mode:
        return True
    if mode != "membership_or_code":
        raise ValueError(f"unknown evidence unit gate: {mode}")
    return bool(re.search(
        r"포함|해당|분류\s*코드|질병\s*코드|수가\s*코드|(?<![A-Za-z])코드",
        raw_query, re.I))


def membership_support_plan(search, raw_query):
    """범주 X에 구성원 Y가 포함되는지 묻는 질문의 답 담화 범위를 찾는다.

    X/Y 표면형만 규칙으로 추출하고, X가 문서 identity의 실제 이름에
    있는 경우만 scope로 쓴다. 대괄호 판본 표지 안에만 X가 있는 다른 문서는
    제외하므로 특정 상품·질병 사전 없이도 질문의 직접 범위를 유지한다.
    """
    match = re.search(
        r"(?P<category>[^\s,?.!]{2,30})(?:이|가|은|는)\s+"
        r"(?P<member>.{2,60}?)(?:도)?\s*(?:포함|해당)", raw_query)
    if not match:
        return None
    category = match.group("category").strip()
    member = re.sub(r"도\s*$", "", match.group("member").strip())
    if len(category) < 2 or len(member) < 2:
        return None

    from structured_search import compact, explicit_identity_name
    identities = []
    for fields in search.ensure_structured().fields:
        identity = fields.get("identity_full", "")
        surface = explicit_identity_name(identity)
        if compact(category) in compact(surface) and identity not in identities:
            identities.append(identity)
    if not identities:
        return None
    return {
        "query": raw_query,
        "slots": {"identity": identities[:12], "subject": [category, member]},
        "category": category,
        "member": member,
        "identity_count": len(identities),
    }


def effective_portfolio_mode(arm, search, current_query):
    """에이전트 재질의가 아니라 원 질문으로 조건부 arm의 발화 여부를 고정한다."""
    mode = arm.get("portfolio")
    if mode not in {"relax_gated", "evidence", "rrf_evidence", "rrf_axes", "rrf_safe_axes",
                    "rrf_safe_axes_expanded"}:
        return mode
    original = original_question(current_query)
    original_slots, _ = search.router.route(original)
    if mode in {"evidence", "rrf_evidence", "rrf_axes", "rrf_safe_axes",
                "rrf_safe_axes_expanded"}:
        roles = search.ensure_structured().infer_evidence_roles(
            original, original_slots, expanded=mode == "rrf_safe_axes_expanded")
        return mode if roles else None
    identities = list(original_slots.get("contract", [])) + list(original_slots.get("identity", []))
    return mode if len(identities) >= 3 else None


def load_search(elements, tags, structured=False):
    """SlotSearch 를 pickle 캐시로 로드(호출당 프로세스 기동 비용 절감). 캐시는 자기 out/ 에 분리."""
    from clm_search import SlotSearch
    suffix = "_bm25f" if structured else ""
    deps = [Path(elements), Path(tags), FS / "clm_search.py"]
    if structured:
        deps += [FS / "structured_search.py", FS / "schema_adapter.py"]
    fingerprint, entries = dependency_fingerprint(deps)
    key = HERE / "out" / f".cache_{Path(elements).stem}_{Path(tags).stem}{suffix}_{fingerprint[:16]}.pkl"
    key.parent.mkdir(exist_ok=True)
    if key.exists():
        search = pickle.load(open(key, "rb"))
        search._cache_provenance = {"fingerprint": fingerprint, "dependencies": entries,
                                    "cache_hit": True, "path": str(key)}
        return search
    S = SlotSearch(elements, tags)
    if structured:
        S.ensure_structured()
    S._cache_provenance = {"fingerprint": fingerprint, "dependencies": entries,
                           "cache_hit": False, "path": str(key)}
    pickle.dump(S, open(key, "wb"))
    return S


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("--q", default=""); s.add_argument("--page", type=int, default=1)
    s.add_argument("--scope", default="", help='계층 경로 "<특약>[/<관>[/<조>]]" — 매치 가산 부스트. --q 없이 주면 browse')
    for f in ("contract", "role", "subject", "qualifier", "schema"):
        s.add_argument(f"--{f}", default="", help="쉼표 구분, 선택")
    for f in GENERIC_AXES:
        s.add_argument(f"--{f}", default="", help="범용 Semantic Tag 축(쉼표 구분, BM25F arm)")
    vs = sub.add_parser("msearch"); vs.add_argument("--q", required=True); vs.add_argument("--page", type=int, default=1)
    vs.add_argument("--strategy", default="hybrid", choices=("hybrid", "bm25", "dense"))
    r = sub.add_parser("read"); r.add_argument("--id", required=True)
    m = sub.add_parser("submit"); m.add_argument("--ids", required=True, help="쉼표 구분 element_id 순위(최대 10)")
    a = ap.parse_args()

    sess = Path(os.environ["SEMTAG_SESSION"]); sess.mkdir(parents=True, exist_ok=True)
    qid = os.environ.get("SEMTAG_QID", "")
    arm = json.loads(os.environ.get("SEMTAG_ARM", "{}"))
    log_path = sess / "calls.jsonl"
    calls = [json.loads(l) for l in open(log_path)] if log_path.exists() else []
    n_search = sum(1 for c in calls if c["cmd"] in ("search", "msearch")); n_read = sum(1 for c in calls if c["cmd"] == "read")

    def log(rec):
        rec.update({"t": time.time(), "cmd": a.cmd})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def out(obj):
        print(json.dumps(obj, ensure_ascii=False))

    S = load_search(str(FS / "out" / arm.get("elements", "elements_u2.jsonl")),
                    str(FS / "out" / arm.get("tags", "tags_u2_rules.jsonl")),
                    structured=arm.get("ranker") == "bm25f")
    S.router.strict_explicit_bracket = bool(arm.get("strict_explicit_bracket", False))
    if not hasattr(S, "_jo"):
        J = [json.loads(l) for l in open(FS / "out" / arm.get("jo", "elements_u2jo.jsonl"), encoding="utf-8")]
        S._jo = J; S._m2j = {mm: j for j, u in enumerate(J) for mm in u["members"]}
        S._eidx = {e["element_id"]: i for i, e in enumerate(S.E)}
    T = None
    if arm.get("expose_tags"):
        T = {json.loads(l)["element_id"]: json.loads(l) for l in open(FS / "out" / arm.get("tags", "tags_u2_rules.jsonl"), encoding="utf-8")}

    def meta_items(q, top_k):
        """msearch 내부 — msearch 커맨드와 폴백이 공유."""
        import bisect
        jo_starts = [unit["char_start"] for unit in S._jo]

        def jo_of_span(c0, c1):
            index = bisect.bisect_right(jo_starts, c0) - 1
            best, best_overlap = None, 0
            for candidate in S._jo[max(0, index - 2):min(len(S._jo), index + 6)]:
                overlap = min(c1, candidate["char_end"]) - max(c0, candidate["char_start"])
                if overlap > best_overlap:
                    best, best_overlap = candidate, overlap
            return best
        meta_python = os.environ.get("SEMTAG_META_PYTHON", "").strip()
        # Do not resolve the venv executable symlink: resolving it collapses the
        # dependency-bearing venv Python into the system interpreter path.
        if (meta_python and
                Path(meta_python).expanduser().absolute() != Path(sys.executable).absolute()):
            env = dict(os.environ)
            env.setdefault("HF_HUB_OFFLINE", "1")
            env.setdefault("TRANSFORMERS_OFFLINE", "1")
            proc = subprocess.run(
                [meta_python, str(VS / "hybrid_search.py"), "--query", q,
                 "--strategy", "hybrid", "--top-k", str(top_k),
                 "--view", arm.get("meta_view", "V9")],
                capture_output=True, text=True, timeout=180, env=env, check=True)
            res = json.loads(proc.stdout)["results"]
        else:
            sys.path.insert(0, str(VS))
            from hybrid_search import ChunkHybridSearch
            hs = ChunkHybridSearch(view=arm.get("meta_view", "V9"))
            res = hs.search(q, strategy="hybrid", top_k=top_k)
        items = []
        for r in res:
            j = jo_of_span(r["char_start"], r["char_end"]) if r.get("char_start") is not None else None
            items.append({"id": r["id"], "jo": j["element_id"] if j else "", "contract": (j or {}).get("contract_scope", "")[:40],
                          "preview": " ".join((r.get("preview") or "").split())[:PREVIEW]})
        return items

    if a.cmd == "search":
        if n_search >= SEARCH_CAP:
            out({"error": f"search 예산 초과({SEARCH_CAP}회). submit 하십시오."}); return
        # browse 모드: --q 없이 --scope 만 (scope_boost arm 한정)
        if not a.q:
            if not arm.get("scope_boost"):
                out({"error": "--q 가 필요합니다."}); return
            b = enhance.browse(S, a.scope)
            log({"browse": a.scope, "level": b.get("level"), "n_items": len(b.get("items", []))})
            out({"browse": True, "scope": a.scope, "search_calls_left": SEARCH_CAP - n_search - 1, **b}); return
        # 라우터: 동결 qtags + 규칙(--q) + 에이전트 지정 슬롯
        slots, toks = S.router.route(a.q)
        conf = slots.pop("_conf", {})
        router = arm.get("router", "union")
        if router in ("llm", "union") and arm.get("qtags"):
            for l in open(FS / arm["qtags"], encoding="utf-8"):
                d = json.loads(l)
                if d["qid"] == qid:
                    llm = {k: d.get(k) or [] for k in ("contract", "role", "subject", "qualifier", "schema")}
                    if router == "llm":
                        slots = {k: v for k, v in llm.items() if v}; conf = {}
                    else:
                        for k, v in llm.items():
                            if v:
                                slots[k] = list(dict.fromkeys(list(slots.get(k, [])) + v))
                        if llm.get("contract"):
                            conf.pop("contract", None)
                    break
        for f in ("contract", "role", "subject", "qualifier", "schema"):
            v = [x.strip() for x in getattr(a, f).split(",") if x.strip()]
            if v:
                slots[f] = list(dict.fromkeys(list(slots.get(f, [])) + v)); conf.pop(f, None)
        for f in GENERIC_AXES:
            v = [x.strip() for x in getattr(a, f).split(",") if x.strip()]
            if v:
                slots[f] = list(dict.fromkeys(list(slots.get(f, [])) + v))
        identity_expand_info, reference_follow_info = {}, {}
        follow_ids, expand_contracts = set(), set()
        if arm.get("identity_expand"):
            config = arm["identity_expand"]
            added, identity_expand_info = identity_expansion_candidates(
                S.router, a.q, slots, conf,
                config if isinstance(config, dict) else {})
            if added:
                slots["contract"] = list(dict.fromkeys(list(slots.get("contract", [])) + added))
        alias_log = {}
        if arm.get("alias"):
            extra, alias_log = enhance.expand_query(a.q, toks)
            toks = toks + extra
        w = {f: v * conf.get(f, 1.0) for f, v in (arm.get("w") or {}).items()}
        M, L = S.match_table(slots, toks)
        if arm.get("ranker") == "bm25f":
            portfolio_mode = effective_portfolio_mode(arm, S, a.q)
            baseline_res = None
            if (portfolio_mode or arm.get("tag_ensemble")) and arm.get("preserve_jo_top"):
                baseline_weights = (arm.get("baseline_sfw") if arm.get("tag_ensemble")
                                    else arm.get("sfw"))
                baseline_res = S.rank_structured(
                    slots, toks, a.q, weights=baseline_weights,
                    profile=arm.get("profile", "core"), limit=400, lexical_counts=L)
            if portfolio_mode in {"evidence", "rrf_evidence", "rrf_axes", "rrf_safe_axes",
                                  "rrf_safe_axes_expanded"} and n_search > 0:
                portfolio_mode = None
            if portfolio_mode:
                res = S.rank_structured_portfolio(
                    slots, toks, a.q, weights=arm.get("sfw"), profile=arm.get("profile", "core"),
                    mode=portfolio_mode, limit=400, window=int(arm.get("portfolio_window", 40)),
                    seed_quota=int(arm.get("seed_quota", 20)), lexical_counts=L,
                    process_query=(original_question(a.q)
                                   if portfolio_mode in {"evidence", "rrf_evidence", "rrf_axes",
                                                         "rrf_safe_axes", "rrf_safe_axes_expanded"}
                                   else None),
                    rrf_k=int(arm.get("rrf_k", 60)),
                    coverage_seed=int(arm.get("coverage_seed", 3)),
                    coverage_per_role=int(arm.get("coverage_per_role", 1)))
            else:
                res = S.rank_structured(slots, toks, a.q, weights=arm.get("sfw"),
                                        profile=arm.get("profile", "full"), limit=400,
                                        lexical_counts=L)
            if arm.get("tag_ensemble"):
                clm = S.rank(
                    M, L, mode="clm", lex=arm.get("ensemble_lex", "count"),
                    weights=arm.get("ensemble_clm_w") or {}, limit=400,
                    rare=bool(arm.get("ensemble_rare")), n_tokens=len(toks))
                if arm.get("tag_ensemble") == "quota":
                    member_to_jo = {member: S._jo[index]["element_id"]
                                    for member, index in S._m2j.items()}
                    rankings = [clm, res]
                    original = original_question(a.q)
                    support_plan = (membership_support_plan(S, original)
                                    if int(arm.get("ensemble_membership_support_quota", 0)) > 0
                                    else None)
                    unit_enabled = evidence_unit_gate(original, arm.get("ensemble_unit_gate"))
                    if arm.get("ensemble_unit_require_support") and not support_plan:
                        unit_enabled = False
                    support_active = bool(support_plan and unit_enabled)
                    quotas = [int(arm.get("ensemble_support_clm_quota", 15)
                                  if support_active else arm.get("ensemble_clm_quota", 17)),
                              int(arm.get("ensemble_support_fact_quota", 7)
                                  if support_active else arm.get("ensemble_fact_quota", 8))]
                    if int(arm.get("ensemble_unit_quota", 0)) > 0 and unit_enabled:
                        unit_ranked = S.rank_structured(
                            slots, toks, a.q,
                            weights=arm.get("ensemble_unit_sfw") or {},
                            profile=arm.get("profile", "core"), limit=400,
                            lexical_counts=[0] * len(S.E))
                        rankings.append(unit_ranked)
                        quotas.append(int(arm["ensemble_unit_quota"]))
                    if support_active:
                        support_ranked = S.rank_structured(
                            support_plan["slots"], toks, support_plan["query"],
                            weights=arm.get("ensemble_membership_support_sfw") or {},
                            profile=arm.get("profile", "core"), limit=400,
                            lexical_counts=[0] * len(S.E))
                        rankings.append(support_ranked)
                        quotas.append(int(arm["ensemble_membership_support_quota"]))
                    if arm.get("reference_follow"):
                        follow = arm["reference_follow"]
                        follow_enabled = reference_follow_enabled(
                            original, follow.get("gate", "table_code_or_document"))
                        follow_ranked, follow_audit = [], {}
                        if follow_enabled:
                            follow_stats = json.load(open(
                                FS / "out" / follow["stats"], encoding="utf-8"))
                            follow_ranked, follow_audit = reference_follow_ranking(
                                S, slots, follow_stats,
                                limit=int(follow.get("limit", 5)),
                                max_hops=int(follow.get("max_hops", 2)))
                        if follow_ranked:
                            follow_ids = {e["element_id"] for e, _ in follow_ranked}
                            rankings.append(follow_ranked)
                            quotas.append(int(follow.get("quota", 5)))
                        else:
                            follow_ids = set()
                        reference_follow_info = {
                            "mode": "tail_quota_after_existing_rankings",
                            "enabled": follow_enabled,
                            "fired": bool(follow_ranked),
                            "quota": int(follow.get("quota", 5)),
                            **follow_audit,
                        }
                    res = unique_jo_quota(
                        rankings, member_to_jo, quotas, limit=400)
                else:
                    res = reciprocal_rank_fuse(
                        [res, clm] + ([baseline_res] if baseline_res is not None else []),
                        limit=400, rrf_k=int(arm.get("ensemble_rrf_k", 60)))
            if baseline_res is not None:
                res = preserve_baseline_jo_prefix(
                    baseline_res, res, {member: S._jo[index]["element_id"]
                                        for member, index in S._m2j.items()},
                    int(arm["preserve_jo_top"]))[:400]
        else:
            res = S.rank(M, L, mode=arm.get("mode", "clm"), lex=arm.get("lex", "count"), weights=w, limit=400, rare=bool(arm.get("rare")), n_tokens=len(toks))
        if a.scope and arm.get("scope_boost"):
            res = enhance.apply_scope_boost(res, S, S._eidx, a.scope, float(arm["scope_boost"]))
        page = res[PAGE * (a.page - 1): PAGE * a.page]
        import re as _re
        def _split_c(sc_):
            m = _re.search(r"\(무배당[^)]*\)", sc_)
            return _re.sub(r"\(무배당[^)]*\)", "", sc_).strip(), (m.group(0)[1:-1].replace("무배당", "").strip(", ") if m else "")
        def _snip(text, tk):
            flat = " ".join(text.split())
            for t in tk:
                p = flat.find(t)
                if p >= 0:
                    st = max(0, p - 40)
                    return ("…" if st else "") + flat[st: st + PREVIEW]
            return flat[:PREVIEW]
        items = []
        identity_audit = (getattr(getattr(S, "_structured", None),
                                  "last_exact_identity", {})
                          if arm.get("audit_payload") else {})
        coverage_audit = (getattr(getattr(S, "_structured", None),
                                  "last_locator_coverage", {})
                          if arm.get("audit_payload") else {})
        role_audit = (getattr(getattr(S, "_structured", None),
                              "last_evidence_role", {})
                      if arm.get("audit_payload") else {})
        portfolio = getattr(S, "_last_portfolio", {})
        phase_by_index = {t["index"]: t for t in portfolio.get("trace", [])}
        for display_rank, (e, sc) in enumerate(page, start=1 + PAGE * (a.page - 1)):
            j = S._jo[S._m2j[e["element_id"]]]
            bc, var = _split_c(e["contract_scope"])
            visible_score = (round(1.0 / display_rank, 8)
                             if arm.get("score_view") == "rank" else sc)
            it = {"id": e["element_id"], "jo": j["element_id"], "score": visible_score,
                  "contract": bc, "variant": var, "jo_title": (j.get("title") or "")[:40],
                  "preview": _snip(e["text"], toks)}
            phase = phase_by_index.get(S._eidx[e["element_id"]])
            if phase and arm.get("portfolio_prompt"):
                it["phase"] = phase["phase"]
            if T:
                t = T[e["element_id"]]; loc = t.get("locator") or {}
                it["tag"] = f"[특약]{t.get('contract_key','')[:30]} [조]{loc.get('article','')} {loc.get('article_title','')[:30]} [역할]{'/'.join(t.get('role') or [])} [유형]{t.get('schema_tag','')}"
            if e["element_id"] in follow_ids:
                it["note"] = "참조표 후보: 질문 특약의 조가 인용하는 표/조 원문"
            items.append(it)
        fb = arm.get("fallback") if arm.get("meta") else None
        fb_used = ""
        fb_status = ""
        fb_error_detail = ""
        gate_info = {}
        if fb and a.page == 1:
            if fb.get("gate") == "axis_coverage":
                gate_info = S.hybrid_fallback_gate(
                    slots, res, top_k=int(fb.get("top_k", 5)),
                    min_axis_coverage=float(fb.get("min_axis_coverage", 0.5)))
                should_fallback = gate_info["fallback"]
                if fb.get("force_first") and n_search == 0:
                    should_fallback = True
                    gate_info["fallback"] = True
                    gate_info["forced"] = "first_search_probe"
                    gate_info["reasons"] = list(dict.fromkeys(
                        list(gate_info.get("reasons", [])) + ["first_search_probe"]))
                if fb.get("first_only", True) and n_search > 0:
                    should_fallback = False
                    gate_info["suppressed"] = "not_first_search"
            else:
                min_n = fb.get("min_n", 5)
                should_fallback = not res or len(res) < min_n
            if should_fallback:
                try:
                    mitems = meta_items(a.q, PAGE)
                except Exception as exc:  # 폴백 실패가 태그 결과까지 죽이면 안 된다
                    mitems, fb_used = [], f"error:{type(exc).__name__}"
                    fb_status = fb_used
                    fb_error_detail = str(exc)
                    if isinstance(exc, subprocess.CalledProcessError):
                        fb_error_detail = " ".join(
                            (fb_error_detail + " " + (exc.stderr or "")[-1000:]).split())
                if mitems:
                    fb_status = "ok:no_new_items"
                    for it in mitems:
                        it["src"] = "meta"
                    if not res:
                        items, fb_used = mitems, "replace"
                        fb_status = "ok:replace"
                    else:
                        # 하위 슬롯 치환: 태그 상위는 보존, 페이지 하위 K칸을 meta 로 확보
                        K = fb.get("merge_k", 10)
                        have = {it["id"] for it in items}
                        add = [x for x in mitems if x["id"] not in have][:K]
                        if add:
                            items = items[: max(0, PAGE - len(add))] + add
                            fb_used = f"merge:{len(add)}"
                            fb_status = f"ok:{fb_used}"
                # 실제 병합 0건이면 라벨을 남기지 않는다(오신호 방지) — fb_used 는 위에서만 설정
        fact_card_overlay_info = {}
        overlay = arm.get("fact_card_overlay")
        if overlay and a.page == 1:
            overlay_search = load_search(
                str(FS / "out" / overlay["elements"]),
                str(FS / "out" / overlay["tags"]), structured=True)
            _, overlay_lexical = overlay_search.match_table(slots, toks)
            overlay_ranked = overlay_search.rank_structured(
                slots, toks, a.q, weights=overlay.get("sfw") or {},
                profile=overlay.get("profile", "core"), limit=400,
                lexical_counts=overlay_lexical)
            existing_jo = {item.get("jo") for item in items if item.get("jo")}
            overlay_items, seen_overlay_jo = [], set()
            rejected_surface = 0
            support_audit = []
            for card, card_score in overlay_ranked:
                jo_id = card.get("parent_jo")
                if not jo_id or jo_id in existing_jo or jo_id in seen_overlay_jo:
                    continue
                support = fact_card_query_support(
                    toks, card.get("text", ""),
                    min_matches=int(overlay.get("min_token_matches", 2)),
                    strong_token_chars=int(overlay.get("strong_token_chars", 5)))
                if overlay.get("surface_gate") and not support["allowed"]:
                    rejected_surface += 1
                    continue
                seen_overlay_jo.add(jo_id)
                base_contract, card_variant = _split_c(card.get("contract_scope", ""))
                overlay_items.append({
                    # j-ID로 노출하면 read/submit/scoring은 기존 고정 JO corpus를
                    # 그대로 사용하고, 검색 preview만 source-derived card를 쓴다.
                    "id": jo_id, "jo": jo_id,
                    "score": (round(1.0 / (len(items) + len(overlay_items) + 1), 8)
                              if arm.get("score_view") == "rank" else card_score),
                    "contract": base_contract, "variant": card_variant,
                    "jo_title": card.get("title", "")[:40],
                    "preview": _snip(card.get("text", ""), toks),
                    "src": "semantic_fact_card",
                })
                support_audit.append({"jo": jo_id, **support})
                if len(overlay_items) >= int(overlay.get("limit", 5)):
                    break
            items.extend(overlay_items)
            fact_card_overlay_info = {
                "mode": "append_after_complete_hybrid_page",
                "added": len(overlay_items),
                "jo_ids": [item["jo"] for item in overlay_items],
                "base_page_preserved": len(items) - len(overlay_items),
                "surface_rejected": rejected_surface,
                "surface_support": support_audit,
                "cache": {key: overlay_search._cache_provenance[key]
                          for key in ("fingerprint", "cache_hit", "path")},
            }
        reference_bundle_info = {}
        reference_bundle = arm.get("reference_bundle")
        if reference_bundle and a.page == 1:
            original = original_question(a.q)
            enabled = reference_bundle_enabled(
                original, reference_bundle.get("gate", "membership_or_code"))
            linked = []
            if enabled:
                stats_path = FS / "out" / reference_bundle["stats"]
                stats = json.load(open(stats_path, encoding="utf-8"))
                jo_index = {row["element_id"]: row for row in S._jo}
                linked = reference_bundles_for_items(
                    items, stats.get("edge_ledger", []), jo_index, toks, _snip,
                    max_source_rank=int(reference_bundle.get("max_source_rank", 12)),
                    max_targets_per_source=int(reference_bundle.get("max_targets_per_source", 2)),
                    max_hops=int(reference_bundle.get("max_hops", 2)))
            reference_bundle_info = {
                "mode": "nested_under_unchanged_hybrid_result",
                "enabled": enabled,
                "linked_source_count": len(linked),
                "linked_target_count": sum(len(row["targets"]) for row in linked),
                "links": linked,
                "base_page_size": len(items),
            }
        reference_projection_info = {}
        reference_projection = arm.get("reference_projection")
        if reference_projection and a.page == 1:
            original = original_question(a.q)
            enabled = reference_projection_enabled(
                original, reference_projection.get("gate", "classification_membership"))
            projected = []
            if enabled:
                stats_path = FS / "out" / reference_projection["stats"]
                stats = json.load(open(stats_path, encoding="utf-8"))
                jo_index = {row["element_id"]: row for row in S._jo}
                selected = reference_projection_for_items(
                    items, stats.get("edge_ledger", []), jo_index, toks, raw_query=original,
                    max_source_rank=int(reference_projection.get("max_source_rank", 12)),
                    limit=int(reference_projection.get("limit", 2)),
                    max_hops=int(reference_projection.get("max_hops", 2)),
                    min_token_matches=int(reference_projection.get("min_token_matches", 2)),
                    strong_token_chars=int(reference_projection.get("strong_token_chars", 5)))
                for rank, selected_item in enumerate(selected, start=1):
                    target = jo_index[selected_item["jo"]]
                    base_contract, variant = _split_c(target.get("contract_scope", ""))
                    projected.append({
                        "id": selected_item["jo"], "jo": selected_item["jo"],
                        "score": round(1.0 / rank, 8), "contract": base_contract,
                        "variant": variant, "jo_title": (target.get("title") or "")[:40],
                        "preview": _snip(target.get("text", ""), toks),
                        "src": "semantic_reference_projection",
                        "source_ranks": selected_item["source_ranks"],
                        "hops": selected_item["hops"],
                        "via": [path["keys"] for path in selected_item["paths"]],
                        "matched_query_surfaces": selected_item["support"]["matched"],
                        **({"explicit_anchor": selected_item["explicit_anchor"]}
                           if selected_item.get("explicit_anchor") else {}),
                    })
            reference_projection_info = {
                "mode": "deduplicated_query_supported_reference_targets",
                "enabled": enabled, "items": projected,
                "base_page_size": len(items), "base_page_unchanged": True,
            }
        reference_evidence_info = {}
        reference_evidence = arm.get("reference_evidence")
        if reference_evidence and a.page == 1:
            original = original_question(a.q)
            enabled = reference_evidence_enabled(
                original, reference_evidence.get("gate", "reference_or_scenario"))
            plan = scenario_intent_query(original) if enabled else None
            normalized_tokens = []
            if plan:
                _, normalized_tokens = S.router.route(plan["query"])
            linked_items = []
            catalog_items = []
            if enabled:
                stats_path = FS / "out" / reference_evidence["stats"]
                stats = json.load(open(stats_path, encoding="utf-8"))
                linked_items = reference_evidence_for_items(
                    items, stats.get("edge_ledger", []),
                    {row["element_id"]: row for row in S._jo},
                    {row["element_id"]: row for row in S.E}, toks,
                    raw_query=original, normalized_tokens=normalized_tokens,
                    max_source_rank=int(reference_evidence.get("max_source_rank", 40)),
                    limit=int(reference_evidence.get("limit", 2)),
                    max_hops=int(reference_evidence.get("max_hops", 2)),
                    min_token_matches=int(reference_evidence.get("min_token_matches", 2)),
                    strong_token_chars=int(reference_evidence.get("strong_token_chars", 5)),
                    region_members=int(reference_evidence.get("region_members", 3)),
                    stop_on_mixed_labels=bool(
                        reference_evidence.get("stop_on_mixed_labels", False)),
                    sanitize_region_title=bool(
                        reference_evidence.get("sanitize_region_title", False)))
                if reference_evidence.get("catalog_title_lookup"):
                    jo_documents = stats.get("jo_documents", {})
                    allowed_documents = [jo_documents.get(item.get("jo", ""), "")
                                         for item in items]
                    catalog_items = table_catalog_evidence(
                        original, stats.get("table_catalog", []), allowed_documents,
                        max_variants=int(
                            reference_evidence.get("catalog_max_variants", 5)))
                    catalog_jos = {item["jo"] for item in catalog_items}
                    linked_items = catalog_items + [item for item in linked_items
                                                     if item.get("jo") not in catalog_jos]
            reference_evidence_info = {
                "mode": "exact_reference_region_without_base_rerank",
                "enabled": enabled, "normalized_intent": plan or {},
                "items": linked_items, "base_page_size": len(items),
                "base_page_unchanged": True,
                **({"catalog_title_lookup": True,
                    "catalog_item_count": len(catalog_items)}
                   if catalog_items else {}),
            }
        intent_bundle_info = {}
        intent_bundle = arm.get("intent_bundle")
        if intent_bundle and a.page == 1:
            plan = scenario_intent_query(original_question(a.q))
            intent_items = []
            if plan:
                intent_slots, intent_tokens = S.router.route(plan["query"])
                intent_slots.pop("_conf", None)
                _, intent_lexical = S.match_table(intent_slots, intent_tokens)
                intent_ranked = S.rank_structured(
                    intent_slots, intent_tokens, plan["query"],
                    weights=intent_bundle.get("sfw") or arm.get("sfw") or {},
                    profile=intent_bundle.get("profile", arm.get("profile", "core")),
                    limit=100, lexical_counts=intent_lexical)
                existing_jo = {item.get("jo") for item in items if item.get("jo")}
                seen_jo = set()
                for element, score in intent_ranked:
                    jo = S._jo[S._m2j[element["element_id"]]]
                    jo_id = jo["element_id"]
                    if jo_id in existing_jo or jo_id in seen_jo:
                        continue
                    seen_jo.add(jo_id)
                    base_contract, variant = _split_c(element.get("contract_scope", ""))
                    intent_items.append({
                        "id": jo_id, "jo": jo_id, "score": score,
                        "contract": base_contract, "variant": variant,
                        "jo_title": (jo.get("title") or "")[:40],
                        "preview": _snip(element.get("text", ""), intent_tokens),
                        "src": "normalized_intent",
                    })
                    if len(intent_items) >= int(intent_bundle.get("limit", 1)):
                        break
            intent_bundle_info = {
                "mode": "separate_normalized_intent_evidence",
                "plan": plan or {},
                "items": intent_items,
                "base_page_size": len(items),
                "base_page_unchanged": True,
            }
            if (plan and intent_items and
                    intent_bundle.get("mode") == "prepend_dedup"):
                intent_jo = {row["jo"] for row in intent_items}
                items = (intent_items + [row for row in items
                                         if row.get("jo") not in intent_jo])[:PAGE]
                intent_bundle_info["mode"] = "prepend_normalized_intent_then_base"
                intent_bundle_info["base_page_unchanged"] = False
                intent_bundle_info["result_page_size"] = len(items)
        claim_bundle_info = {}
        claim_bundle = arm.get("claim_bundle")
        if claim_bundle and a.page == 1:
            original = original_question(a.q)
            original_slots, _ = S.router.route(original)
            original_slots.pop("_conf", None)
            allowed_contracts = explicit_contract_scopes(original, original_slots)
            roles = requested_claim_roles(
                original, original_slots, limit=int(claim_bundle.get("max_roles", 3)))
            if claim_bundle.get("explicit_identity_only") and not allowed_contracts:
                roles = []
            groups = []
            if roles:
                raw_groups = role_bundle_items(
                    S, original, original_slots, roles,
                    claim_bundle.get("sfw") or arm.get("sfw") or {},
                    claim_bundle.get("profile", arm.get("profile", "core")),
                    per_role=int(claim_bundle.get("per_role", 2)),
                    allowed_contracts=(set(allowed_contracts)
                                       if claim_bundle.get("explicit_identity_only") else None))
                for group in raw_groups:
                    candidates = []
                    for element, jo, score, role_tokens in group["candidates"]:
                        base_contract, variant = _split_c(element.get("contract_scope", ""))
                        candidates.append({
                            "id": jo["element_id"], "jo": jo["element_id"],
                            "score": score, "contract": base_contract,
                            "variant": variant,
                            "jo_title": (jo.get("title") or "")[:40],
                            "preview": _snip(element.get("text", ""), role_tokens),
                        })
                    groups.append({"role": group["role"], "label": group["label"],
                                   "items": candidates})
            promoted = []
            if roles and claim_bundle.get("inject_after") is not None:
                items, promoted = inject_claim_role_items(
                    items, groups, after=int(claim_bundle["inject_after"]),
                    limit=int(claim_bundle.get("max_injected", len(groups))))
            claim_bundle_info = {
                "mode": ("explicit_identity_role_slots_after_protected_prefix"
                         if claim_bundle.get("inject_after") is not None else
                         "role_decomposed_evidence_without_base_rerank"),
                "roles": roles,
                "allowed_contracts": allowed_contracts,
                "groups": groups,
                "promoted": [item.get("jo") for item in promoted],
                "base_page_size": len(items),
                "base_page_unchanged": not bool(promoted),
            }
        compact_info = {}
        if arm.get("compact_jo"):
            before = len(items)
            items = compact_items_by_jo(items)
            compact_info = {"before": before, "after": len(items), "refill": 0}
        elif arm.get("group_jo"):
            before = len(items)
            items = group_items_by_jo(items, int(arm.get("group_max_evidence", 0)))
            compact_info = {"before": before, "after": len(items), "refill": 0,
                            "mode": "grouped_evidence"}
        facets = {}
        if arm.get("facet"):
            # 상위 200 후보의 특약·조 분포 — 에이전트가 범위를 좁혀 재검색할 수 있게 하는 참고 정보(필터 아님)
            import collections as _c
            fc = _c.Counter(e["contract_scope"][:40] for e, _ in res[:200])
            fj = _c.Counter((S._jo[S._m2j[e["element_id"]]]["title"] or S._jo[S._m2j[e["element_id"]]]["element_id"])[:30] for e, _ in res[:200])
            facets = {"contract_top": fc.most_common(8), "article_top": fj.most_common(8)}
        returned = [(it["id"], it.get("score", 0)) for it in items]
        if arm.get("group_jo"):
            returned = [(variant["id"], variant.get("score", 0))
                        for it in items for variant in it.get("evidence_variants", [])]
            returned.sort(key=lambda pair: next(
                variant["raw_rank"] for it in items for variant in it.get("evidence_variants", [])
                if variant["id"] == pair[0]))
        log({"q": a.q, "slots": slots, "n_tokens": len(toks), "total": len(res), "page": a.page,
             **({"cache": {k: S._cache_provenance[k] for k in ("fingerprint", "cache_hit", "path")}}
                if hasattr(S, "_cache_provenance") else {}),
             **({"scope": a.scope} if a.scope else {}), **({"alias_expanded": alias_log} if alias_log else {}),
             **({"hybrid_gate": gate_info} if gate_info else {}),
             **({"fallback": fb_used} if fb_used else {}),
             **({"fallback_status": fb_status} if fb_status else {}),
             **({"fallback_error": fb_error_detail} if fb_error_detail else {}),
             **({"fact_card_overlay": fact_card_overlay_info} if fact_card_overlay_info else {}),
             **({"reference_bundle": reference_bundle_info} if reference_bundle_info else {}),
             **({"reference_projection": reference_projection_info}
                if reference_projection_info else {}),
             **({"reference_evidence": reference_evidence_info}
                if reference_evidence_info else {}),
             **({"intent_bundle": intent_bundle_info} if intent_bundle_info else {}),
             **({"claim_bundle": claim_bundle_info} if claim_bundle_info else {}),
             **({"compact_jo": compact_info} if compact_info else {}),
             **({"portfolio": {"specs": portfolio.get("specs", []),
                                "phases": [it.get("phase", "") for it in items]}} if portfolio else {}),
             **({"explicit_identity": identity_audit} if identity_audit else {}),
             **({"identity_expand": identity_expand_info} if identity_expand_info else {}),
             **({"reference_follow": reference_follow_info} if reference_follow_info else {}),
             **({"locator_coverage": coverage_audit} if coverage_audit else {}),
             **({"evidence_role": role_audit} if role_audit else {}),
             **({"display_results": items} if arm.get("audit_payload") else {}),
             "returned": returned})
        out({"query": a.q, "slots_used": slots, "total_candidates": len(res), "page": a.page, "page_size": len(items),
             **({"raw_page_size": len(returned)} if arm.get("group_jo") else {}),
             "search_calls_left": SEARCH_CAP - n_search - 1,
             **({"channel": "tag+fallback:meta"} if fb_used else {}),
             **({"hybrid_gate": gate_info} if gate_info else {}),
             **({"fallback_status": fb_status} if fb_status else {}),
             **({"fallback_error": fb_error_detail} if fb_error_detail else {}),
             **({"fact_card_overlay": fact_card_overlay_info} if fact_card_overlay_info else {}),
             **({"reference_bundle": reference_bundle_info} if reference_bundle_info else {}),
             **({"reference_projection": reference_projection_info}
                if reference_projection_info else {}),
             **({"reference_evidence": reference_evidence_info}
                if reference_evidence_info else {}),
             **({"intent_bundle": intent_bundle_info} if intent_bundle_info else {}),
             **({"claim_bundle": claim_bundle_info} if claim_bundle_info else {}),
             **({"compact_jo": compact_info} if compact_info else {}),
             **({"search_process": portfolio.get("specs", [])}
                if portfolio and arm.get("portfolio_prompt") else {}),
             **({"explicit_identity": identity_audit} if identity_audit else {}),
             **({"identity_expand": identity_expand_info} if identity_expand_info else {}),
             **({"reference_follow": reference_follow_info} if reference_follow_info else {}),
             **({"locator_coverage": coverage_audit} if coverage_audit else {}),
             **({"evidence_role": role_audit} if role_audit else {}),
             **({"facets": facets} if facets else {}), "results": items})
    elif a.cmd == "msearch":
        # 메타데이터(청크) 검색 — vector_search.ChunkHybridSearch(BM25+Dense RRF). 이 채널은 고정이며 우리 변인이 아니다.
        if n_search >= SEARCH_CAP:
            out({"error": f"search 예산 초과({SEARCH_CAP}회). submit 하십시오."}); return
        if arm.get("search_only"):
            out({"error": "이 실험 arm은 변인 통제를 위해 직접 msearch를 금지합니다. search를 사용하십시오."}); return
        if not arm.get("meta"):
            out({"error": "이 arm 에서는 msearch 를 사용할 수 없습니다."}); return
        all_items = meta_items(a.q, PAGE * a.page)
        page_items = all_items[PAGE * (a.page - 1): PAGE * a.page]
        log({"q": a.q, "strategy": a.strategy, "page": a.page, "returned": [(r["id"], 0) for r in page_items]})
        out({"query": a.q, "channel": f"meta:{a.strategy}", "page": a.page, "page_size": PAGE, "search_calls_left": SEARCH_CAP - n_search - 1, "results": page_items})
    elif a.cmd == "read":
        if n_read >= READ_CAP:
            out({"error": f"read 예산 초과({READ_CAP}회)."}); return
        eid = a.id.strip()
        chunk = None
        if eid.startswith("c"):
            from units import Units
            unit_index = Units(jo_path=FS / "out" / arm.get("jo", "elements_u2jo.jsonl"),
                               chunks_path=VS / "out/chunks.jsonl")
            chunk = unit_index.C.get(eid)
            if not chunk:
                out({"error": "unknown id"}); return
            j = unit_index.jo_of_span(chunk["char_start"], chunk["char_end"])
            text = (chunk.get("text") or "")[:4000]
            members = (j or {}).get("members", [])
        elif eid.startswith("j"):
            j = next((u for u in S._jo if u["element_id"] == eid), None)
            text = j["text"][:4000] if j else ""; members = j["members"] if j else []
        else:
            i = S._eidx.get(eid)
            if i is None:
                out({"error": "unknown id"}); return
            j = S._jo[S._m2j[eid]]
            text = j["text"][:4000]; members = j["members"]
        resp = {"id": eid, "jo": j["element_id"] if j else "", "contract": (j or {}).get("contract_scope", ""),
                "members": members[:60], "text": text, "read_calls_left": READ_CAP - n_read - 1}
        if chunk:
            resp["chunk_span"] = [chunk.get("char_start"), chunk.get("char_end")]
        if arm.get("ref_expand") and j:
            refs = enhance.load_refs().get(j["element_id"], [])[:4]
            if refs:
                jix = {u["element_id"]: u for u in S._jo}
                resp["refs"] = [{"jo": rj, "title": jix[rj].get("title", ""),
                                 "preview": " ".join((jix[rj].get("text") or "").split())[:PREVIEW]}
                                for rj in refs if rj in jix]
        log({"id": eid, "chars": len(text), **({"n_refs": len(resp.get("refs", []))} if arm.get("ref_expand") else {})})
        out(resp)
    elif a.cmd == "submit":
        ids = [x.strip() for x in a.ids.split(",") if x.strip()][:SUBMIT_MAX]
        log({"ids": ids})
        json.dump({"qid": qid, "ranked": ids, "n_search": n_search, "n_read": n_read}, open(sess / "submit.json", "w"), ensure_ascii=False)
        out({"ok": True, "submitted": ids, "note": "세션 종료. 더 이상 도구를 호출하지 마십시오."})


if __name__ == "__main__":
    main()

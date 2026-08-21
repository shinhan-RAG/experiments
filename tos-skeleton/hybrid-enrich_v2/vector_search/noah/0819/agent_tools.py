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
        if meta_python and Path(meta_python).resolve() != Path(sys.executable).resolve():
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
            items.append(it)
        fb = arm.get("fallback") if arm.get("meta") else None
        fb_used = ""
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
                if mitems:
                    for it in mitems:
                        it["src"] = "meta"
                    if not res:
                        items, fb_used = mitems, "replace"
                    else:
                        # 하위 슬롯 치환: 태그 상위는 보존, 페이지 하위 K칸을 meta 로 확보
                        K = fb.get("merge_k", 10)
                        have = {it["id"] for it in items}
                        add = [x for x in mitems if x["id"] not in have][:K]
                        if add:
                            items = items[: max(0, PAGE - len(add))] + add
                            fb_used = f"merge:{len(add)}"
                # 실제 병합 0건이면 라벨을 남기지 않는다(오신호 방지) — fb_used 는 위에서만 설정
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
             **({"compact_jo": compact_info} if compact_info else {}),
             **({"portfolio": {"specs": portfolio.get("specs", []),
                                "phases": [it.get("phase", "") for it in items]}} if portfolio else {}),
             **({"explicit_identity": identity_audit} if identity_audit else {}),
             **({"locator_coverage": coverage_audit} if coverage_audit else {}),
             **({"evidence_role": role_audit} if role_audit else {}),
             **({"display_results": items} if arm.get("audit_payload") else {}),
             "returned": returned})
        out({"query": a.q, "slots_used": slots, "total_candidates": len(res), "page": a.page, "page_size": len(items),
             **({"raw_page_size": len(returned)} if arm.get("group_jo") else {}),
             "search_calls_left": SEARCH_CAP - n_search - 1,
             **({"channel": "tag+fallback:meta"} if fb_used else {}),
             **({"hybrid_gate": gate_info} if gate_info else {}),
             **({"compact_jo": compact_info} if compact_info else {}),
             **({"search_process": portfolio.get("specs", [])}
                if portfolio and arm.get("portfolio_prompt") else {}),
             **({"explicit_identity": identity_audit} if identity_audit else {}),
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

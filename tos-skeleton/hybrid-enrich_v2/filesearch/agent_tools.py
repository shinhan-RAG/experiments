#!/usr/bin/env python3
"""에이전트 도구 CLI (search / read / submit) — S5 예산 하네스.

세션 환경변수: SEMTAG_SESSION(세션 디렉터리) · SEMTAG_QID · SEMTAG_ARM(json)
ARM 예: {"lex":"count","w":{"contract":2},"router":"llm|rule|union","qtags":"out/qtags_haiku.jsonl","expose_tags":0}
예산(S5): search ≤20회 · read ≤8회 · page 40 · preview 160자 · submit 최대 10개. 초과 호출은 거부.
모든 호출의 인자와 반환(후보 id·점수)을 세션 로그(calls.jsonl)에 남긴다.
검색기 = clm_search.SlotSearch(CLM). 라우터: 사전 동결 qtags(질문 단위) + 규칙 라우터(에이전트가 준 --q) + 에이전트 슬롯 지정(--contract 등, 덧셈).
"""
import argparse, hashlib, json, os, pickle, subprocess, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
VS = HERE.parent / "vector_search"
sys.path.insert(0, str(HERE))
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


def original_question(current_query):
    """세션 원 질문을 반환한다. 파일이 없는 수동 호출은 현재 질의를 쓴다."""
    question_file = Path(os.environ.get("SEMTAG_SESSION", ".")) / "question.json"
    if question_file.exists():
        return json.load(open(question_file, encoding="utf-8"))["question"]
    return os.environ.get("SEMTAG_QUESTION", current_query)


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
    """SlotSearch 를 pickle 캐시로 로드(호출당 프로세스 기동 비용 절감)."""
    from clm_search import SlotSearch
    suffix = "_bm25f" if structured else ""
    deps = [Path(elements), Path(tags), HERE / "clm_search.py"]
    if structured:
        deps += [HERE / "structured_search.py", HERE / "schema_adapter.py"]
    fingerprint, entries = dependency_fingerprint(deps)
    key = HERE / "out" / f".cache_{Path(elements).stem}_{Path(tags).stem}{suffix}_{fingerprint[:16]}.pkl"
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
    s = sub.add_parser("search"); s.add_argument("--q", required=True); s.add_argument("--page", type=int, default=1)
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

    S = load_search(str(HERE / "out" / arm.get("elements", "elements_u2.jsonl")),
                    str(HERE / "out" / arm.get("tags", "tags_u2_rules.jsonl")),
                    structured=arm.get("ranker") == "bm25f")
    if not hasattr(S, "_jo"):
        J = [json.loads(l) for l in open(HERE / "out" / arm.get("jo", "elements_u2jo.jsonl"), encoding="utf-8")]
        S._jo = J; S._m2j = {mm: j for j, u in enumerate(J) for mm in u["members"]}
        S._eidx = {e["element_id"]: i for i, e in enumerate(S.E)}
    T = None
    if arm.get("expose_tags"):
        T = {json.loads(l)["element_id"]: json.loads(l) for l in open(HERE / "out" / arm.get("tags", "tags_u2_rules.jsonl"), encoding="utf-8")}

    def meta_items(q, top_k):
        """msearch와 자동 폴백이 공유하는 고정 BM25+Dense RRF 채널."""
        from units import Units
        U = Units()
        meta_python = os.environ.get("SEMTAG_META_PYTHON", "").strip()
        # Preserve the venv executable symlink; resolve() would select the
        # dependency-free system interpreter instead.
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
            hits = json.loads(proc.stdout)["results"]
        else:
            sys.path.insert(0, str(VS))
            from hybrid_search import ChunkHybridSearch
            hs = ChunkHybridSearch(view=arm.get("meta_view", "V9"))
            hits = hs.search(q, strategy="hybrid", top_k=top_k)
        items = []
        for hit in hits:
            j = U.jo_of_span(hit["char_start"], hit["char_end"]) if hit.get("char_start") is not None else None
            items.append({"id": hit["id"], "jo": j["element_id"] if j else "",
                          "contract": (j or {}).get("contract_scope", "")[:40],
                          "preview": " ".join((hit.get("preview") or "").split())[:PREVIEW]})
        return items

    if a.cmd == "search":
        if n_search >= SEARCH_CAP:
            out({"error": f"search 예산 초과({SEARCH_CAP}회). submit 하십시오."}); return
        # 라우터: 동결 qtags + 규칙(--q) + 에이전트 지정 슬롯
        slots, toks = S.router.route(a.q)
        conf = slots.pop("_conf", {})
        router = arm.get("router", "union")
        if router in ("llm", "union") and arm.get("qtags"):
            for l in open(HERE / arm["qtags"], encoding="utf-8"):
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
        w = {f: v * conf.get(f, 1.0) for f, v in (arm.get("w") or {}).items()}
        S.variant_rr = bool(arm.get("vrr"))
        M, L = S.match_table(slots, toks)
        if arm.get("ranker") == "bm25f":
            portfolio_mode = effective_portfolio_mode(arm, S, a.q)
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
        else:
            res = S.rank(M, L, mode=arm.get("mode", "clm"), lex=arm.get("lex", "count"), weights=w, limit=400, rare=bool(arm.get("rare")), n_tokens=len(toks))
        page = res[PAGE * (a.page - 1): PAGE * a.page]
        import re as _re
        def split_contract(sc_):
            m = _re.search(r"\(무배당[^)]*\)", sc_)
            variant = (m.group(0)[1:-1].replace("무배당", "").strip(", ") if m else "")
            return _re.sub(r"\(무배당[^)]*\)", "", sc_).strip(), variant
        def snippet(text, toks):
            flat = " ".join(text.split())
            for t in toks:
                p = flat.find(t)
                if p >= 0:
                    st = max(0, p - 40)
                    return ("…" if st else "") + flat[st: st + PREVIEW]
            return flat[:PREVIEW]
        items = []
        portfolio = getattr(S, "_last_portfolio", {})
        phase_by_index = {t["index"]: t for t in portfolio.get("trace", [])}
        for e, sc in page:
            j = S._jo[S._m2j[e["element_id"]]]
            base_c, variant = split_contract(e["contract_scope"])
            it = {"id": e["element_id"], "jo": j["element_id"], "score": sc,
                  "contract": base_c, "variant": variant, "jo_title": (j.get("title") or "")[:40],
                  "preview": snippet(e["text"], toks)}
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
                should_fallback = not res or len(res) < int(fb.get("min_n", 5))
            if should_fallback:
                try:
                    mitems = meta_items(a.q, PAGE)
                except Exception as exc:
                    mitems, fb_used = [], f"error:{type(exc).__name__}"
                if mitems:
                    for it in mitems:
                        it["src"] = "meta"
                    if not res:
                        items, fb_used = mitems, "replace"
                    else:
                        merge_k = int(fb.get("merge_k", 10))
                        have = {it["id"] for it in items}
                        add = [it for it in mitems if it["id"] not in have][:merge_k]
                        if add:
                            items = items[:max(0, PAGE - len(add))] + add
                            fb_used = f"merge:{len(add)}"
        facets = {}
        if arm.get("facet"):
            # 상위 200 후보의 특약·조 분포 — 에이전트가 범위를 좁혀 재검색할 수 있게 하는 참고 정보(필터 아님)
            import collections as _c
            fc = _c.Counter(_re.sub(r"\(무배당[^)]*\)", "", e["contract_scope"]).strip() for e, _ in res[:200])
            fj = _c.Counter((S._jo[S._m2j[e["element_id"]]]["title"] or S._jo[S._m2j[e["element_id"]]]["element_id"])[:30] for e, _ in res[:200])
            facets = {"contract_top": fc.most_common(8), "article_top": fj.most_common(8)}
        log({"q": a.q, "slots": slots, "n_tokens": len(toks), "total": len(res), "page": a.page,
             **({"cache": {k: S._cache_provenance[k] for k in ("fingerprint", "cache_hit", "path")}}
                if hasattr(S, "_cache_provenance") else {}),
             **({"hybrid_gate": gate_info} if gate_info else {}),
             **({"fallback": fb_used} if fb_used else {}),
             **({"portfolio": {"specs": portfolio.get("specs", []),
                                "phases": [it.get("phase", "") for it in items]}} if portfolio else {}),
             "returned": [(e["element_id"], sc) for e, sc in page]})
        out({"query": a.q, "slots_used": slots, "total_candidates": len(res), "page": a.page, "page_size": PAGE,
             "search_calls_left": SEARCH_CAP - n_search - 1,
             **({"channel": "tag+fallback:meta"} if fb_used else {}),
             **({"hybrid_gate": gate_info} if gate_info else {}),
             **({"search_process": portfolio.get("specs", [])}
                if portfolio and arm.get("portfolio_prompt") else {}),
             **({"facets": facets} if facets else {}), "results": items})
    elif a.cmd == "msearch":
        # 메타데이터(청크) 검색 — vector_search.ChunkHybridSearch(BM25+Dense RRF). 이 채널은 고정이며 우리 변인이 아니다.
        if n_search >= SEARCH_CAP:
            out({"error": f"search 예산 초과({SEARCH_CAP}회). submit 하십시오."}); return
        if not arm.get("meta"):
            out({"error": "이 arm 에서는 msearch 를 사용할 수 없습니다."}); return
        sys.path.insert(0, str(HERE.parent / "vector_search"))
        from hybrid_search import ChunkHybridSearch
        from units import Units
        hs = ChunkHybridSearch(view=arm.get("meta_view", "V9"))
        U = Units()
        res = hs.search(a.q, strategy=a.strategy, top_k=PAGE * a.page)
        page = res[PAGE * (a.page - 1): PAGE * a.page]
        items = []
        for r in page:
            j = U.jo_of_span(r["char_start"], r["char_end"]) if r.get("char_start") is not None else None
            items.append({"id": r["id"], "jo": j["element_id"] if j else "", "contract": (j or {}).get("contract_scope", "")[:40],
                          "preview": " ".join((r.get("preview") or "").split())[:PREVIEW]})
        log({"q": a.q, "strategy": a.strategy, "page": a.page, "returned": [(r["id"], r.get("rrf_score") or r.get("dense_sim") or 0) for r in page]})
        out({"query": a.q, "channel": f"meta:{a.strategy}", "page": a.page, "page_size": PAGE, "search_calls_left": SEARCH_CAP - n_search - 1, "results": items})
    elif a.cmd == "read":
        if n_read >= READ_CAP:
            out({"error": f"read 예산 초과({READ_CAP}회)."}); return
        eid = a.id.strip()
        if eid.startswith("j"):
            j = next((u for u in S._jo if u["element_id"] == eid), None)
            text = j["text"][:4000] if j else ""; members = j["members"] if j else []
        else:
            i = S._eidx.get(eid)
            if i is None:
                out({"error": "unknown id"}); return
            j = S._jo[S._m2j[eid]]
            text = j["text"][:4000]; members = j["members"]
        log({"id": eid, "chars": len(text)})
        out({"id": eid, "jo": j["element_id"] if j else "", "contract": (j or {}).get("contract_scope", ""),
             "members": members[:60], "text": text, "read_calls_left": READ_CAP - n_read - 1})
    elif a.cmd == "submit":
        ids = [x.strip() for x in a.ids.split(",") if x.strip()][:SUBMIT_MAX]
        log({"ids": ids})
        json.dump({"qid": qid, "ranked": ids, "n_search": n_search, "n_read": n_read}, open(sess / "submit.json", "w"), ensure_ascii=False)
        out({"ok": True, "submitted": ids, "note": "세션 종료. 더 이상 도구를 호출하지 마십시오."})


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""에이전트 도구 CLI (search / msearch / read / submit) — 0820 재설계판.

0819 사본에 침묵(silent) 개선 3종 + 행동 강제 1종을 얹었다. 신규 arm 플래그가 전부 꺼져
있으면 0819/원본과 동일하게 동작한다(기준선 재현 게이트).
  [0819 유지]
  alias:1            aliases.json 질의 확장(lex 채널 한정)
  scope_boost:3.0    search --scope soft-descent 가산 부스트 + browse 모드
  fallback:{...}     태그 결과 0/빈약 시 msearch 자동 폴백 — 프롬프트 안내 줄 포함(0819 방식)
  ref_expand:1       read 에 참조 조 미리보기, search 상위 10건에 ref_jo 첨부
  [0820 신규 — 전부 프롬프트 무언급(silent), build_extra 가 줄을 만들지 않는다]
  fallback_silent:{...} fallback 과 동일 로직, 안내 줄 없음(0819 구축 효과 제거 재실험)
  submit_pad:1       submit 시 조 dedup 후 세션 검색결과 RRF 로 10개 distinct 조까지 자동 패딩
  submit_ref_merge:1 submit 시 제출 조의 참조 조(refs_jo)를 빈 슬롯에 병합(다중 근거 겨냥)
  force_msearch:1    msearch 0회 상태의 첫 submit 을 1회 거부(행동 강제, p_active_hard 전용)
데이터는 전부 filesearch/out/ 을 참조한다(복사 금지). pickle 캐시만 이 디렉터리 out/ 에 둔다.
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
VS = HERE.parents[1]
sys.path.insert(0, str(FS))
sys.path.insert(0, str(HERE))
import enhance
PAGE, PREVIEW, SEARCH_CAP, READ_CAP, SUBMIT_MAX = 40, 160, 20, 8, 10


def load_search(elements, tags):
    """SlotSearch 를 pickle 캐시로 로드(호출당 프로세스 기동 비용 절감). 캐시는 자기 out/ 에 분리."""
    from clm_search import SlotSearch
    key = HERE / "out" / f".cache_{Path(elements).stem}_{Path(tags).stem}.pkl"
    key.parent.mkdir(exist_ok=True)
    if key.exists() and key.stat().st_mtime > max(Path(elements).stat().st_mtime, Path(tags).stat().st_mtime):
        return pickle.load(open(key, "rb"))
    S = SlotSearch(elements, tags)
    pickle.dump(S, open(key, "wb"))
    return S


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("--q", default=""); s.add_argument("--page", type=int, default=1)
    s.add_argument("--scope", default="", help='계층 경로 "<특약>[/<관>[/<조>]]" — 매치 가산 부스트. --q 없이 주면 browse')
    for f in ("contract", "role", "subject", "qualifier", "schema"):
        s.add_argument(f"--{f}", default="", help="쉼표 구분, 선택")
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

    S = load_search(str(FS / "out" / arm.get("elements", "elements_u2.jsonl")), str(FS / "out" / arm.get("tags", "tags_u2_rules.jsonl")))
    if not hasattr(S, "_jo"):
        J = [json.loads(l) for l in open(FS / "out" / arm.get("jo", "elements_u2jo.jsonl"), encoding="utf-8")]
        S._jo = J; S._m2j = {mm: j for j, u in enumerate(J) for mm in u["members"]}
        S._eidx = {e["element_id"]: i for i, e in enumerate(S.E)}
    T = None
    if arm.get("expose_tags"):
        T = {json.loads(l)["element_id"]: json.loads(l) for l in open(FS / "out" / arm.get("tags", "tags_u2_rules.jsonl"), encoding="utf-8")}

    def meta_items(q, top_k):
        """msearch 내부 — msearch 커맨드와 폴백이 공유."""
        sys.path.insert(0, str(VS))
        from hybrid_search import ChunkHybridSearch
        from units import Units
        hs = ChunkHybridSearch(view=arm.get("meta_view", "V9"))
        U = Units()
        res = hs.search(q, strategy="hybrid", top_k=top_k)
        items = []
        for r in res:
            j = U.jo_of_span(r["char_start"], r["char_end"]) if r.get("char_start") is not None else None
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
        alias_log = {}
        if arm.get("alias"):
            extra, alias_log = enhance.expand_query(a.q, toks)
            toks = toks + extra
        w = {f: v * conf.get(f, 1.0) for f, v in (arm.get("w") or {}).items()}
        M, L = S.match_table(slots, toks)
        res = S.rank(M, L, mode=arm.get("mode", "clm"), lex=arm.get("lex", "count"), weights=w, limit=400, rare=bool(arm.get("rare")), n_tokens=len(toks))
        if a.scope and arm.get("scope_boost"):
            res = enhance.apply_scope_boost(res, S, S._eidx, a.scope, float(arm["scope_boost"]))
        page = res[PAGE * (a.page - 1): PAGE * a.page]
        refs = enhance.load_refs() if arm.get("ref_expand") else {}
        items = []
        for e, sc in page:
            j = S._jo[S._m2j[e["element_id"]]]
            it = {"id": e["element_id"], "jo": j["element_id"], "score": sc,
                  "contract": e["contract_scope"][:40], "preview": " ".join(e["text"].split())[:PREVIEW]}
            if T:
                t = T[e["element_id"]]; loc = t.get("locator") or {}
                it["tag"] = f"[특약]{t.get('contract_key','')[:30]} [조]{loc.get('article','')} {loc.get('article_title','')[:30]} [역할]{'/'.join(t.get('role') or [])} [유형]{t.get('schema_tag','')}"
            if refs and len(items) < 10 and j["element_id"] in refs:
                it["ref_jo"] = refs[j["element_id"]][:4]
            items.append(it)
        # 자동 폴백: 태그 결과가 없거나 빈약하면 같은 호출 안에서 msearch 결과 병합/대체
        # fallback_silent 는 동일 로직 — 차이는 build_extra 가 프롬프트 안내 줄을 만들지 않는 것뿐
        fb = (arm.get("fallback") or arm.get("fallback_silent")) if arm.get("meta") else None
        fb_used = ""
        if fb and a.page == 1:
            min_n = fb.get("min_n", 5); tau = fb.get("tau")
            top_sc = res[0][1] if res else 0.0
            if not res or len(res) < min_n or (tau is not None and top_sc < float(tau)):
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
                        have = {it["id"] for it in items}
                        items = items + [x for x in mitems if x["id"] not in have][:max(0, PAGE - len(items))]
                        fb_used = "append"
        facets = {}
        if arm.get("facet"):
            # 상위 200 후보의 특약·조 분포 — 에이전트가 범위를 좁혀 재검색할 수 있게 하는 참고 정보(필터 아님)
            import collections as _c
            fc = _c.Counter(e["contract_scope"][:40] for e, _ in res[:200])
            fj = _c.Counter((S._jo[S._m2j[e["element_id"]]]["title"] or S._jo[S._m2j[e["element_id"]]]["element_id"])[:30] for e, _ in res[:200])
            facets = {"contract_top": fc.most_common(8), "article_top": fj.most_common(8)}
        log({"q": a.q, "slots": slots, "n_tokens": len(toks), "total": len(res), "page": a.page,
             **({"scope": a.scope} if a.scope else {}), **({"alias_expanded": alias_log} if alias_log else {}),
             **({"fallback": fb_used} if fb_used else {}),
             "returned": [(it["id"], it.get("score", 0)) for it in items]})
        out({"query": a.q, "slots_used": slots, "total_candidates": len(res), "page": a.page, "page_size": PAGE,
             "search_calls_left": SEARCH_CAP - n_search - 1,
             **({"channel": "tag+fallback:meta"} if fb_used else {}),
             **({"facets": facets} if facets else {}), "results": items})
    elif a.cmd == "msearch":
        # 메타데이터(청크) 검색 — vector_search.ChunkHybridSearch(BM25+Dense RRF). 이 채널은 고정이며 우리 변인이 아니다.
        if n_search >= SEARCH_CAP:
            out({"error": f"search 예산 초과({SEARCH_CAP}회). submit 하십시오."}); return
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
        if eid.startswith("j"):
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
        # A2(force_msearch): msearch 0회 상태의 첫 submit 을 1회 거부. 두 번째 submit 은 무조건 수리(no_submit 증가 방지).
        if arm.get("force_msearch") and arm.get("meta"):
            n_ms = sum(1 for c in calls if c["cmd"] == "msearch")
            already_rejected = any(c["cmd"] == "submit" and c.get("rejected") for c in calls)
            if n_ms == 0 and not already_rejected and n_search < SEARCH_CAP:
                log({"ids": ids, "rejected": "no_msearch"})
                out({"error": "제출 전에 msearch 로 교차 확인하십시오. msearch 를 1회 실행한 뒤 다시 submit 하십시오."}); return
        pad_note = {}
        if arm.get("submit_pad") or arm.get("submit_ref_merge"):
            # 채점과 동일한 id→조 사상(units.Units.resolve)을 그대로 사용한다.
            from units import Units
            U = Units(str(FS / "out" / arm.get("jo", "elements_u2jo.jsonl")))

            def jo_id(x):
                r = U.resolve([x])
                return r[0]["element_id"] if r else None
            # 1) 조 dedup: 같은 조로 접히는 뒷순위 id 제거 — 채점 규칙과 동일하므로 점수 불변, 슬롯만 회수
            kept, covered = [], set()
            for x in ids:
                j = jo_id(x)
                if j is None or j not in covered:
                    kept.append(x)
                    if j:
                        covered.add(j)
            pad_note["deduped"] = len(ids) - len(kept)
            ids = kept
            # 2) S3(submit_ref_merge): 제출 조가 참조하는 조를 뒤 순위에 병합 — 다중 근거 문항 겨냥
            if arm.get("submit_ref_merge"):
                refs = enhance.load_refs()
                added = []
                for x in list(ids):
                    j = jo_id(x)
                    for rj in (refs.get(j) or [])[:4]:
                        if len(ids) + len(added) >= SUBMIT_MAX:
                            break
                        if rj not in covered:
                            added.append(rj); covered.add(rj)
                ids = ids + added
                pad_note["ref_merged"] = len(added)
            # 3) S1(submit_pad): 세션에서 본 검색결과 전체의 RRF 융합 순위로 빈 슬롯을 distinct 조로 채움
            if arm.get("submit_pad") and len(ids) < SUBMIT_MAX:
                rrf = {}
                for c in calls:
                    if c["cmd"] in ("search", "msearch") and c.get("returned"):
                        for rank, (rid, _s) in enumerate(c["returned"]):
                            rrf[rid] = rrf.get(rid, 0.0) + 1.0 / (60 + rank)
                n_pad = 0
                for rid, _ in sorted(rrf.items(), key=lambda kv: -kv[1]):
                    if len(ids) >= SUBMIT_MAX:
                        break
                    j = jo_id(rid)
                    if j and j not in covered:
                        ids.append(rid); covered.add(j); n_pad += 1
                pad_note["padded"] = n_pad
        log({"ids": ids, **pad_note})
        json.dump({"qid": qid, "ranked": ids, "n_search": n_search, "n_read": n_read}, open(sess / "submit.json", "w"), ensure_ascii=False)
        out({"ok": True, "submitted": ids, "note": "세션 종료. 더 이상 도구를 호출하지 마십시오."})


if __name__ == "__main__":
    main()

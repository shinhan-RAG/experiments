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
  [0820 신규 2차 — silent]
  fanout:1           search 시 role/subject 값이 2개 이상이면 단일 값 서브 검색(≤6)을 내부 실행해
                     통합(가중 2)+서브(가중 1) RRF(k=60) 융합으로 순위 재구성. 출력 스키마 불변.
  submit_cover:1     submit 최종 리스트(pad/merge 이후)에서 top-5 근사중복 조 압축 — 같은 특약
                     base + 제목 동일 or 텍스트 앞 200자 bigram 유사도 ≥0.85 군은 top-5에 최대
                     2개, 밀려난 조는 6위 이후 원 순서 유지.
데이터는 전부 filesearch/out/ 을 참조한다(복사 금지). pickle 캐시만 이 디렉터리 out/ 에 둔다.
"""
import argparse, json, os, pickle, re, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
FS = HERE.parents[2] / "filesearch"
VS = HERE.parents[1]
sys.path.insert(0, str(FS))
sys.path.insert(0, str(HERE))
import enhance
PAGE, PREVIEW, SEARCH_CAP, READ_CAP, SUBMIT_MAX = 40, 160, 20, 8, 10

_TOKEN_RE = re.compile(r"[가-힣]{2,}|[A-Za-z]{2,}|\d+(?:\.\d+)?%?")


def _q_tokens(q):
    return _TOKEN_RE.findall(q or "")[:8]


def split_contract(sc_):
    """filesearch PR#65 이식: 특약명 40자 절단 대신 전체 표기 + 판본(variant) 분리."""
    m = re.search(r"\(무배당[^)]*\)", sc_ or "")
    variant = (m.group(0)[1:-1].replace("무배당", "").strip(", ") if m else "")
    return re.sub(r"\(무배당[^)]*\)", "", sc_ or "").strip(), variant


def snippet(text, toks):
    """filesearch PR#65 이식: preview 를 앞 160자 대신 질의 토큰 매치 주변 스니펫으로."""
    flat = " ".join((text or "").split())
    for t in toks or []:
        p = flat.find(t)
        if p >= 0:
            st = max(0, p - 40)
            return ("…" if st else "") + flat[st: st + PREVIEW]
    return flat[:PREVIEW]


def _norm_bigrams(text, n_chars=200):
    """P2 근사중복 판정용: 공백 제거 정규화 텍스트 앞 n_chars 자의 bigram 집합."""
    s = "".join((text or "").split())[:n_chars]
    return set(s[i:i + 2] for i in range(len(s) - 1)) if len(s) > 1 else ({s} if s else set())


def _near_dup(a, b):
    """같은 특약 base 이고 (제목 동일 or 텍스트 앞 200자 bigram Dice ≥ 0.85) 이면 근사중복 조."""
    if a is None or b is None:
        return False
    if split_contract(a.get("contract_scope", ""))[0] != split_contract(b.get("contract_scope", ""))[0]:
        return False
    ta, tb = (a.get("title") or "").strip(), (b.get("title") or "").strip()
    if ta and ta == tb:
        return True
    ga, gb = _norm_bigrams(a.get("text")), _norm_bigrams(b.get("text"))
    if not ga or not gb:
        return False
    return 2.0 * len(ga & gb) / (len(ga) + len(gb)) >= 0.85


def cover_reorder(ids, jo_of, top_n=5, max_dup=2):
    """P2(submit_cover): 제출 순서대로 top-5 를 채우되, 이미 든 조와 근사중복이 max_dup 개
    이상이면 그 id 를 뒤로 미루고 다음 후보를 승격. 미룬 id 는 6위 이후 원 순서 유지.
    반환: (재정렬 ids, 미룬 ids). jo_of: id → 조 dict(title/contract_scope/text) or None."""
    sel_idx, sel_jos, moved_idx = [], [], []
    for i, x in enumerate(ids):
        if len(sel_idx) >= top_n:
            break
        j = jo_of(x)
        if j is not None and sum(1 for sj in sel_jos if _near_dup(j, sj)) >= max_dup:
            moved_idx.append(i)
            continue
        sel_idx.append(i); sel_jos.append(j)
    picked = set(sel_idx)
    new_ids = [ids[i] for i in sel_idx] + [x for i, x in enumerate(ids) if i not in picked]
    return new_ids, [ids[i] for i in moved_idx]


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
        qtoks = _q_tokens(q)
        items = []
        for r in res:
            j = U.jo_of_span(r["char_start"], r["char_end"]) if r.get("char_start") is not None else None
            base_c, variant = split_contract((j or {}).get("contract_scope", ""))
            items.append({"id": r["id"], "jo": j["element_id"] if j else "", "contract": base_c, "variant": variant,
                          "preview": snippet(r.get("preview") or "", qtoks)})
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
        # alias_cond: 태그 top 점수가 τ 미만일 때만 alias 확장 재랭킹 (silent — 프롬프트 무언급).
        # 결정론 검증(0820): 상시 alias 는 R@5 −2.6%p 희석, 조건부 τ=5 는 전 지표 + (R@5 +0.5, R@40 +2.7).
        if arm.get("alias_cond") and not arm.get("alias"):
            top_sc = res[0][1] if res else 0.0
            if not res or top_sc < float(arm["alias_cond"]):
                extra, alog = enhance.expand_query(a.q, toks)
                if extra:
                    toks_c = toks + extra
                    Mc, Lc = S.match_table(slots, toks_c)
                    res = S.rank(Mc, Lc, mode=arm.get("mode", "clm"), lex=arm.get("lex", "count"), weights=w, limit=400, rare=bool(arm.get("rare")), n_tokens=len(toks_c))
                    alias_log = dict(alog, _cond="fired")
        # P1(fanout): role/subject 값이 2개 이상이면 각 단일 값으로 좁힌 서브 검색(상한 6)을 내부
        # 실행하고, 통합 res(가중 2) + 서브(가중 1) 를 RRF(k=60) 융합해 순위를 재구성 (silent).
        fanout_log = {}; fanout_top_sc = None
        if arm.get("fanout"):
            subs = []
            for f in ("role", "subject"):
                vals = slots.get(f) or []
                if len(vals) >= 2:
                    subs.extend((f, v) for v in vals)
            subs = subs[:6]
            if subs:
                fanout_top_sc = res[0][1] if res else 0.0  # 폴백 τ 판정은 융합 전 태그 점수 기준 유지
                ftoks = toks
                if arm.get("alias_cond") and not arm.get("alias") and alias_log.get("_cond") == "fired":
                    ftoks = toks_c
                rrf, emap = {}, {}
                for rk, (e, _sc) in enumerate(res):
                    rrf[e["element_id"]] = rrf.get(e["element_id"], 0.0) + 2.0 / (60 + rk)
                    emap.setdefault(e["element_id"], e)
                for f, v in subs:
                    s_slots = dict(slots); s_slots[f] = [v]
                    Mf, Lf = S.match_table(s_slots, ftoks)
                    sub_res = S.rank(Mf, Lf, mode=arm.get("mode", "clm"), lex=arm.get("lex", "count"), weights=w, limit=100, rare=bool(arm.get("rare")), n_tokens=len(ftoks))
                    for rk, (e, _sc) in enumerate(sub_res):
                        rrf[e["element_id"]] = rrf.get(e["element_id"], 0.0) + 1.0 / (60 + rk)
                        emap.setdefault(e["element_id"], e)
                res = [(emap[eid], sc) for eid, sc in sorted(rrf.items(), key=lambda kv: (-kv[1], kv[0]))]
                fanout_log = {"n_sub": len(subs), "slots": [f"{f}={v}" for f, v in subs]}
        if a.scope and arm.get("scope_boost"):
            res = enhance.apply_scope_boost(res, S, S._eidx, a.scope, float(arm["scope_boost"]))
        page = res[PAGE * (a.page - 1): PAGE * a.page]
        refs = enhance.load_refs() if arm.get("ref_expand") else {}
        items = []
        for e, sc in page:
            j = S._jo[S._m2j[e["element_id"]]]
            base_c, variant = split_contract(e["contract_scope"])
            it = {"id": e["element_id"], "jo": j["element_id"], "score": sc,
                  "contract": base_c, "variant": variant, "jo_title": (j.get("title") or "")[:40],
                  "preview": snippet(e["text"], toks)}
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
            top_sc = fanout_top_sc if fanout_top_sc is not None else (res[0][1] if res else 0.0)
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
            fc = _c.Counter(split_contract(e["contract_scope"])[0] for e, _ in res[:200])
            fj = _c.Counter((S._jo[S._m2j[e["element_id"]]]["title"] or S._jo[S._m2j[e["element_id"]]]["element_id"])[:30] for e, _ in res[:200])
            facets = {"contract_top": fc.most_common(8), "article_top": fj.most_common(8)}
        log({"q": a.q, "slots": slots, "n_tokens": len(toks), "total": len(res), "page": a.page,
             **({"scope": a.scope} if a.scope else {}), **({"alias_expanded": alias_log} if alias_log else {}),
             **({"fanout": fanout_log} if fanout_log else {}),
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
        # P2(submit_cover): pad/merge 이후 최종 리스트에 top-5 근사중복 조 압축 적용 (silent)
        cover_note = {}
        if arm.get("submit_cover"):
            from units import Units as _Units
            _U = _Units(str(FS / "out" / arm.get("jo", "elements_u2jo.jsonl")))
            _jc = {}

            def _jo_of(x):
                if x not in _jc:
                    r = _U.resolve([x])
                    _jc[x] = r[0] if r else None
                return _jc[x]
            ids, moved = cover_reorder(ids, _jo_of)
            cover_note["cover_reorder"] = {"moved": moved}
        log({"ids": ids, **pad_note, **cover_note})
        json.dump({"qid": qid, "ranked": ids, "n_search": n_search, "n_read": n_read}, open(sess / "submit.json", "w"), ensure_ascii=False)
        out({"ok": True, "submitted": ids, "note": "세션 종료. 더 이상 도구를 호출하지 마십시오."})


if __name__ == "__main__":
    main()

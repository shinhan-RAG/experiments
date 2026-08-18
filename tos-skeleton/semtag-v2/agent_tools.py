#!/usr/bin/env python3
"""에이전트 도구 CLI (search / read / submit) — S5 예산 하네스.

세션 환경변수: SEMTAG_SESSION(세션 디렉터리) · SEMTAG_QID · SEMTAG_ARM(json)
ARM 예: {"lex":"count","w":{"contract":2},"router":"llm|rule|union","qtags":"out/qtags_haiku.jsonl","expose_tags":0}
예산(S5): search ≤20회 · read ≤8회 · page 40 · preview 160자 · submit 최대 10개. 초과 호출은 거부.
모든 호출의 인자와 반환(후보 id·점수)을 세션 로그(calls.jsonl)에 남긴다.
검색기 = clm_search.SlotSearch(CLM). 라우터: 사전 동결 qtags(질문 단위) + 규칙 라우터(에이전트가 준 --q) + 에이전트 슬롯 지정(--contract 등, 덧셈).
"""
import argparse, json, os, pickle, sys, time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
PAGE, PREVIEW, SEARCH_CAP, READ_CAP, SUBMIT_MAX = 40, 160, 20, 8, 10


def load_search(elements, tags):
    """SlotSearch 를 pickle 캐시로 로드(호출당 프로세스 기동 비용 절감)."""
    from clm_search import SlotSearch
    key = HERE / "out" / f".cache_{Path(elements).stem}_{Path(tags).stem}.pkl"
    if key.exists() and key.stat().st_mtime > max(Path(elements).stat().st_mtime, Path(tags).stat().st_mtime):
        return pickle.load(open(key, "rb"))
    S = SlotSearch(elements, tags)
    pickle.dump(S, open(key, "wb"))
    return S


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("search"); s.add_argument("--q", required=True); s.add_argument("--page", type=int, default=1)
    for f in ("contract", "role", "subject", "qualifier", "schema"):
        s.add_argument(f"--{f}", default="", help="쉼표 구분, 선택")
    r = sub.add_parser("read"); r.add_argument("--id", required=True)
    m = sub.add_parser("submit"); m.add_argument("--ids", required=True, help="쉼표 구분 element_id 순위(최대 10)")
    a = ap.parse_args()

    sess = Path(os.environ["SEMTAG_SESSION"]); sess.mkdir(parents=True, exist_ok=True)
    qid = os.environ.get("SEMTAG_QID", "")
    arm = json.loads(os.environ.get("SEMTAG_ARM", "{}"))
    log_path = sess / "calls.jsonl"
    calls = [json.loads(l) for l in open(log_path)] if log_path.exists() else []
    n_search = sum(1 for c in calls if c["cmd"] == "search"); n_read = sum(1 for c in calls if c["cmd"] == "read")

    def log(rec):
        rec.update({"t": time.time(), "cmd": a.cmd})
        with open(log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def out(obj):
        print(json.dumps(obj, ensure_ascii=False))

    S = load_search(str(HERE / "out" / arm.get("elements", "elements_u2.jsonl")), str(HERE / "out" / arm.get("tags", "tags_u2_rules.jsonl")))
    if not hasattr(S, "_jo"):
        J = [json.loads(l) for l in open(HERE / "out" / arm.get("jo", "elements_u2jo.jsonl"), encoding="utf-8")]
        S._jo = J; S._m2j = {mm: j for j, u in enumerate(J) for mm in u["members"]}
        S._eidx = {e["element_id"]: i for i, e in enumerate(S.E)}
    T = None
    if arm.get("expose_tags"):
        T = {json.loads(l)["element_id"]: json.loads(l) for l in open(HERE / "out" / arm.get("tags", "tags_u2_rules.jsonl"), encoding="utf-8")}

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
        w = {f: v * conf.get(f, 1.0) for f, v in (arm.get("w") or {}).items()}
        M, L = S.match_table(slots, toks)
        res = S.rank(M, L, mode=arm.get("mode", "clm"), lex=arm.get("lex", "count"), weights=w, limit=400)
        page = res[PAGE * (a.page - 1): PAGE * a.page]
        items = []
        for e, sc in page:
            j = S._jo[S._m2j[e["element_id"]]]
            it = {"id": e["element_id"], "jo": j["element_id"], "score": sc,
                  "contract": e["contract_scope"][:40], "preview": " ".join(e["text"].split())[:PREVIEW]}
            if T:
                t = T[e["element_id"]]; loc = t.get("locator") or {}
                it["tag"] = f"[특약]{t.get('contract_key','')[:30]} [조]{loc.get('article','')} {loc.get('article_title','')[:30]} [역할]{'/'.join(t.get('role') or [])} [유형]{t.get('schema_tag','')}"
            items.append(it)
        log({"q": a.q, "slots": slots, "n_tokens": len(toks), "total": len(res), "page": a.page,
             "returned": [(e["element_id"], sc) for e, sc in page]})
        out({"query": a.q, "slots_used": slots, "total_candidates": len(res), "page": a.page, "page_size": PAGE,
             "search_calls_left": SEARCH_CAP - n_search - 1, "results": items})
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

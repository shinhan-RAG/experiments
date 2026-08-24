# -*- coding: utf-8 -*-
"""
하이브리드 8슬롯 태그 — 결정론 3 + LLM 5 (codex CLI / gpt-5.6-luna).

  결정론 3 : contract_key, locator, schema_tag        <- tags_u2_rules.jsonl 그대로
  LLM    5 : subject_key, role, qualifier, reference, search_text

`search_text` 는 독립 슬롯이 아니라 나머지 슬롯의 조합 문자열이다(tag_rules.py:76-83).
따라서 LLM 이 실제로 생성하는 것은 subject/role/qualifier/reference 4개이고,
search_text 는 그 값들로 규칙본과 동일한 포맷으로 재조합한다 — 내용의 출처가 LLM 이므로 LLM 슬롯이다.

호출 단위는 **조(u2jo)** 다. 조 안의 항(element)에 배분한다:
  - subject / qualifier / reference : 그 문자열이 실제로 들어 있는 항에만 (verbatim 검증)
  - role                            : 조 전체에 부여

본문 30자 미만인 조(4,364/7,599 = 57%)는 제목·머리글이라 호출하지 않고 규칙본 값을 승계한다.
그렇게 하지 않으면 그 항들의 태그가 통째로 비어 어휘 검색이 무너진다.

사용:
  python build_tags_h35.py --workers 8
  python build_tags_h35.py --limit 20        # 비용 시험
"""
import argparse, json, re, subprocess, sys, io, os, shutil, time, collections
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
R = HERE.parents[2] / "tos-skeleton" / "hybrid-enrich_v2"
FS = R / "filesearch"
sys.path.insert(0, str(FS))
from patterns import ROLE_KO                      # noqa: E402

ROLES = list(ROLE_KO.keys())
OUT = HERE / "out"; OUT.mkdir(exist_ok=True)
VERSION = "semtag-h35-luna-1.0"
MIN_TEXT = 30

PROMPT = """보험 약관의 한 조(條)입니다. 검색 색인용 태그를 JSON 하나로만 출력하세요(설명 금지).

모든 값은 **본문에 있는 표기 그대로** 복사하세요. 요약·동의어·의역·창작 금지.

- "subject": 이 조가 다루는 핵심 대상어 2~6개 (급여금명, 질병·수술명, 정의되는 용어 등).
- "role": 이 조의 역할 코드 1~3개. 아래 목록에서만 고르세요.
- "qualifier": 본문에 있는 조건·수치 그대로 (기간, 횟수, 비율, 금액 등). 없으면 [].
- "reference": 본문이 가리키는 다른 조항·부표·별첨 표기 그대로 (예: "제17조", "<부표2-1>"). 없으면 [].

[역할 코드]
{roles}

[조 원문]
{text}

JSON:"""


def _codex_bin():
    for c in ("codex.cmd", "codex"):
        p = shutil.which(c)
        if p and (os.name != "nt" or p.lower().endswith(".cmd")):
            return p
    p = Path(os.environ.get("APPDATA", "")) / "npm" / "codex.cmd"
    return str(p) if p.exists() else "codex"


CODEX = _codex_bin()


def call_codex(model, prompt, cwd, timeout=300):
    r = subprocess.run([CODEX, "exec", "-m", model, "--skip-git-repo-check", "-"],
                       input=prompt, capture_output=True, text=True, timeout=timeout,
                       cwd=str(cwd), encoding="utf-8", errors="replace")
    txt = r.stdout or ""
    for b in reversed(re.findall(r"\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}", txt, re.S)):
        try:
            d = json.loads(b)
            if isinstance(d, dict) and any(k in d for k in ("subject", "role", "qualifier", "reference")):
                return d
        except Exception:
            continue
    return {}


def nz(s):
    return "".join(c for c in s if not c.isspace())


def compose(schema_tag, ck, subjects, roles, loc, quals, refs):
    """tag_rules.py:76-83 과 동일 포맷."""
    art, art_title = loc.get("article", ""), loc.get("article_title", "")
    sec = loc.get("section", "")
    parts = [
        "[schema] %s" % schema_tag,
        "[contract] %s" % ck if ck else "",
        "[subject] %s" % " | ".join(subjects) if subjects else "",
        "[role] %s | %s" % (" | ".join(roles), " | ".join(ROLE_KO.get(r, r) for r in roles)) if roles else "",
        ("[article] %s %s" % (art, art_title)).strip() if art else "",
        "[section] %s" % sec if sec else "",
        "[qualifier] %s" % " | ".join(quals) if quals else "",
        "[reference] %s" % " | ".join(refs) if refs else "",
    ]
    return " | ".join(x for x in parts if x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=-1)
    ap.add_argument("--out", default=str(OUT / "tags_u2_h35.jsonl"))
    a = ap.parse_args()

    J = [json.loads(l) for l in (FS / "out" / "elements_u2jo.jsonl").open(encoding="utf-8")]
    E = [json.loads(l) for l in (FS / "out" / "elements_u2.jsonl").open(encoding="utf-8")]
    rules = {d["element_id"]: d for d in
             (json.loads(l) for l in (FS / "out" / "tags_u2_rules.jsonl").open(encoding="utf-8"))}

    cache_path = OUT / ("tagllm_cache_%s.jsonl" % a.model.replace("/", "_"))
    cache = {}
    if cache_path.exists():
        for l in cache_path.open(encoding="utf-8"):
            try:
                d = json.loads(l); cache[d["jo"]] = d
            except Exception:
                pass

    # 실패(ok=False)는 캐시에 남기지 않고 재시도한다 — 네트워크 단절 구간이 빈 태그로 굳는 것을 막는다.
    todo = [u for u in J if len((u.get("text") or "")) >= MIN_TEXT
            and not (cache.get(u["element_id"]) or {}).get("ok")]
    if a.limit > 0:
        todo = todo[:a.limit]
    skipped = sum(1 for u in J if len((u.get("text") or "")) < MIN_TEXT)
    print("조 %d | 30자미만 승계 %d | 캐시 %d | 호출대상 %d (model=%s)"
          % (len(J), skipped, len(cache), len(todo), a.model), flush=True)

    roles_doc = "\n".join("- %s: %s" % (r, ROLE_KO[r]) for r in ROLES)
    t0 = time.time()
    fail = 0

    def work(u):
        try:
            js = call_codex(a.model, PROMPT.format(roles=roles_doc, text=(u["text"] or "")[:3500]), HERE)
        except Exception:
            js = {}
        body = u.get("text") or ""
        nb = nz(body)
        def keep(vals, cap):
            out = []
            for v in (vals or []):
                v = str(v).strip()
                if v and nz(v) in nb and v not in out:      # verbatim 검증
                    out.append(v)
            return out[:cap]
        return {"jo": u["element_id"],
                "subject": keep(js.get("subject"), 6),
                "role": [r for r in (js.get("role") or []) if r in ROLES][:3],
                "qualifier": keep(js.get("qualifier"), 8),
                "reference": keep(js.get("reference"), 12),
                "ok": bool(js)}

    if todo:
        with ThreadPoolExecutor(a.workers) as ex, cache_path.open("a", encoding="utf-8") as f:
            for i, rec in enumerate(ex.map(work, todo), 1):
                if not rec["ok"]:
                    fail += 1
                    continue                      # 캐시에 남기지 않음 -> 다음 실행에서 재시도
                cache[rec["jo"]] = rec
                f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
                if i % 25 == 0 or i == len(todo):
                    el = time.time() - t0
                    print("  %d/%d  실패 %d  %.0fs  (남은 %.0f분)"
                          % (i, len(todo), fail, el, (len(todo) - i) * el / i / 60), flush=True)

    # ---- 조 -> 항 배분 + 병합 ----
    jo_of = {}
    for u in J:
        for m in (u.get("members") or []):
            jo_of[m if isinstance(m, str) else m.get("element_id", "")] = u["element_id"]

    out, stat = [], collections.Counter()
    for e in E:
        eid = e["element_id"]
        base = rules.get(eid)
        if not base:
            continue
        jo = jo_of.get(eid)
        c = cache.get(jo)
        et = nz(e.get("text") or "")
        if c:
            subj = [s for s in c["subject"] if nz(s) in et]
            qual = [s for s in c["qualifier"] if nz(s) in et]
            refs = [s for s in c["reference"] if nz(s) in et]
            role = list(c["role"])
            stat["llm"] += 1
        else:
            subj, qual, refs, role = base["subject_key"], base["qualifier"], base["reference"], base["role"]
            stat["rules_승계"] += 1
        rec = {"element_id": eid, "schema_version": VERSION,
               "schema_tag": base["schema_tag"],          # 결정론
               "contract_key": base["contract_key"],      # 결정론
               "locator": base["locator"],                # 결정론
               "subject_key": subj, "role": role,         # LLM
               "qualifier": qual, "reference": refs,      # LLM
               "search_text": compose(base["schema_tag"], base["contract_key"],
                                      subj, role, base["locator"], qual, refs)}
        if not nz(rec["search_text"]):                    # 빈 태그 방지
            rec["search_text"] = base["search_text"]
            stat["search_text_fallback"] += 1
        out.append(rec)

    with open(a.out, "w", encoding="utf-8") as f:
        for r in out:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    n = len(out)
    cov = {k: round(sum(1 for r in out if r[k]) / n, 3)
           for k in ("contract_key", "subject_key", "role", "qualifier", "reference")}
    meta = {"version": VERSION, "n": n, "model": a.model,
            "deterministic": ["contract_key", "locator", "schema_tag"],
            "llm": ["subject_key", "role", "qualifier", "reference", "search_text"],
            "jo_called": len([1 for v in cache.values() if v.get("ok")]),
            "jo_skipped_short": skipped, "min_text": MIN_TEXT, "failed_uncached": fail,
            "element_source": dict(stat), "coverage": cov}
    Path(a.out.replace(".jsonl", "_meta.json")).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(meta, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8")
    main()

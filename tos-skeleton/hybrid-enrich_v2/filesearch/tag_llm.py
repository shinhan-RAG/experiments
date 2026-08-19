#!/usr/bin/env python3
"""태그 내용 슬롯(subject·role) LLM 추출 — 규칙본(tag_rules)과 1변수 비교용.

설계서 2.1 준수 조건:
  - subject 는 조(u2jo) 원문에 실제로 있는 문자열만(verbatim). 추출 후 원문 substring 검증으로 걸러낸다.
  - role 은 폐쇄 코드 11종에서만 고른다.
  - 요약·동의어·의역 생성 금지. 1회 생성 후 캐시 동결(재실행 시 재호출 없음).
조 단위로 1회 호출해 조 안의 element(항)들에 배분한다: subject 는 그 문자열이 실제로 들어 있는 항에만,
role 은 조 전체에 부여(조 제목·본문 기준). 다른 슬롯(contract/article/qualifier/…)은 규칙본 그대로 승계.
출력: out/tags_u2_llm.jsonl (tag_rules 와 같은 스키마), 캐시 out/tagllm_cache.jsonl(조 단위).
"""
import argparse, hashlib, json, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from patterns import ROLE_KO  # noqa: E402

ROLES = list(ROLE_KO.keys())
PROMPT = """보험 약관의 한 조(條)입니다. 검색 색인용 태그를 JSON 하나로만 출력하세요(설명 금지).

- "subject": 이 조가 다루는 핵심 대상어를 **본문에 있는 표기 그대로** 2~6개 복사하세요(급여금명, 질병·수술명, 정의되는 용어 등). 본문에 없는 말을 만들지 마세요. 요약·동의어 금지.
- "role": 이 조의 역할을 아래 코드에서 1~3개 고르세요.
{roles}

[조 원문]
{text}

JSON:"""


def call(model, prompt):
    r = subprocess.run(["claude", "-p", "--model", model, "--output-format", "json"], input=prompt,
                       capture_output=True, text=True, timeout=240)
    try:
        txt = json.loads(r.stdout).get("result", "")
    except Exception:
        txt = r.stdout
    m = re.search(r"\{.*\}", txt, re.S)
    try:
        return json.loads(m.group(0)) if m else {}
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="haiku")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--limit", type=int, default=-1, help="조 수 제한(비용 시험용)")
    a = ap.parse_args()
    J = [json.loads(l) for l in open(HERE / "out/elements_u2jo.jsonl", encoding="utf-8")]
    E = {json.loads(l)["element_id"]: json.loads(l) for l in open(HERE / "out/elements_u2.jsonl", encoding="utf-8")}
    base = {json.loads(l)["element_id"]: json.loads(l) for l in open(HERE / "out/tags_u2_rules.jsonl", encoding="utf-8")}
    cache_path = HERE / "out" / f"tagllm_cache_{a.model}.jsonl"
    cache = {}
    if cache_path.exists():
        for l in open(cache_path, encoding="utf-8"):
            d = json.loads(l); cache[d["jo"]] = d
    roles_doc = "\n".join(f"- {r}: {ROLE_KO[r]}" for r in ROLES)
    todo = [u for u in J if u["element_id"] not in cache and len(u["text"].strip()) >= 40]
    if a.limit > 0:
        todo = todo[: a.limit]
    print(f"jo total={len(J)} cached={len(cache)} todo={len(todo)}", flush=True)

    def norm(s):
        return re.sub(r"\s", "", s)

    def work(u):
        try:
            js = call(a.model, PROMPT.format(roles=roles_doc, text=u["text"][:3500]))
        except Exception:
            try:
                js = call(a.model, PROMPT.format(roles=roles_doc, text=u["text"][:2000]))
            except Exception:
                js = {}
        nt = norm(u["text"])
        subj = []
        for s in (js.get("subject") or [])[:8]:
            s = str(s).strip()
            if 2 <= len(s) <= 60 and norm(s) in nt and s not in subj:
                subj.append(s)  # verbatim 검증: 조 원문에 실제 존재하는 문자열만
        role = [r for r in (js.get("role") or []) if r in ROLES][:3]
        return {"jo": u["element_id"], "subject": subj[:6], "role": role, "model": a.model,
                "parse_ok": bool(js), "n_rejected": len(js.get("subject") or []) - len(subj)}

    with ThreadPoolExecutor(a.workers) as ex, open(cache_path, "a", encoding="utf-8") as f:
        for i, rec in enumerate(ex.map(work, todo)):
            cache[rec["jo"]] = rec
            f.write(json.dumps(rec, ensure_ascii=False) + "\n"); f.flush()
            if i % 50 == 0:
                print(i, rec["jo"], rec["subject"][:2], rec["role"], flush=True)

    # 배분: subject 는 문자열이 실제 들어 있는 항에만, role 은 조 전체
    out_path = HERE / "out/tags_u2_llm.jsonl"
    n_subj = n_role = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for u in J:
            c = cache.get(u["element_id"], {})
            for eid in u["members"]:
                t = dict(base[eid])
                et = norm(E[eid]["text"])
                subj_here = [s for s in c.get("subject", []) if norm(s) in et]
                if subj_here:
                    t["subject_key"] = list(dict.fromkeys(subj_here + t.get("subject_key", [])))[:10]
                    n_subj += 1
                if c.get("role"):
                    t["role"] = list(dict.fromkeys(c["role"] + t.get("role", [])))[:5]
                    n_role += 1
                t["schema_version"] = f"semtag-v2-llm-{a.model}-1.0"
                f.write(json.dumps(t, ensure_ascii=False) + "\n")
    n = len(base)
    stats = {"model": a.model, "jo_tagged": len(cache), "elements_subject_added": n_subj, "elements_role_added": n_role,
             "coverage_subject": round(sum(1 for l in open(out_path, encoding='utf-8') if json.loads(l)["subject_key"]) / n, 3),
             "sha256": hashlib.sha256(open(out_path, "rb").read()).hexdigest()[:16]}
    json.dump(stats, open(HERE / "out/tags_u2_llm_stats.json", "w"), ensure_ascii=False, indent=1)
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

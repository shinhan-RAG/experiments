#!/usr/bin/env python3
"""3b — (특약,조) 그룹별 예상질문 생성(haiku) + Doc2Query--식 필터 + 동결.

근거: doc2query(arXiv:1904.08375)·docTTTTTquery — 예상질문 부착으로 검색 개선;
Doc2Query--(ECIR 2023) — 환각 필터 필수. 생성물은 SHA 동결(schema_version 스탬프).
"""
import hashlib, json, os, re, subprocess, sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
sys.path.insert(0, ".")
import semtag_experiment as SE
from context_v2 import annotate_v2

PROMPT = """보험 약관의 한 조항이다. 고객이 콜센터에 이 조항 내용을 물을 때 쓸 법한
짧은 질문 표현을 3개 만들어라. 조건: 각 5~14자, 일상 구어 어휘(약관 문어체 금지),
조항 원문에 그대로 나오는 표현은 금지, 명사구 또는 짧은 의문형.

조항: {jo}
발췌: {excerpt}

한 줄 JSON만 출력: {{"q": ["표현1", "표현2", "표현3"]}}"""


def gen_one(key, jo, excerpt, outdir):
    fid = hashlib.sha256((key).encode()).hexdigest()[:16]
    f = outdir / f"{fid}.json"
    if f.exists():
        d = json.loads(f.read_text())
        if d.get("q"):
            return key, d["q"]
    env = dict(os.environ); env.pop("ANTHROPIC_API_KEY", None)
    try:
        r = subprocess.run(["claude", "-p", "--model", "haiku"],
                           input=PROMPT.format(jo=jo, excerpt=excerpt[:400]),
                           capture_output=True, text=True, timeout=90, env=env)
        m = re.findall(r"\{[^{}]*\"q\"[\s\S]*?\}", r.stdout)
        d = json.loads(m[-1]) if m else {"q": []}
        qs = [str(x)[:16] for x in d.get("q", [])][:3]
    except Exception:
        qs = []
    f.write_text(json.dumps({"q": qs}, ensure_ascii=False))
    return key, qs


def main():
    doc = sys.argv[1]
    lines = SE.nfc(Path(doc).read_text(encoding="utf-8", errors="ignore")).splitlines()
    els = annotate_v2(SE.split_elements(lines), lines)
    groups = {}
    for e in els:
        if e["jo"]:
            k = e["scope"] + "\x1f" + e["jo"]
            if k not in groups:
                groups[k] = (e["jo"], e["text"][:600])
    print(f"groups={len(groups)}", flush=True)
    outdir = Path("out/qgen"); outdir.mkdir(parents=True, exist_ok=True)
    results = {}
    with ThreadPoolExecutor(max_workers=8) as ex:
        futs = {ex.submit(gen_one, k, jo, ex_txt, outdir): k
                for k, (jo, ex_txt) in groups.items()}
        for i, fu in enumerate(as_completed(futs), 1):
            k, qs = fu.result()
            results[k] = qs
            if i % 50 == 0 or i == len(futs):
                ok = sum(1 for v in results.values() if v)
                print(f"  {i}/{len(futs)} 생성 성공={ok}", flush=True)
    # Doc2Query--식 필터: 원문 재기입 제거 + 환각 가드(주제 bigram 최소 겹침)
    doc_norm = SE.ns("\n".join(lines))
    kept = {}
    drop_verbatim = drop_haluc = 0
    for k, qs in results.items():
        jo = groups[k][0]; src = SE.ns(groups[k][1])
        keep = []
        for q in qs:
            qn = SE.ns(q)
            if len(qn) < 4:
                continue
            if qn in src or qn in doc_norm[:0]:   # 원문 그대로 → 새 어휘 아님
                drop_verbatim += 1
                continue
            qb = set(SE.bigrams(q)); jb = set(SE.bigrams(jo + groups[k][1][:200]))
            if not (qb & jb) and len(qn) > 6:      # 주제와 완전 무관 → 환각 의심
                drop_haluc += 1
                continue
            keep.append(q)
        kept[k] = keep[:3]
    payload = {"schema_version": "qtag.gen.v1", "model": "haiku",
               "prompt_sha": hashlib.sha256(PROMPT.encode()).hexdigest()[:12],
               "n_groups": len(groups),
               "dropped": {"verbatim": drop_verbatim, "hallucination": drop_haluc},
               "tags": kept}
    out = Path("out/qgen_frozen.json")
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1), encoding="utf-8")
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    print(f"frozen: {out} sha={sha[:16]} 필터 제거 verbatim={drop_verbatim} 환각={drop_haluc}",
          flush=True)


if __name__ == "__main__":
    main()

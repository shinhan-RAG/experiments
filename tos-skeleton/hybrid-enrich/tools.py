#!/usr/bin/env python3
"""에이전트 검색 도구 CLI. Arm 격리 + 세션당 호출 상한 8회.
usage:
  python3 tools.py vector "질의"
  python3 tools.py grep "패턴"
  python3 tools.py read c01234
환경변수: ARM(BASE|META|TAG|BOTH), SESSION_DIR(카운터/로그 저장)"""
import json, os, sys, re, subprocess, urllib.request
import numpy as np

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(BASE, "out")
MAX_CALLS = 8

ARM = os.environ.get("ARM")
SESSION_DIR = os.environ.get("SESSION_DIR")
assert ARM in ("BASE", "META", "TAG", "BOTH"), f"bad ARM {ARM}"
assert SESSION_DIR, "SESSION_DIR required"

VEC = {"BASE": "vec_base", "TAG": "vec_base", "META": "vec_meta_v2", "BOTH": "vec_meta_v2"}[ARM]
GREP = {"BASE": "grep_base", "META": "grep_base", "TAG": "grep_tag_v2", "BOTH": "grep_tag_v2"}[ARM]


def charge(kind, arg):
    cf = os.path.join(SESSION_DIR, "calls.jsonl")
    n = 0
    if os.path.exists(cf):
        n = sum(1 for _ in open(cf))
    if n >= MAX_CALLS:
        print(json.dumps({"error": f"도구 호출 상한({MAX_CALLS}회) 도달. 지금까지의 정보로 최종 JSON을 출력하라."}, ensure_ascii=False))
        sys.exit(0)
    with open(cf, "a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": kind, "arg": arg[:200]}, ensure_ascii=False) + "\n")
    return n + 1


def vector(query):
    n = charge("vector_search", query)
    req = urllib.request.Request("http://localhost:11434/api/embed",
        data=json.dumps({"model": "bge-m3", "input": [query]}).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=60) as r:
        q = np.array(json.load(r)["embeddings"][0], dtype=np.float32)
    q /= (np.linalg.norm(q) or 1)
    vecs = np.load(os.path.join(OUT, f"{VEC}.npy"), mmap_mode="r")
    ids = json.load(open(os.path.join(OUT, f"{VEC}_ids.json")))["ids"]
    sims = vecs @ q
    top = np.argsort(-sims)[:10]
    chunks = _chunk_texts([ids[i] for i in top])
    res = [{"chunk_id": ids[i], "score": round(float(sims[i]), 4),
            "preview": chunks[ids[i]][:200]} for i in top]
    print(json.dumps({"calls_used": n, "results": res}, ensure_ascii=False))


def grep(pattern):
    n = charge("grep_search", pattern)
    target = os.path.join(OUT, f"{GREP}.jsonl")
    try:
        p = subprocess.run(["rg", "-i", "--no-line-number", pattern, target],
                           capture_output=True, text=True, timeout=30)
        lines = p.stdout.splitlines()
    except Exception as e:
        print(json.dumps({"calls_used": n, "error": str(e)}, ensure_ascii=False)); return
    total = len(lines)
    # 20개 초과 시 문서 전역 균등 분산 샘플 (파일 앞 편중 제거)
    if total > 20:
        step = total / 20.0
        picked = [lines[int(i * step)] for i in range(20)]
    else:
        picked = lines
    res = []
    for l in picked:
        try:
            row = json.loads(l)
        except Exception:
            continue
        g = row["g"]
        m = re.search(pattern, g, re.I)
        s = max(0, (m.start() if m else 0) - 60)
        res.append({"element_id": row["element_id"], "match": g[s:s + 180]})
    out = {"calls_used": n, "total_matches": total, "results": res}
    if total > 20:
        out["note"] = f"총 {total}건 매치 — 문서 전체에서 균등 샘플 20건만 표시. 패턴을 좁히면 정밀해진다."
    print(json.dumps(out, ensure_ascii=False))


def _unit_texts(uids):
    want = set(uids)
    outm = {}
    fname, key = ("chunks.jsonl", "chunk_id")
    if any(u.startswith("e") for u in want):
        pass
    for fname, key in (("chunks.jsonl", "chunk_id"), ("elements.jsonl", "element_id")):
        if not any((u.startswith("c") if key == "chunk_id" else u.startswith("e")) for u in want):
            continue
        for l in open(os.path.join(OUT, fname)):
            c = json.loads(l)
            if c[key] in want:
                outm[c[key]] = c["text"]
        if len(outm) == len(want):
            break
    return outm


def read_chunk(uid):
    n = charge("read_chunk", uid)
    m = _unit_texts([uid])
    if uid not in m:
        print(json.dumps({"calls_used": n, "error": "no such unit (c*=청크, e*=엘리먼트)"}, ensure_ascii=False)); return
    print(json.dumps({"calls_used": n, "id": uid, "text": m[uid][:4000]}, ensure_ascii=False))


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    {"vector": vector, "grep": grep, "read": read_chunk}[cmd](arg)

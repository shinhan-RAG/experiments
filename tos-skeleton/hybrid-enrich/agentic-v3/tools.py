#!/usr/bin/env python3
"""에이전트 검색 도구 CLI. Arm 격리 + 세션당 호출 상한 8회.

`hybrid-enrich/tools.py`의 250212 코퍼스 포트다.

원본과 같은 것 (계약)
    도구 3종(vector / grep / read) · **세션당 호출 상한 8회**(calls.jsonl 줄 수로 과금) ·
    arm 격리 · vector top-10 + preview 200자 · grep 20건 초과 시 문서 전역 균등 표본 20건 ·
    read 4,000자 절단 · 출력은 JSON 한 줄.

원본과 다른 것 (인용할 때 같이 말할 것)
    (a) 코퍼스. 원본은 260507판(13,780청크, sha 279f134b)이고 그 산출물
        (chunks.jsonl · elements.jsonl · grep_*.jsonl · vec_*.npy)이 저장소에 하나도
        남아 있지 않다. 여기서는 250212판(5,335청크, sha 40a471d5)을 쓴다.
    (b) grep 단위. 원본은 **엘리먼트**(표/단락/산식)였다. 250212판에는 엘리먼트 산출물이
        없어 **청크** 단위로 grep한다. 표 헤더·행라벨 채널이 없어지므로 어휘 경로가
        원본보다 불리하게 잡힌다.
    (c) 임베딩. 원본은 ollama `bge-m3`(:11434), 여기는 `dragonkue/BGE-m3-ko`(:8378).
        문서 벡터가 그 모델로 캐시돼 있어 질의도 같은 모델이어야 한다.
    (d) arm 하나가 벡터와 뷰 텍스트를 **동시에** 정한다(뷰 프리픽스가 그대로 임베딩된다).
        원본의 BOTH 구성에 해당하고, `A0_BASE`가 원본 BASE에 해당한다.

usage:
  python tools.py vector "질의"
  python tools.py grep "패턴"
  python tools.py read 1040aa492c::c00012
환경변수: ARM · SESSION_DIR · (선택) V4_OUT · BGE_URL · BGE_EMBED_MODEL
"""
import json, os, sys, re, urllib.request
import numpy as np

# Windows 기본 콘솔 인코딩(cp949)으로는 한국어 JSON을 파이프로 못 내보낸다.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

BASE = os.path.dirname(os.path.abspath(__file__))
OUT = os.environ.get("V4_OUT") or os.path.join(
    BASE, "..", "metajson-v6", "meta-search-v4", "out")
OUT = os.path.abspath(OUT)
MAX_CALLS = 8
CHUNKSET = "fixed600"

ARMS = ["A0_BASE", "J2_JSONPURE", "A6_HIER2", "V7_DENSE",
        "V9_QSURF", "V10_ASKS", "V11_ALIAS"]

ARM = os.environ.get("ARM")
SESSION_DIR = os.environ.get("SESSION_DIR")
assert ARM in ARMS, f"bad ARM {ARM}"
assert SESSION_DIR, "SESSION_DIR required"

VEC = f"vec_{CHUNKSET}_{ARM}"
GREP = f"view_{CHUNKSET}_{ARM}"
CHUNKS = f"chunks_{CHUNKSET}.jsonl"

BGE_URL = os.environ.get("BGE_URL", "http://localhost:8378").rstrip("/")
BGE_MODEL = os.environ.get("BGE_EMBED_MODEL", "dragonkue/BGE-m3-ko")


def charge(kind, arg):
    cf = os.path.join(SESSION_DIR, "calls.jsonl")
    n = 0
    if os.path.exists(cf):
        n = sum(1 for _ in open(cf, encoding="utf-8"))
    if n >= MAX_CALLS:
        print(json.dumps({"error": f"도구 호출 상한({MAX_CALLS}회) 도달. 지금까지의 정보로 최종 JSON을 출력하라."}, ensure_ascii=False))
        sys.exit(0)
    with open(cf, "a", encoding="utf-8") as f:
        f.write(json.dumps({"tool": kind, "arg": arg[:200]}, ensure_ascii=False) + "\n")
    return n + 1


def vector(query):
    n = charge("vector_search", query)
    # BGE 서버는 서버측에서 L2 정규화까지 끝내 준다(bge_server.py: normalize_embeddings=True).
    body = json.dumps({"model": BGE_MODEL, "input": [query]}, ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        f"{BGE_URL}/api/embed", data=body, method="POST",
        headers={"Content-Type": "application/json; charset=utf-8"})
    with urllib.request.urlopen(req, timeout=60) as r:
        q = np.array(json.load(r)["embeddings"][0], dtype=np.float32)
    q /= (np.linalg.norm(q) or 1)
    vecs = np.load(os.path.join(OUT, f"{VEC}.npy"), mmap_mode="r")
    ids = json.load(open(os.path.join(OUT, f"{VEC}_ids.json"), encoding="utf-8"))["ids"]
    sims = vecs @ q
    top = np.argsort(-sims)[:10]
    chunks = _chunk_texts([ids[i] for i in top])
    res = [{"chunk_id": ids[i], "score": round(float(sims[i]), 4),
            "preview": chunks.get(ids[i], "")[:200]} for i in top]
    print(json.dumps({"calls_used": n, "results": res}, ensure_ascii=False))


def grep(pattern):
    """원본은 `rg -i --no-line-number` 서브프로세스였다. 이 환경의 `rg`는 Git Bash
    셸 함수(claude.exe 심)라 서브프로세스에서 실행되지 않는다. 같은 것(대소문자 무시 ·
    줄 단위 매치 · 매치 줄 수 = total_matches)을 파이썬 `re`로 그대로 한다.
    원본도 매치 창을 뜰 때 이미 `re.search(pattern, g, re.I)`를 썼으므로 정규식
    방언은 그때부터 파이썬이었다."""
    n = charge("grep_search", pattern)
    target = os.path.join(OUT, f"{GREP}.jsonl")
    try:
        rx = re.compile(pattern, re.I)
    except re.error as e:
        print(json.dumps({"calls_used": n, "error": f"정규식 오류: {e}"}, ensure_ascii=False)); return
    lines = [l for l in open(target, encoding="utf-8") if rx.search(l)]
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
        g = row["text"]
        m = rx.search(g)
        s = max(0, (m.start() if m else 0) - 60)
        res.append({"chunk_id": row["chunk_id"], "match": g[s:s + 180]})
    out = {"calls_used": n, "total_matches": total, "results": res}
    if total > 20:
        out["note"] = f"총 {total}건 매치 — 문서 전체에서 균등 샘플 20건만 표시. 패턴을 좁히면 정밀해진다."
    print(json.dumps(out, ensure_ascii=False))


def _chunk_texts(uids):
    want = set(uids)
    outm = {}
    for l in open(os.path.join(OUT, CHUNKS), encoding="utf-8"):
        c = json.loads(l)
        if c["chunk_id"] in want:
            outm[c["chunk_id"]] = c["text"]
            if len(outm) == len(want):
                break
    return outm


def read_chunk(uid):
    n = charge("read_chunk", uid)
    m = _chunk_texts([uid])
    if uid not in m:
        print(json.dumps({"calls_used": n, "error": "no such chunk (형식: 1040aa492c::c00012)"}, ensure_ascii=False)); return
    print(json.dumps({"calls_used": n, "id": uid, "text": m[uid][:4000]}, ensure_ascii=False))


_STRUCT_CACHE = {}

def _norm(s):
    """Normalize for fuzzy matching: lowercase, strip whitespace, remove parens content."""
    s = s.strip().lower()
    s = re.sub(r'\s+', '', s)
    s = re.sub(r'\(.*?\)', '', s)
    return s

def _load_struct_index():
    """Lazy-load tags and bridge map. Returns (tags_list, bridge_dict) or (None, None)."""
    if _STRUCT_CACHE:
        return _STRUCT_CACHE.get("tags"), _STRUCT_CACHE.get("bridge")

    # Paths: try DOC_CONFIG env, fall back to defaults
    base = os.path.dirname(os.path.abspath(__file__))
    he_base = os.path.abspath(os.path.join(base, ".."))

    tags_path = os.path.join(he_base, "semtag-schema", "out", "v2ds", "tags_run1.jsonl")
    bridge_path = os.path.join(he_base, "out", "chunk_map_rev4_to_fixed600.json")

    if not os.path.exists(tags_path) or not os.path.exists(bridge_path):
        _STRUCT_CACHE["tags"] = None
        _STRUCT_CACHE["bridge"] = None
        return None, None

    tags = []
    for line in open(tags_path, encoding="utf-8"):
        row = json.loads(line)
        tags.append(row)

    bridge = json.load(open(bridge_path, encoding="utf-8"))

    _STRUCT_CACHE["tags"] = tags
    _STRUCT_CACHE["bridge"] = bridge
    return tags, bridge


def structured(filters_json):
    """Slot-based structured search over semtag element tags.

    Usage: python tools.py structured '{"contract":"암진단특약","role":"payment_trigger"}'

    Active slots: contract, subject, role, qualifier, schema
    Slot-to-slot: AND. Within one slot with comma-separated values: OR.
    Ranking: CLM (coordination level matching) — descending count of matched slots,
    ties broken by document order (line_start).
    Returns chunk_ids (bridged from element_ids).
    """
    n = charge("structured_search", filters_json)

    try:
        filters = json.loads(filters_json)
    except json.JSONDecodeError as e:
        print(json.dumps({"calls_used": n, "error": f"JSON 파싱 오류: {e}"}, ensure_ascii=False))
        return

    ACTIVE_SLOTS = {"contract", "subject", "role", "qualifier", "schema"}
    filters = {k: v for k, v in filters.items() if k in ACTIVE_SLOTS and v}

    if not filters:
        print(json.dumps({"calls_used": n, "error": "활성 슬롯(contract/subject/role/qualifier/schema) 중 하나 이상 지정 필요", "active_slots": list(ACTIVE_SLOTS)}, ensure_ascii=False))
        return

    # Load tags and bridge (lazy, cached)
    tags, bridge = _load_struct_index()
    if tags is None:
        print(json.dumps({"calls_used": n, "error": "structured_search 색인 파일 없음. semtag 산출물을 서버에서 반입하세요."}, ensure_ascii=False))
        return

    # Filter and rank
    results = []
    for tag_row in tags:
        matched_slots = []
        for slot, query_val in filters.items():
            tag_val = tag_row.get(slot, "")
            if not tag_val:
                continue
            # fuzzy: normalize whitespace, case-insensitive containment
            query_vals = [v.strip() for v in query_val.split(",")]
            tag_norm = _norm(tag_val)
            if any(_norm(qv) in tag_norm or tag_norm in _norm(qv) for qv in query_vals):
                matched_slots.append(slot)

        if matched_slots:
            results.append({
                "element_id": tag_row["element_id"],
                "matched_slots": matched_slots,
                "n_matched": len(matched_slots),
                "line_start": tag_row.get("line_start", 0),
            })

    # CLM ranking: by n_matched desc, then line_start asc (document order)
    results.sort(key=lambda x: (-x["n_matched"], x["line_start"]))
    results = results[:20]

    # Bridge to chunk_ids and add previews
    chunk_results = []
    preview_cids = []
    for r in results:
        eid = r["element_id"]
        cids = bridge.get(eid, [])
        for cid in cids:
            chunk_results.append({
                "chunk_id": cid,
                "element_id": eid,
                "matched_slots": r["matched_slots"],
                "n_matched": r["n_matched"],
            })
            preview_cids.append(cid)

    # Deduplicate by chunk_id, keep highest n_matched
    seen = {}
    for cr in chunk_results:
        cid = cr["chunk_id"]
        if cid not in seen or cr["n_matched"] > seen[cid]["n_matched"]:
            seen[cid] = cr
    chunk_results = sorted(seen.values(), key=lambda x: (-x["n_matched"], x["chunk_id"]))[:20]

    # Add previews
    preview_map = _chunk_texts([cr["chunk_id"] for cr in chunk_results])
    for cr in chunk_results:
        cr["preview"] = preview_map.get(cr["chunk_id"], "")[:200]

    out = {"calls_used": n, "total_element_hits": len(results), "results": chunk_results}
    if not chunk_results:
        out["hint"] = "0건 반환. 슬롯을 줄이거나 값을 넓혀 재시도하라."
    print(json.dumps(out, ensure_ascii=False))


if __name__ == "__main__":
    cmd, arg = sys.argv[1], sys.argv[2]
    {"vector": vector, "grep": grep, "read": read_chunk, "structured": structured}[cmd](arg)

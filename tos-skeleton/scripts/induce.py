#!/usr/bin/env python3
"""뼈대(skeleton) + 슬롯 귀납 및 라운드트립 검증.

입력: data/segmented/{vid}.json (버전 집합은 인자로)
방법:
  - 조 키 = (unit, no, title). 전 버전에 존재하는 조만 뼈대 대상, 나머지는 잔차(override).
  - 전 버전에서 텍스트 동일 → 고정 조. 다르면 토큰 열에 대해 반복 공통부분수열
    (iterative LCS)로 뼈대 추출, 뼈대 사이 갭 = 슬롯. 같은 (조, 슬롯번호) = 같은 키.
  - 라운드트립: 뼈대 + 버전별 슬롯 값으로 재조립 → 원 토큰열과 완전 일치해야 함.
출력: results/{tag}_skeleton.json, {tag}_values.json, {tag}_metrics.json
"""
import json
import sys
from difflib import SequenceMatcher
from pathlib import Path

BASE = Path("/Users/seyoung/workspace/braincrew/experiments/tos-skeleton")
SEG = BASE / "data" / "segmented"
RES = BASE / "results"


def load(vid):
    return json.loads((SEG / f"{vid}.json").read_text(encoding="utf-8"))


def article_map(units):
    m = {}
    for u in units:
        for a in u["articles"]:
            key = f"{u['unit']}\t{a['no']}\t{a['title']}"
            text = " ".join(a["text"]).split()
            if key in m:  # 같은 키 중복(헤더 유실로 쪼개진 조) → 이어붙임
                m[key] = m[key] + text
            else:
                m[key] = text
    return m


def common_subseq(a, b):
    sm = SequenceMatcher(None, a, b, autojunk=False)
    out = []
    for i, _, n in sm.get_matching_blocks():
        out.extend(a[i:i + n])
    return out


def gaps_against(skel, seq):
    """skel(부분수열)을 seq에 정렬해 갭(슬롯 값)들을 추출."""
    sm = SequenceMatcher(None, skel, seq, autojunk=False)
    blocks = [b for b in sm.get_matching_blocks() if b.size]
    matched = sum(b.size for b in blocks)
    if matched != len(skel):
        return None  # 정렬 실패 (뼈대가 온전히 배치되지 않음)
    gaps, pos_j, slots = [], 0, []
    for b in blocks:
        if b.b > pos_j:
            slots.append((b.a, seq[pos_j:b.b]))  # 뼈대 내 위치 a 앞에 오는 갭
        pos_j = b.b + b.size
    if pos_j < len(seq):
        slots.append((len(skel), seq[pos_j:]))
    return slots


def main(vids, tag):
    RES.mkdir(exist_ok=True)
    maps = {v: article_map(load(v)) for v in vids}
    all_keys = set().union(*[set(m) for m in maps.values()])
    shared = [k for k in all_keys if all(k in m for m in maps.values())]
    residual_articles = sorted(all_keys - set(shared))

    skeletons, values, stats = {}, {v: {} for v in vids}, {
        "versions": vids,
        "articles_total_union": len(all_keys),
        "articles_shared": len(shared),
        "articles_residual": len(residual_articles),
        "articles_identical": 0,
        "articles_slotted": 0,
        "articles_align_fail": 0,
        "roundtrip_ok": 0,
        "roundtrip_fail": 0,
        "tokens_total": 0,
        "tokens_skeleton": 0,
        "slot_count": 0,
    }

    for key in sorted(shared):
        seqs = [maps[v][key] for v in vids]
        stats["tokens_total"] += sum(map(len, seqs))
        if all(s == seqs[0] for s in seqs[1:]):
            skeletons[key] = {"fixed": True, "tokens": seqs[0]}
            stats["articles_identical"] += 1
            stats["tokens_skeleton"] += sum(map(len, seqs))
            continue
        skel = seqs[0]
        for s in seqs[1:]:
            skel = common_subseq(skel, s)
        ok = True
        per_ver = {}
        for v, s in zip(vids, seqs):
            slots = gaps_against(skel, s)
            if slots is None:
                ok = False
                break
            per_ver[v] = slots
        if not ok:
            # 정렬 실패 → 조 전체를 버전별 override로 저장 (무손실 유지)
            skeletons[key] = {"fixed": False, "align_fail": True}
            for v, s in zip(vids, seqs):
                values[v][key] = {"__override__": s}
            stats["articles_align_fail"] += 1
            continue
        skeletons[key] = {"fixed": False, "tokens": skel}
        slot_positions = sorted({pos for sl in per_ver.values() for pos, _ in sl})
        stats["slot_count"] += len(slot_positions)
        stats["articles_slotted"] += 1
        stats["tokens_skeleton"] += len(skel) * len(vids)
        for v in vids:
            d = {pos: [] for pos in slot_positions}
            for pos, toks in per_ver[v]:
                d[pos] = toks
            values[v][key] = {str(p): d[p] for p in slot_positions}

    # 라운드트립 검증
    for key in sorted(shared):
        sk = skeletons[key]
        for v in vids:
            orig = maps[v][key]
            if sk.get("fixed"):
                recon = sk["tokens"]
            elif sk.get("align_fail"):
                recon = values[v][key]["__override__"]
            else:
                skel = sk["tokens"]
                slots = {int(p): t for p, t in values[v][key].items()}
                recon, prev = [], 0
                for pos in sorted(slots):
                    recon += skel[prev:pos] + slots[pos]
                    prev = pos
                recon += skel[prev:]
            if recon == orig:
                stats["roundtrip_ok"] += 1
            else:
                stats["roundtrip_fail"] += 1

    stats["skeleton_coverage"] = round(
        stats["tokens_skeleton"] / stats["tokens_total"], 4)
    stats["residual_article_keys_sample"] = residual_articles[:40]

    (RES / f"{tag}_skeleton.json").write_text(
        json.dumps(skeletons, ensure_ascii=False), encoding="utf-8")
    (RES / f"{tag}_values.json").write_text(
        json.dumps(values, ensure_ascii=False), encoding="utf-8")
    (RES / f"{tag}_metrics.json").write_text(
        json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    for k, v in stats.items():
        if not isinstance(v, list):
            print(f"{k:28s} {v}")


if __name__ == "__main__":
    tag = sys.argv[1]
    main(sys.argv[2:], tag)

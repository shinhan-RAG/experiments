#!/usr/bin/env python3
"""페어 노출 게이트 비교 — 같은 qid 집합에 대한 두 exposure_check summary 의 문항별 차이."""
import argparse
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent


def load(tag):
    p = HERE / "out" / "exposure_check" / f"summary_{tag}.json"
    d = json.loads(p.read_text(encoding="utf-8"))
    return {r["qid"]: r for r in d["rows"] if "error" not in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-tag", required=True)
    ap.add_argument("--cand-tag", required=True)
    a = ap.parse_args()
    B, C = load(a.base_tag), load(a.cand_tag)
    qids = sorted(set(B) & set(C))
    gain = [q for q in qids if C[q]["exposed"] and not B[q]["exposed"]]
    loss = [q for q in qids if B[q]["exposed"] and not C[q]["exposed"]]
    rank_up, rank_dn = [], []
    for q in qids:
        rb = (B[q]["ranks"] or [999])[0]
        rc = (C[q]["ranks"] or [999])[0]
        if rc < rb:
            rank_up.append((q, rb, rc))
        elif rc > rb:
            rank_dn.append((q, rb, rc))
    print(json.dumps({
        "n": len(qids),
        "base_exposed": sum(B[q]["exposed"] for q in qids),
        "cand_exposed": sum(C[q]["exposed"] for q in qids),
        "exposure_gain": gain, "exposure_loss": loss,
        "first_rank_up": rank_up, "first_rank_down": rank_dn,
    }, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fail-closed preflight for retriever A/B first 25."""
import hashlib
import json
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent
OUT = BASE / "out"


def load_jsonl(name):
    return [json.loads(line) for line in (OUT / name).read_text(encoding="utf-8").splitlines()]


def sha(name):
    h = hashlib.sha256()
    with (OUT / name).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    gold = load_jsonl("train350_gold_repaired_v1.jsonl")
    manifest_path = OUT / "retriever_ab25_manifest.json"
    if not manifest_path.exists():
        manifest_path.write_text(json.dumps({
            "version": "filesearch-ab25-v2",
            "source": "train350_gold_repaired_v1.jsonl",
            "selection": "first 25 in frozen source order",
            "qids": [row["qid"] for row in gold[:25]],
            "arms": {
                "OLD_RET": ["vec_meta_v3_concat", "grep_tag_new_repaired_v1"],
                "NEW_RET": ["vec_meta_v3_concat", "grep_tag_new_repaired_v1"],
            },
        }, ensure_ascii=False, indent=2), encoding="utf-8")
    chunks = load_jsonl("chunks.jsonl")
    elements = load_jsonl("elements_repaired_v1.jsonl")
    chunk_ids = [row["chunk_id"] for row in chunks]
    element_ids = [row["element_id"] for row in elements]
    vector_ready = (OUT / "vec_meta_v3_concat_ids.json").exists() and (OUT / "vec_meta_v3_concat.npy").exists()
    vec_ids = json.loads((OUT / "vec_meta_v3_concat_ids.json").read_text())["ids"] if vector_ready else []
    matrix = np.load(OUT / "vec_meta_v3_concat.npy", mmap_mode="r") if vector_ready else np.empty((0, 1024))
    old_tag = load_jsonl("grep_tag_new_repaired_v1.jsonl")
    manifest = json.loads(manifest_path.read_text())
    by_qid = {row["qid"]: row for row in gold}
    selected = [by_qid[qid] for qid in manifest["qids"]]
    gc = {uid for row in selected for uid in row["gold_chunk_ids"]}
    ge = {uid for row in selected for uid in row["gold_element_ids"]}
    checks = {
        "gold_350": len(gold) == 350,
        "batch_25": len(selected) == 25 and len(set(manifest["qids"])) == 25,
        "vector_artifacts_ready": vector_ready,
        "vector_rows": matrix.shape == (len(chunk_ids), 1024),
        "vector_id_alignment": vec_ids == chunk_ids,
        "vector_finite": bool(np.isfinite(matrix).all()),
        "vector_nonzero": bool(np.all(np.linalg.norm(matrix, axis=1) > 0)),
        "tag_rows": len(old_tag) == len(element_ids),
        "tag_id_alignment": [row["element_id"] for row in old_tag] == element_ids,
        "batch_gold_chunk_coverage": gc <= set(chunk_ids),
        "batch_gold_element_coverage": ge <= set(element_ids),
        "vector_arm_invariant": manifest["arms"]["OLD_RET"][0] == manifest["arms"]["NEW_RET"][0],
    }
    result = {
        "status": "pass" if all(checks.values()) else "fail",
        "checks": checks,
        "counts": {"gold": len(gold), "batch": len(selected), "chunks": len(chunk_ids),
                   "elements": len(element_ids), "batch_gold_chunks": len(gc),
                   "batch_gold_elements": len(ge)},
        "hashes": {name: sha(name) for name in [
            "train350_gold_repaired_v1.jsonl", "retriever_ab25_manifest.json",
            "vec_meta_v3_concat_ids.json", "grep_tag_new_repaired_v1.jsonl",
            "element_tags_new_repaired_full_v1.jsonl"]},
    }
    (OUT / "preflight_retriever_ab25.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result["status"] == "pass" else 1)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Build audited v5 train/test gold without mutating the frozen v4 files."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
DOC = ROOT / "vector_search" / "doc" / "판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md"
SOURCE_DIR = ROOT / "out" / "noah"
OUT_DIR = HERE / "out" / "gold_v5"
CORRECTIONS = HERE / "qa_gold_corrections_v5.jsonl"

import sys

sys.path.insert(0, str(FS))
from units import Units  # noqa: E402


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]


def all_occurrences(text: str, quote: str) -> list[tuple[int, int]]:
    starts, offset = [], 0
    while True:
        pos = text.find(quote, offset)
        if pos < 0:
            return starts
        starts.append((pos, pos + len(quote)))
        offset = pos + 1


def groups_from_citations(text: str, citations: list[str], units: Units) -> list[dict]:
    groups = []
    for quote in citations:
        spans = all_occurrences(text, quote)
        if not spans:
            raise SystemExit(f"교정 인용문이 현재 코퍼스에 없습니다: {quote}")
        members, jos = [], set()
        for c0, c1 in spans:
            unit = units.jo_of_span(c0, c1)
            if not unit:
                continue
            members.append({"c0": c0, "c1": c1, "src": "v5_manual"})
            jos.add(unit["element_id"])
        if not members:
            raise SystemExit(f"교정 인용문이 조 인덱스에 매핑되지 않습니다: {quote}")
        groups.append({"c0": members[0]["c0"], "c1": members[0]["c1"], "members": members, "jos": sorted(jos)})
    return groups


def normalize_groups(groups: list[dict], units: Units) -> tuple[list[dict], int]:
    """Normalize legacy v3 singleton groups to the explicit v5 OR-member shape."""
    normalized, dropped = [], 0
    for source in groups:
        group = dict(source)
        members = list(group.get("members") or [])
        if not members and group.get("c0") is not None and group.get("c1") is not None:
            members = [{"c0": group["c0"], "c1": group["c1"], "src": group.get("src", "v3")}]
        reachable = []
        jos = {jo for jo in (group.get("jos") or []) if jo and jo != "?"}
        for member in members:
            unit = units.jo_of_span(member["c0"], member["c1"])
            if unit:
                reachable.append(member)
                jos.add(unit["element_id"])
        if reachable and jos:
            group.update(c0=reachable[0]["c0"], c1=reachable[0]["c1"], members=reachable, jos=sorted(jos))
            normalized.append(group)
        else:
            dropped += 1
    return normalized, dropped


def validate(rows: list[dict], split: str, units: Units) -> dict:
    qids = set()
    for row in rows:
        qid = row.get("qid")
        if not qid or qid in qids:
            raise SystemExit(f"{split}: qid 누락/중복: {qid}")
        qids.add(qid)
        if not str(row.get("q", "")).strip() or row.get("status") != "ok":
            raise SystemExit(f"{split}: 비정상 질문/status: {qid}")
        if row.get("c3_partial"):
            raise SystemExit(f"{split}: c3_partial 잔존: {qid}")
        if row.get("task_type") == "not_answerable":
            raise SystemExit(f"{split}: not_answerable 잔존: {qid}")
        groups = row.get("groups") or []
        if not groups or len(groups) > 10:
            raise SystemExit(f"{split}: evidence group 수 위반: {qid} groups={len(groups)}")
        for group in groups:
            if not group.get("members") or not group.get("jos"):
                raise SystemExit(f"{split}: 빈 gold group: {qid}")
            for member in group["members"]:
                if not (0 <= member["c0"] < member["c1"]):
                    raise SystemExit(f"{split}: 잘못된 span: {qid}")
                if not units.jo_of_span(member["c0"], member["c1"]):
                    raise SystemExit(f"{split}: 도달 불가능 span: {qid}")
    return {"n": len(rows), "n_partial": 0, "n_not_answerable": 0, "n_groups_gt_10": 0}


def main() -> None:
    text = DOC.read_text(encoding="utf-8")
    units = Units(str(FS / "out" / "elements_u2jo.jsonl"))
    corrections = read_jsonl(CORRECTIONS)
    correction_map = {entry["qid"]: entry for entry in corrections}
    if len(correction_map) != len(corrections):
        raise SystemExit("correction manifest에 중복 qid가 있습니다.")
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    audit = []
    outputs = {}
    seen_corrections = set()

    for split in ("train", "test"):
        source = SOURCE_DIR / f"gold_v4_{split}_full.jsonl"
        rows = []
        for original in read_jsonl(source):
            row = dict(original)
            correction = correction_map.get(row["qid"])
            if correction:
                if correction["split"] != split:
                    raise SystemExit(f"split 불일치: {row['qid']}")
                seen_corrections.add(row["qid"])
                old_q = row["q"]
                row.update(
                    q=correction["q"],
                    task_type=correction["task_type"],
                    groups=groups_from_citations(text, correction["citations"], units),
                    status="ok",
                    gold_src="v5_manual",
                    v5_revision="manual",
                )
                row.pop("c3_partial", None)
                audit.append({
                    "qid": row["qid"], "split": split, "kind": "manual", "old_q": old_q,
                    "new_q": row["q"], "reason": correction["reason"], "n_groups": len(row["groups"]),
                })
            elif row.get("c3_partial"):
                old_q = row["q"]
                row["q"] = old_q.rstrip() + " 현재 250212 판매약관에서 확인되는 근거 범위만 답해줘."
                row["gold_src"] = "v5_scoped_existing"
                row["v5_revision"] = "scope_to_current_corpus"
                row.pop("c3_partial", None)
                audit.append({
                    "qid": row["qid"], "split": split, "kind": "scope_to_current_corpus",
                    "old_q": old_q, "new_q": row["q"],
                    "reason": "구판 인용 일부가 현재 코퍼스에 없어 질문 범위를 매핑 성공 근거로 한정",
                    "n_groups": len(row["groups"]),
                })
            row["groups"], dropped = normalize_groups(row.get("groups") or [], units)
            if dropped:
                old_q = row["q"]
                if row.get("v5_revision") != "scope_to_current_corpus":
                    row["q"] = old_q.rstrip() + " 현재 250212 판매약관에서 확인되는 근거 범위만 답해줘."
                row["v5_revision"] = "scope_to_current_corpus"
                row["gold_src"] = "v5_scoped_existing"
                audit.append({
                    "qid": row["qid"], "split": split, "kind": "drop_unreachable_group",
                    "old_q": old_q, "new_q": row["q"],
                    "reason": f"현재 조 인덱스에 도달할 수 없는 evidence group {dropped}개 제거 후 질문 범위 한정",
                    "n_groups": len(row["groups"]),
                })
            rows.append(row)
        stats = validate(rows, split, units)
        out = OUT_DIR / f"gold_v5_{split}_full.jsonl"
        out.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")
        outputs[split] = {"path": str(out.resolve()), "sha256": digest(out), **stats}

    missing = sorted(set(correction_map) - seen_corrections)
    if missing:
        raise SystemExit(f"원본 gold에 없는 correction qid: {missing}")
    train_qids = {row["qid"] for row in read_jsonl(OUT_DIR / "gold_v5_train_full.jsonl")}
    test_qids = {row["qid"] for row in read_jsonl(OUT_DIR / "gold_v5_test_full.jsonl")}
    if train_qids & test_qids:
        raise SystemExit(f"train/test qid 중복: {sorted(train_qids & test_qids)}")

    audit_path = OUT_DIR / "correction_audit.json"
    audit_path.write_text(json.dumps(audit, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = {
        "version": "v5", "corpus": {"path": str(DOC.resolve()), "sha256": digest(DOC)},
        "corrections": {"path": str(CORRECTIONS.resolve()), "sha256": digest(CORRECTIONS), "n": len(audit)},
        "train": outputs["train"], "test": outputs["test"], "train_test_qid_overlap": 0,
    }
    manifest_path = OUT_DIR / "gold_v5_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

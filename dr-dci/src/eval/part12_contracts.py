"""Preflight contracts for Peter's Part 1 and Part 2 experiments.

The checks are deliberately model-free.  They prevent an expensive run when
the compared arms are duplicates, required augmentation artifacts are absent,
or a scale arm changes evidence/augmentation content in addition to corpus
size.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any


FEATURE_KEYS = ("taxonomy", "tags", "prefix", "metadata", "pull_backend")


def arm_signature(step: dict[str, Any]) -> tuple[tuple[str, Any], ...]:
    """Return only treatment variables; labels and descriptions are excluded."""
    return tuple((key, step.get(key, False)) for key in FEATURE_KEYS)


def duplicate_arms(steps: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_signature: dict[tuple[tuple[str, Any], ...], list[str]] = defaultdict(list)
    for step in steps:
        by_signature[arm_signature(step)].append(str(step["name"]))
    return [
        {"arms": names, "signature": dict(signature)}
        for signature, names in by_signature.items()
        if len(names) > 1
    ]


def _load_json(path: Path) -> Any:
    with path.open() as stream:
        return json.load(stream)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open() as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _artifact_path(data_dir: Path, dataset: str, size: int, feature: str,
                   tag_approach: str = "A") -> Path:
    size_key = f"{size // 1000}k"
    if feature == "tags":
        return data_dir / "tags" / dataset / f"approach_{tag_approach.lower()}" / f"{size_key}.json"
    return data_dir / feature / f"{dataset}_{size_key}.json"


def required_features(steps: list[dict[str, Any]]) -> dict[str, set[str]]:
    required = {"taxonomy": set(), "tags": set(), "prefix": set(), "metadata": set()}
    for step in steps:
        for feature in ("taxonomy", "prefix", "metadata"):
            if step.get(feature):
                required[feature].add("enabled")
        if step.get("tags"):
            required["tags"].add(str(step["tags"]).upper())
    return required


def audit_subsets(data_dir: Path, dataset: str, sizes: list[int]) -> dict[str, Any]:
    raw_dir = data_dir / "raw" / dataset
    qrels = _load_jsonl(raw_dir / "qrels.jsonl")
    positive_gold = {
        str(row["corpus-id"]) for row in qrels if float(row.get("score", 0)) >= 1
    }

    reports = []
    previous_ids: set[str] | None = None
    blockers = []
    subset_ids_by_size: dict[int, set[str]] = {}
    for size in sizes:
        path = data_dir / "subsets" / dataset / f"{size // 1000}k.json"
        if not path.exists():
            blockers.append(f"missing subset manifest: {path}")
            continue
        payload = _load_json(path)
        raw_ids = [str(value) for value in payload.get("doc_ids", [])]
        ids = set(raw_ids)
        subset_ids_by_size[size] = ids
        missing_gold = sorted(positive_gold - ids)
        nested = previous_ids is None or previous_ids <= ids
        report = {
            "size": size,
            "declared_size": payload.get("subset_size"),
            "actual_unique_size": len(ids),
            "duplicate_id_count": len(raw_ids) - len(ids),
            "positive_gold_count": len(positive_gold),
            "missing_positive_gold_count": len(missing_gold),
            "nested_with_previous": nested,
        }
        reports.append(report)
        if len(ids) != size:
            blockers.append(f"subset {size} has {len(ids)} unique documents")
        if report["duplicate_id_count"]:
            blockers.append(f"subset {size} contains duplicate document IDs")
        if missing_gold:
            blockers.append(f"subset {size} omits {len(missing_gold)} positive gold documents")
        if not nested:
            blockers.append(f"subset {size} is not a superset of the previous scale")
        previous_ids = ids

    return {
        "dataset": dataset,
        "subsets": reports,
        "blockers": blockers,
        "_ids": subset_ids_by_size,
    }


def _artifact_by_doc(path: Path, feature: str) -> dict[str, Any]:
    payload = _load_json(path)
    if feature != "tags":
        if not isinstance(payload, dict):
            raise ValueError(f"{feature} artifact must be an object: {path}")
        return {str(key): value for key, value in payload.items()}

    grouped: dict[str, list[Any]] = defaultdict(list)
    if not isinstance(payload, list):
        raise ValueError(f"tags artifact must be an array: {path}")
    for row in payload:
        grouped[str(row["doc_id"])].append(row)
    return dict(grouped)


def audit_augmentations(data_dir: Path, dataset: str, sizes: list[int],
                        steps: list[dict[str, Any]],
                        subset_ids: dict[int, set[str]]) -> dict[str, Any]:
    required = required_features(steps)
    reports = []
    blockers = []

    feature_variants: list[tuple[str, str]] = []
    for feature in ("taxonomy", "prefix", "metadata"):
        if required[feature]:
            feature_variants.append((feature, ""))
    for approach in sorted(required["tags"]):
        feature_variants.append(("tags", approach))

    for feature, variant in feature_variants:
        previous: dict[str, Any] | None = None
        previous_size: int | None = None
        for size in sizes:
            path = _artifact_path(data_dir, dataset, size, feature, variant or "A")
            if not path.exists():
                blockers.append(f"missing {feature} artifact: {path}")
                reports.append({"feature": feature, "variant": variant, "size": size,
                                "present": False})
                continue
            by_doc = _artifact_by_doc(path, feature)
            expected = subset_ids.get(size, set())
            missing = expected - set(by_doc)
            shared_changes = 0
            if previous is not None:
                shared = set(previous) & set(by_doc)
                shared_changes = sum(previous[doc_id] != by_doc[doc_id] for doc_id in shared)
                if shared_changes:
                    blockers.append(
                        f"{feature} changes {shared_changes} shared documents between "
                        f"{previous_size} and {size}; scale is not the only variable"
                    )
            reports.append({
                "feature": feature,
                "variant": variant,
                "size": size,
                "present": True,
                "document_count": len(by_doc),
                "missing_subset_document_count": len(missing),
                "changed_shared_document_count": shared_changes,
            })
            # Taxonomy/prefix/metadata are defined per document. Tags can be
            # legitimately absent when no element was annotated, so report but
            # do not block that case.
            if missing and feature != "tags":
                blockers.append(f"{feature} {size} misses {len(missing)} subset documents")
            previous = by_doc
            previous_size = size

    return {"requirements": {k: sorted(v) for k, v in required.items()},
            "artifacts": reports, "blockers": blockers}


def audit_part12(config: dict[str, Any], data_dir: Path, *,
                 step_names: set[str] | None = None,
                 sizes: list[int] | None = None) -> dict[str, Any]:
    part1 = config["parts"]["part1_stacking"]
    part2 = config["parts"]["part2_scaling"]
    dataset = str(part2["dataset"])
    if dataset != str(part1["dataset"]):
        raise ValueError("Part 1 and Part 2 must use the same dataset for this audit")

    sizes = sizes or [int(size) for size in part2["subsets"]]
    selected_steps = [
        step for step in part1["steps"]
        if step_names is None or str(step["name"]) in step_names
    ]
    if not selected_steps:
        raise ValueError("no Part 1 treatment arms selected")
    subset_report = audit_subsets(data_dir, dataset, sizes)
    duplicates = duplicate_arms(part1["steps"])
    augmentation_report = audit_augmentations(
        data_dir,
        dataset,
        sizes,
        selected_steps,
        subset_report.pop("_ids"),
    )
    blockers = [*subset_report["blockers"], *augmentation_report["blockers"]]
    return {
        "status": "blocked" if blockers else "ready",
        "selected_arms": [str(step["name"]) for step in selected_steps],
        "part1_duplicate_arms": duplicates,
        "subsets": subset_report,
        "augmentations": augmentation_report,
        "blockers": blockers,
        "warnings": (["Part 1 contains duplicate treatment arms"] if duplicates else []),
    }

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
    with path.open(encoding="utf-8") as stream:
        return json.load(stream)


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _iter_jsonl(path: Path):
    with path.open(encoding="utf-8") as stream:
        for line in stream:
            if line.strip():
                yield json.loads(line)


# aihub 계열(법률)은 data/raw/<ds>가 아닌 data/aihub/<variant>에 있고,
# 서브셋을 doc-id가 아닌 parent-id 목록({size}_parent_ids.json)으로 정의한다.
# run_experiment.dataset_dir / scripts.utils.dataset_dir 와 동일하게 유지한다.
AIHUB_DATASET_DIRS = {
    "aihub-full": ("aihub", "full"),
    "aihub-smoke20k": ("aihub", "smoke20k"),
}


def _dataset_dir(data_dir: Path, dataset: str) -> Path:
    if dataset in AIHUB_DATASET_DIRS:
        group, variant = AIHUB_DATASET_DIRS[dataset]
        return data_dir / group / variant
    return data_dir / "raw" / dataset


def _parent_subset_path(data_dir: Path, dataset: str, size: int) -> Path:
    return _dataset_dir(data_dir, dataset) / f"{size // 1000}k_parent_ids.json"


def _is_chunked(data_dir: Path, dataset: str, sizes: list[int]) -> bool:
    return any(_parent_subset_path(data_dir, dataset, size).exists() for size in sizes)


def _augmentation_expected_ids(
    data_dir: Path, dataset: str, sizes: list[int],
    subset_ids: dict[int, set[str]],
) -> dict[int, set[str]]:
    """Return per-size id sets at the *augmentation* unit.

    Augmentations (taxonomy/prefix/metadata/tags) are keyed by chunk ``_id``.
    For chunked (법률) corpora the subset is defined by parent ids, so expand
    each parent to its chunk ids via the corpus. BEIR corpora already carry
    doc(=chunk) ids, so the subset ids pass through unchanged.
    """
    if not _is_chunked(data_dir, dataset, sizes):
        return subset_ids
    chunks_by_parent: dict[str, set[str]] = defaultdict(set)
    for row in _iter_jsonl(_dataset_dir(data_dir, dataset) / "corpus.jsonl"):
        chunks_by_parent[str(row.get("parent_id", row["_id"]))].add(str(row["_id"]))
    expanded: dict[int, set[str]] = {}
    for size, parents in subset_ids.items():
        chunk_ids: set[str] = set()
        for parent in parents:
            chunk_ids |= chunks_by_parent.get(parent, set())
        expanded[size] = chunk_ids
    return expanded


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
    ds_dir = _dataset_dir(data_dir, dataset)
    qrels = _load_jsonl(ds_dir / "qrels.jsonl")
    positive_gold = {
        str(row["corpus-id"]) for row in qrels if float(row.get("score", 0)) >= 1
    }

    reports = []
    previous_ids: set[str] | None = None
    blockers = []
    subset_ids_by_size: dict[int, set[str]] = {}
    for size in sizes:
        parent_path = _parent_subset_path(data_dir, dataset, size)
        if parent_path.exists():
            # 청크형(법률) corpus: 서브셋은 parent 문서 ID의 평문 배열이다.
            raw_ids = [str(value) for value in _load_json(parent_path)]
            declared_size = None
        else:
            path = data_dir / "subsets" / dataset / f"{size // 1000}k.json"
            if not path.exists():
                blockers.append(f"missing subset manifest: {path}")
                continue
            payload = _load_json(path)
            raw_ids = [str(value) for value in payload.get("doc_ids", [])]
            declared_size = payload.get("subset_size")
        ids = set(raw_ids)
        subset_ids_by_size[size] = ids
        missing_gold = sorted(positive_gold - ids)
        nested = previous_ids is None or previous_ids <= ids
        report = {
            "size": size,
            "declared_size": declared_size,
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
    # 청크형 corpus는 서브셋이 parent 단위지만 augmentation은 chunk 단위이므로,
    # 커버리지 검사는 parent 서브셋을 chunk id 집합으로 펼쳐서 비교한다.
    aug_expected = _augmentation_expected_ids(
        data_dir, dataset, sizes, subset_report.pop("_ids")
    )
    augmentation_report = audit_augmentations(
        data_dir,
        dataset,
        sizes,
        selected_steps,
        aug_expected,
    )
    blockers = [*subset_report["blockers"], *augmentation_report["blockers"]]
    if duplicates:
        labels = ["/".join(item["arms"]) for item in duplicates]
        blockers.append(
            "duplicate treatment arms must not be executed independently: "
            + ", ".join(labels)
        )
    return {
        "status": "blocked" if blockers else "ready",
        "selected_arms": [str(step["name"]) for step in selected_steps],
        "part1_duplicate_arms": duplicates,
        "subsets": subset_report,
        "augmentations": augmentation_report,
        "blockers": blockers,
        "warnings": [],
    }

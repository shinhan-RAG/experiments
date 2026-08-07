"""Run deterministic EDA over downloaded experiment datasets."""

import argparse
import json
from pathlib import Path

from src.eval.dataset_eda import analyze_augmentations, analyze_dataset, render_markdown


BASE_DIR = Path(__file__).parent.parent


def main():
    parser = argparse.ArgumentParser(description="Analyze BEIR-style experiment datasets")
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["trec-covid", "fiqa", "ko-strategyqa"],
    )
    parser.add_argument("--data-dir", type=Path, default=BASE_DIR / "data")
    parser.add_argument("--output-dir", type=Path, default=BASE_DIR / "analysis" / "eda")
    parser.add_argument("--augmentation-size", default="20k")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for dataset in args.datasets:
        report = analyze_dataset(dataset, args.data_dir / "raw" / dataset)
        augmentations = analyze_augmentations(dataset, args.data_dir, args.augmentation_size)
        if augmentations:
            report["augmentations"] = augmentations
        (args.output_dir / f"{dataset}.json").write_text(
            json.dumps(report, ensure_ascii=False, indent=2)
        )
        (args.output_dir / f"{dataset}.md").write_text(render_markdown(report))
        summary[dataset] = {
            "documents": report["corpus"]["document_count"],
            "queries": report["queries"]["query_count"],
            "document_token_p50": report["corpus"]["tokens_regex"].get("p50"),
            "document_token_p95": report["corpus"]["tokens_regex"].get("p95"),
            "multi_gold_query_rate": report["relevance"]["multi_gold_query_rate"],
            "zero_overlap_query_rate": report["lexical_alignment"]["zero_token_overlap_query_rate"],
            "query_token_coverage_p50": report["lexical_alignment"]["query_token_coverage_in_gold"].get("p50"),
        }

    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2)
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

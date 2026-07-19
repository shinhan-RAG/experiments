"""
9단계: HuggingFace 업로드
- braincrew-dev/dr-dci-augmented에 전처리 데이터셋 업로드
"""

import json
from pathlib import Path
from huggingface_hub import HfApi, create_repo

DATA_DIR = Path(__file__).parent.parent / "data"
REPO_ID = "braincrew-dev/dr-dci-augmented"


def upload():
    print(f"=== Uploading to HuggingFace: {REPO_ID} ===")

    api = HfApi()

    # repo 생성 (이미 있으면 무시)
    try:
        create_repo(REPO_ID, repo_type="dataset", exist_ok=True)
    except Exception as e:
        print(f"  Repo already exists or error: {e}")

    # 업로드할 디렉토리 구조
    upload_dirs = [
        ("subsets", "subsets"),
        ("taxonomy", "taxonomy"),
        ("tags", "tags"),
        ("prefix", "prefix"),
        ("metadata", "metadata"),
        ("reference_answers", "reference_answers"),
    ]

    for local_name, remote_name in upload_dirs:
        local_path = DATA_DIR / local_name
        if not local_path.exists():
            print(f"  SKIP: {local_name} (not found)")
            continue

        print(f"  Uploading {local_name}...")
        api.upload_folder(
            folder_path=str(local_path),
            path_in_repo=remote_name,
            repo_id=REPO_ID,
            repo_type="dataset",
        )
        print(f"    Done: {remote_name}/")

    # README 생성 & 업로드
    readme = generate_readme()
    readme_path = DATA_DIR / "README.md"
    with open(readme_path, "w") as f:
        f.write(readme)

    api.upload_file(
        path_or_fileobj=str(readme_path),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
    )
    print("  Uploaded README.md")

    print(f"\n=== Upload complete: https://huggingface.co/datasets/{REPO_ID} ===")


def generate_readme() -> str:
    return """---
license: mit
task_categories:
  - text-retrieval
language:
  - en
  - ko
tags:
  - retrieval
  - dr-dci
  - augmented
size_categories:
  - 100K<n<1M
---

# DR-DCI Augmented Dataset

DR-DCI (arxiv 2606.14885) 고도화 실험용 전처리 데이터셋.

## Datasets

| Dataset | Corpus | Subsets | Queries | Language | Role |
|---------|--------|---------|---------|----------|------|
| TREC-COVID | 171K medical papers | 20K/50K/110K | 50 | EN | Main (scale degradation test) |
| FiQA | 57K financial docs | 20K | 50 sampled | EN | Cross-domain generalization |
| Ko-StrategyQA | 9.2K docs | 9.2K (full) | 50 sampled | EN | Cross-domain generalization |

## Structure

```
subsets/
  trec-covid/{20k,50k,110k}.json
  fiqa/{20k,sampled_queries}.json
  ko-strategyqa/{20k,sampled_queries}.json

taxonomy/
  {dataset}_{size}.json         # L1/L2/L3 per doc (dataset-specific categories)

tags/
  {dataset}/approach_{a,b,c}/{size}.json  # @el: semantic tags

prefix/
  {dataset}_{size}.json         # contextual prefix per doc

metadata/
  {dataset}_{size}.json         # structured metadata (schema-driven)

reference_answers/
  trec-covid.json               # LLM-generated answers from gold docs
```

## Augmentation Details

### Document Taxonomy (L1/L2/L3) — Dataset-specific
- **TREC-COVID**: Treatment, Diagnosis, Prevention, Mechanism, Epidemiology, Other
- **FiQA**: Investing, Personal_Finance, Markets, Banking, Tax, Insurance, Other
- **Ko-StrategyQA**: Science, History, Geography, Arts_Culture, Sports, Politics, Technology, Biology, Society, Other

### Semantic Tags (@el:)
- Elements: paragraph, table, list, figure
- Approach A (semantic role): definition, condition, procedure, example, exception, comparison, summary, evidence, criteria
- Approach B (question type): what, how, when, who, how_much, why, which, if
- Approach C (type only): no sub-tag

### Contextual Prefix
- 50-100 token summary prepended to each chunk
- Improves embedding quality for Pull retriever

### Metadata — Schema-driven per dataset
- **TREC-COVID**: study_type, year, population, entities (disease/drug/gene_protein/...)
- **FiQA**: asset_class, topic_type, time_horizon, entities (company/index/instrument/...)
- **Ko-StrategyQA**: topic_domain, document_type, time_period, entities (person/org/location/...)

## Generation

All augmentations generated with Qwen3-8B (vLLM, async 32 concurrent).

## Experiment Design

See: https://github.com/HwangIsAce/experiments/tree/dev/dr-dci
"""


if __name__ == "__main__":
    upload()

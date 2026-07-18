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

| Dataset | Corpus | Queries | Language | Role |
|---------|--------|---------|----------|------|
| TREC-COVID | 171K medical papers | 50 | EN | Main (high intra-domain similarity) |
| FiQA | 57K financial docs | 50 sampled | EN | Generalization |
| Ko-StrategyQA | 27.8K docs | 50 sampled | KO | Korean verification |

## Structure

```
subsets/
  trec-covid/
    10k.json          # doc_ids for 10K subset (gold + noise)
    50k.json          # 50K subset (10K ⊂ 50K)
    110k.json         # 110K subset (50K ⊂ 110K)
  fiqa/
    sampled_queries.json
  ko-strategyqa/
    sampled_queries.json

taxonomy/
  trec-covid_10k.json   # L1/L2/L3 classification per doc

tags/
  trec-covid/
    approach_a/10k.json  # @el:type/semantic_role
    approach_b/10k.json  # @el:type/question_type
    approach_c/10k.json  # @el:type only

prefix/
  trec-covid_10k.json   # contextual prefix per doc

metadata/
  trec-covid_10k.json   # topic, entities, year, study_type

reference_answers/
  trec-covid.json       # LLM-generated answers from gold docs
```

## Augmentation Details

### Document Taxonomy (L1/L2/L3)
- L1: Treatment, Diagnosis, Prevention, Mechanism, Epidemiology, Other
- L2: Sub-categories per L1
- L3: Free-form topic keyword

### Semantic Tags (@el:)
- Elements: paragraph, table, list, figure
- Approach A (semantic role): definition, condition, procedure, example, exception, comparison, summary, evidence, criteria
- Approach B (question type): what, how, when, who, how_much, why, which, if
- Approach C (type only): no sub-tag
- Small elements: caption/footnote → merged into parent; header/footer/page-number → removed

### Contextual Prefix
- 50-100 token summary prepended to each chunk
- Improves embedding quality for Pull retriever

### Metadata
- topic, entities, year, study_type, population
- Used for Pull pre-filtering

## Experiment Design

See: https://github.com/HwangIsAce/experiments/tree/main/dr-dci
"""


if __name__ == "__main__":
    upload()

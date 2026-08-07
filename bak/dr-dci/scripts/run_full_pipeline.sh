#!/bin/bash
# 전체 전처리 파이프라인 (TREC-COVID 20K 이후 나머지 전부)
# nohup bash scripts/run_full_pipeline.sh > logs/full_pipeline.log 2>&1 &

set -e
cd "$(dirname "$0")/.."
mkdir -p logs

echo "$(date) === Full Pipeline Start ==="

# ============================================================
# 1. TREC-COVID 50K, 110K
# ============================================================
echo "$(date) === TREC-COVID 50K/110K ==="

echo "$(date) [trec-covid] taxonomy --all"
python -u scripts/build_taxonomy.py --all >> logs/taxonomy_all.log 2>&1

echo "$(date) [trec-covid] tags --all"
python -u scripts/build_tags.py --all >> logs/tags_all.log 2>&1

echo "$(date) [trec-covid] prefix --all"
python -u scripts/build_prefix.py --all >> logs/prefix_all.log 2>&1

echo "$(date) [trec-covid] metadata --all"
python -u scripts/build_metadata.py --all >> logs/metadata_all.log 2>&1

# ============================================================
# 2. FiQA 다운로드 + 서브셋 + 전처리
# ============================================================
echo "$(date) === FiQA ==="

echo "$(date) [fiqa] download"
python -u scripts/download_datasets.py fiqa >> logs/download_fiqa.log 2>&1

echo "$(date) [fiqa] subsets"
python -u scripts/build_subsets.py fiqa >> logs/subsets_fiqa.log 2>&1

echo "$(date) [fiqa] taxonomy"
python -u scripts/build_taxonomy.py fiqa >> logs/taxonomy_fiqa.log 2>&1

echo "$(date) [fiqa] tags"
python -u scripts/build_tags.py fiqa >> logs/tags_fiqa.log 2>&1

echo "$(date) [fiqa] prefix"
python -u scripts/build_prefix.py fiqa >> logs/prefix_fiqa.log 2>&1

echo "$(date) [fiqa] metadata"
python -u scripts/build_metadata.py fiqa >> logs/metadata_fiqa.log 2>&1

# ============================================================
# 3. Ko-StrategyQA 다운로드 + 서브셋 + 전처리
# ============================================================
echo "$(date) === Ko-StrategyQA ==="

echo "$(date) [ko-strategyqa] download"
python -u scripts/download_datasets.py ko-strategyqa >> logs/download_kosqa.log 2>&1

echo "$(date) [ko-strategyqa] subsets"
python -u scripts/build_subsets.py ko-strategyqa >> logs/subsets_kosqa.log 2>&1

echo "$(date) [ko-strategyqa] taxonomy"
python -u scripts/build_taxonomy.py ko-strategyqa >> logs/taxonomy_kosqa.log 2>&1

echo "$(date) [ko-strategyqa] tags"
python -u scripts/build_tags.py ko-strategyqa >> logs/tags_kosqa.log 2>&1

echo "$(date) [ko-strategyqa] prefix"
python -u scripts/build_prefix.py ko-strategyqa >> logs/prefix_kosqa.log 2>&1

echo "$(date) [ko-strategyqa] metadata"
python -u scripts/build_metadata.py ko-strategyqa >> logs/metadata_kosqa.log 2>&1

# ============================================================
echo "$(date) === Full Pipeline Complete ==="

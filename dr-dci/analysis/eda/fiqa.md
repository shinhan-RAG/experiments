# fiqa EDA

## Dataset Size

- Documents: 57,638
- Queries: 6,648
- Queries with positive gold: 648
- Positive qrels: 1,706
- Unique gold documents: 1,706

## Document Shape

- Regex-token p50/p95/p99/max: 95.0 / 380.0 / 664.6 / 3,056.0
- Documents <=256 tokens: 88.09%
- Documents <=4,000 characters: 99.23%
- Contains Korean: 0.00%
- Contains Latin: 99.93%

## Relevance Structure

- Multi-gold query rate: 66.05%
- Gold documents/query p50/p95/max: 2.0 / 7.0 / 15.0
- Missing gold documents: 0

## Lexical Alignment

- Evaluated queries: 648 (positive-gold queries only)
- Query-token coverage in gold p50: 66.67%
- Zero token-overlap query rate: 0.62%
- Rare query-term ratio p50: 0.00%

Token counts use a deterministic Unicode regex and are diagnostic values, not model-tokenizer lengths.

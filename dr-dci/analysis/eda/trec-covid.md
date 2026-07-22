# trec-covid EDA

## Dataset Size

- Documents: 171,332
- Queries: 50
- Queries with positive gold: 50
- Positive qrels: 24,673
- Unique gold documents: 17,537

## Document Shape

- Regex-token p50/p95/p99/max: 176.0 / 355.0 / 465.0 / 19,387.0
- Documents <=256 tokens: 73.66%
- Documents <=4,000 characters: 99.76%
- Contains Korean: 0.00%
- Contains Latin: 99.99%

## Relevance Structure

- Multi-gold query rate: 100.00%
- Gold documents/query p50/p95/max: 478.0 / 863.9 / 1,266.0
- Missing gold documents: 0

## Lexical Alignment

- Evaluated queries: 50 (positive-gold queries only)
- Query-token coverage in gold p50: 100.00%
- Zero token-overlap query rate: 0.00%
- Rare query-term ratio p50: 0.00%

Token counts use a deterministic Unicode regex and are diagnostic values, not model-tokenizer lengths.

# ko-strategyqa EDA

## Dataset Size

- Documents: 9,251
- Queries: 2,833
- Queries with positive gold: 592
- Positive qrels: 1,145
- Unique gold documents: 1,077

## Document Shape

- Regex-token p50/p95/p99/max: 64.0 / 150.0 / 266.0 / 1,121.0
- Documents <=256 tokens: 98.91%
- Documents <=4,000 characters: 99.96%
- Contains Korean: 99.96%
- Contains Latin: 99.96%

## Relevance Structure

- Multi-gold query rate: 64.36%
- Gold documents/query p50/p95/max: 2.0 / 3.0 / 7.0
- Missing gold documents: 0

## Lexical Alignment

- Evaluated queries: 592 (positive-gold queries only)
- Query-token coverage in gold p50: 25.00%
- Zero token-overlap query rate: 25.84%
- Rare query-term ratio p50: 50.00%

Token counts use a deterministic Unicode regex and are diagnostic values, not model-tokenizer lengths.

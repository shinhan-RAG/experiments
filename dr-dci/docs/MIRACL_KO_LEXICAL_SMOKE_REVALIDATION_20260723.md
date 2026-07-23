# MIRACL Korean lexical smoke — result-contract revalidation

## Scope

This report revalidates existing raw rankings without running Anserini again. It remains a standalone Korean passage-retrieval plumbing smoke, not a taxonomy, Agentic RAG, focused Part 1/2, insurance-domain, or physical-document operations result.

## Provenance

- Smoke-run Git commit: `39ff3d21dae821170c1b77e0572c92b769999adc`.
- Original report Git commit: `ee916d3f4e8858521d83df5388021f50f1b2ede6`.
- Original report SHA-256: `efd61f7d2778e04e739e7139e9883002eda7eaafb14afb856b0ee0cb1d999d45`.
- Historical execution contract SHA-256: `0dca1cb4decfe79d9fb28a121418e83e809760c7aac5d03cfa03d38c985b88ff`.
- Historical contract covers runner, lexical result validation, paired-bootstrap code, MIRACL preparation validator, backend config, Docker recipe, revision lock, and current subset-manifest hash.

## Contract checks

- Every raw per-query ranking was rescored against the verified dev qrels and its scale corpus; stored raw-row and aggregate passage metrics matched.
- Raw-result, runtime, runner, validation-code, Docker-recipe, query/qrel, subset, and subset-manifest hashes matched the historical source/data contract.
- No Anserini command, taxonomy generation, Agent/LLM call, embedding call, or focused Part 1/2 run occurred.

## Corrected latency handling

The historical rows repeat one batch elapsed/query descriptive mean. It is retained as batch metadata below, but is excluded from paired bootstrap and has no query-level CI.
- 20000: batch `4.814675s`; batch/query mean `0.022604s`; paired latency CI `not reported`.
- 50000: batch `2.077328s`; batch/query mean `0.009753s`; paired latency CI `not reported`.
- 110000: batch `2.349579s`; batch/query mean `0.011031s`; paired latency CI `not reported`.

## Revalidated retrieval comparison

- 110K−20K passage Recall@20: `-0.073730`; 95% CI [`-0.102176`, `-0.047463`]; paired queries `213`.
- The paired section includes retrieval metrics only; no latency metric appears.

## Interpretation boundary

The fixture fixes all judged passages and increases unjudged distractors. It supports controlled distractor-scaling plumbing only, not natural-corpus growth, taxonomy effectiveness, Agentic RAG effectiveness, insurance-domain transfer, or Shinhan 110K physical-document performance.

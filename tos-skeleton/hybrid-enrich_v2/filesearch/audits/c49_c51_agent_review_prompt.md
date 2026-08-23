You are an independent reviewer. Review only the files and artifacts listed below. Do not modify or execute experiments. Do not trust the author's interpretation; reproduce agent metrics from raw results and inspect traces/code directly.

Objective: determine whether C49-C51 improve only the Semantic Tag/data/reference-evidence side of a fixed hybrid search, without changing Vector/Meta retrieval, using a non-agent LLM reranker, or reading QA/Gold/results during index construction.

Code/data to inspect:
- filesearch/build_tags_u5_reference_graph.py
- filesearch/test_build_tags_u5_reference_graph.py
- filesearch/out/tags_u5_reference_graph_v12_stats.json
- filesearch/out/tags_u5_reference_graph_v13_stats.json
- vector_search/noah/0819/agent_tools.py
- vector_search/noah/0819/arms.json (C48, C49, C50, C51)
- vector_search/noah/0819/test_experiment_arms.py

Agent runs to inspect:
- vector_search/noah/0819/out/host_agent/gpt56luna_medium_c48c49_devtest10x1_goldtestv1_20260823
- vector_search/noah/0819/out/host_agent/gpt56luna_medium_c45c50_colloquial1x3_goldreviewedv1_20260823
- vector_search/noah/0819/out/host_agent/gpt56luna_medium_c50c51_tablecatalog2x3_goldtestv1_20260823

Questions:
1. Are C48/C49/C50/C51 arm differences accurately isolated? Confirm whether the Vector/Meta lane, agent model/prompt/runner, scoring, and Gold are fixed within each paired run.
2. Does the graph/index builder import, read, or derive features from QA, Gold, qids, result files, or agent output? Search for direct and indirect leakage.
3. Is v1.2 per-local-section 2-hop mapping provenance-safe, fail-closed on ambiguity, and document scoped? Identify false links or namespace collisions.
4. Is v1.3 table catalog document scoped and based only on exact table headings? Check region start/end logic, repeated-page headings, split headings, duplicate titles, and all submit-compatible JO variants. Identify overlong or cross-table regions.
5. Is C50 colloquial normalization generic and fail-closed? Find false positives such as contract-state wording, copied activity names, or insurance-domain hardcoding.
6. Does C51 leave the base hybrid results unchanged and expose exact-title table evidence only as additive Semantic metadata? Is this effectively a non-LLM reranker/fusion despite the claim otherwise?
7. Recompute agent metrics from results.jsonl. Distinguish mechanism/dev evidence from holdout/generalization. Pay particular attention to C50 baseline protocol_errors=5 in the C51 run and whether gains remain causally supported by traces.
8. Verify whether C51 q0309/q0310 submitted the exact catalog JOs in every rep. Explain failures/gains using agent traces, not deterministic scores.
9. Identify generalization limitations: current corpus document count, consumed development questions, and whether any fresh feature-active holdout exists.
10. Give a verdict for each of C49, C50, C51: reject / mechanism-only / eligible for fresh holdout / adopt. State minimal next action and exact safety gates.

Output in Korean. Separate verified facts, inference, limitations, and verdict. Report agent metrics only; deterministic retrieval numbers may be used internally but do not headline them as experiment performance.

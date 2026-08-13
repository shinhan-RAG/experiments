#!/usr/bin/env python3
"""Retriever-level paired evaluation on the old dev questions that are in Train.

No Test-150 question is loaded.  Historical agent results are recomputed on the
same subset for context; direct retrieval metrics are intentionally kept
separate from agent-final metrics.
"""
from __future__ import annotations

import collections
import json
import math
import re
import urllib.request
from pathlib import Path

import numpy as np

BASE = Path(__file__).resolve().parent.parent
OUT = BASE / "out"
MODEL = "bge-m3"
FIELD_WEIGHTS = {
    "contract": 3.0, "subject": 3.0, "role": 2.5, "locator": 2.0,
    "qualifier": 1.7, "reference": 1.2, "schema": 0.8, "tag": 0.8,
    "raw": 0.35,
}
STOP = {
    "무엇", "어떻게", "인가요", "있나요", "되나요", "경우", "대한", "관련", "해당",
    "알려", "주세요", "에서는", "으로", "하는", "되는", "정한", "보험", "특약",
}
QUERY_ROLE_RULES = [
    (r"지급하지|보상하지|면책|제외|예외", "exclusion_exception"),
    (r"납입.?면제", "premium_waiver"),
    (r"지급사유|지급.?조건|언제.?지급", "payment_trigger"),
    (r"지급금액|얼마|지급률|산정|계산", "payment_amount"),
    (r"한도|몇.?회|횟수|며칠", "limit_frequency"),
    (r"보장개시|책임개시|대기기간|감액기간|언제부터|기간", "timing_period"),
    (r"정의|뜻|무엇을.?말", "definition"),
    (r"진단확정|판정|기준", "criteria_rule"),
    (r"갱신|해지|소멸|무효|환급", "contract_lifecycle"),
    (r"청구|서류|절차", "claim_procedure"),
    (r"분류코드|질병코드|수가코드|부표", "code_reference"),
]


def load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def embed(texts: list[str]) -> np.ndarray:
    request = urllib.request.Request(
        "http://localhost:11434/api/embed",
        data=json.dumps({"model": MODEL, "input": texts}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        result = np.asarray(json.load(response)["embeddings"], dtype=np.float32)
    norms = np.linalg.norm(result, axis=1, keepdims=True)
    norms[norms == 0] = 1
    return result / norms


def tokens(text: str) -> set[str]:
    text = text.lower()
    values: set[str] = set()
    for word in re.findall(r"[가-힣]+|[a-z]+|\d+(?:\.\d+)?", text):
        if word in STOP or len(word) < 2:
            continue
        values.add(word)
        if re.fullmatch(r"[가-힣]+", word) and len(word) >= 4:
            values.update(word[index:index + 3] for index in range(len(word) - 2))
    return values


def lexical_index(rows, tagged: bool):
    inverted: dict[str, dict[str, float]] = collections.defaultdict(dict)
    for row in rows:
        eid = row["element_id"]
        fields = row["fields"]
        names = list(FIELD_WEIGHTS) if tagged else ["raw"]
        for name in names:
            weight = FIELD_WEIGHTS[name]
            for token in tokens(fields.get(name, "")):
                inverted[token][eid] = max(inverted[token].get(eid, 0.0), weight)
    n_docs = len(rows)
    return inverted, {token: math.log(1 + n_docs / len(postings)) for token, postings in inverted.items()}


def lexical_search(question: str, inverted, idf, top_k=20):
    scores: dict[str, float] = collections.defaultdict(float)
    for token in tokens(question):
        for eid, weight in inverted.get(token, {}).items():
            scores[eid] += idf[token] * weight
    return [eid for eid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


def normalize_key(text: str) -> str:
    text = re.sub(r"\(무배당[^)]*\)|해약환급금미지급형|간편가입|간편", "", text)
    return re.sub(r"[^가-힣a-z0-9]", "", text.lower())


def field_aware_search(question: str, inverted, idf, rows_by_eid, top_k=20):
    query_tokens = tokens(question)
    query_roles = [role for pattern, role in QUERY_ROLE_RULES if re.search(pattern, question)]
    query_tokens.update(tokens(" ".join(query_roles)))

    # Exact lexical namespace selection when the user names a rider.
    contract_mentions = re.findall(r"[가-힣A-Za-z0-9·()\[\]\-]{3,100}?특약", question)
    normalized_mentions = [normalize_key(value) for value in contract_mentions]
    allowed = None
    if normalized_mentions:
        matched = {
            eid for eid, row in rows_by_eid.items()
            if any(mention and (mention in normalize_key(row["fields"]["contract"]) or normalize_key(row["fields"]["contract"]) in mention)
                   for mention in normalized_mentions)
        }
        if matched:
            allowed = matched

    scores: dict[str, float] = collections.defaultdict(float)
    for token in query_tokens:
        for eid, weight in inverted.get(token, {}).items():
            if allowed is None or eid in allowed:
                scores[eid] += idf[token] * weight
    # Role is a controlled exact field, so reward agreement after namespace filtering.
    if query_roles:
        for eid in list(scores):
            element_roles = rows_by_eid[eid]["fields"]["role"]
            scores[eid] += 8.0 * sum(role in element_roles for role in query_roles)
    return [eid for eid, _ in sorted(scores.items(), key=lambda item: (-item[1], item[0]))[:top_k]]


UNIT_SPANS: dict[str, tuple[int, int]] = {}


def first_rank(ids: list[str], gold_ids: set[str], gold_spans: list[list[int]] | None = None) -> int | None:
    for rank, uid in enumerate(ids, 1):
        span = UNIT_SPANS.get(uid)
        overlaps = span and gold_spans and any(span[0] < end and span[1] > start for start, end in gold_spans)
        if uid in gold_ids or overlaps:
            return rank
    return None


def summarize(ranks: dict[str, int | None]):
    n = len(ranks)
    return {
        "n": n,
        "recall@1": sum(rank == 1 for rank in ranks.values()) / n,
        "recall@5": sum(rank is not None and rank <= 5 for rank in ranks.values()) / n,
        "recall@10": sum(rank is not None and rank <= 10 for rank in ranks.values()) / n,
        "recall@20": sum(rank is not None and rank <= 20 for rank in ranks.values()) / n,
        "mrr@10": sum((1 / rank) if rank and rank <= 10 else 0 for rank in ranks.values()) / n,
    }


def main():
    for name, key in (("chunks.jsonl", "chunk_id"), ("elements.jsonl", "element_id")):
        for row in load_jsonl(OUT / name):
            UNIT_SPANS[row[key]] = (row["char_start"], row["char_end"])
    manifest = load_jsonl(OUT / "qa500_train_test_split_manifest.jsonl")
    train_field_nos = {row["no"] for row in manifest if row["split"] == "train" and row["origin"] == "field"}
    gold_all = load_jsonl(OUT / "qa100_gold.jsonl")
    gold = {row["qid"]: row for row in gold_all if row["qid"] in train_field_nos}
    assert len(gold) == 72, len(gold)

    questions = [gold[qid]["question"] for qid in sorted(gold)]
    qids = sorted(gold)
    qvecs = embed(questions)

    vector_rankings = {}
    for arm, stem in {
        "VECTOR_RAW": "vec_base",
        "VECTOR_META_CONCAT_V2": "vec_meta_v2",
        "VECTOR_META_CONCAT_V3": "vec_meta_v3_concat",
    }.items():
        matrix = np.load(OUT / f"{stem}.npy", mmap_mode="r")
        ids = json.loads((OUT / f"{stem}_ids.json").read_text())["ids"]
        ranks = {}
        rows = []
        for qid, qvec in zip(qids, qvecs, strict=True):
            top = np.argsort(-(matrix @ qvec))[:20]
            retrieved = [ids[index] for index in top]
            rank = first_rank(retrieved, set(gold[qid]["gold_chunk_ids"]), gold[qid]["gold_spans"])
            ranks[qid] = rank
            rows.append({"qid": qid, "arm": arm, "rank": rank, "ids": retrieved})
        vector_rankings[arm] = (ranks, rows)

    lex_rows = load_jsonl(OUT / "lex_tag_v3.jsonl")
    lex_by_eid = {row["element_id"]: row for row in lex_rows}
    raw_index, raw_idf = lexical_index(lex_rows, tagged=False)
    tag_index, tag_idf = lexical_index(lex_rows, tagged=True)
    lexical_rankings = {}
    for arm, index, idf in (
        ("LEXICAL_RAW_FIELDED", raw_index, raw_idf),
        ("LEXICAL_TAG_V3_FIELDED", tag_index, tag_idf),
    ):
        ranks = {}
        rows = []
        for qid in qids:
            retrieved = lexical_search(gold[qid]["question"], index, idf)
            rank = first_rank(retrieved, set(gold[qid]["gold_element_ids"]), gold[qid]["gold_spans"])
            ranks[qid] = rank
            rows.append({"qid": qid, "arm": arm, "rank": rank, "ids": retrieved})
        lexical_rankings[arm] = (ranks, rows)

    arm = "LEXICAL_TAG_V3_SCHEMA_AWARE"
    ranks = {}
    rows = []
    for qid in qids:
        retrieved = field_aware_search(gold[qid]["question"], tag_index, tag_idf, lex_by_eid)
        rank = first_rank(retrieved, set(gold[qid]["gold_element_ids"]), gold[qid]["gold_spans"])
        ranks[qid] = rank
        rows.append({"qid": qid, "arm": arm, "rank": rank, "ids": retrieved})
    lexical_rankings[arm] = (ranks, rows)

    historical = load_jsonl(OUT / "retrieval_v2_final.jsonl")
    historical_rankings = {}
    for arm in ("BASE", "META", "TAG", "BOTH"):
        by_qid = {row["qid"]: row for row in historical if row["arm"] == arm and row["qid"] in gold}
        historical_rankings[f"YESTERDAY_AGENT_{arm}"] = {
            qid: first_rank(by_qid.get(qid, {}).get("ranked_chunk_ids", []), set(gold[qid]["gold_element_ids"]), gold[qid]["gold_spans"])
            for qid in qids
        }

    # Candidate availability for an agent with independently callable tools.
    coverage = {}
    combos = {
        "TOOLS_RAW": ("VECTOR_RAW", "LEXICAL_RAW_FIELDED"),
        "TOOLS_META_V3": ("VECTOR_META_CONCAT_V3", "LEXICAL_RAW_FIELDED"),
        "TOOLS_TAG_V3": ("VECTOR_RAW", "LEXICAL_TAG_V3_FIELDED"),
        "TOOLS_BOTH_V3": ("VECTOR_META_CONCAT_V3", "LEXICAL_TAG_V3_FIELDED"),
        "TOOLS_TAG_V3_SCHEMA_AWARE": ("VECTOR_RAW", "LEXICAL_TAG_V3_SCHEMA_AWARE"),
        "TOOLS_BOTH_V3_SCHEMA_AWARE": ("VECTOR_META_CONCAT_V3", "LEXICAL_TAG_V3_SCHEMA_AWARE"),
    }
    for label, (vector_arm, lexical_arm) in combos.items():
        vranks = vector_rankings[vector_arm][0]
        lranks = lexical_rankings[lexical_arm][0]
        coverage[label] = {
            "n": len(qids),
            "candidate_recall_vector@10_or_lexical@20": sum(
                (vranks[qid] is not None and vranks[qid] <= 10) or
                (lranks[qid] is not None and lranks[qid] <= 20)
                for qid in qids
            ) / len(qids),
            "vector_unique_hits": sum(
                vranks[qid] is not None and vranks[qid] <= 10 and not (lranks[qid] is not None and lranks[qid] <= 20)
                for qid in qids
            ),
            "lexical_unique_hits": sum(
                lranks[qid] is not None and lranks[qid] <= 20 and not (vranks[qid] is not None and vranks[qid] <= 10)
                for qid in qids
            ),
        }

    result = {
        "scope": {"document_count": 1, "split": "train_only", "n": len(qids), "qids": qids},
        "historical_agent_metrics_same_72": {arm: summarize(ranks) for arm, ranks in historical_rankings.items()},
        "direct_vector_metrics": {arm: summarize(data[0]) for arm, data in vector_rankings.items()},
        "direct_lexical_metrics": {arm: summarize(data[0]) for arm, data in lexical_rankings.items()},
        "independent_tool_candidate_coverage": coverage,
        "notes": [
            "Direct retriever metrics and historical agent-final metrics are different levels and are not compared as the same metric.",
            "Candidate coverage uses independently callable tools; it is not a fixed keyword/vector fusion ranking.",
            "Test-150 was not loaded.",
        ],
    }
    (OUT / "eval_retrieval_fields_v3.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    with (OUT / "retrieval_fields_v3_rankings.jsonl").open("w", encoding="utf-8") as handle:
        for _, rows in vector_rankings.values():
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        for _, rows in lexical_rankings.values():
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

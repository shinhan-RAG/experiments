#!/usr/bin/env python3
"""Build an Index/Frontmatter-independent document-search QA candidate set.

Only raw Markdown paths and contents are used. Generated Index and Frontmatter
artifacts are intentionally not read. Each item has one target document and raw
evidence. Content items remain candidates until corpus-wide relevance review.
"""
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path


RAW = Path("/Users/seyoung/Downloads/parsed_md")
HERE = Path(__file__).parent
OUT = HERE / "out"
DEST = OUT / "document_search_qa_500_candidate_v1.jsonl"
MANIFEST = OUT / "document_search_qa_500_candidate_v1_manifest.json"
REVIEW = OUT / "document_search_qa_500_candidate_v1_review.csv"
VERSION = "document-search-qa-candidate-v1"

BAD_HEADINGS = {
    "목차", "가입자 유의사항", "주요내용 요약서", "주요보험용어 해설",
    "보험약관", "보통보험약관", "특약약관", "별표", "부표",
}
CONTAMINATION = (
    "preserve the original", "never translate", "final translation",
    "main article is about", "if any part is unreadable", "add commentary",
)
DOC_TYPE_PATTERNS = (
    ("사업방법서", ("사업방법", "사방서", "사방")),
    ("상품요약서", ("상품요약", "요약서")),
    ("공시약관", ("공시",)),
    ("판매약관", ("판매약관", "약관")),
)
PARAPHRASES = (
    ("보험금을 지급하지 않는 사유", "보험금이 지급되지 않는 조건"),
    ("보험금 등을 지급하지 않는 사유", "보험금 등이 지급되지 않는 조건"),
    ("보험금의 지급사유", "보험금을 받을 수 있는 조건"),
    ("보험금 지급사유", "보험금을 받을 수 있는 조건"),
    ("보험금의 청구", "보험금을 청구하는 절차"),
    ("보험금 등의 청구", "보험금 등을 청구하는 절차"),
    ("계약 전 알릴 의무", "보험 가입 전에 알려야 하는 사항"),
    ("알릴 의무 위반의 효과", "고지의무를 위반했을 때 발생하는 조치"),
    ("계약의 소멸", "보험계약이 종료되는 조건"),
    ("계약의 해지", "보험계약을 해지하는 조건"),
    ("청약의 철회", "보험 가입 청약을 철회하는 방법"),
    ("보험료의 납입", "보험료를 납부하는 기준"),
    ("보험료 납입", "보험료를 납부하는 기준"),
    ("주소변경 통지", "주소가 바뀌었을 때 알려야 하는 절차"),
    ("약관의 해석", "보험약관의 의미가 불명확할 때 적용하는 해석 기준"),
    ("분쟁의 조정", "보험 관련 분쟁을 조정하는 절차"),
    ("소멸시효", "보험금 청구 권리가 소멸하는 기간"),
)


def stable(value):
    return hashlib.sha256((VERSION + "\0" + value).encode("utf-8")).hexdigest()


def nfc(value):
    return unicodedata.normalize("NFC", value)


def document_id(relative_path):
    digest = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()[:16]
    return f"doc-{digest}"


def doc_type(filename):
    lowered = nfc(filename).lower()
    for label, patterns in DOC_TYPE_PATTERNS:
        if any(pattern.lower() in lowered for pattern in patterns):
            return label
    return None


def parse_date(filename):
    filename = nfc(filename)
    # Conservative filename facts only: prefer standalone YYYYMMDD, then YYMMDD.
    matches = re.findall(r"(?<!\d)((?:19|20)\d{6})(?!\d)", filename)
    if matches:
        raw = matches[-1]
        return f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
    matches = re.findall(r"(?<!\d)(\d{6})(?!\d)", filename)
    if matches:
        raw = matches[-1]
        yy = int(raw[:2])
        year = 2000 + yy if yy <= 40 else 1900 + yy
        return f"{year:04d}-{raw[2:4]}-{raw[4:6]}"
    return None


def clean_heading(value):
    value = re.sub(r"^#+\s*", "", value)
    value = re.sub(r"^제\s*\d+(?:-\d+)?조(?:의\d+)?\s*", "", value)
    value = value.strip(" []【】<>-–—:：·.0123456789")
    value = re.sub(r"\s+", " ", value)
    for source, replacement in PARAPHRASES:
        if source in value:
            value = value.replace(source, replacement)
    value = value.replace("사항 사항", "사항")
    return value.strip()


def extract_evidence(path):
    try:
        lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    except OSError:
        return None
    candidates = []
    for index, raw in enumerate(lines):
        if not raw.lstrip().startswith("#"):
            continue
        heading = clean_heading(raw)
        compact = re.sub(r"\s+", "", heading)
        if not 9 <= len(heading) <= 65 or heading in BAD_HEADINGS:
            continue
        tokens = heading.split()
        if len(tokens) >= 4 and sum(len(token) == 1 for token in tokens) / len(tokens) >= 0.6:
            continue
        if heading.startswith(("년 ", "월 ", "일부터 ", "까지 ")):
            continue
        if len(re.findall(r"[가-힣]", heading)) < max(5, len(compact) * 0.45):
            continue
        body = []
        for following in lines[index + 1:index + 9]:
            following = following.strip()
            if following.startswith("#"):
                break
            if following and not following.startswith(("|", "![", "```")):
                body.append(following)
            if sum(map(len, body)) >= 100:
                break
        quote = " ".join(body)
        lowered_quote = quote.lower()
        if any(marker in lowered_quote for marker in CONTAMINATION):
            continue
        if 60 <= len(quote) <= 600:
            candidates.append((heading, quote[:500], index + 1))
    if not candidates:
        return None
    candidates.sort(key=lambda item: stable(path.as_posix() + "\0" + item[0]))
    return candidates[0]


def inventory():
    rows = []
    for path in sorted(RAW.rglob("*.md")):
        relative = nfc(path.relative_to(RAW).as_posix())
        dtype = doc_type(path.name)
        if not dtype:
            continue
        rows.append({
            "document_id": document_id(relative),
            "path": path,
            "relative": relative,
            "product": nfc(path.relative_to(RAW).parts[0]),
            "doc_type": dtype,
            "date": parse_date(path.name),
        })
    return rows


def base_record(qid, intent, query, target, evidence, constraints, status):
    heading, quote, line = evidence
    return {
        "qid": qid,
        "dataset_version": VERSION,
        "query": query,
        "intent_type": intent,
        "gold_document_id": target["document_id"],
        "gold_source_path": target["relative"],
        "evidence": {"heading": heading, "quote": quote, "line": line},
        "constraints": constraints,
        "provenance": "raw_markdown_only",
        "review_status": status,
    }


def build_identity(rows):
    dated = [row for row in rows if row["date"]]
    groups = defaultdict(list)
    for row in dated:
        groups[(row["product"], row["doc_type"], row["date"][:4])].append(row)
    unique = [group[0] for group in groups.values() if len(group) == 1]
    unique.sort(key=lambda row: stable(row["relative"]))

    selected = []
    product_use = Counter()
    type_use = Counter()
    # The corpus currently contains three usable document types. Keep the two
    # dominant types balanced and retain every available product-diverse slot
    # needed for the smaller disclosure-terms class.
    type_limit = {"판매약관": 65, "사업방법서": 65, "공시약관": 20}
    for row in unique:
        if product_use[row["product"]] or type_use[row["doc_type"]] >= type_limit[row["doc_type"]]:
            continue
        selected.append(row)
        product_use[row["product"]] += 1
        type_use[row["doc_type"]] += 1
        if len(selected) == 150:
            break
    if len(selected) < 150:
        selected_paths = {row["relative"] for row in selected}
        for row in unique:
            if row["relative"] in selected_paths or product_use[row["product"]] >= 2:
                continue
            selected.append(row)
            selected_paths.add(row["relative"])
            product_use[row["product"]] += 1
            type_use[row["doc_type"]] += 1
            if len(selected) == 150:
                break
    if len(selected) != 150:
        raise RuntimeError(f"identity candidates: {len(selected)}")

    result = []
    for row in selected:
        year = row["date"][:4]
        query = f"{row['product']}의 {year}년 {row['doc_type']} 문서를 찾아줘."
        evidence = ("원본 문서 식별정보", row["path"].name, 0)
        result.append(base_record(
            f"identity-{len(result) + 1:04d}", "identity", query, row, evidence,
            {"product_name": row["product"], "doc_type": row["doc_type"], "year": year},
            "mechanically_verified_unique_identity",
        ))
    return result, {row["relative"] for row in selected}


def content_pool(rows, excluded):
    pool = []
    product_seen = set()
    for row in sorted(rows, key=lambda item: stable(item["relative"])):
        if row["relative"] in excluded or row["product"] in product_seen:
            continue
        evidence = extract_evidence(row["path"])
        if not evidence:
            continue
        product_seen.add(row["product"])
        pool.append((row, evidence))
    return pool


def build_content(rows, excluded):
    pool = content_pool(rows, excluded)
    result = []
    used = set(excluded)
    queries = set()
    for row, evidence in pool:
        heading = evidence[0]
        query = f"{heading}에 관한 기준이나 처리 방법을 설명한 문서를 찾아줘."
        if query in queries:
            continue
        queries.add(query)
        result.append(base_record(
            f"content-{len(result) + 1:04d}", "content", query, row, evidence, {},
            "source_verified_candidate",
        ))
        used.add(row["relative"])
        if len(result) == 200:
            break
    if len(result) < 200:
        raise RuntimeError(f"unique content candidates: {len(result)}")
    return result, used


def build_mixed(rows, excluded):
    dated = [row for row in rows if row["date"] and row["relative"] not in excluded]
    unique_key = Counter((row["product"], row["doc_type"], row["date"][:4]) for row in dated)
    pool = content_pool(
        [row for row in dated if unique_key[(row["product"], row["doc_type"], row["date"][:4])] == 1],
        excluded,
    )
    if len(pool) < 150:
        raise RuntimeError(f"mixed candidates: {len(pool)}")
    result = []
    for row, evidence in pool[:150]:
        heading = evidence[0]
        year = row["date"][:4]
        query = (f"{row['product']}의 {year}년 {row['doc_type']} 중 "
                 f"{heading}에 관한 기준을 설명한 문서를 찾아줘.")
        result.append(base_record(
            f"mixed-{len(result) + 1:04d}", "mixed", query, row, evidence,
            {"product_name": row["product"], "doc_type": row["doc_type"], "year": year,
             "content_topic": heading},
            "source_and_identity_verified_candidate",
        ))
    return result


def assign_splits(records):
    quotas = {"identity": 30, "content": 40, "mixed": 30}
    grouped = defaultdict(list)
    for record in records:
        grouped[record["intent_type"]].append(record)
    for intent, rows in grouped.items():
        ordered = sorted(rows, key=lambda row: stable(row["query"]))
        dev = {row["qid"] for row in ordered[:quotas[intent]]}
        for row in rows:
            row["split"] = "dev" if row["qid"] in dev else "test"


def audit(records):
    errors = []
    if len(records) != 500:
        errors.append(f"count={len(records)}")
    if len({row["qid"] for row in records}) != 500:
        errors.append("duplicate qid")
    if len({row["query"] for row in records}) != 500:
        errors.append("duplicate query")
    if any(not row["gold_document_id"] or not row["evidence"]["quote"] for row in records):
        errors.append("missing gold/evidence")
    if errors:
        raise RuntimeError(", ".join(errors))


def main():
    rows = inventory()
    identity, used = build_identity(rows)
    content, used = build_content(rows, used)
    mixed = build_mixed(rows, used)
    records = identity + content + mixed
    assign_splits(records)
    audit(records)

    with DEST.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    with REVIEW.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(("qid", "intent_type", "query", "gold_source_path", "heading",
                         "evidence_quote", "review_status", "reviewer_verdict", "note"))
        for row in records:
            writer.writerow((row["qid"], row["intent_type"], row["query"],
                             row["gold_source_path"], row["evidence"]["heading"],
                             row["evidence"]["quote"], row["review_status"], "", ""))

    digest = hashlib.sha256(DEST.read_bytes()).hexdigest()
    manifest = {
        "dataset_version": VERSION,
        "n_questions": 500,
        "intent_counts": dict(Counter(row["intent_type"] for row in records)),
        "split_counts": dict(Counter(row["split"] for row in records)),
        "unique_gold_documents": len({row["gold_document_id"] for row in records}),
        "unique_products": len({row["gold_source_path"].split("/", 1)[0] for row in records}),
        "sha256": digest,
        "independence": {
            "reads_generated_index": False,
            "reads_generated_frontmatter": False,
            "source": "raw Markdown paths and contents only",
        },
        "status": "candidate",
        "limitations": [
            "Content questions were created by deterministic templates because no LLM endpoint was available.",
            "A source document and evidence are verified, but alternative relevant documents were not pooled/reviewed.",
            "Use the review CSV before promoting content/mixed records to approved gold.",
        ],
    }
    MANIFEST.write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()

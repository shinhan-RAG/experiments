#!/usr/bin/env python3
"""Offline, integrity-gated EDA for the two proposed Peter Part 1/2 datasets.

The script never modifies the supplied archives, creates no taxonomy/query/qrel,
and makes no network request.  ZIP members are read through ``ZipInfo`` objects
and staged under safe temporary filenames because the legal archive contains
legacy member paths that macOS ``unzip`` cannot materialize as filesystem paths.
Only nested ZIPs with a central directory and a passing CRC test contribute to
numeric EDA.  A noncanonical nested ZIP is retained in the hash manifest but is
explicitly excluded from all aggregate statistics.
"""

from __future__ import annotations

import argparse
from array import array
from collections import Counter, defaultdict
from contextlib import contextmanager
from datetime import datetime, timezone
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile
from typing import BinaryIO, Iterator
import zipfile


EDA_SCHEMA_VERSION = "dr-dci.new-dataset-eda.v1"
TOKEN_RE = re.compile(r"\w+", re.UNICODE)
WHITESPACE_RE = re.compile(r"\s+")
EMAIL_RE = re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}")
PHONE_RE = re.compile(r"(?:0\d{1,2}[- ]?)?\d{3,4}[- ]?\d{4}")
RRN_RE = re.compile(r"\b\d{6}[- ]?[1-4]\d{6}\b")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def sha256_text(text: str) -> bytes:
    # ``split``/``join`` has the same whitespace-collapsing semantics here as
    # the former regular expression and is materially cheaper for the large
    # provider label JSONs.
    normalized = " ".join(text.split())
    return hashlib.sha256(normalized.encode("utf-8")).digest()


def distribution(values: array) -> dict:
    if not values:
        return {"count": 0}
    ordered = sorted(values)

    def percentile(percent: float) -> float:
        index = (len(ordered) - 1) * percent / 100
        lower = int(index)
        upper = min(lower + 1, len(ordered) - 1)
        return ordered[lower] + (ordered[upper] - ordered[lower]) * (index - lower)

    return {
        "count": len(values),
        "min": ordered[0],
        "mean": round(sum(values) / len(values), 4),
        **{f"p{percent}": round(percentile(percent), 4)
           for percent in (25, 50, 75, 90, 95, 99)},
        "max": ordered[-1],
    }


class TextStats:
    """Bounded-memory corpus statistics; raw text is never retained."""

    def __init__(self) -> None:
        self.documents = 0
        self.ids: set[str] = set()
        self.duplicate_id_documents = 0
        self.content_hashes: set[bytes] = set()
        self.duplicate_content_documents = 0
        self.empty_documents = 0
        self.title_present = 0
        self.characters = array("Q")
        self.tokens = array("I")
        self.paragraphs = array("I")
        self.sensitive_document_hits = Counter()

    def add(self, doc_id: str | int | None, text: str, *, title: str = "") -> None:
        self.documents += 1
        normalized_id = "" if doc_id is None else str(doc_id).strip()
        if not normalized_id:
            self.duplicate_id_documents += 1
        elif normalized_id in self.ids:
            self.duplicate_id_documents += 1
        else:
            self.ids.add(normalized_id)

        combined = f"{title} {text}".strip()
        content_hash = sha256_text(combined)
        if content_hash in self.content_hashes:
            self.duplicate_content_documents += 1
        else:
            self.content_hashes.add(content_hash)

        self.empty_documents += not bool(combined)
        self.title_present += bool(title.strip())
        self.characters.append(len(combined))
        self.tokens.append(len(TOKEN_RE.findall(combined)))
        self.paragraphs.append(len([part for part in re.split(r"\n\s*\n", text) if part.strip()]))
        for name, pattern in (("email", EMAIL_RE), ("phone_like", PHONE_RE),
                              ("resident_registration_like", RRN_RE)):
            if pattern.search(combined):
                self.sensitive_document_hits[name] += 1

    def report(self) -> dict:
        return {
            "documents": self.documents,
            "unique_ids": len(self.ids),
            "duplicate_id_documents": self.duplicate_id_documents,
            "duplicate_content_documents": self.duplicate_content_documents,
            "duplicate_content_document_rate": round(
                self.duplicate_content_documents / self.documents, 6
            ) if self.documents else 0.0,
            "empty_documents": self.empty_documents,
            "empty_document_rate": round(self.empty_documents / self.documents, 6)
            if self.documents else 0.0,
            "title_present_documents": self.title_present,
            "character_length": distribution(self.characters),
            "heuristic_token_length": distribution(self.tokens),
            "paragraph_count": distribution(self.paragraphs),
            "sensitive_pattern_document_hits": dict(self.sensitive_document_hits),
        }


def json_value_types(value: object) -> str:
    if value is None:
        return "null"
    if isinstance(value, dict):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, (int, float)):
        return "number"
    return "string"


def schema_fields(record: dict, field_types: dict[str, Counter]) -> None:
    for name, value in record.items():
        field_types[name][json_value_types(value)] += 1


def counter_dict(counter: Counter, *, limit: int | None = None) -> dict:
    """Serialize a counter without making a summary artifact unbounded."""
    pairs = counter.most_common(limit)
    return {str(key): int(value) for key, value in pairs}


def decode_json(raw: bytes) -> dict | list:
    return json.loads(raw.decode("utf-8-sig"))


def iter_json_data(stream: BinaryIO) -> Iterator[dict]:
    """Stream objects from a top-level ``{..., "data": [...]}`` JSON object."""
    text = io.TextIOWrapper(stream, encoding="utf-8-sig")
    decoder = json.JSONDecoder()
    buffer = ""
    position = 0
    data_array_started = False
    eof = False

    def refill() -> None:
        nonlocal buffer, position, eof
        if eof:
            return
        if position:
            buffer = buffer[position:]
            position = 0
        chunk = text.read(1024 * 1024)
        if chunk:
            buffer += chunk
        else:
            eof = True

    while not data_array_started:
        match = re.search(r'"data"\s*:\s*\[', buffer)
        if match:
            position = match.end()
            data_array_started = True
            break
        if eof:
            raise ValueError("top-level data array not found")
        refill()

    while True:
        while position < len(buffer) and buffer[position] in " \t\r\n,":
            position += 1
        if position >= len(buffer):
            if eof:
                raise ValueError("unterminated top-level data array")
            refill()
            continue
        if buffer[position] == "]":
            return
        try:
            record, end = decoder.raw_decode(buffer, position)
        except json.JSONDecodeError:
            if eof:
                raise
            refill()
            continue
        if not isinstance(record, dict):
            raise ValueError("top-level data array contains a non-object record")
        yield record
        position = end
        if position > 2 * 1024 * 1024:
            buffer = buffer[position:]
            position = 0


@contextmanager
def nested_member_to_temp(outer: zipfile.ZipFile, info: zipfile.ZipInfo):
    """Stage one outer member under a safe temporary filename and hash it."""
    digest = hashlib.sha256()
    with tempfile.NamedTemporaryFile(suffix=".zip") as temporary:
        with outer.open(info) as source:
            while chunk := source.read(1024 * 1024):
                digest.update(chunk)
                temporary.write(chunk)
        temporary.flush()
        yield Path(temporary.name), digest.hexdigest()


def archive_signature(path: Path) -> dict:
    with path.open("rb") as stream:
        first = stream.read(4)
        size = path.stat().st_size
        stream.seek(max(0, size - 65_557))
        tail = stream.read()
    return {
        "local_file_header_signature": first.hex(),
        "has_local_file_header": first == b"PK\x03\x04",
        "has_eocd_signature": b"PK\x05\x06" in tail,
        "has_zip64_eocd_signature": b"PK\x06\x06" in tail,
        "has_zip64_locator_signature": b"PK\x06\x07" in tail,
    }


class NestedArchiveScanner:
    def __init__(self, archive_path: Path, dataset_key: str) -> None:
        self.archive_path = archive_path
        self.dataset_key = dataset_key
        self.nested_members: list[dict] = []
        self.outer_report: dict = {}
        self.outer_regular_files_verified_readable = 0

    def scan(self, on_canonical_zip) -> None:
        with zipfile.ZipFile(self.archive_path) as outer:
            infos = outer.infolist()
            outer_bad = outer.testzip()
            regular_infos = [info for info in infos if not info.is_dir()]
            self.outer_report = {
                "path": str(self.archive_path),
                "size_bytes": self.archive_path.stat().st_size,
                "sha256": sha256_file(self.archive_path),
                "file_type": "zip archive (validated by Python zipfile)",
                "signature": archive_signature(self.archive_path),
                "central_directory_present": True,
                "central_directory_entries": len(infos),
                "regular_file_entries_expected": len(regular_infos),
                "archive_test": {
                    "status": "passed" if outer_bad is None else "failed",
                    "bad_member": outer_bad,
                },
            }
            for outer_index, info in enumerate(infos):
                if info.is_dir():
                    continue
                member = {
                    "outer_index": outer_index,
                    "logical_path": info.filename,
                    "compressed_size": info.compress_size,
                    "uncompressed_size": info.file_size,
                }
                if not info.filename.lower().endswith(".zip"):
                    member["status"] = "non_zip_regular_member"
                    self.nested_members.append(member)
                    continue
                with nested_member_to_temp(outer, info) as (temporary_path, digest):
                    self.outer_regular_files_verified_readable += 1
                    member["sha256"] = digest
                    member["signature"] = archive_signature(temporary_path)
                    try:
                        with zipfile.ZipFile(temporary_path) as nested:
                            bad_member = nested.testzip()
                            if bad_member is not None:
                                member.update({
                                    "status": "failed_crc",
                                    "bad_member": bad_member,
                                    "central_directory_present": True,
                                })
                            else:
                                regular = [entry for entry in nested.infolist() if not entry.is_dir()]
                                extensions = Counter(
                                    Path(entry.filename).suffix.lower() or "<none>"
                                    for entry in regular
                                )
                                member.update({
                                    "status": "passed",
                                    "central_directory_present": True,
                                    "inner_regular_members": len(regular),
                                    "inner_member_extensions": counter_dict(extensions),
                                    "inner_uncompressed_bytes": sum(
                                        entry.file_size for entry in regular
                                    ),
                                })
                                on_canonical_zip(info, nested, member)
                    except zipfile.BadZipFile as error:
                        member.update({
                            "status": "noncanonical_no_central_directory",
                            "central_directory_present": False,
                            "error": str(error),
                        })
                self.nested_members.append(member)
            self.outer_report["regular_file_entries_verified_readable"] = (
                self.outer_regular_files_verified_readable
            )


def law_source_text(record: dict) -> str:
    fields = (
        "사건명", "판시사항", "판결요지", "참조조문", "참조판례", "판례내용",
    )
    return "\n".join(str(record.get(name) or "") for name in fields).strip()


def law_label_text(record: dict) -> str:
    summary = record.get("Summary") or []
    summary_text = []
    for item in summary:
        if isinstance(item, dict):
            summary_text.extend(str(item.get(key) or "") for key in ("summ_contxt", "summ_pass"))
    return "\n".join([str(record.get("jdgmn") or ""), *summary_text]).strip()


def analyze_law_case(path: Path) -> tuple[dict, dict]:
    scanner = NestedArchiveScanner(path, "law_case")
    source_stats = TextStats()
    label_stats = TextStats()
    qa_stats = TextStats()
    source_case_numbers: set[str] = set()
    label_case_numbers: set[str] = set()
    source_field_types: dict[str, Counter] = defaultdict(Counter)
    label_field_types: dict[str, Counter] = defaultdict(Counter)
    qa_field_types: dict[str, Counter] = defaultdict(Counter)
    source_case_types = Counter()
    label_class_names = Counter()
    label_instance_names = Counter()
    label_questions = 0
    qa_questions = 0
    qa_reference_rule_present = 0
    parse_failures = Counter()
    leaf_digest = hashlib.sha256()
    leaf_count = 0

    def consume(info: zipfile.ZipInfo, nested: zipfile.ZipFile, member: dict) -> None:
        nonlocal label_questions, qa_questions, qa_reference_rule_present, leaf_count
        logical_path = info.filename
        if "/Other/" in logical_path:
            role = "qa"
        elif "/02.라벨링데이터/" in logical_path:
            role = "label"
        else:
            role = "source"
        member["eda_role"] = role
        parsed = 0
        for entry in nested.infolist():
            if entry.is_dir():
                continue
            raw = nested.read(entry)
            leaf_digest.update(entry.filename.encode("utf-8"))
            leaf_digest.update(hashlib.sha256(raw).digest())
            leaf_count += 1
            try:
                record = decode_json(raw)
            except (UnicodeDecodeError, json.JSONDecodeError):
                parse_failures[role] += 1
                continue
            if not isinstance(record, dict):
                parse_failures[f"{role}_non_object"] += 1
                continue
            parsed += 1
            if role == "source":
                schema_fields(record, source_field_types)
                case_number = str(record.get("사건번호") or "").strip()
                serial_number = str(record.get("판례일련번호") or "").strip()
                if case_number:
                    source_case_numbers.add(case_number)
                source_case_types[str(record.get("사건종류명") or "<null>")] += 1
                source_stats.add(
                    serial_number or case_number,
                    law_source_text(record),
                    title=str(record.get("사건명") or ""),
                )
            elif role == "label":
                schema_fields(record, label_field_types)
                info_record = record.get("info") if isinstance(record.get("info"), dict) else {}
                case_number = str(info_record.get("caseNoID") or "").strip()
                if case_number:
                    label_case_numbers.add(case_number)
                label_stats.add(
                    info_record.get("id") or case_number,
                    law_label_text(record),
                    title=str(info_record.get("caseTitle") or ""),
                )
                class_info = record.get("Class_info") if isinstance(record.get("Class_info"), dict) else {}
                label_class_names[str(class_info.get("class_name") or "<null>")] += 1
                label_instance_names[str(class_info.get("instance_name") or "<null>")] += 1
                for question_answer in record.get("jdgmnInfo") or []:
                    if isinstance(question_answer, dict) and str(question_answer.get("question") or "").strip():
                        label_questions += 1
            else:
                schema_fields(record, qa_field_types)
                question = str(record.get("question") or "")
                if question.strip():
                    qa_questions += 1
                if str(record.get("reference_rules") or "").strip():
                    qa_reference_rule_present += 1
                qa_stats.add(record.get("id"), str(record.get("commentary") or ""),
                             title=str(record.get("title") or ""))
        member["eda_parsed_regular_members"] = parsed

    scanner.scan(consume)
    canonical = [member for member in scanner.nested_members if member.get("status") == "passed"]
    failed = [member for member in scanner.nested_members if member.get("status") != "passed"]
    # The outer ZIP is intact, but an expected labelled component is not a
    # canonical ZIP.  The caller's integrity rule treats this as a hard stop
    # for dataset-level statistics.  We intentionally retain only a schema
    # reference from the readable components below; no partial counts are
    # exposed as corpus facts.
    if failed:
        summary = {
            "dataset_key": "law_case",
            "integrity_gate": {
                "status": "blocked_for_full_dataset_eda",
                "reason": (
                    "one expected Training labelled ZIP has no central directory; "
                    "the outer archive is valid, but the dataset is not an immutable, "
                    "fully extractable corpus"
                ),
                "outer_archive": scanner.outer_report,
                "nested_zip_count": len(scanner.nested_members),
                "nested_zip_passed": len(canonical),
                "nested_zip_failed_or_noncanonical": len(failed),
                "required_redownload_or_replacement": [
                    {
                        "logical_path": member["logical_path"],
                        "sha256_of_received_member": member.get("sha256"),
                        "failure": member.get("error", member.get("status")),
                    }
                    for member in failed
                ],
            },
            "numeric_scope": {
                "status": "not_performed",
                "reason": (
                    "numeric inventory, duplicate rates, length distributions, and query "
                    "counts are withheld because the supplied dataset is incomplete"
                ),
            },
            "schema_reference_only": {
                "scope": "field presence observed in canonical readable components; not dataset-level counts",
                "file_formats": {"outer": "zip", "canonical_nested": "zip", "leaf": "json"},
                "splits_observed": ["Training", "Validation", "Other", "Sublabel"],
                "source_case_top_level_fields": sorted(source_field_types),
                "label_record_top_level_fields": sorted(label_field_types),
                "qa_record_top_level_fields": sorted(qa_field_types),
                "field_descriptions": {
                    "source_case": {
                        "판례일련번호": "source case serial identifier",
                        "사건번호": "case number; candidate linkage only, not a qrel",
                        "사건명": "case title",
                        "판시사항/판결요지/판례내용": "legal decision body sections",
                        "참조조문/참조판례": "cited statutes and cases",
                    },
                    "label_record": {
                        "info": "case metadata including id and caseNoID",
                        "jdgmnInfo": "question/answer-style objects",
                        "Summary": "case summary objects",
                        "keyword_tagg": "keyword objects",
                        "Class_info": "class_name and instance_name taxonomy candidates",
                    },
                    "qa_record": {
                        "id": "QA record id",
                        "question/answer/commentary": "query-like and answer text",
                        "reference_rules/reference_court_case": "reference metadata",
                    },
                },
            },
            "search_evaluation": {
                "status": "not_certified_due_to_integrity_gate",
                "query_like_fields_observed": ["jdgmnInfo.question", "question"],
                "explicit_qrels_or_gold_mapping_observed": False,
                "candidate_link_field": "info.caseNoID ↔ source 사건번호",
                "judgment": (
                    "do not infer qrels from case-number overlap. A complete replacement archive and "
                    "an explicit qid-to-gold contract are both required before retrieval evaluation."
                ),
                "leakage_risk": (
                    "high if label summaries, answers, or questions are indexed as evidence for the "
                    "same cases"
                ),
            },
            "taxonomy_and_structure": {
                "candidate_taxonomy_fields": [
                    "Class_info.class_name", "Class_info.instance_name", "keyword_tagg.keyword",
                    "Reference_info.reference_rules", "사건종류명",
                ],
                "semantic_structure": [
                    "판시사항", "판결요지", "참조조문", "참조판례", "판례내용", "Summary",
                ],
                "label_circularity_note": (
                    "taxonomy/category labels must remain separate from qrel construction and must not "
                    "be indexed as gold evidence for their own questions"
                ),
            },
            "license_and_sensitive_data": {
                "license_artifact_found": False,
                "license_status": "unknown; obtain provider license and permitted-use terms",
                "sensitive_data_note": (
                    "legal judgments and QA may contain names, addresses, or case-specific facts; "
                    "screening has not established safe handling status"
                ),
            },
        }
        manifest = {"outer_archive": scanner.outer_report, "nested_members": scanner.nested_members}
        return summary, manifest

    summary = {
        "dataset_key": "law_case",
        "integrity_gate": {
            "status": "blocked_for_full_dataset_eda" if failed else "passed",
            "reason": (
                "one nested labelled ZIP has no central directory; all numeric EDA below "
                "excludes that component"
                if failed else "all nested ZIPs passed"
            ),
            "outer_archive": scanner.outer_report,
            "nested_zip_count": len(scanner.nested_members),
            "nested_zip_passed": len(canonical),
            "nested_zip_failed_or_noncanonical": len(failed),
            "safe_virtual_extraction": {
                "expected_outer_regular_files": scanner.outer_report["regular_file_entries_expected"],
                "verified_outer_regular_files": scanner.outer_report[
                    "regular_file_entries_verified_readable"
                ],
                "note": "archive CRC test passed; safe virtual extraction uses entry IDs, not legacy paths",
            },
        },
        "numeric_scope": {
            "includes": "54 canonical nested ZIP components only",
            "excludes": [
                member["logical_path"] for member in failed
            ],
        },
        "inventory": {
            "file_formats": {"outer": "zip", "canonical_nested": "zip", "leaf": "json"},
            "splits_observed": ["Training", "Validation", "Other", "Sublabel"],
            "parse_failures": counter_dict(parse_failures),
            "source_document_candidate": source_stats.report(),
            "label_records": label_stats.report(),
            "qa_records": qa_stats.report(),
            "source_case_type_distribution": counter_dict(source_case_types),
            "label_class_distribution": counter_dict(label_class_names),
            "label_instance_distribution": counter_dict(label_instance_names),
            "schemas": {
                "source_case": {key: counter_dict(value) for key, value in source_field_types.items()},
                "label_record": {key: counter_dict(value) for key, value in label_field_types.items()},
                "qa_record": {key: counter_dict(value) for key, value in qa_field_types.items()},
            },
            "leaf_content_hash_aggregate_sha256": leaf_digest.hexdigest(),
            "leaf_files_included": leaf_count,
        },
        "search_evaluation": {
            "query_sources": {
                "label_jdgmnInfo_questions": label_questions,
                "other_qa_questions": qa_questions,
            },
            "explicit_qrels_provided": False,
            "explicit_gold_document_id_field": False,
            "candidate_label_to_source_case_number_links": {
                "label_case_numbers": len(label_case_numbers),
                "source_case_numbers": len(source_case_numbers),
                "matched_case_numbers": len(label_case_numbers & source_case_numbers),
                "label_case_number_coverage_in_source": round(
                    len(label_case_numbers & source_case_numbers) / len(label_case_numbers), 6
                ) if label_case_numbers else 0.0,
            },
            "qa_reference_rule_present_records": qa_reference_rule_present,
            "judgment": (
                "queries exist in label/QA records, but no explicit qrel or gold-document "
                "connection is supplied; case-number overlap is only a candidate linkage and "
                "must not be promoted to qrels in this EDA"
            ),
            "leakage_risk": (
                "high if label records become corpus: label summaries, answers, and questions "
                "describe the same case as their candidate source link"
            ),
        },
        "taxonomy_and_structure": {
            "candidate_taxonomy_fields": [
                "Class_info.class_name", "Class_info.instance_name", "keyword_tagg.keyword",
                "Reference_info.reference_rules", "사건종류명",
            ],
            "semantic_structure": [
                "판시사항", "판결요지", "참조조문", "참조판례", "판례내용", "Summary",
            ],
            "label_circularity_note": (
                "taxonomy/category labels are usable only if kept separate from qrel construction; "
                "label records must not be indexed as gold evidence for the same questions"
            ),
        },
        "license_and_sensitive_data": {
            "license_artifact_found": False,
            "license_status": "unknown; obtain provider license and permitted-use terms",
            "sensitive_data_note": (
                "legal judgments and free-text QA can contain names, addresses, or case-specific "
                "facts; regex hits are screening only and do not establish safe handling status"
            ),
        },
    }
    manifest = {"outer_archive": scanner.outer_report, "nested_members": scanner.nested_members}
    return summary, manifest


def declared_totalcount(nested: zipfile.ZipFile, entry: zipfile.ZipInfo) -> int | None:
    with nested.open(entry) as stream:
        prefix = stream.read(4096)
    match = re.search(rb'"totalcount"\s*:\s*(\d+)', prefix)
    return int(match.group(1)) if match else None


def analyze_books(path: Path) -> tuple[dict, dict]:
    scanner = NestedArchiveScanner(path, "medical_legal_books")
    source_stats: dict[tuple[str, str], TextStats] = defaultdict(TextStats)
    overall_source_stats = TextStats()
    source_content_hashes: dict[str, bytes] = {}
    source_ids_by_group: dict[tuple[str, str], set[str]] = defaultdict(set)
    label_counts = Counter()
    label_categories: dict[str, Counter] = defaultdict(Counter)
    label_keyword_counts = Counter()
    entity_types = Counter()
    label_schema_types: dict[str, Counter] = defaultdict(Counter)
    label_totalcount_declared = {}
    label_record_count = 0
    label_source_id_matches = 0
    label_source_exact_text_matches = 0
    label_question_field_records = 0
    parse_failures = Counter()
    leaf_digest = hashlib.sha256()
    leaf_count = 0

    def path_group(logical_path: str) -> tuple[str, str]:
        split = "Training" if "/Training/" in logical_path else "Validation"
        domain = "medical" if "의료" in logical_path else "legal"
        return split, domain

    def consume(info: zipfile.ZipInfo, nested: zipfile.ZipFile, member: dict) -> None:
        nonlocal label_record_count, label_source_id_matches, label_source_exact_text_matches
        nonlocal label_question_field_records, leaf_count
        split, domain = path_group(info.filename)
        is_label = "/02.라벨링데이터/" in info.filename
        member["eda_role"] = "label" if is_label else "source"
        member["split"] = split
        member["domain"] = domain
        regular = [entry for entry in nested.infolist() if not entry.is_dir()]
        if is_label:
            if len(regular) != 1:
                parse_failures["label_member_count"] += 1
                return
            entry = regular[0]
            label_totalcount_declared[f"{split}_{domain}"] = declared_totalcount(nested, entry)
            try:
                with nested.open(entry) as stream:
                    for record in iter_json_data(stream):
                        label_record_count += 1
                        label_counts[f"{split}_{domain}"] += 1
                        schema_fields(record, label_schema_types)
                        if str(record.get("question") or "").strip():
                            label_question_field_records += 1
                        book_id = str(record.get("book_id") or "").strip()
                        text = str(record.get("text") or "")
                        if book_id in source_content_hashes:
                            label_source_id_matches += 1
                            if source_content_hashes[book_id] == sha256_text(text):
                                label_source_exact_text_matches += 1
                        label_categories[domain][str(record.get("category") or "<null>")] += 1
                        for keyword in record.get("keyword") or []:
                            if isinstance(keyword, str) and keyword.strip():
                                label_keyword_counts[keyword] += 1
                        for entity in record.get("NE") or []:
                            if isinstance(entity, dict):
                                entity_types[str(entity.get("type") or "<null>")] += 1
            except (UnicodeDecodeError, json.JSONDecodeError, ValueError):
                parse_failures[f"label_{split}_{domain}"] += 1
            return

        parsed = 0
        stats = source_stats[(split, domain)]
        for entry in regular:
            raw = nested.read(entry)
            leaf_digest.update(entry.filename.encode("utf-8"))
            leaf_digest.update(hashlib.sha256(raw).digest())
            leaf_count += 1
            try:
                text = raw.decode("utf-8-sig")
            except UnicodeDecodeError:
                parse_failures[f"source_decode_{split}_{domain}"] += 1
                continue
            book_id = Path(entry.filename).stem
            content_hash = sha256_text(text)
            source_content_hashes[book_id] = content_hash
            source_ids_by_group[(split, domain)].add(book_id)
            stats.add(book_id, text)
            overall_source_stats.add(book_id, text)
            parsed += 1
        member["eda_parsed_regular_members"] = parsed

    scanner.scan(consume)
    canonical = [member for member in scanner.nested_members if member.get("status") == "passed"]
    failed = [member for member in scanner.nested_members if member.get("status") != "passed"]
    per_group = {
        f"{split}_{domain}": stats.report()
        for (split, domain), stats in sorted(source_stats.items())
    }
    per_domain_docs = {
        domain: sum(stats.documents for (split, candidate_domain), stats in source_stats.items()
                    if candidate_domain == domain)
        for domain in ("medical", "legal")
    }
    summary = {
        "dataset_key": "medical_legal_books",
        "integrity_gate": {
            "status": "passed" if not failed else "blocked_for_full_dataset_eda",
            "outer_archive": scanner.outer_report,
            "nested_zip_count": len(scanner.nested_members),
            "nested_zip_passed": len(canonical),
            "nested_zip_failed_or_noncanonical": len(failed),
            "safe_virtual_extraction": {
                "expected_outer_regular_files": scanner.outer_report["regular_file_entries_expected"],
                "verified_outer_regular_files": scanner.outer_report[
                    "regular_file_entries_verified_readable"
                ],
                "note": "all nested ZIPs are canonical and CRC-tested",
            },
        },
        "numeric_scope": {"includes": "all 8 canonical nested ZIP components", "excludes": []},
        "inventory": {
            "file_formats": {"outer": "zip", "canonical_nested": "zip", "source_leaf": "txt", "label_leaf": "json"},
            "splits_observed": ["Training", "Validation"],
            "source_documents": overall_source_stats.report(),
            "source_documents_by_split_domain": per_group,
            "source_documents_by_domain": per_domain_docs,
            "label_records_by_split_domain": counter_dict(label_counts),
            "label_totalcount_declared": label_totalcount_declared,
            "label_source_id_matches": label_source_id_matches,
            "label_source_exact_text_matches": label_source_exact_text_matches,
            "parse_failures": counter_dict(parse_failures),
            "category_distribution": {
                domain: counter_dict(counter) for domain, counter in label_categories.items()
            },
            "top_keywords": counter_dict(label_keyword_counts, limit=100),
            "entity_type_distribution": counter_dict(entity_types),
            "label_schema": {key: counter_dict(value) for key, value in label_schema_types.items()},
            "source_leaf_content_hash_aggregate_sha256": leaf_digest.hexdigest(),
            "source_leaf_files_included": leaf_count,
        },
        "search_evaluation": {
            "query_provided": False,
            "question_field_records": label_question_field_records,
            "qrels_provided": False,
            "gold_document_connection_provided": False,
            "relevance_grades_provided": False,
            "judgment": "corpus candidate only; no provided query/gold evaluation contract",
            "leakage_risk": (
                "high if label JSON is indexed together with its matched source TXT because label "
                "records reproduce the source text"
            ),
        },
        "taxonomy_and_structure": {
            "candidate_taxonomy_fields": ["category", "keyword", "NE.type", "popularity"],
            "semantic_structure": ["NE entity spans with begin/end offsets", "word_segment"],
            "field_descriptions": {
                "book_id": "matches the source TXT filename stem; document identifier, not a query id",
                "category": "provider category; candidate taxonomy field",
                "keyword": "provider keyword array; candidate semantic tag field",
                "NE": "entity objects with entity, type, begin, and end offsets",
                "text": "label-side text; do not index alongside matching source TXT",
                "word_segment": "provider numeric segmentation field",
                "publication_ymd": "publication date string",
                "popularity": "provider numeric metadata",
            },
            "note": (
                "labels can support a taxonomy/semantic-tag treatment only after a separate, "
                "non-circular retrieval evaluation contract exists"
            ),
        },
        "license_and_sensitive_data": {
            "license_artifact_found": False,
            "license_status": "unknown; obtain provider license and permitted-use terms",
            "sensitive_data_note": (
                "medical and legal text can contain personal or sensitive facts; regex hits in the "
                "source-document report are screening only"
            ),
        },
    }
    manifest = {"outer_archive": scanner.outer_report, "nested_members": scanner.nested_members}
    return summary, manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--law-zip", type=Path, required=True)
    parser.add_argument("--books-zip", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--stamp", default="20260723")
    args = parser.parse_args()

    for path in (args.law_zip, args.books_zip):
        if not path.is_file():
            raise FileNotFoundError(path)

    law_summary, law_manifest = analyze_law_case(args.law_zip)
    books_summary, books_manifest = analyze_books(args.books_zip)
    generated_at = datetime.now(timezone.utc).isoformat()
    summary = {
        "schema_version": EDA_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "network_or_model_calls": False,
        "datasets": [law_summary, books_summary],
    }
    manifest = {
        "schema_version": EDA_SCHEMA_VERSION,
        "generated_at_utc": generated_at,
        "archives": {
            "law_case": law_manifest,
            "medical_legal_books": books_manifest,
        },
        "scope_note": (
            "Nested component hashes are hashes of the files virtually extracted from each outer "
            "archive. Leaf-file aggregate hashes cover the canonical components scanned for EDA; "
            "no raw source text is written to this manifest."
        ),
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    summary_path = args.output_dir / f"NEW_DATASET_EDA_{args.stamp}.json"
    manifest_path = args.output_dir / f"NEW_DATASET_HASH_MANIFEST_{args.stamp}.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(summary_path)
    print(manifest_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""U3 규칙 태그를 범용 계층형 fact-card 태그로 확장한다.

QA·Gold·검색 결과를 읽지 않는다. 원문 element, 기존 태그, JO 경계만 사용한다.
추가되는 신호는 다음 네 종류다.

* document_key: 상위 문서 identity와 하위 section/contract identity를 분리
* evidence_anchor: 문장·표 행 단위의 답변 근거(전체 원문 덤프가 아님)
* benefit_aliases: ``X급여금/보험금/진단비``의 보수적 표면형 정규화
* reference edge: namespace를 보존한 명시적 표 참조의 source/target provenance

모든 파생값은 원 element id와 parent JO를 보존하므로 독립 감사가 가능하다.
"""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
from collections import defaultdict
from pathlib import Path

from patterns import ROLE_RULES, VALUE_RE, unique


HERE = Path(__file__).resolve().parent
VERSION = "semtag-u4-fact-rules-1.1"
SOURCE_PATH_RE = re.compile(r"<!--\s*원본:\s*([^>]+?)\s*-->")
TABLE_KEY_RE = re.compile(
    r"(?P<schedule>[<\[]?\s*부표\s*(?P<schedule_no>\d+(?:-\d+)?)\s*[>\]]?)"
    r"|(?P<appendix>[<\[]?\s*별첨\s*(?P<appendix_no>\d+)\s*[>\]]?\s*"
    r"\[?\s*표\s*(?P<appendix_table_no>\d+(?:-\d+)?)\s*\]?)"
    r"|(?P<table>\[?\s*(?<!부)표\s*(?P<table_no>\d+(?:-\d+)?)\s*\]?)",
    re.I,
)
BENEFIT_RE = re.compile(
    r"([가-힣A-Za-z0-9·]{2,50}?)(급여금|보험금|진단비|치료비|수술비|지원비)"
)
CODE_RE = re.compile(r"(?<![A-Za-z0-9])([A-Z]\d{2}(?:\.\d{1,2})?)(?![A-Za-z])", re.I)
QUOTED_RE = re.compile(r'["“「『](.{2,160}?)["”」』]')
BOILERPLATE_RE = re.compile(
    r"The following is|The solution|CHINA NATIONAL LIFE|SHINHAN LIFE|"
    r"text in the image cannot be extracted|^[_\-\s$\\{}]+$",
    re.I,
)
FACT_ROLE_RULES = (
    (r"사고증명서|장해진단서|사망진단서|진료기록부|제출.{0,12}서류|구비서류", "claim_procedure"),
    (r"내용.{0,20}다른 경우|우선.{0,12}적용|유리한 내용|효력", "conflict_priority"),
    (r"최초\s*1회|연간\s*\d+회|\d+회.{0,10}한도|횟수", "limit_frequency"),
    (r"분류\s*코드|[A-Z]\d{2}(?:\.\d+)?|분류표", "code_reference"),
    (r"지급사유|지급합니다|지급함|보장합니다", "payment_trigger"),
    (r"지급금액|가입금액.{0,20}%|\d+(?:,\d{3})*(?:만)?원", "payment_amount"),
    (r"포함|해당하는 항목|말합니다|이라 함은|라 함은", "criteria_rule"),
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.open(encoding="utf-8")]


def table_keys(text: str) -> list[str]:
    """표 번호의 법적 namespace를 보존한다.

    ``부표1``과 ``별첨2 [표1]``은 서로 다른 표이므로 같은 key로 축약하지
    않는다. namespace가 없는 ``표1``도 별도 fail-closed key로 둔다.
    """
    output = []
    for match in TABLE_KEY_RE.finditer(text):
        if match.group("schedule"):
            key = "부표:" + match.group("schedule_no")
        elif match.group("appendix"):
            key = ("별첨:" + match.group("appendix_no") + ":표:" +
                   match.group("appendix_table_no"))
        else:
            key = "표:" + match.group("table_no")
        if key not in output:
            output.append(key)
    return output


def document_keys(elements: list[dict]) -> list[str]:
    """각 element의 문서 경계를 보존한다.

    대규모 corpus에서 한 파일의 ``표2-1`` 참조가 다른 파일의 같은 표 번호로
    연결되지 않도록 document metadata/source marker를 순서대로 전파한다.
    """
    current = ""
    output = []
    for row in elements:
        explicit = ""
        for key in ("source_file", "source_document", "document_name", "document_title"):
            if row.get(key):
                explicit = str(row[key]).strip()
                break
        match = SOURCE_PATH_RE.search(str(row.get("text", "")))
        if match:
            explicit = match.group(1).strip()
        if explicit:
            normalized = str(Path(explicit).expanduser())
            current = "doc:" + hashlib.sha256(normalized.encode("utf-8")).hexdigest()[:20]
        output.append(current)
    return output


def clean_text(value: str, limit: int = 500) -> str:
    value = re.sub(r"<[^>]+>", " ", str(value))
    value = re.sub(r"\$[^$]{0,300}\$", " ", value)
    value = re.sub(r"\s+", " ", value).strip(" |,;:#*_-\t")
    if len(value) < 2 or BOILERPLATE_RE.search(value):
        return ""
    return value[:limit]


def ordered_unique(values: list[str], limit: int) -> list[str]:
    """근거 문장을 subject 전용 BAD_KEY 필터로 버리지 않는 순서 보존 dedupe."""
    seen, output = set(), []
    for value in values:
        value = str(value).strip()
        if not value or value in seen:
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def table_rows(text: str) -> list[str]:
    rows = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [clean_text(cell, 180) for cell in line.strip().strip("|").split("|")]
        cells = [cell for cell in cells if cell and not re.fullmatch(r"[-: ]+", cell)]
        if cells:
            rows.append(" | ".join(cells)[:500])
    return rows


def evidence_anchors(element: dict) -> list[str]:
    """질문 독립적으로 답 후보가 되는 문장/표 행/열거 항목만 보존한다."""
    text = str(element.get("text", ""))
    anchors = table_rows(text)
    if element.get("element_type") != "table":
        for block in re.split(r"\n+|(?<=[.!?다요함됨])\s+(?=[①-⑳가-힣A-Z0-9\"“「])", text):
            cleaned = clean_text(block)
            if cleaned and re.search(
                    r"지급|보장|진단|서류|증명|포함|제외|한도|횟수|유리|다른 경우|"
                    r"참조|코드|효력|책임|산정|계산|말합니다|이라 함은|라 함은|해야 합니다",
                    cleaned):
                anchors.append(cleaned)
    # 긴 인용 열거에서 태거가 놓치던 개별 답 항목을 독립 anchor로 만든다.
    for quoted in QUOTED_RE.findall(text):
        for item in re.split(r"[,·]|\s+(?:및|또는)\s+", quoted):
            cleaned = clean_text(item, 120)
            if cleaned:
                anchors.append(cleaned)
    return ordered_unique(anchors, 40)


def benefit_aliases(text: str) -> list[str]:
    aliases = []
    for match in BENEFIT_RE.finditer(text):
        base = re.sub(r"\s+", "", match.group(1)).strip()
        if len(base) < 2:
            continue
        aliases.extend((base + "금", base + "급여금", base + "보험금"))
    return unique(aliases, 24)


def fact_roles(text: str) -> list[str]:
    rules = tuple(ROLE_RULES) + FACT_ROLE_RULES
    return unique([role for pattern, role in rules if re.search(pattern, text, re.I)], 16)


def explicit_table_refs(text: str) -> list[str]:
    refs = []
    for line in text.splitlines():
        if "참조" not in line:
            continue
        # 참조 뒤의 목차/다음 문장 표 번호를 잘못 붙이지 않도록 참조 토큰까지의
        # 제한된 문맥만 해석한다.
        before = line[:line.find("참조")]
        refs.extend(table_keys(before[-120:]))
    return list(dict.fromkeys(refs))


def appendix_context(elements: list[dict], doc_keys: list[str] | None = None) -> dict[str, str]:
    """별첨 heading과 바로 이어진 table run에만 table key를 전파한다.

    문서 전체의 마지막 ``표 N`` heading을 뒤의 모든 table에 전파하면 서로 다른
    특약/부록이 같은 표로 연결된다. 따라서 heading 다음의 연속 table element만
    허용하고, 다른 paragraph를 만나거나 line gap이 벌어지면 context를 닫는다.
    """
    context = ""
    frontier_line = -10_000
    output = {}
    prior_doc = None
    for index, row in enumerate(elements):
        doc_key = doc_keys[index] if doc_keys is not None else ""
        if prior_doc is not None and doc_key != prior_doc:
            context = ""
            frontier_line = -10_000
        prior_doc = doc_key
        text = str(row.get("text", ""))
        non_reference = "\n".join(line for line in text.splitlines() if "참조" not in line)
        labels = table_keys(non_reference)
        if labels:
            context = labels[-1]
            frontier_line = int(row.get("line_end", row.get("line_start", 0)) or 0)
            continue
        start = int(row.get("line_start", 0) or 0)
        if (context and row.get("element_type") == "table"
                and 0 <= start - frontier_line <= 8):
            output[row["element_id"]] = context
            frontier_line = int(row.get("line_end", start) or start)
        elif context:
            context = ""
            frontier_line = -10_000
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--elements", default=str(HERE / "out/elements_u3.jsonl"))
    parser.add_argument("--tags", default=str(HERE / "out/tags_u3_rules.jsonl"))
    parser.add_argument("--jo", default=str(HERE / "out/elements_u3jo.jsonl"))
    parser.add_argument("--out", default=str(HERE / "out/tags_u4_fact_rules.jsonl"))
    args = parser.parse_args()

    element_path, tag_path, jo_path, output_path = map(Path, (args.elements, args.tags, args.jo, args.out))
    elements, tags, jo_rows = map(load_jsonl, (element_path, tag_path, jo_path))
    if len(elements) != len(tags):
        raise ValueError(f"element/tag length mismatch: {len(elements)} != {len(tags)}")
    if [e["element_id"] for e in elements] != [t["element_id"] for t in tags]:
        raise ValueError("element/tag id order mismatch")

    member_to_jo = {}
    for jo in jo_rows:
        for member in jo.get("members", []):
            if member in member_to_jo:
                raise ValueError(f"duplicate JO member: {member}")
            member_to_jo[member] = jo["element_id"]
    if set(member_to_jo) != {e["element_id"] for e in elements}:
        raise ValueError("JO membership is not an exact element partition")

    doc_keys = document_keys(elements)
    contexts = appendix_context(elements, doc_keys)
    incoming_sources = defaultdict(list)
    for element, tag, doc_key in zip(elements, tags, doc_keys):
        identity = str(tag.get("contract_key") or element.get("contract_scope") or "").strip()
        for label in explicit_table_refs(str(element.get("text", ""))):
            if identity:
                # 같은 약관 bundle 안에는 서로 다른 특약의 표2-1이 반복된다.
                # target parser scope와 source referencer scope가 정확히 같을 때만
                # edge를 열어 label-only cross-contract collision을 fail-closed 한다.
                key = (doc_key, identity, label)
                if element["element_id"] not in incoming_sources[key]:
                    incoming_sources[key].append(element["element_id"])

    output = []
    counts = defaultdict(int)
    for element, source_tag, doc_key in zip(elements, tags, doc_keys):
        tag = copy.deepcopy(source_tag)
        text = str(element.get("text", ""))
        anchors = evidence_anchors(element)
        aliases = benefit_aliases(text)
        roles = fact_roles(text)
        refs = explicit_table_refs(text)
        appendix_key = contexts.get(element["element_id"], "")
        target_identity = str(
            source_tag.get("contract_key") or element.get("contract_scope") or "").strip()
        reference_sources = sorted(
            incoming_sources.get((doc_key, target_identity, appendix_key), ())
        )

        # 기존 U3 schema_version/role을 그대로 둬 fact 축 weight=0일 때 순위를
        # byte-for-byte 동작 호환되게 한다. 새 provenance는 adapter가 무시한다.
        tag["fact_tag_version"] = VERSION
        if doc_key:
            tag["document_key"] = doc_key
        tag["parent_jo"] = member_to_jo[element["element_id"]]
        if anchors:
            tag["evidence_anchor"] = anchors
        values = unique(VALUE_RE.findall(text) + CODE_RE.findall(text), 30)
        if values:
            tag["answer_values"] = values
        if aliases:
            tag["benefit_aliases"] = aliases
        if roles:
            tag["fact_roles"] = roles
        if refs:
            tag["explicit_table_references"] = refs
        if appendix_key:
            tag["appendix_key"] = appendix_key
        if reference_sources:
            # 검색 evidence로 identity 문자열을 복제하지 않는다. source/target
            # 원 ID는 provenance 전용이며 schema adapter가 점수 필드로 쓰지 않는다.
            tag["reference_source_element_ids"] = reference_sources
            tag["reference_target_element_id"] = element["element_id"]

        counts["anchors"] += len(anchors)
        counts["rows_with_anchor"] += bool(anchors)
        counts["rows_with_alias"] += bool(aliases)
        counts["rows_with_appendix_context"] += bool(appendix_key)
        counts["rows_with_reference_edge"] += bool(reference_sources)
        counts["reference_edges"] += len(reference_sources)
        output.append(tag)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as handle:
        for row in output:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    stats = {
        "version": VERSION,
        "n": len(output),
        "inputs": {"elements": sha256(element_path), "tags": sha256(tag_path), "jo": sha256(jo_path)},
        "output_sha256": sha256(output_path),
        "document_key": doc_keys[0] if len(set(doc_keys)) == 1 else "",
        "document_count": len(set(doc_keys)),
        "reference_scope_key": "document_id+exact_contract_key+table_namespace+table_id",
        "reference_key_count": len(incoming_sources),
        "multi_source_reference_key_count": sum(
            1 for sources in incoming_sources.values() if len(sources) > 1),
        "reference_namespaces": dict(sorted(
            (namespace, sum(1 for key in incoming_sources if key[2].startswith(namespace + ":")))
            for namespace in ("부표", "별첨", "표")
        )),
        "counts": dict(sorted(counts.items())),
        "invariants": {
            "qa_or_gold_read": False,
            "element_id_order_preserved": True,
            "jo_membership_exact": True,
        },
    }
    stats_path = output_path.with_name(output_path.stem + "_stats.json")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(stats, ensure_ascii=False))


if __name__ == "__main__":
    main()

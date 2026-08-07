"""
DR-DCI Workspace: 에이전트가 Pull한 문서를 저장하고 DCI 도구로 탐색하는 공간

가이드 반영 사항:
- P0-6: tag grammar 통일 매칭 (@el:paragraph/evidence == @el:evidence == evidence)
- P0-6: tag가 없는 문서는 tag_data_missing으로 명시적 보고 (silent raw fallback 제거)
- P1-3: read한 문서 추적 (read recall 계산용)
- P1-6: metadata filter AND semantics + entity name/category 필드 분리
- P0-9: max_docs 설정화 (config의 workspace_max_docs가 실제로 반영되도록)
"""

import re
from dataclasses import dataclass, field


@dataclass
class Document:
    doc_id: str
    title: str
    text: str
    taxonomy: dict = field(default_factory=dict)
    tags: list = field(default_factory=list)
    prefix: str = ""
    metadata: dict = field(default_factory=dict)


def normalize_tag(tag: str) -> str:
    """tag 문자열을 canonical 형태(마지막 하위 태그)로 정규화.

    '@el:paragraph/evidence' -> 'evidence'
    '@el:evidence'           -> 'evidence'
    'evidence'               -> 'evidence'
    """
    t = tag.strip().lower()
    if t.startswith("@el:"):
        t = t[len("@el:"):]
    if "/" in t:
        t = t.rsplit("/", 1)[-1]
    return t


def tag_matches(doc_tag: str, tag_filter: str) -> bool:
    """저장된 tag와 filter를 canonical 형태로 비교. 전체 경로 일치도 허용."""
    dt = doc_tag.strip().lower()
    tf = tag_filter.strip().lower()
    if dt == tf:
        return True
    return normalize_tag(dt) == normalize_tag(tf)


class Workspace:
    def __init__(self, max_docs: int = 100):
        self.docs: dict[str, Document] = {}
        self.max_docs = max_docs
        self.read_ids: set[str] = set()
        self.rejected_due_to_capacity: int = 0

    def add(self, doc: Document) -> bool:
        if doc.doc_id in self.docs:
            return False
        if len(self.docs) >= self.max_docs:
            self.rejected_due_to_capacity += 1
            return False
        self.docs[doc.doc_id] = doc
        return True

    def grep(self, pattern: str, tag_filter: str = None) -> dict:
        """workspace 내 문서에서 패턴 검색 (DCI grep).

        반환: {"matches": [...], "docs_searched": n, "tag_data_missing": m}
        tag_filter가 주어졌는데 tag가 없는 문서는 검색 대상에서 빠지며
        그 수를 tag_data_missing으로 명시한다 (raw fallback 없음).
        """
        matches = []
        tag_missing = 0
        docs_searched = 0
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        for doc in self.docs.values():
            if tag_filter:
                if not doc.tags:
                    tag_missing += 1
                    continue
                docs_searched += 1
                for elem in doc.tags:
                    if tag_matches(elem.get("tag", ""), tag_filter):
                        if regex.search(elem.get("text", "")):
                            matches.append({
                                "doc_id": doc.doc_id,
                                "title": doc.title[:50],
                                "match": elem["text"][:200],
                                "tag": elem.get("tag", ""),
                            })
            else:
                docs_searched += 1
                doc_matches = 0
                for line in doc.text.split("\n"):
                    if regex.search(line):
                        matches.append({
                            "doc_id": doc.doc_id,
                            "title": doc.title[:50],
                            "match": line[:200],
                        })
                        doc_matches += 1
                        if doc_matches >= 5:
                            break

        result = {"matches": matches, "docs_searched": docs_searched}
        if tag_filter:
            result["tag_data_missing"] = tag_missing
        return result

    def find(self, taxonomy_filter: dict = None, metadata_filter: dict = None) -> dict:
        """workspace 내 문서 필터링 (DCI find).

        metadata_filter는 AND semantics: 모든 field 조건을 만족해야 매칭.
        entity 검색은 field를 분리한다.
        - entity_names: list[str] — 문서 entities에 이름 중 하나라도 있으면 해당 field 만족
        - entity_category: str — 문서 entities에 해당 category가 있으면 만족
        (구버전 'entities' 키도 type에 따라 위 두 의미로 해석해 하위 호환)
        """
        results = []
        for doc in self.docs.values():
            if taxonomy_filter and not all(
                doc.taxonomy.get(k) == v for k, v in taxonomy_filter.items()
            ):
                continue

            if metadata_filter and not self._metadata_matches(doc, metadata_filter):
                continue

            results.append(doc.doc_id)

        return {
            "doc_ids": results,
            "matched": len(results),
            "searched": len(self.docs),
            "semantics": "AND",
        }

    @staticmethod
    def _metadata_matches(doc: Document, metadata_filter: dict) -> bool:
        for key, val in metadata_filter.items():
            if key in ("entities", "entity_names", "entity_category"):
                doc_entities = doc.metadata.get("entities", [])
                doc_names = {e.get("name", "").lower() for e in doc_entities if isinstance(e, dict)}
                doc_cats = {e.get("category", "") for e in doc_entities if isinstance(e, dict)}
                if key == "entity_names" or (key == "entities" and isinstance(val, list)):
                    names = val if isinstance(val, list) else [val]
                    if not any(str(v).lower() in doc_names for v in names):
                        return False
                else:  # entity_category, or legacy 'entities' with str value
                    if str(val) not in doc_cats:
                        return False
            else:
                if doc.metadata.get(key) != val:
                    return False
        return True

    def read(self, doc_id: str) -> str | None:
        """workspace 내 특정 문서 읽기 (DCI read). 읽은 문서를 추적한다."""
        doc = self.docs.get(doc_id)
        if doc:
            self.read_ids.add(doc_id)
            parts = [f"[{doc.doc_id}] {doc.title}"]
            if doc.prefix:
                parts.append(f"Summary: {doc.prefix}")
            parts.append(doc.text)
            return "\n".join(parts)
        return None

    def summary(self) -> dict:
        return {
            "doc_count": len(self.docs),
            "doc_ids": list(self.docs.keys()),
            "read_count": len(self.read_ids),
            "capacity": self.max_docs,
            "rejected_due_to_capacity": self.rejected_due_to_capacity,
        }

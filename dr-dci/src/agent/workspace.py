"""
DR-DCI Workspace: 에이전트가 Pull한 문서를 저장하고 DCI 도구로 탐색하는 공간
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


class Workspace:
    def __init__(self, max_docs: int = 100):
        self.docs: dict[str, Document] = {}
        self.max_docs = max_docs

    def add(self, doc: Document):
        if len(self.docs) >= self.max_docs:
            return False
        self.docs[doc.doc_id] = doc
        return True

    def grep(self, pattern: str, tag_filter: str = None) -> list[dict]:
        """workspace 내 문서에서 패턴 검색 (DCI grep)"""
        results = []
        try:
            regex = re.compile(pattern, re.IGNORECASE)
        except re.error:
            regex = re.compile(re.escape(pattern), re.IGNORECASE)

        for doc in self.docs.values():
            # tag_filter가 있으면 해당 태그의 element만 검색
            if tag_filter and doc.tags:
                for elem in doc.tags:
                    if tag_filter in elem.get("tag", ""):
                        if regex.search(elem.get("text", "")):
                            results.append({
                                "doc_id": doc.doc_id,
                                "match": elem["text"][:200],
                                "tag": elem.get("tag", ""),
                            })
            else:
                if regex.search(doc.text):
                    # 매칭 라인 추출
                    for line in doc.text.split("\n"):
                        if regex.search(line):
                            results.append({
                                "doc_id": doc.doc_id,
                                "match": line[:200],
                            })
                            break

        return results

    def find(self, taxonomy_filter: dict = None, metadata_filter: dict = None) -> list[str]:
        """workspace 내 문서 필터링 (DCI find)"""
        results = []
        for doc in self.docs.values():
            match = True

            if taxonomy_filter:
                for key, val in taxonomy_filter.items():
                    if doc.taxonomy.get(key) != val:
                        match = False
                        break

            if metadata_filter and match:
                meta_match = False
                for key, val in metadata_filter.items():
                    if key == "entities":
                        doc_entities = doc.metadata.get("entities", [])
                        doc_names = {e.get("name", "").lower() for e in doc_entities if isinstance(e, dict)}
                        if isinstance(val, list):
                            if any(v.lower() in doc_names for v in val):
                                meta_match = True
                        elif isinstance(val, str):
                            doc_cats = {e.get("category", "") for e in doc_entities if isinstance(e, dict)}
                            if val in doc_cats:
                                meta_match = True
                    elif doc.metadata.get(key) == val:
                        meta_match = True
                match = meta_match

            if match:
                results.append(doc.doc_id)

        return results

    def read(self, doc_id: str) -> str | None:
        """workspace 내 특정 문서 읽기 (DCI read)"""
        doc = self.docs.get(doc_id)
        if doc:
            return f"[{doc.doc_id}] {doc.title}\n{doc.text}"
        return None

    def summary(self) -> dict:
        return {
            "doc_count": len(self.docs),
            "doc_ids": list(self.docs.keys()),
        }

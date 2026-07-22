"""P0-6, P1-3, P1-6, P2: workspace 동작 검증"""
from src.agent.workspace import Workspace, Document, normalize_tag, tag_matches


def make_doc(doc_id, text="", tags=None, taxonomy=None, metadata=None, prefix=""):
    return Document(doc_id=doc_id, title=f"title-{doc_id}", text=text,
                    tags=tags or [], taxonomy=taxonomy or {},
                    metadata=metadata or {}, prefix=prefix)


# ---- capacity / dedup ----

def test_capacity_enforced():
    ws = Workspace(max_docs=2)
    assert ws.add(make_doc("a"))
    assert ws.add(make_doc("b"))
    assert not ws.add(make_doc("c"))  # 초과
    assert ws.rejected_due_to_capacity == 1
    assert len(ws.docs) == 2


def test_duplicate_add_ignored():
    ws = Workspace(max_docs=10)
    assert ws.add(make_doc("a"))
    assert not ws.add(make_doc("a"))  # 중복
    assert len(ws.docs) == 1


# ---- tag grammar (P0-6) ----

def test_normalize_tag():
    assert normalize_tag("@el:paragraph/evidence") == "evidence"
    assert normalize_tag("@el:evidence") == "evidence"
    assert normalize_tag("evidence") == "evidence"


def test_tag_matches_across_grammars():
    assert tag_matches("@el:paragraph/evidence", "@el:evidence")
    assert tag_matches("@el:paragraph/evidence", "evidence")
    assert tag_matches("@el:table/definition", "definition")
    assert not tag_matches("@el:paragraph/evidence", "definition")


def test_grep_tag_filter_matches_stored_grammar():
    ws = Workspace()
    ws.add(make_doc("a", tags=[{"tag": "@el:paragraph/evidence", "text": "vaccine reduces risk"}]))
    # agent가 짧은 형식으로 필터해도 매칭돼야 함
    res = ws.grep("vaccine", tag_filter="evidence")
    assert len(res["matches"]) == 1
    assert res["tag_data_missing"] == 0


def test_grep_reports_missing_tag_data():
    ws = Workspace()
    ws.add(make_doc("a", text="vaccine", tags=[]))  # tag 없음
    res = ws.grep("vaccine", tag_filter="evidence")
    assert res["matches"] == []
    assert res["tag_data_missing"] == 1  # silent fallback 아님


# ---- metadata AND semantics (P1-6) ----

def test_metadata_filter_is_AND():
    ws = Workspace()
    ws.add(make_doc("both", metadata={"study_type": "clinical_trial", "population": "elderly"}))
    ws.add(make_doc("one", metadata={"study_type": "clinical_trial", "population": "young"}))
    res = ws.find(metadata_filter={"study_type": "clinical_trial", "population": "elderly"})
    assert res["doc_ids"] == ["both"]  # OR였다면 둘 다 반환됐음
    assert res["semantics"] == "AND"


def test_entity_names_vs_category_separated():
    ws = Workspace()
    ws.add(make_doc("a", metadata={"entities": [{"name": "COVID-19", "category": "disease"}]}))
    assert ws.find(metadata_filter={"entity_names": ["covid-19"]})["doc_ids"] == ["a"]
    assert ws.find(metadata_filter={"entity_category": "disease"})["doc_ids"] == ["a"]
    assert ws.find(metadata_filter={"entity_category": "drug"})["doc_ids"] == []


# ---- read tracking (P1-3) ----

def test_read_tracks_ids():
    ws = Workspace()
    ws.add(make_doc("a", text="body"))
    assert ws.read("a") is not None
    assert ws.read("missing") is None
    assert ws.read_ids == {"a"}
    assert ws.summary()["read_count"] == 1

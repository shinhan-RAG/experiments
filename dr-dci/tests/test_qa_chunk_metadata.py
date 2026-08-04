import json
from pathlib import Path

from run_experiment import part6_build_aug_texts, write_part6_comparison_report
from scripts.build_tos_qa_chunk_metadata_openai import build_payload
from src.retrieval.qa_metadata import (
    JSON_SCHEMA,
    QA_COMBINED_FIELDS,
    QUESTION_INTENTS,
    metadata_quality_report,
    normalize_metadata,
    serialize_metadata_fields,
    validate_metadata,
)


def valid_metadata(**overrides):
    value = {
        "핵심주제": "암 진단급여",
        "질문의도": ["지급 조건·진단 요건", "금액·비율·계산"],
        "시맨틱태그": ["질병보장>지급 조건·진단 요건>일반암 진단"],
        "핵심대상": ["일반암진단급여금"],
        "수치조건": ["계약일부터 90일", "보험가입금액의 50%"],
        "제외제한": ["90일 이전 진단 제외"],
        "결과급부": ["진단급여금 지급"],
        "질의별칭": ["암 진단금은 언제부터 받을 수 있나요?", "암 진단금은 얼마인가요?"],
        "검색키워드": ["암진단금", "암보험금", "90일 면책"],
    }
    value.update(overrides)
    return value


def test_schema_and_application_validation_accept_contract():
    assert JSON_SCHEMA["additionalProperties"] is False
    assert JSON_SCHEMA["properties"]["질문의도"]["maxItems"] == 3
    assert JSON_SCHEMA["properties"]["질의별칭"]["minItems"] == 2
    assert JSON_SCHEMA["properties"]["검색키워드"]["maxItems"] == 8
    assert validate_metadata(valid_metadata()) == []


def test_rejects_unknown_intent_bad_tag_and_array_overflow():
    bad = valid_metadata(
        질문의도=["임의 의도"],
        시맨틱태그=["두단계>태그"],
        검색키워드=[str(i) for i in range(9)],
    )
    errors = " | ".join(validate_metadata(bad))
    assert "controlled vocabulary" in errors
    assert "exactly three" in errors
    assert "0..8" in errors
    assert set(QUESTION_INTENTS) == set(JSON_SCHEMA["properties"]["질문의도"]["items"]["enum"])


def test_normalization_removes_generic_terms_and_exact_qa_alias():
    metadata = valid_metadata(
        핵심대상=["보험", "일반암진단급여금"],
        질의별칭=["암 진단금은 언제부터 받을 수 있나요?", "다른 질문인가요?"],
    )
    cleaned, removed = normalize_metadata(
        metadata, qa_questions={"암 진단금은 언제부터 받을 수 있나요"}
    )
    assert cleaned["핵심대상"] == ["일반암진단급여금"]
    assert cleaned["질의별칭"] == ["다른 질문인가요?"]
    assert removed == 1


def test_selected_field_serialization_is_capped_at_240_characters():
    metadata = valid_metadata(검색키워드=["매우긴검색어" * 20 for _ in range(8)])
    rendered = serialize_metadata_fields(metadata, QA_COMBINED_FIELDS, max_chars=240)
    assert len(rendered) <= 240
    assert "핵심주제:" not in rendered
    assert rendered.startswith("시맨틱태그:")


def test_part6_metadata_fields_produce_ablation_from_one_source():
    corpus = [{"_id": "c1"}]
    metadata = valid_metadata()
    maps = {"qa_semantic": {"c1": metadata}}
    facets = part6_build_aug_texts(
        corpus,
        {
            "index_metadata": True,
            "metadata_source": "qa_semantic",
            "metadata_fields": ["시맨틱태그", "수치조건"],
            "metadata_max_chars": 240,
        },
        maps,
    )["c1"]
    questions = part6_build_aug_texts(
        corpus,
        {
            "index_metadata": True,
            "metadata_source": "qa_semantic",
            "metadata_fields": ["질의별칭"],
            "metadata_max_chars": 240,
        },
        maps,
    )["c1"]
    assert "시맨틱태그:" in facets and "수치조건:" in facets
    assert "질의별칭:" not in facets
    assert questions.startswith("질의별칭:")


def test_api_payload_has_chunk_only_content_and_structured_schema():
    chunk = {
        "_id": "c1",
        "section": "제3조",
        "element_type": "formula",
        "text": "보험가입금액의 50%를 지급한다.",
        "answer": "must not appear",
        "gold_chunk_id": "must not appear",
        "supporting_spans": ["must not appear"],
    }
    payload_text = json.dumps(build_payload(chunk, "gpt-4o-mini"), ensure_ascii=False)
    assert "보험가입금액의 50%" in payload_text
    assert "must not appear" not in payload_text
    assert '"strict": true' in payload_text


def test_quality_report_enforces_coverage_uniqueness_and_zero_copy():
    corpus = [
        {"_id": "c1", "doc": "d"},
        {"_id": "c2", "doc": "d"},
    ]
    metadata = {
        "c1": valid_metadata(),
        "c2": valid_metadata(
            핵심주제="수술 횟수한도",
            시맨틱태그=["수술보장>횟수·한도>연간 수술"],
            핵심대상=["수술급여금"],
        ),
    }
    report = metadata_quality_report(corpus, metadata, {"평가 질문 원문"})
    assert report["coverage"] == 1.0
    assert report["within_document_unique_ratio"] == 1.0
    assert report["exact_qa_copy_count"] == 0
    assert all(report["gates"].values())


def test_comparison_report_falls_back_to_best_ablation(tmp_path: Path):
    def result(r5, r20, rank):
        return {"metrics": {
            "recall_at_1": 0.2, "recall_at_5": r5, "recall_at_10": 0.6,
            "recall_at_20": r20, "mrr": 0.3, "ndcg_at_10": 0.4,
            "mean_gold_rank": rank,
        }}

    results = {
        "baseline__hybrid_rrf": result(0.465, 0.8, 9.0),
        "qa_facets__hybrid_rrf": result(0.47, 0.81, 8.5),
        "qa_questions__hybrid_rrf": result(0.46, 0.80, 9.1),
        "qa_combined__hybrid_rrf": result(0.464, 0.81, 8.7),
    }
    path = tmp_path / "report.md"
    write_part6_comparison_report(
        path, results, {}, baseline_name="baseline", target_recall_at_5=0.465
    )
    text = path.read_text(encoding="utf-8")
    assert "최종 권장 조건: `qa_facets`" in text


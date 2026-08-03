"""측정 설계 교정 G: 질의 variant 파일 선택 로직 검증"""
import json
import pytest

import run_experiment as rx


def _write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows) + "\n",
                    encoding="utf-8")


@pytest.fixture
def toy_dataset(tmp_path, monkeypatch):
    d = tmp_path / "raw" / "toy"
    d.mkdir(parents=True)
    _write_jsonl(d / "queries.jsonl", [{"_id": "1", "text": "orig"}])
    _write_jsonl(d / "queries_paraphrase.jsonl", [{"_id": "1", "text": "para"}])
    _write_jsonl(d / "qrels.jsonl", [{"query-id": 1, "corpus-id": "c1", "score": 1}])
    monkeypatch.setattr(rx, "DATA_DIR", tmp_path)
    return d


def test_default_uses_original_queries(toy_dataset):
    queries, qrels = rx.load_queries("toy")
    assert queries[0]["text"] == "orig"
    assert qrels[0]["corpus-id"] == "c1"


def test_variant_selects_paraphrase_file(toy_dataset):
    queries, _ = rx.load_queries("toy", variant="paraphrase")
    assert queries[0]["text"] == "para"


def test_missing_variant_fails_loudly(toy_dataset):
    # 조용한 원본 폴백 금지 — 결과 라벨과 실제 질의가 어긋나는 것을 차단
    with pytest.raises(FileNotFoundError):
        rx.load_queries("toy", variant="nope")

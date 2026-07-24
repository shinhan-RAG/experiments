import pytest

from scripts import run_pilot


def test_pilot_model_failure_is_not_silently_converted(monkeypatch):
    monkeypatch.setattr(run_pilot, "OAI_KEY", "test-key")
    monkeypatch.setattr(run_pilot.time, "sleep", lambda _: None)

    def fail(*args, **kwargs):
        raise OSError("endpoint unavailable")

    monkeypatch.setattr(run_pilot.requests, "post", fail)
    with pytest.raises(RuntimeError, match="failed after retries"):
        run_pilot.oai([{"role": "user", "content": "test"}])


def test_pilot_rejects_malformed_structured_output():
    with pytest.raises(ValueError, match="valid JSON"):
        run_pilot.parse_json("not-json")


def _mk_corpus(ids):
    return {d: {"_id": d, "title": d, "text": f"text about {d}"} for d in ids}


def _patch_scale(monkeypatch, n_queries, k_gold, scales):
    monkeypatch.setattr(run_pilot, "N_QUERIES", n_queries)
    monkeypatch.setattr(run_pilot, "K_GOLD_PER_Q", k_gold)
    monkeypatch.setattr(run_pilot, "SCALES", scales)


def test_distractors_never_contain_positive_qrels(monkeypatch):
    """선택된 질의의 전체 positive qrel(미선택 gold 포함)은 distractor가 될 수 없다."""
    _patch_scale(monkeypatch, n_queries=2, k_gold=2, scales=[6, 14])
    gold = {
        "q1": ["g1", "g2", "g3", "g4"],  # K=2 → g3, g4는 평가에서 제외되는 positive
        "q2": ["g5", "g6", "g7"],
    }
    corpus = _mk_corpus([f"g{i}" for i in range(1, 8)] + [f"d{i}" for i in range(10)])

    chosen, query_gold, subsets, _ = run_pilot.build_pilot_data(corpus, {}, gold)

    all_positives = set()
    for qid, _ in chosen:
        all_positives.update(gold[qid])
    picked = set()
    for ids in query_gold.values():
        picked.update(ids)

    for s, ids in subsets.items():
        assert len(ids) == s
        distractors = set(ids) - picked
        assert not (distractors & all_positives), (
            f"scale {s}: positive qrel이 distractor로 포함됨: "
            f"{sorted(distractors & all_positives)}"
        )


def test_pilot_rejects_gold_exceeding_min_scale(monkeypatch):
    """gold_union이 최소 scale보다 크면 subset이 scale을 초과하므로 명시적으로 거부한다."""
    _patch_scale(monkeypatch, n_queries=2, k_gold=2, scales=[3])
    gold = {"q1": ["g1", "g2"], "q2": ["g3", "g4"]}
    corpus = _mk_corpus(["g1", "g2", "g3", "g4"] + [f"d{i}" for i in range(5)])

    with pytest.raises(ValueError, match="min.*scale|scale.*gold|gold.*scale"):
        run_pilot.build_pilot_data(corpus, {}, gold)


def test_pilot_rejects_insufficient_distractors(monkeypatch):
    """distractor 후보가 부족해 subset이 scale에 못 미치면 조용히 줄이지 않고 거부한다."""
    _patch_scale(monkeypatch, n_queries=2, k_gold=2, scales=[20])
    gold = {"q1": ["g1", "g2"], "q2": ["g3", "g4"]}
    corpus = _mk_corpus(["g1", "g2", "g3", "g4"] + [f"d{i}" for i in range(5)])

    with pytest.raises(ValueError, match="distractor"):
        run_pilot.build_pilot_data(corpus, {}, gold)

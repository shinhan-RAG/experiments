"""P1-1, P1-11: retrieval metrics + 통계 도구 검증"""
from src.eval.metrics import (
    recall_at_k, ndcg_at_k, mrr_at_k, bootstrap_ci, paired_bootstrap_test,
)


def test_recall_at_k():
    ranked = ["a", "b", "c", "d"]
    gold = {"a", "d", "z"}
    assert recall_at_k(ranked, gold, 2) == 1 / 3   # a만 top2
    assert recall_at_k(ranked, gold, 4) == 2 / 3   # a, d


def test_recall_empty_gold():
    assert recall_at_k(["a"], set(), 10) == 0.0


def test_ndcg_perfect_ranking_is_one():
    ranked = ["a", "b", "c"]
    scores = {"a": 2, "b": 1, "c": 1}
    assert abs(ndcg_at_k(ranked, scores, 3) - 1.0) < 1e-9


def test_ndcg_worse_ranking_below_one():
    scores = {"a": 2, "b": 1, "c": 1}
    good = ndcg_at_k(["a", "b", "c"], scores, 3)
    bad = ndcg_at_k(["c", "b", "a"], scores, 3)
    assert bad < good


def test_mrr():
    assert mrr_at_k(["x", "a", "y"], {"a"}, 10) == 0.5
    assert mrr_at_k(["x", "y"], {"a"}, 10) == 0.0


def test_bootstrap_ci_bounds():
    values = [0.5] * 50
    ci = bootstrap_ci(values)
    assert ci["mean"] == 0.5
    assert ci["ci_low"] == 0.5 and ci["ci_high"] == 0.5


def test_bootstrap_ci_contains_mean():
    values = [0.0, 1.0] * 25
    ci = bootstrap_ci(values)
    assert ci["ci_low"] <= ci["mean"] <= ci["ci_high"]


def test_paired_bootstrap_detects_difference():
    # a가 항상 b보다 0.3 높음 → 유의한 양의 차이
    a = [0.8] * 40
    b = [0.5] * 40
    res = paired_bootstrap_test(a, b)
    assert res["mean_diff"] == 0.3
    assert res["p_value"] < 0.05
    assert res["ci_low"] > 0


def test_paired_bootstrap_no_difference():
    a = [0.5, 0.6, 0.4] * 10
    res = paired_bootstrap_test(a, a)
    assert res["mean_diff"] == 0.0

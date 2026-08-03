"""측정 설계 교정 C: paired delta 블록의 절대값·상대%·p-value 검증"""
from src.eval.comparison import paired_bootstrap_delta


def test_delta_reports_absolute_means_and_relative_pct():
    control = [0.5] * 40
    treatment = [0.6] * 40
    res = paired_bootstrap_delta(control, treatment, seed=42)
    assert res["control_mean"] == 0.5
    assert res["treatment_mean"] == 0.6
    assert abs(res["mean_delta"] - 0.1) < 1e-9
    assert abs(res["relative_pct"] - 20.0) < 1e-6


def test_relative_pct_none_when_control_zero():
    res = paired_bootstrap_delta([0.0] * 20, [0.1] * 20, seed=42)
    assert res["relative_pct"] is None
    assert res["control_mean"] == 0.0


def test_pvalue_small_for_consistent_difference():
    # 전 질의에서 일관된 +0.3 → sign-flip 귀무분포에서 극단
    control = [0.5] * 40
    treatment = [0.8] * 40
    res = paired_bootstrap_delta(control, treatment, seed=42)
    assert res["p_value"] < 0.01


def test_pvalue_large_for_no_difference():
    vals = [0.5, 0.6, 0.4, 0.7] * 10
    res = paired_bootstrap_delta(vals, vals, seed=42)
    # 델타 전부 0 → randomization도 전부 0 → p=1
    assert res["p_value"] >= 0.99
    assert res["mean_delta"] == 0.0


def test_pvalue_large_for_noise():
    # 부호가 뒤섞인 작은 노이즈 → 유의하지 않아야 한다
    control = [0.5, 0.6, 0.4, 0.7, 0.5, 0.6] * 8
    treatment = [0.51, 0.58, 0.42, 0.69, 0.52, 0.58] * 8
    res = paired_bootstrap_delta(control, treatment, seed=42)
    assert res["p_value"] > 0.05

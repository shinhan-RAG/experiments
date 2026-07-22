"""P0-1: judge exact parsing 검증"""
from src.eval.judge import parse_judgment, compute_metrics


def test_incorrect_is_not_correct():
    # 기존 버그: 'incorrect'에 'correct'가 포함되어 correct로 오판정
    assert parse_judgment("incorrect") == "incorrect"


def test_correct():
    assert parse_judgment("correct") == "correct"
    assert parse_judgment("Correct") == "correct"
    assert parse_judgment(" CORRECT\n") == "correct"
    assert parse_judgment('"correct".') == "correct"


def test_empty_is_format_error():
    assert parse_judgment("") == "format_error"
    assert parse_judgment(None) == "format_error"


def test_explanatory_response_is_format_error():
    assert parse_judgment("The answer is correct because...") == "format_error"
    assert parse_judgment("Judgment: correct") == "format_error"


def test_single_line_correct_allowed():
    assert parse_judgment("correct\n") == "correct"


def test_compute_metrics_excludes_na_from_accuracy():
    results = [
        {"judgment": "correct"},
        {"judgment": "incorrect"},
        {"judgment": "n/a"},
    ]
    m = compute_metrics(results)
    assert m["n"] == 3
    assert m["n_judged"] == 2
    assert m["accuracy"] == 0.5  # 2개만 분모


def test_compute_metrics_all_na_gives_none_accuracy():
    results = [{"judgment": "n/a"}, {"judgment": "n/a"}]
    m = compute_metrics(results)
    assert m["accuracy"] is None


def test_compute_metrics_counts_judge_errors():
    results = [
        {"judgment": "correct"},
        {"judgment": "format_error"},
        {"judgment": "error"},
    ]
    m = compute_metrics(results)
    assert m["judge_error_count"] == 2
    assert m["accuracy"] == 1.0  # correct 1 / judged 1

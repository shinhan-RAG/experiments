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

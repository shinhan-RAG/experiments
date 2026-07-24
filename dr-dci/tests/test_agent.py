"""P0-3, P0-9, single-pull: agent harness 규칙 검증 (mock LLM + mock retriever)"""
import numpy as np
import pytest
from src.agent.dci_agent import DCIAgent, normalize_query
from src.agent.retriever import PullRetriever, RetrieverConfig


class MockRetriever(PullRetriever):
    def __init__(self, n_docs, config):
        super().__init__(config)
        self.n_docs = n_docs

    def _embed_batch(self, texts, batch_size=256):
        out = []
        for t in texts:
            idx = int("".join(ch for ch in t if ch.isdigit()) or 0)
            vec = np.array([1.0 / (1 + abs(j - idx)) for j in range(self.n_docs)])
            out.append(vec)
        return out


def build_retriever(n=30):
    cfg = RetrieverConfig(embedding_url="mock", embedding_model="mock", top_k=5,
                          query_instruction=None)
    r = MockRetriever(n, cfg)
    docs = [{"_id": f"d{i}", "title": f"d{i}", "text": f"doc text {i}"} for i in range(n)]
    r.index(docs)
    return r, {d["_id"]: d for d in docs}


class ScriptedAgent(DCIAgent):
    """_call_llm을 스크립트로 대체해 결정적 tool 호출 시퀀스를 재생한다."""

    def __init__(self, script, **kw):
        super().__init__(llm_url="mock", model_name="mock", **kw)
        self.script = list(script)
        self.step = 0

    def _call_llm(self, messages):
        if self.step >= len(self.script):
            return {"content": "done", "tool_calls": None}
        calls = self.script[self.step]
        self.step += 1
        if calls is None:
            return {"content": "plain text answer", "tool_calls": None}
        tool_calls = []
        for i, (name, args) in enumerate(calls):
            import json
            tool_calls.append({
                "id": f"c{self.step}_{i}",
                "function": {"name": name, "arguments": json.dumps(args)},
            })
        return {"content": None, "tool_calls": tool_calls}


def make_agent(script, single_pull=False, min_pulls=2, max_turns=10):
    r, corpus = build_retriever()
    return ScriptedAgent(
        script,
        retriever=r, corpus=corpus,
        max_turns=max_turns, workspace_max_docs=100,
        min_pulls=min_pulls, single_pull=single_pull,
    )


def test_answer_rejected_before_min_pulls():
    # 첫 turn에 바로 answer 시도 → 거부되고, 이후 정상 흐름
    script = [
        [("answer", {"text": "too early"})],
        [("pull", {"query": "q1"})],
        [("pull", {"query": "q2"})],
        [("answer", {"text": "final"})],
    ]
    agent = make_agent(script)
    out = agent.run("question")
    assert out["answer"] == "final"
    assert out["pull_count"] == 2
    assert "answer_rejected_min_pull_rule" in out["rule_violations"]


def test_answer_is_not_accepted_on_last_turn_before_min_pulls():
    script = [
        [("pull", {"query": "q1"})],
        [("answer", {"text": "must not escape the rule"})],
    ]
    out = make_agent(script, max_turns=2).run("question")
    assert out["answer"] == ""
    assert out["budget_exhausted"] is True
    assert "answer_rejected_min_pull_rule" in out["rule_violations"]


def test_duplicate_pull_queries_flagged():
    script = [
        [("pull", {"query": "same"})],
        [("pull", {"query": "SAME  "})],  # 정규화하면 동일
        [("answer", {"text": "final"})],
    ]
    agent = make_agent(script)
    out = agent.run("question")
    assert out["distinct_pull_queries"] == 1
    assert "duplicate_pull_queries" in out["rule_violations"]


def test_normal_two_distinct_pulls_no_violation():
    script = [
        [("pull", {"query": "covid vaccine"})],
        [("pull", {"query": "treatment options"})],
        [("read", {"doc_id": "d0"})],
        [("answer", {"text": "final"})],
    ]
    agent = make_agent(script)
    out = agent.run("question")
    assert out["answer"] == "final"
    assert out["distinct_pull_queries"] == 2
    assert out["rule_violations"] == []
    assert "d0" in out["read_docs"]


def test_pull_stats_recorded():
    script = [
        [("pull", {"query": "q0", "top_k": 5})],
        [("pull", {"query": "q0", "top_k": 5})],  # 같은 query지만 dedup+backfill 동작 확인
        [("answer", {"text": "final"})],
    ]
    agent = make_agent(script)
    out = agent.run("question")
    assert len(out["pull_stats"]) == 2
    # 두 번째 pull은 새 문서로 채워지므로 newly_added > 0
    assert out["pull_stats"][1]["newly_added"] > 0


def test_single_pull_mode_blocks_second_pull():
    script = [
        [("pull", {"query": "q0"})],
        [("pull", {"query": "q1"})],   # single-pull 모드에서 거부돼야 함
        [("answer", {"text": "final"})],
    ]
    agent = make_agent(script, single_pull=True)
    out = agent.run("question")
    assert out["pull_count"] == 1
    assert "extra_pull_in_single_pull_mode" in out["rule_violations"]
    assert out["answer"] == "final"  # single-pull은 min_pulls=1이라 answer 허용


def test_budget_exhausted_flag():
    # answer 없이 pull만 반복 → turn 소진
    script = [[("pull", {"query": f"q{i}"})] for i in range(10)]
    agent = make_agent(script, max_turns=3)
    out = agent.run("question")
    assert out["turns"] == 3
    assert out["budget_exhausted"] is True


def test_plain_text_early_answer_rejected_and_continues():
    # min_pulls 미달 상태의 일반 텍스트 답변은 answer 도구와 동일하게
    # 거부된 뒤 다음 턴이 계속 진행돼야 한다 (즉시 종료 금지)
    script = [
        [("pull", {"query": "q1"})],
        None,  # 텍스트로 조기 답변 시도
        [("pull", {"query": "q2"})],
        [("answer", {"text": "final"})],
    ]
    out = make_agent(script).run("question")
    assert out["answer"] == "final"
    assert "answered_without_min_pulls" in out["rule_violations"]
    assert out["termination_reason"] == "answered"


def test_termination_reason_answered():
    script = [
        [("pull", {"query": "q1"})],
        [("pull", {"query": "q2"})],
        [("answer", {"text": "final"})],
    ]
    out = make_agent(script).run("question")
    assert out["termination_reason"] == "answered"
    assert out["budget_exhausted"] is False


def test_termination_reason_turn_budget_exhausted():
    script = [[("pull", {"query": f"q{i}"})] for i in range(10)]
    out = make_agent(script, max_turns=3).run("question")
    assert out["termination_reason"] == "turn_budget_exhausted"
    assert out["budget_exhausted"] is True


def test_repeated_early_text_answers_end_as_budget_exhausted():
    # 거부가 반복되다 turn이 소진되면 answered가 아니라 budget 소진으로 끝난다
    script = [[("pull", {"query": "q1"})], None, None]
    out = make_agent(script, max_turns=3).run("question")
    assert out["answer"] == ""
    assert out["termination_reason"] == "turn_budget_exhausted"
    assert out["budget_exhausted"] is True


def test_normalize_query():
    assert normalize_query("  COVID  Vaccine ") == "covid vaccine"


# ---------------------------------------------------------------- _call_llm 오류 처리

class _FakeResponse:
    def __init__(self, status_code=200, body=None, text=""):
        self.status_code = status_code
        self._body = body
        self.text = text

    def json(self):
        if self._body is None:
            raise ValueError("no json")
        return self._body


def _bare_agent():
    return DCIAgent(llm_url="http://mock", model_name="m",
                    retriever=None, corpus={})


def _ok_body(content="ok"):
    return {"choices": [{"message": {"content": content, "tool_calls": None}}],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1}}


def test_call_llm_retries_timeout_and_5xx(monkeypatch):
    import requests
    from src.agent import dci_agent as mod

    responses = [
        requests.exceptions.Timeout("timed out"),
        _FakeResponse(status_code=503),
        _FakeResponse(status_code=200, body=_ok_body()),
    ]
    calls = {"n": 0}

    def post(*args, **kwargs):
        item = responses[calls["n"]]
        calls["n"] += 1
        if isinstance(item, Exception):
            raise item
        return item

    monkeypatch.setattr(mod.requests, "post", post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    out = _bare_agent()._call_llm([])
    assert out["content"] == "ok"
    assert calls["n"] == 3


def test_call_llm_raises_explicit_error_after_retries(monkeypatch):
    import requests
    from src.agent import dci_agent as mod
    from src.agent.dci_agent import LLMCallError

    def post(*args, **kwargs):
        raise requests.exceptions.Timeout("timed out")

    monkeypatch.setattr(mod.requests, "post", post)
    monkeypatch.setattr(mod.time, "sleep", lambda s: None)
    with pytest.raises(LLMCallError):
        _bare_agent()._call_llm([])


def test_call_llm_400_raises_instead_of_fake_answer(monkeypatch):
    # 400을 가짜 정상 답변으로 바꾸면 장애가 성능 결과로 둔갑한다
    from src.agent import dci_agent as mod
    from src.agent.dci_agent import LLMCallError

    monkeypatch.setattr(
        mod.requests, "post",
        lambda *a, **k: _FakeResponse(status_code=400, text="bad request"),
    )
    with pytest.raises(LLMCallError):
        _bare_agent()._call_llm([])


def test_run_records_llm_error_termination():
    from src.agent.dci_agent import LLMCallError

    agent = make_agent([])

    def boom(messages):
        raise LLMCallError("HTTP 500 after retries")

    agent._call_llm = boom
    out = agent.run("question")
    assert out["termination_reason"] == "llm_error"
    assert out["answer"] == ""
    assert "HTTP 500" in out["error"]

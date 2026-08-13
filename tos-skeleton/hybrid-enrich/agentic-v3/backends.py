"""Agent backend abstraction: CLI subprocess vs HTTP chat API."""
import json
import os
import shutil
import subprocess


class AgentBackend:
    """Protocol for agent backends."""
    name: str = "base"

    def run(self, prompt: str, session_dir: str, env: dict, cwd: str) -> tuple[str, dict]:
        """Run one agent session.
        Returns (response_text, usage_meta).
        usage_meta: {input_tokens, output_tokens, cache_read_tokens,
                     cache_creation_tokens, model, backend}
        """
        raise NotImplementedError


class ClaudeCliBackend(AgentBackend):
    """Current approach: claude -p subprocess."""
    name = "cli"

    def __init__(self, model: str = "claude-haiku-4-5-20251001", claude_bin: str | None = None):
        self.model = model
        # Windows: `claude` is an extensionless shell script subprocess can't find directly
        # (WinError 2) — same resolution order as run_retrieval.py.
        self.claude_bin = (
            claude_bin
            or os.environ.get("CLAUDE_BIN")
            or os.environ.get("CLAUDE_CODE_EXECPATH")
            or shutil.which("claude.cmd")
            or shutil.which("claude")
            or "claude"
        )

    def _empty_usage(self):
        return {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                 "cache_creation_tokens": 0, "model": self.model, "backend": self.name}

    def run(self, prompt: str, session_dir: str, env: dict, cwd: str) -> tuple[str, dict]:
        try:
            p = subprocess.run(
                [self.claude_bin, "-p", prompt, "--model", self.model,
                 "--allowedTools", "Bash(python tools.py:*)",
                 "--disallowedTools", "Read,Write,Edit,Grep,Glob,WebFetch,WebSearch,Agent,Task",
                 "--output-format", "json", "--max-turns", "24"],
                capture_output=True, text=True, timeout=420, cwd=cwd, env=env,
                encoding="utf-8", errors="replace")
        except subprocess.TimeoutExpired:
            return "", dict(self._empty_usage(), model=self.model)
        except Exception as e:
            return f"error: {e}", self._empty_usage()

        raw = p.stdout
        try:
            # `claude -p --output-format json` outer JSON:
            #   {"result": "...", "usage": {"input_tokens": N, ...}, "model": "...", ...}
            outer = json.loads(raw)
        except Exception:
            # Not valid outer JSON (crash, non-JSON stderr-on-stdout, etc.) — return raw
            # text and no usage rather than raising, so callers can still salvage a result.
            return raw, self._empty_usage()

        text = outer.get("result", "")
        u = outer.get("usage") or {}
        usage = {
            "input_tokens": u.get("input_tokens", 0),
            "output_tokens": u.get("output_tokens", 0),
            "cache_read_tokens": u.get("cache_read_input_tokens", 0),
            "cache_creation_tokens": u.get("cache_creation_input_tokens", 0),
            "model": outer.get("model", self.model),
            "backend": self.name,
        }
        return text, usage


class HttpChatBackend(AgentBackend):
    """OpenAI-compatible HTTP chat API with local tool-call loop."""
    name = "http"

    TOOL_DEFS = [
        {"type": "function", "function": {
            "name": "vector_search", "description": "의미 기반 벡터 검색 top-10",
            "parameters": {"type": "object", "properties": {
                "query": {"type": "string", "description": "자연어 질의"}
            }, "required": ["query"]}}},
        {"type": "function", "function": {
            "name": "grep_search", "description": "정규식 키워드 검색, 최대 20 hit",
            "parameters": {"type": "object", "properties": {
                "pattern": {"type": "string", "description": "정규식 패턴"}
            }, "required": ["pattern"]}}},
        {"type": "function", "function": {
            "name": "read_chunk", "description": "청크 전문 읽기 (4000자)",
            "parameters": {"type": "object", "properties": {
                "chunk_id": {"type": "string", "description": "청크 ID (예: 1040aa492c::c00012)"}
            }, "required": ["chunk_id"]}}},
        {"type": "function", "function": {
            "name": "structured_search", "description": "시맨틱 태그 슬롯 기반 구조 검색",
            "parameters": {"type": "object", "properties": {
                "filters": {"type": "string", "description": "JSON 필터 (예: {\"contract\":\"암진단특약\"})"}
            }, "required": ["filters"]}}},
    ]

    TOOL_CMD_MAP = {
        "vector_search": lambda a: ("vector", a.get("query", "")),
        "grep_search": lambda a: ("grep", a.get("pattern", "")),
        "read_chunk": lambda a: ("read", a.get("chunk_id", "")),
        "structured_search": lambda a: ("structured", a.get("filters", "{}")),
    }

    def __init__(self, base_url: str = "http://localhost:8002/v1",
                 api_key: str = "dummy", model: str = "Qwen/Qwen2.5-1.5B-Instruct",
                 **_ignored):
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.model = model
        self.max_turns = 24

    def _post(self, messages, tools=None):
        import urllib.request
        body = {"model": self.model, "messages": messages, "max_tokens": 1024,
                "temperature": 0}
        if tools:
            body["tools"] = tools
        data = json.dumps(body, ensure_ascii=False).encode("utf-8")
        req = urllib.request.Request(
            f"{self.base_url}/chat/completions", data=data, method="POST",
            headers={"Content-Type": "application/json; charset=utf-8",
                     "Authorization": f"Bearer {self.api_key}"})
        with urllib.request.urlopen(req, timeout=120) as r:
            return json.load(r)

    def _exec_tool(self, name, args, env, cwd):
        mapping = self.TOOL_CMD_MAP.get(name)
        if not mapping:
            return f"unknown tool: {name}"
        cmd, arg = mapping(args if isinstance(args, dict) else json.loads(args))
        try:
            p = subprocess.run(
                ["python", "tools.py", cmd, arg],
                capture_output=True, text=True, timeout=60, cwd=cwd, env=env,
                encoding="utf-8", errors="replace")
            return p.stdout or p.stderr or "empty"
        except Exception as e:
            return f"tool error: {e}"

    @staticmethod
    def _parse_text_tool_calls(text):
        """Fallback: parse <tool_call>{"name":..., "arguments":...}</tool_call> from text."""
        import re
        calls = []
        for m in re.finditer(r'<tool_call>\s*(\{.*?\})\s*</tool_call>', text, re.S):
            try:
                obj = json.loads(m.group(1))
                name = obj.get("name", "")
                args = obj.get("arguments", {})
                if isinstance(args, str):
                    args = json.loads(args)
                calls.append({"name": name, "args": args})
            except Exception:
                continue
        return calls

    def run(self, prompt: str, session_dir: str, env: dict, cwd: str) -> tuple[str, dict]:
        usage_total = {"input_tokens": 0, "output_tokens": 0, "cache_read_tokens": 0,
                       "cache_creation_tokens": 0, "model": self.model, "backend": self.name}
        messages = [{"role": "user", "content": prompt}]
        tools = [t for t in self.TOOL_DEFS
                 if t["function"]["name"] != "structured_search"
                 or env.get("STRUCT_ENABLED") == "1"]

        for turn in range(self.max_turns):
            try:
                resp = self._post(messages, tools)
            except Exception as e:
                return f"http error: {e}", usage_total

            u = resp.get("usage", {})
            usage_total["input_tokens"] += u.get("prompt_tokens", 0)
            usage_total["output_tokens"] += u.get("completion_tokens", 0)

            choice = resp["choices"][0]
            msg = choice["message"]
            content = msg.get("content", "") or ""

            # Try structured tool_calls first (VLLM auto-tool-choice)
            tool_calls = msg.get("tool_calls")
            if tool_calls:
                messages.append(msg)
                for tc in tool_calls:
                    fn = tc["function"]
                    args = fn.get("arguments", "{}")
                    if isinstance(args, str):
                        try:
                            args = json.loads(args)
                        except Exception:
                            args = {}
                    result = self._exec_tool(fn["name"], args, env, cwd)
                    messages.append({"role": "tool", "tool_call_id": tc["id"], "content": result})
                continue

            # Fallback: parse <tool_call> tags from text (Qwen 1.5B emits these)
            text_calls = self._parse_text_tool_calls(content)
            if text_calls:
                messages.append({"role": "assistant", "content": content})
                tool_results = []
                for tc in text_calls:
                    result = self._exec_tool(tc["name"], tc["args"], env, cwd)
                    tool_results.append(f"[{tc['name']}] {result}")
                messages.append({"role": "user", "content": "도구 결과:\n" + "\n".join(tool_results)})
                continue

            # No tool calls — final response
            return content, usage_total

        last = messages[-1].get("content", "") if messages else ""
        return last, usage_total


def get_backend(backend_type: str = "cli", **kwargs) -> AgentBackend:
    """Factory. backend_type: 'cli' or 'http'."""
    if backend_type == "cli":
        return ClaudeCliBackend(**kwargs)
    if backend_type == "http":
        return HttpChatBackend(**kwargs)
    raise ValueError(f"unknown backend_type: {backend_type!r}")

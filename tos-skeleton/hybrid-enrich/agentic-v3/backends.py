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
    """OpenAI/Anthropic-compatible HTTP chat API with local tool loop."""
    name = "http"

    def __init__(self, base_url: str, api_key: str, model: str,
                 tools_module_path: str | None = None):
        self.base_url = base_url
        self.api_key = api_key
        self.model = model
        # Points at tools.py for in-process tool calls (vector/grep/read dispatch).
        self.tools_module_path = tools_module_path
        self.max_turns = 24

    def run(self, prompt: str, session_dir: str, env: dict, cwd: str) -> tuple[str, dict]:
        # TODO(KT Cloud integration):
        #   1. POST {base_url}/chat/completions with the initial message + tool defs
        #      derived from tools_module_path.
        #   2. If the response carries tool_calls: import tools_module_path, execute
        #      each call in-process, and feed the results back as tool-result messages.
        #   3. Repeat until the model returns a final message with no tool_calls, or
        #      self.max_turns is reached.
        #   4. Accumulate usage (input/output/cache tokens) across every turn into the
        #      returned usage_meta before returning.
        raise NotImplementedError(
            "HTTP backend requires KT Cloud integration — use CLI backend for now")


def get_backend(backend_type: str = "cli", **kwargs) -> AgentBackend:
    """Factory. backend_type: 'cli' or 'http'."""
    if backend_type == "cli":
        return ClaudeCliBackend(**kwargs)
    if backend_type == "http":
        return HttpChatBackend(**kwargs)
    raise ValueError(f"unknown backend_type: {backend_type!r}")

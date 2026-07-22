"""
DR-DCI Agent: Pull + DCI (workspace tools) 기반 질의응답 에이전트

가이드 반영 사항:
- P0-2: pull이 ranked preview(rank, doc_id, title)와 pull별 통계를 반환,
        workspace 기존 문서 제외 + backfill
- P0-3: 최소 2회 서로 다른 query pull 규칙을 harness에서 강제, 위반 기록
- P0-6: tag grammar를 실제 데이터 형식과 통일해 prompt에 안내
- P0-9: workspace 상한 / turn 예산을 config에서 주입, budget 소진 기록
- P1-3: read한 문서 추적
- P1-4: 전체 tool trace(event log) 저장
- Part 2 추가 baseline: single_pull 모드 (pull 1회 후 workspace 동결)
"""

import json
import re
import requests
from .workspace import Workspace, Document, normalize_tag
from .retriever import PullRetriever

AGENT_SYSTEM_BASE = """You are a research assistant that answers questions by searching and analyzing documents in your workspace.

You have access to the following tools:
1. pull(query, top_k?, taxonomy_filter?) - Search and retrieve relevant documents into your workspace. Returns a ranked preview of newly added documents. Use top_k to control how many documents to retrieve. Documents already in your workspace are excluded automatically and replaced with fresh candidates.
2. grep(pattern, tag_filter?) - Search text patterns in workspace documents. Use tag_filter for semantic element types.
3. find(taxonomy_filter?, metadata_filter?) - Filter workspace documents by category or metadata. All metadata conditions must match (AND).
4. read(doc_id) - Read the full content of a specific document
5. answer(text) - Provide your final answer

IMPORTANT WORKFLOW:
1. Pull documents with a relevant query
2. Check the ranked preview, then use grep/find/read to inspect the most promising documents
3. Pull AGAIN with a DIFFERENT query or different taxonomy category to get more diverse results
4. Repeat until you have covered multiple angles, then answer

IMPORTANT RULES:
- You MUST pull at least 2 times with different queries before answering. The answer tool is rejected until then.
- The preview shows rank and title only; read() the promising documents before answering.
- find() only filters documents ALREADY in your workspace. It does NOT search new documents.
- When taxonomy categories are available, pull from MULTIPLE relevant categories, not just one.
- More pulls with diverse queries = better coverage = better answer."""

SINGLE_PULL_NOTE = """

NOTE: You are in SINGLE-PULL mode. You may call pull exactly ONCE. Choose your query carefully, then explore the workspace with grep/find/read and answer."""


def build_system_prompt(taxonomy_schema: dict = None, metadata_schema: dict = None,
                        tags_data: dict = None, single_pull: bool = False) -> str:
    """Build system prompt with available schema information for workspace tools."""
    prompt = AGENT_SYSTEM_BASE
    if single_pull:
        prompt = prompt.replace(
            "- You MUST pull at least 2 times with different queries before answering. The answer tool is rejected until then.\n", "")
        prompt += SINGLE_PULL_NOTE

    if taxonomy_schema:
        prompt += "\n\n## Taxonomy Categories (for pull and find)\n"
        prompt += "Use taxonomy_filter in pull() to focus retrieval on a category, or in find() to filter workspace documents.\n"
        l1_cats = taxonomy_schema.get("L1", [])
        prompt += f"- L1 categories: {', '.join(l1_cats)}\n"
        l2_map = taxonomy_schema.get("L2", {})
        for l1, l2_list in l2_map.items():
            prompt += f"  - {l1} → L2: {', '.join(l2_list)}\n"

    if metadata_schema:
        prompt += "\n\n## Workspace Metadata (for find tool)\n"
        prompt += "Use metadata_filter in find(). ALL conditions must match (AND semantics).\n"
        prompt += "For entities use: entity_names (list of names) or entity_category (single category).\n"
        fields = metadata_schema.get("fields", {})
        for field_name, field_info in fields.items():
            if field_info.get("type") == "enum":
                values = [v for v in field_info.get("values", []) if v is not None]
                prompt += f"- {field_name}: {', '.join(values[:10])}\n"
            elif field_name == "entities":
                item_schema = field_info.get("item_schema", {})
                cats = [c for c in item_schema.get("category", {}).get("values", []) if c is not None]
                prompt += f"- entity_category values: {', '.join(cats)}\n"

    if tags_data:
        # P0-6: prompt 예시를 실제 저장된 tag 형식에서 추출해 grammar 불일치 제거
        seen = []
        for elems in tags_data.values():
            for elem in elems:
                tag = elem.get("tag", "")
                if tag and tag not in seen:
                    seen.append(tag)
            if len(seen) >= 12:
                break
        prompt += "\n\n## Semantic Tags (for grep tool)\n"
        prompt += "Use tag_filter in grep() to search specific element types in documents.\n"
        if seen:
            prompt += f"Tags present in this corpus (use exactly these, or their last component): {', '.join(sorted(set(seen))[:12])}\n"
            subs = sorted({normalize_tag(t) for t in seen})
            prompt += f"Short forms accepted: {', '.join(subs[:12])}\n"

    return prompt


TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "pull",
            "description": "Search and retrieve documents into workspace. Returns ranked preview of new documents. Duplicates already in workspace are excluded and backfilled with next-ranked candidates.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "top_k": {
                        "type": "integer", "minimum": 1, "maximum": 200,
                        "description": "Optional: number of new documents to retrieve (default set by system)",
                    },
                    "taxonomy_filter": {
                        "type": "object",
                        "description": "Optional: focus retrieval on a taxonomy category (e.g. {\"L1\": \"Treatment\"})",
                        "properties": {
                            "L1": {"type": "string"},
                            "L2": {"type": "string"},
                        }
                    },
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search patterns in workspace documents. Returns matching lines. If tag_filter is set, only tagged elements are searched and documents without tag data are reported as tag_data_missing.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text pattern to search"},
                    "tag_filter": {"type": "string", "description": "Optional semantic tag filter, e.g. 'evidence' or '@el:paragraph/evidence'"},
                },
                "required": ["pattern"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Filter documents in workspace by taxonomy or metadata (AND semantics). Entity filters: entity_names (list) / entity_category (string).",
            "parameters": {
                "type": "object",
                "properties": {
                    "taxonomy_filter": {"type": "object", "description": "Filter by L1/L2 category"},
                    "metadata_filter": {"type": "object", "description": "Filter by metadata fields; all conditions must match"},
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read full content of a document in workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "doc_id": {"type": "string"},
                },
                "required": ["doc_id"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "answer",
            "description": "Provide final answer after analyzing documents. Rejected until the minimum pull rule is satisfied.",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "Your final answer based on document analysis"},
                },
                "required": ["text"],
            }
        }
    },
]


def normalize_query(q: str) -> str:
    return re.sub(r"\s+", " ", q.strip().lower())


class DCIAgent:
    def __init__(self, llm_url: str, model_name: str, retriever: PullRetriever,
                 corpus: dict, tags_data: dict = None, taxonomy_data: dict = None,
                 metadata_data: dict = None, prefix_data: dict = None,
                 max_turns: int = 10,
                 workspace_max_docs: int = 100,
                 min_pulls: int = 2,
                 single_pull: bool = False,
                 preview_size: int = 10,
                 taxonomy_schema: dict = None, metadata_schema: dict = None,
                 api_key: str = None):
        self.llm_url = llm_url
        self.model_name = model_name
        self.api_key = api_key
        self.retriever = retriever
        self.corpus = corpus
        self.tags_data = tags_data
        self.taxonomy_data = taxonomy_data
        self.metadata_data = metadata_data
        self.prefix_data = prefix_data
        self.max_turns = max_turns
        self.workspace_max_docs = workspace_max_docs
        self.min_pulls = 1 if single_pull else min_pulls
        self.single_pull = single_pull
        self.preview_size = preview_size
        self.system_prompt = build_system_prompt(
            taxonomy_schema=taxonomy_schema,
            metadata_schema=metadata_schema,
            tags_data=tags_data,
            single_pull=single_pull,
        )

    def run(self, query: str) -> dict:
        """쿼리에 대해 에이전트 실행, 결과 반환"""
        workspace = Workspace(max_docs=self.workspace_max_docs)
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Answer this question: {query}"},
        ]

        state = {
            "pull_count": 0,
            "pull_queries": [],
            "pull_stats": [],
            "trace": [],
            "rule_violations": [],
            "final_answer": "",
            "answered": False,
            "retrieved_candidates": 0,
            "added_documents": 0,
            "tool_call_counts": {"pull": 0, "grep": 0, "find": 0, "read": 0, "answer": 0},
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "taxonomy_filtered_pulls": 0,
            "system_fingerprints": set(),
        }

        turns_used = self.max_turns
        for turn in range(self.max_turns):
            response = self._call_llm(messages)
            usage = response.pop("_usage", None) or {}
            fingerprint = response.pop("_system_fingerprint", None)
            state["prompt_tokens"] += int(usage.get("prompt_tokens") or 0)
            state["completion_tokens"] += int(usage.get("completion_tokens") or 0)
            if fingerprint:
                state["system_fingerprints"].add(str(fingerprint))

            if not response.get("tool_calls"):
                distinct = len({normalize_query(q) for q in state["pull_queries"]})
                if state["pull_count"] < self.min_pulls or distinct < self.min_pulls:
                    state["rule_violations"].append("answered_without_min_pulls")
                else:
                    state["final_answer"] = response.get("content", "") or ""
                    state["answered"] = True
                turns_used = turn + 1
                break

            messages.append({
                "role": "assistant",
                "content": response.get("content", None),
                "tool_calls": response["tool_calls"],
            })

            answered_this_turn = False
            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]
                state["tool_call_counts"][func_name] = (
                    state["tool_call_counts"].get(func_name, 0) + 1
                )
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                result = self._execute_tool(func_name, args, workspace, state)

                state["trace"].append({
                    "turn": turn + 1,
                    "tool": func_name,
                    "args": args,
                    "result_summary": self._summarize_result(func_name, result),
                })

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False)[:2000],
                })

                if func_name == "answer" and result.get("status") == "answered":
                    answered_this_turn = True

            if answered_this_turn:
                turns_used = turn + 1
                break

        distinct_queries = len({normalize_query(q) for q in state["pull_queries"]})
        if state["pull_count"] >= self.min_pulls and distinct_queries < self.min_pulls:
            state["rule_violations"].append("duplicate_pull_queries")

        return {
            "answer": state["final_answer"],
            "pull_count": state["pull_count"],
            "distinct_pull_queries": distinct_queries,
            "pull_stats": state["pull_stats"],
            "workspace_docs": list(workspace.docs.keys()),
            "read_docs": sorted(workspace.read_ids),
            "turns": turns_used,
            "budget_exhausted": not state["answered"] and turns_used >= self.max_turns,
            "rule_violations": state["rule_violations"],
            "trace": state["trace"],
            "retrieved_candidates": state["retrieved_candidates"],
            "added_documents": state["added_documents"],
            "tool_call_counts": dict(state["tool_call_counts"]),
            "tool_calls_total": sum(state["tool_call_counts"].values()),
            "llm_prompt_tokens": state["prompt_tokens"],
            "llm_completion_tokens": state["completion_tokens"],
            "taxonomy_filtered_pulls": state["taxonomy_filtered_pulls"],
            "system_fingerprints": sorted(state["system_fingerprints"]),
        }

    def _execute_tool(self, name: str, args: dict, workspace: Workspace,
                      state: dict) -> dict:
        if name == "pull":
            if self.single_pull and state["pull_count"] >= 1:
                state["rule_violations"].append("extra_pull_in_single_pull_mode")
                return {"error": "pull budget exhausted: single-pull mode allows exactly one pull"}

            query = str(args.get("query") or "").strip()
            if not query:
                state["rule_violations"].append("invalid_pull_arguments")
                return {"error": "pull requires a non-empty query"}
            try:
                pulled = self.retriever.pull(
                    query=query,
                    taxonomy_filter=args.get("taxonomy_filter"),
                    top_k=args.get("top_k"),
                    exclude_ids=set(workspace.docs.keys()),
                )
            except (TypeError, ValueError) as exc:
                state["rule_violations"].append("invalid_pull_arguments")
                return {"error": str(exc)}
            state["pull_count"] += 1
            state["pull_queries"].append(query)
            if args.get("taxonomy_filter"):
                state["taxonomy_filtered_pulls"] += 1

            added = 0
            preview = []
            for r in pulled["results"]:
                doc_id = r["doc_id"]
                if doc_id in self.corpus:
                    raw = self.corpus[doc_id]
                    doc = Document(
                        doc_id=doc_id,
                        title=raw.get("title", ""),
                        text=raw.get("text", ""),
                        tags=self.tags_data.get(doc_id, []) if self.tags_data else [],
                        taxonomy=self.taxonomy_data.get(doc_id, {}) if self.taxonomy_data else {},
                        metadata=self.metadata_data.get(doc_id, {}) if self.metadata_data else {},
                        prefix=self.prefix_data.get(doc_id, "") if self.prefix_data else "",
                    )
                    if workspace.add(doc):
                        added += 1
                        if len(preview) < self.preview_size:
                            preview.append({
                                "rank": r.get("rank"),
                                "doc_id": doc_id,
                                "title": (raw.get("title", "") or "")[:80],
                            })

            stats = {
                "requested": pulled["requested"],
                "retrieved": len(pulled["results"]),
                "newly_added": added,
                "duplicate_count": pulled["duplicates_excluded"],
            }
            state["pull_stats"].append(stats)
            state["retrieved_candidates"] += stats["retrieved"]
            state["added_documents"] += added
            if added == 0:
                state["rule_violations"].append("pull_did_not_expand_workspace")

            return {
                **stats,
                "total_in_workspace": len(workspace.docs),
                "preview": preview,
            }

        elif name == "grep":
            pattern = str(args.get("pattern") or "")
            if not pattern:
                return {"error": "grep requires a non-empty pattern"}
            result = workspace.grep(pattern, args.get("tag_filter"))
            result["matches"] = result["matches"][:20]
            return result

        elif name == "find":
            result = workspace.find(args.get("taxonomy_filter"), args.get("metadata_filter"))
            result["doc_ids"] = result["doc_ids"][:20]
            return result

        elif name == "read":
            content = workspace.read(args.get("doc_id", ""))
            return {"content": content[:2000] if content else "Document not found in workspace"}

        elif name == "answer":
            # P0-3: 최소 pull 규칙을 harness에서 강제
            distinct = len({normalize_query(q) for q in state["pull_queries"]})
            if state["pull_count"] < self.min_pulls or distinct < self.min_pulls:
                state["rule_violations"].append("answer_rejected_min_pull_rule")
                return {
                    "error": (
                        f"Answer rejected: you must pull at least {self.min_pulls} times "
                        f"with different queries first (pulls so far: {state['pull_count']}, "
                        f"distinct queries: {distinct})."
                    )
                }
            state["final_answer"] = args.get("text", "")
            state["answered"] = True
            return {"status": "answered"}

        return {"error": f"Unknown tool: {name}"}

    @staticmethod
    def _summarize_result(name: str, result: dict) -> dict:
        if name == "pull":
            return {k: result.get(k) for k in ("newly_added", "duplicate_count", "total_in_workspace")}
        if name == "grep":
            return {"matches": len(result.get("matches", [])),
                    "tag_data_missing": result.get("tag_data_missing")}
        if name == "find":
            return {"matched": result.get("matched")}
        if name == "read":
            return {"found": "content" in result and result["content"] != "Document not found in workspace"}
        if name == "answer":
            return {"status": result.get("status", result.get("error", ""))[:80]}
        return {}

    def _call_llm(self, messages: list) -> dict:
        payload = {
            "model": self.model_name,
            "messages": messages,
            "tools": TOOL_DEFINITIONS,
            "temperature": 0,
            "max_tokens": 1024,
        }

        # vLLM 로컬 서버면 thinking 비활성화
        if "openai.com" not in self.llm_url:
            payload["chat_template_kwargs"] = {"enable_thinking": False}

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(3):
            resp = requests.post(self.llm_url, json=payload, headers=headers, timeout=120)
            if resp.status_code == 400:
                return {"content": "I cannot process this query due to context limitations.", "tool_calls": None}
            if resp.status_code == 429:
                import time
                time.sleep(5 * (attempt + 1))
                continue
            resp.raise_for_status()
            body = resp.json()
            choice = dict(body["choices"][0]["message"])
            choice["_usage"] = body.get("usage") or {}
            choice["_system_fingerprint"] = body.get("system_fingerprint")
            return choice

        return {"content": "Max retries exceeded.", "tool_calls": None}

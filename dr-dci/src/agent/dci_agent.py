"""
DR-DCI Agent: Pull + DCI (workspace tools) 기반 질의응답 에이전트
"""

import json
import requests
from .workspace import Workspace, Document
from .retriever import PullRetriever

AGENT_SYSTEM_BASE = """You are a research assistant that answers questions by searching and analyzing documents in your workspace.

You have access to the following tools:
1. pull(query, taxonomy_filter?) - Search and retrieve relevant documents into your workspace. Use taxonomy_filter to focus on a specific category.
2. grep(pattern, tag_filter?) - Search text patterns in workspace documents. Use tag_filter for semantic element types.
3. find(taxonomy_filter?, metadata_filter?) - Filter workspace documents by category or metadata
4. read(doc_id) - Read the full content of a specific document
5. answer(text) - Provide your final answer

IMPORTANT WORKFLOW:
1. Pull documents with a relevant query
2. Use grep/find/read to explore what you retrieved
3. Pull AGAIN with a DIFFERENT query or different taxonomy category to get more diverse results
4. Repeat until you have covered multiple angles, then answer

IMPORTANT RULES:
- You MUST pull at least 2 times with different queries before answering.
- find() only filters documents ALREADY in your workspace. It does NOT search new documents.
- When taxonomy categories are available, pull from MULTIPLE relevant categories, not just one.
- More pulls with diverse queries = better coverage = better answer."""


def build_system_prompt(taxonomy_schema: dict = None, metadata_schema: dict = None, tags_enabled: bool = False) -> str:
    """Build system prompt with available schema information for workspace tools."""
    prompt = AGENT_SYSTEM_BASE

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
        prompt += "Use metadata_filter in find() to filter workspace documents by attributes.\n"
        fields = metadata_schema.get("fields", {})
        for field_name, field_info in fields.items():
            if field_info.get("type") == "enum":
                values = [v for v in field_info.get("values", []) if v is not None]
                prompt += f"- {field_name}: {', '.join(values[:10])}\n"
            elif field_name == "entities":
                item_schema = field_info.get("item_schema", {})
                cats = [c for c in item_schema.get("category", {}).get("values", []) if c is not None]
                prompt += f"- entities: filter by entity name list. Categories: {', '.join(cats)}\n"

    if tags_enabled:
        prompt += "\n\n## Semantic Tags (for grep tool)\n"
        prompt += "Use tag_filter in grep() to search specific element types in documents.\n"
        prompt += "Available tags: @el:definition, @el:condition, @el:procedure, @el:example, @el:exception, @el:comparison, @el:summary, @el:evidence, @el:criteria\n"

    return prompt

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "pull",
            "description": "Search and retrieve documents into workspace. Use taxonomy_filter to focus on a specific category.",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
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
            "description": "Search patterns in workspace documents. Returns matching lines.",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string", "description": "Text pattern to search"},
                    "tag_filter": {"type": "string", "description": "Optional @el: tag filter (e.g. @el:evidence)"},
                },
                "required": ["pattern"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Filter documents in workspace by taxonomy or metadata",
            "parameters": {
                "type": "object",
                "properties": {
                    "taxonomy_filter": {"type": "object", "description": "Filter by L1/L2 category"},
                    "metadata_filter": {"type": "object", "description": "Filter by metadata fields"},
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
            "description": "Provide final answer after analyzing documents",
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


class DCIAgent:
    def __init__(self, llm_url: str, model_name: str, retriever: PullRetriever,
                 corpus: dict, tags_data: dict = None, taxonomy_data: dict = None,
                 metadata_data: dict = None, prefix_data: dict = None,
                 max_turns: int = 10,
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
        self.system_prompt = build_system_prompt(
            taxonomy_schema=taxonomy_schema,
            metadata_schema=metadata_schema,
            tags_enabled=tags_data is not None,
        )

    def run(self, query: str) -> dict:
        """쿼리에 대해 에이전트 실행, 결과 반환"""
        workspace = Workspace()
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"Answer this question: {query}"},
        ]

        pull_count = 0
        final_answer = ""

        for turn in range(self.max_turns):
            response = self._call_llm(messages)

            if not response.get("tool_calls"):
                final_answer = response.get("content", "")
                break

            messages.append({
                "role": "assistant",
                "content": response.get("content", None),
                "tool_calls": response["tool_calls"],
            })

            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                result = self._execute_tool(func_name, args, workspace)

                if func_name == "pull":
                    pull_count += 1
                elif func_name == "answer":
                    final_answer = args.get("text", "")
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call["id"],
                        "content": json.dumps(result, ensure_ascii=False),
                    })
                    return {
                        "answer": final_answer,
                        "pull_count": pull_count,
                        "workspace_docs": list(workspace.docs.keys()),
                        "turns": turn + 1,
                    }

                messages.append({
                    "role": "tool",
                    "tool_call_id": tool_call["id"],
                    "content": json.dumps(result, ensure_ascii=False)[:2000],
                })

        return {
            "answer": final_answer,
            "pull_count": pull_count,
            "workspace_docs": list(workspace.docs.keys()),
            "turns": self.max_turns,
        }

    def _execute_tool(self, name: str, args: dict, workspace: Workspace) -> dict:
        if name == "pull":
            results = self.retriever.pull(
                query=args["query"],
                taxonomy_filter=args.get("taxonomy_filter"),
            )
            added = 0
            for r in results:
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
            return {"retrieved": len(results), "added_to_workspace": added, "total_in_workspace": len(workspace.docs)}

        elif name == "grep":
            results = workspace.grep(args["pattern"], args.get("tag_filter"))
            return {"matches": results[:20]}

        elif name == "find":
            results = workspace.find(args.get("taxonomy_filter"), args.get("metadata_filter"))
            return {"doc_ids": results[:20]}

        elif name == "read":
            content = workspace.read(args["doc_id"])
            return {"content": content[:2000] if content else "Document not found in workspace"}

        elif name == "answer":
            return {"status": "answered"}

        return {"error": f"Unknown tool: {name}"}

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
            choice = resp.json()["choices"][0]["message"]
            return choice

        return {"content": "Max retries exceeded.", "tool_calls": None}

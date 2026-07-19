"""
DR-DCI Agent: Pull + DCI (workspace tools) 기반 질의응답 에이전트
"""

import json
import requests
from .workspace import Workspace, Document
from .retriever import PullRetriever

AGENT_SYSTEM = """You are a research assistant that answers questions by searching and analyzing documents.

You have access to the following tools:
1. pull(query) - Search and retrieve relevant documents into your workspace
2. grep(pattern, tag_filter?) - Search text patterns in your workspace documents
3. find(taxonomy_filter?, metadata_filter?) - Filter documents in workspace
4. read(doc_id) - Read full content of a specific document
5. answer(text) - Provide your final answer

Process:
1. Use pull() to retrieve potentially relevant documents
2. Use grep/find/read to analyze documents in your workspace
3. When you have enough information, use answer() to respond

Always provide a final answer. If you cannot find relevant information, say so."""

TOOL_DEFINITIONS = [
    {
        "type": "function",
        "function": {
            "name": "pull",
            "description": "Search and retrieve documents into workspace",
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Search query"},
                    "taxonomy_filter": {
                        "type": "object",
                        "description": "Optional taxonomy filter (L1, L2)",
                        "properties": {
                            "L1": {"type": "string"},
                            "L2": {"type": "string"},
                        }
                    },
                    "metadata_filter": {
                        "type": "object",
                        "description": "Optional metadata filter",
                    }
                },
                "required": ["query"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "grep",
            "description": "Search patterns in workspace documents",
            "parameters": {
                "type": "object",
                "properties": {
                    "pattern": {"type": "string"},
                    "tag_filter": {"type": "string", "description": "Optional @el: tag filter"},
                },
                "required": ["pattern"],
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "find",
            "description": "Filter documents in workspace by taxonomy/metadata",
            "parameters": {
                "type": "object",
                "properties": {
                    "taxonomy_filter": {"type": "object"},
                    "metadata_filter": {"type": "object"},
                },
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "read",
            "description": "Read full content of a document",
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
            "description": "Provide final answer to the query",
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string"},
                },
                "required": ["text"],
            }
        }
    },
]


class DCIAgent:
    def __init__(self, llm_url: str, model_name: str, retriever: PullRetriever,
                 corpus: dict, tags_data: dict = None, max_turns: int = 30):
        self.llm_url = llm_url
        self.model_name = model_name
        self.retriever = retriever
        self.corpus = corpus  # doc_id → doc dict
        self.tags_data = tags_data  # doc_id → [elements]
        self.max_turns = max_turns

    def run(self, query: str) -> dict:
        """쿼리에 대해 에이전트 실행, 결과 반환"""
        workspace = Workspace()
        messages = [
            {"role": "system", "content": AGENT_SYSTEM},
            {"role": "user", "content": f"Answer this question: {query}"},
        ]

        pull_count = 0
        final_answer = ""

        for turn in range(self.max_turns):
            # LLM 호출
            response = self._call_llm(messages)

            if not response.get("tool_calls"):
                # tool call 없이 응답 → 강제 종료
                final_answer = response.get("content", "")
                break

            # assistant message (with tool_calls) 추가
            messages.append({
                "role": "assistant",
                "content": response.get("content", None),
                "tool_calls": response["tool_calls"],
            })

            # 각 tool call 실행 및 결과 추가
            for tool_call in response["tool_calls"]:
                func_name = tool_call["function"]["name"]
                try:
                    args = json.loads(tool_call["function"]["arguments"])
                except json.JSONDecodeError:
                    args = {}

                result = self._execute_tool(
                    func_name, args, workspace, pull_count
                )

                if func_name == "pull":
                    pull_count += 1
                elif func_name == "answer":
                    final_answer = args.get("text", "")
                    # tool 응답도 추가 후 리턴
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

    def _execute_tool(self, name: str, args: dict, workspace: Workspace, pull_count: int) -> dict:
        if name == "pull":
            results = self.retriever.pull(
                query=args["query"],
                taxonomy_filter=args.get("taxonomy_filter"),
                metadata_filter=args.get("metadata_filter"),
            )
            # workspace에 문서 추가
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
                    )
                    if workspace.add(doc):
                        added += 1
            return {"retrieved": len(results), "added_to_workspace": added, "total_in_workspace": len(workspace.docs)}

        elif name == "grep":
            results = workspace.grep(args["pattern"], args.get("tag_filter"))
            return {"matches": results[:10]}

        elif name == "find":
            results = workspace.find(args.get("taxonomy_filter"), args.get("metadata_filter"))
            return {"doc_ids": results[:20]}

        elif name == "read":
            content = workspace.read(args["doc_id"])
            return {"content": content[:1500] if content else "Document not found in workspace"}

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
            "chat_template_kwargs": {"enable_thinking": False},
        }

        resp = requests.post(self.llm_url, json=payload, timeout=120)
        resp.raise_for_status()
        choice = resp.json()["choices"][0]["message"]
        return choice

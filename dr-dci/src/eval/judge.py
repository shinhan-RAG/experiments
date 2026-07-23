"""
LLM-as-Judge 평가기
- Accuracy: correct/incorrect 판정
- Gold R@W: workspace recall
- Efficiency: Gold R@W / Pull 횟수
"""

import json
import os
import time
import requests
import numpy as np


class Judge:
    def __init__(self, llm_url: str, model_name: str, prompt_template: str, api_key: str = None):
        self.llm_url = llm_url
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def evaluate_accuracy(self, query: str, reference_answer: str, candidate_answer: str) -> str:
        """LLM-as-Judge로 정답 여부 판정 (retry with backoff)"""
        prompt = self.prompt_template.format(
            query=query,
            reference_answer=reference_answer,
            candidate_answer=candidate_answer,
        )

        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        payload = {
            "model": self.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "max_tokens": 10,
        }

        for attempt in range(5):
            try:
                resp = requests.post(self.llm_url, json=payload, headers=headers, timeout=60)
                if resp.status_code == 429:
                    wait = 10 * (attempt + 1)  # 10, 20, 30, 40, 50s
                    print(f"  Judge rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()["choices"][0]["message"]["content"]
                return self.parse_judgment(result)
            except requests.exceptions.Timeout:
                time.sleep(10)
                continue
            except Exception as e:
                print(f"  Judge error: {e}")
                return "error"
        print(f"  Judge error: max retries exceeded")
        return "error"

    @staticmethod
    def parse_judgment(value: str) -> str:
        normalized = value.strip().strip('"\'').lower().rstrip(".!")
        if normalized == "correct":
            return "correct"
        if normalized == "incorrect":
            return "incorrect"
        return "error"

    @staticmethod
    def gold_recall_at_workspace(workspace_docs: list[str], gold_doc_ids: list[str]) -> float:
        """Gold R@W = |Workspace ∩ gold_docs| / |gold_docs|"""
        if not gold_doc_ids:
            return 0.0
        intersection = set(workspace_docs) & set(gold_doc_ids)
        return len(intersection) / len(gold_doc_ids)

    @staticmethod
    def efficiency(gold_recall: float, pull_count: int) -> float:
        """Efficiency = Gold R@W / Pull 횟수"""
        if pull_count == 0:
            return 0.0
        return gold_recall / pull_count

    @staticmethod
    def degradation_rate(acc_small: float, acc_large: float) -> float:
        """Degradation Rate = (small_acc - large_acc) / small_acc * 100%"""
        if acc_small == 0:
            return 0.0
        return (acc_small - acc_large) / acc_small * 100


def compute_metrics(results: list[dict]) -> dict:
    """실험 결과 리스트에서 집계 메트릭 계산"""
    n = len(results)
    if n == 0:
        return {}

    judged = [
        r for r in results
        if r.get("judgment") in {"correct", "incorrect"}
    ]
    accuracy = (
        sum(1 for r in judged if r["judgment"] == "correct") / len(judged)
        if judged else None
    )
    avg_recall = sum(r.get("gold_recall", 0) for r in results) / n
    avg_efficiency = sum(r.get("efficiency", 0) for r in results) / n
    avg_pulls = sum(r.get("pull_count", 0) for r in results) / n
    avg_taxonomy_pulls = sum(r.get("taxonomy_filtered_pulls", 0) for r in results) / n
    avg_taxonomy_eligible = sum(
        r.get("taxonomy_boost_eligible_documents", 0) for r in results
    ) / n
    avg_taxonomy_returned = sum(
        r.get("taxonomy_boosted_returned_documents", 0) for r in results
    ) / n
    avg_candidates = sum(r.get("retrieved_candidates", 0) for r in results) / n
    avg_workspace_docs = sum(len(r.get("workspace_docs", r.get("retrieved_docs", []))) for r in results) / n
    avg_turns = sum(r.get("turns", 0) for r in results) / n
    avg_latency = sum(r.get("latency_seconds", 0) for r in results) / n
    latencies = [r.get("latency_seconds", 0) for r in results]
    candidate_efficiencies = [
        r.get("gold_recall", 0) * 100 / r.get("retrieved_candidates", 0)
        for r in results
        if r.get("retrieved_candidates", 0) > 0
    ]

    return {
        "n": n,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "judged_n": len(judged),
        "judge_error_n": sum(1 for r in results if r.get("judgment") == "error"),
        "avg_gold_recall": round(avg_recall, 4),
        "avg_efficiency": round(avg_efficiency, 4),
        "avg_pulls": round(avg_pulls, 2),
        "avg_taxonomy_filtered_pulls": round(avg_taxonomy_pulls, 2),
        "avg_taxonomy_boost_eligible_documents": round(avg_taxonomy_eligible, 2),
        "avg_taxonomy_boosted_returned_documents": round(avg_taxonomy_returned, 2),
        "avg_retrieved_candidates": round(avg_candidates, 2),
        "avg_workspace_docs": round(avg_workspace_docs, 2),
        "avg_turns": round(avg_turns, 2),
        "avg_latency_seconds": round(avg_latency, 3),
        "p50_latency_seconds": round(float(np.percentile(latencies, 50)), 3),
        "p95_latency_seconds": round(float(np.percentile(latencies, 95)), 3),
        "avg_gold_recall_per_100_candidates": round(
            sum(candidate_efficiencies) / len(candidate_efficiencies), 4
        ) if candidate_efficiencies else 0.0,
    }

"""
LLM-as-Judge 평가기
- Accuracy: correct/incorrect 판정
- Gold R@W: workspace recall
- Efficiency: Gold R@W / Pull 횟수
"""

import os
import time
import requests
import numpy as np

VALID_JUDGMENTS = {"correct", "incorrect"}


def parse_judgment(raw: str) -> str:
    """Judge 출력 전체가 허용된 단어일 때만 인정한다."""
    if raw is None:
        return "format_error"
    cleaned = raw.strip().lower().strip("\"'`.,!:; \n\t")
    return cleaned if cleaned in VALID_JUDGMENTS else "format_error"


class Judge:
    def __init__(self, llm_url: str, model_name: str, prompt_template: str,
                 api_key: str = None, temperature: float = 0.0,
                 max_tokens: int = 10, llm_seed: int = None):
        self.llm_url = llm_url
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.llm_seed = llm_seed

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
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
        }
        if self.llm_seed is not None:
            payload["seed"] = self.llm_seed

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
        return parse_judgment(value)

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
        if r.get("judgment") in VALID_JUDGMENTS
    ]
    accuracy = (
        sum(1 for r in judged if r["judgment"] == "correct") / len(judged)
        if judged else None
    )
    avg_recall = sum(r.get("gold_recall", 0) for r in results) / n
    avg_efficiency = sum(r.get("efficiency", 0) for r in results) / n
    avg_pulls = sum(r.get("pull_count", 0) for r in results) / n
    avg_taxonomy_pulls = sum(r.get("taxonomy_filtered_pulls", 0) for r in results) / n
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

    recall_vals = [r.get("gold_recall", 0) for r in results]
    ci_lo, ci_hi = bootstrap_ci(recall_vals)

    out = {
        "n": n,
        "accuracy": round(accuracy, 4) if accuracy is not None else None,
        "judged_n": len(judged),
        "n_judged": len(judged),
        "judge_error_n": sum(
            1 for r in results if r.get("judgment") in {"error", "format_error"}
        ),
        "judge_error_count": sum(
            1 for r in results if r.get("judgment") in {"error", "format_error"}
        ),
        "avg_gold_recall": round(avg_recall, 4),
        "recall_ci95": [ci_lo, ci_hi],
        "avg_efficiency": round(avg_efficiency, 4),
        "avg_pulls": round(avg_pulls, 2),
        "avg_taxonomy_filtered_pulls": round(avg_taxonomy_pulls, 2),
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

    # 청크단위 검색 지표 집계 (recall@k/ndcg@k + span coverage/density/f1).
    # 질의마다 키가 다를 수 있어(span 없는 질의는 span 키 없음) 키별로 존재하는 값만 평균.
    chunk_evals = [r["span_metrics"] for r in results if r.get("span_metrics")]
    if chunk_evals:
        all_keys = set()
        for e in chunk_evals:
            all_keys |= set(e.keys())
        cm = {}
        for k in sorted(all_keys):
            vals = [e[k] for e in chunk_evals if k in e]
            if vals:
                cm[k] = round(sum(vals) / len(vals), 4)
        out["chunk_metrics"] = cm
        out["chunk_metrics_n"] = len(chunk_evals)

    return out


def bootstrap_ci(values: list[float], n_boot: int = 1000, seed: int = 42) -> tuple:
    """평균의 95% bootstrap 신뢰구간. 50문항 ±5%p 노이즈 판별용.
    (Date.now/Math.random 미사용 — 고정 seed 의 결정적 리샘플링)"""
    import random
    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    N = len(values)
    means = []
    for _ in range(n_boot):
        s = sum(values[rng.randrange(N)] for _ in range(N)) / N
        means.append(s)
    means.sort()
    lo = means[int(0.025 * n_boot)]
    hi = means[int(0.975 * n_boot)]
    return (round(lo, 4), round(hi, 4))

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
from pathlib import Path


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
                result = resp.json()["choices"][0]["message"]["content"].strip().lower()
                return "correct" if "correct" in result else "incorrect"
            except requests.exceptions.Timeout:
                time.sleep(10)
                continue
            except Exception as e:
                print(f"  Judge error: {e}")
                return "error"
        print(f"  Judge error: max retries exceeded")
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

    accuracy = sum(1 for r in results if r.get("judgment") == "correct") / n
    avg_recall = sum(r.get("gold_recall", 0) for r in results) / n
    avg_efficiency = sum(r.get("efficiency", 0) for r in results) / n
    avg_pulls = sum(r.get("pull_count", 0) for r in results) / n

    recall_vals = [r.get("gold_recall", 0) for r in results]
    ci_lo, ci_hi = bootstrap_ci(recall_vals)

    out = {
        "n": n,
        "accuracy": round(accuracy, 4),
        "avg_gold_recall": round(avg_recall, 4),
        "recall_ci95": [ci_lo, ci_hi],
        "avg_efficiency": round(avg_efficiency, 4),
        "avg_pulls": round(avg_pulls, 2),
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

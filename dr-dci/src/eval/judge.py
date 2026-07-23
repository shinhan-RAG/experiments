"""
LLM-as-Judge 평가기

가이드 반영 사항:
- P0-1: exact-match 판정 파싱 — 'correct'/'incorrect'만 인정, 그 외는 format_error
- P0-8/Part4: reference 없는 query는 'n/a'로 두고 accuracy 분모에서 제외 (nullable accuracy)
- P1-2/P1-3: read recall, duplicate pull rate 등 mechanism metric 집계
"""

import os
import re
import time
import requests

VALID_JUDGMENTS = {"correct", "incorrect"}


def parse_judgment(raw: str) -> str:
    """judge 응답을 엄격하게 파싱한다.

    - 앞뒤 공백/따옴표/구두점 제거 후 전체가 정확히 correct 또는 incorrect일 때만 인정
    - 그 외(빈 응답, 설명이 붙은 응답, 형식 위반)는 'format_error'
    """
    if raw is None:
        return "format_error"
    cleaned = raw.strip().lower()
    cleaned = cleaned.strip("\"'`.,!:; \n\t")
    if cleaned in VALID_JUDGMENTS:
        return cleaned
    # 단독 단어로 한 줄에만 등장하는 경우 허용 (예: "Judgment: correct" 는 불허)
    lines = [ln.strip().strip("\"'`.,!:;") for ln in cleaned.splitlines() if ln.strip()]
    if len(lines) == 1 and lines[0] in VALID_JUDGMENTS:
        return lines[0]
    return "format_error"


class Judge:
    def __init__(self, llm_url: str, model_name: str, prompt_template: str, api_key: str = None):
        self.llm_url = llm_url
        self.model_name = model_name
        self.prompt_template = prompt_template
        self.api_key = api_key or os.getenv("OPENAI_API_KEY", "")

    def evaluate_accuracy(self, query: str, reference_answer: str, candidate_answer: str) -> str:
        """LLM-as-Judge로 정답 여부 판정 (retry with backoff).

        반환: 'correct' | 'incorrect' | 'format_error' | 'error'
        """
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
                    wait = 10 * (attempt + 1)
                    print(f"  Judge rate limited, waiting {wait}s...")
                    time.sleep(wait)
                    continue
                resp.raise_for_status()
                result = resp.json()["choices"][0]["message"]["content"]
                return parse_judgment(result)
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
    """실험 결과 리스트에서 집계 메트릭 계산.

    accuracy는 실제 판정된(correct/incorrect) query만 분모로 사용하고,
    판정 대상이 없으면 None으로 보고한다 (P0-8).
    """
    n = len(results)
    if n == 0:
        return {}

    judged = [r for r in results if r.get("judgment") in VALID_JUDGMENTS]
    n_correct = sum(1 for r in judged if r["judgment"] == "correct")
    accuracy = round(n_correct / len(judged), 4) if judged else None

    judge_errors = sum(1 for r in results if r.get("judgment") in ("error", "format_error"))

    def avg(key, default=0):
        return sum(r.get(key, default) or 0 for r in results) / n

    metrics = {
        "n": n,
        "n_judged": len(judged),
        "accuracy": accuracy,
        "judge_error_count": judge_errors,
        "avg_gold_recall": round(avg("gold_recall"), 4),
        "avg_efficiency": round(avg("efficiency"), 4),
        "avg_pulls": round(avg("pull_count"), 2),
    }

    # mechanism metrics (있을 때만 집계)
    if any("read_recall" in r for r in results):
        metrics["avg_read_recall"] = round(avg("read_recall"), 4)
    if any("distinct_pull_queries" in r for r in results):
        metrics["avg_distinct_queries"] = round(avg("distinct_pull_queries"), 2)
    if any("rule_violations" in r for r in results):
        metrics["violation_rate"] = round(
            sum(1 for r in results if r.get("rule_violations")) / n, 4)
    if any("budget_exhausted" in r for r in results):
        metrics["budget_exhausted_rate"] = round(
            sum(1 for r in results if r.get("budget_exhausted")) / n, 4)
    if any(r.get("pull_stats") for r in results):
        dup_rates = []
        for r in results:
            stats = r.get("pull_stats") or []
            requested = sum(s.get("requested", 0) for s in stats)
            dups = sum(s.get("duplicate_count", 0) for s in stats)
            if requested:
                dup_rates.append(dups / requested)
        if dup_rates:
            metrics["avg_duplicate_pull_rate"] = round(sum(dup_rates) / len(dup_rates), 4)
    for k in ("recall@20", "recall@100", "ndcg@10"):
        if any(k in r for r in results):
            metrics[f"avg_{k}"] = round(avg(k), 4)

    return metrics

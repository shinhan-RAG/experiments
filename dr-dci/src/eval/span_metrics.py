"""
청크단위 검색 평가 — 세 계열 지표를 한 곳에서.

입력(질의별):
  ranked_chunks: [{ "chunk_id": str, "text": str }, ...]   # 검색이 반환한 순위 목록(청크)
  gold_chunk_ids: set[str]                                  # qrels의 정답 청크
  supporting_spans: [{ "text": str }, ...]                  # 근거 문장(원문 인용)

1) qrels 계열 (BEIR 표준):   recall@k, ndcg@k   ← 정답 청크를 top-k에 넣었나
2) supporting-span 계열:      coverage           ← top-k 텍스트가 근거를 얼마나 포함(문자 기준)
                              density            ← 가져온 텍스트 중 근거 비율(노이즈 반비례)
                              span_f1            ← coverage·density의 조화평균
모두 순수 계산(LLM 심판 불필요). k별로 잰다.
"""
import math
import re


def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


# ---------- qrels 계열 ----------
def recall_at_k(ranked_ids, gold_ids, k):
    if not gold_ids:
        raise ValueError("recall_at_k requires non-empty gold")
    return len(set(ranked_ids[:k]) & set(gold_ids)) / len(gold_ids)


def ndcg_at_k(ranked_ids, gold_ids, k):
    if not gold_ids:
        raise ValueError("ndcg_at_k requires non-empty gold")
    gold = set(gold_ids)
    dcg = sum(1.0 / math.log2(r + 2) for r, cid in enumerate(ranked_ids[:k]) if cid in gold)
    idcg = sum(1.0 / math.log2(r + 2) for r in range(min(k, len(gold))))
    return dcg / idcg if idcg > 0 else 0.0


# ---------- supporting-span 계열 ----------
def _char_overlap(evidence_norm: str, retrieved_norm: str) -> int:
    """근거 문장이 검색 텍스트에 포함된 문자 수(부분 포함도 근사).

    완전 포함이면 len(evidence). 아니면 evidence를 앞에서부터 가장 긴 접두-부분
    일치로 근사(청크 경계에서 잘린 경우 부분 점수). 간단·결정적."""
    if not evidence_norm:
        return 0
    if evidence_norm in retrieved_norm:
        return len(evidence_norm)
    # 가장 긴 연속 부분일치 길이(근사): evidence의 토큰을 이어붙이며 최장 포함 구간
    best = 0
    toks = evidence_norm.split(" ")
    for i in range(len(toks)):
        acc = ""
        for j in range(i, len(toks)):
            cand = (acc + " " + toks[j]).strip()
            if cand in retrieved_norm:
                acc = cand
                best = max(best, len(cand))
            else:
                break
    return best


def span_scores(ranked_chunks, supporting_spans, k):
    """coverage/density/f1 (top-k 청크 텍스트 합본 기준)."""
    retrieved = " ".join(c.get("text", "") for c in ranked_chunks[:k])
    r_norm = _norm(retrieved)
    ev_norms = [_norm(s.get("text", "")) for s in supporting_spans if _norm(s.get("text", ""))]
    if not ev_norms:
        return {"coverage": 0.0, "density": 0.0, "span_f1": 0.0}

    ev_total = sum(len(e) for e in ev_norms)
    ev_hit = sum(_char_overlap(e, r_norm) for e in ev_norms)
    coverage = ev_hit / ev_total if ev_total else 0.0          # 근거를 얼마나 담았나
    density = ev_hit / len(r_norm) if r_norm else 0.0          # 가져온 것 중 근거 비율
    density = min(density, 1.0)
    f1 = (2 * coverage * density / (coverage + density)) if (coverage + density) else 0.0
    return {"coverage": round(coverage, 4), "density": round(density, 4), "span_f1": round(f1, 4)}


# ---------- 통합 ----------
def evaluate_query(ranked_chunks, gold_chunk_ids, supporting_spans=None, ks=(5, 10, 20)):
    """qrels 지표(recall/ndcg)는 gold만 있으면 항상 계산.
    span 지표(coverage/density/f1)는 supporting_spans가 있을 때만 추가."""
    ranked_ids = [c["chunk_id"] for c in ranked_chunks]
    out = {}
    for k in ks:
        out[f"recall@{k}"] = round(recall_at_k(ranked_ids, gold_chunk_ids, k), 4)
        out[f"ndcg@{k}"] = round(ndcg_at_k(ranked_ids, gold_chunk_ids, k), 4)
    if supporting_spans:
        for k in ks:
            s = span_scores(ranked_chunks, supporting_spans, k)
            out[f"coverage@{k}"] = s["coverage"]
            out[f"density@{k}"] = s["density"]
            out[f"span_f1@{k}"] = s["span_f1"]
    return out


def aggregate(per_query):
    """질의별 dict 리스트 → 지표별 평균."""
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: round(sum(q[k] for q in per_query) / len(per_query), 4) for k in keys}

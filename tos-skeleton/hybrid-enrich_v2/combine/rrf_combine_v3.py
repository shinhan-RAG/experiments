#!/usr/bin/env python3
"""rrf_combine_v2.py 위에 2026-08-24 검증된 개선 4종을 추가한 최종본.
기존 filesearch/vector_search 공유 코드는 무수정. 이 파일 자체도 새 파일(v2는 그대로 둠).

추가된 것 (전부 combine 레이어 자체 후처리, RRF+scope-boost 계산 이후 top-40 확정 전 단계):
1. scope-boost: qtags(LLM 예측 contract)와 후보의 contract_scope가 일치하면 x2.5 (v2와 동일, 변경 없음)
2. 목차 감점: '제N-M조' 패턴이 5회 이상 나오는 조각(=목차/색인일 가능성) 감점
3. 안내문구(stub) 감점: 30자 미만이면서 조/항/사유/지급/정의/급여금/특약보험기간 등 실질 키워드가
   하나도 없는 조각(=상품명만 있는 표지/구분선) 감점
4. 구비서류 동의어: 질문 토큰에 '구비서류/필요서류/제출서류'가 있으면 '사고증명서/진단서/증명서' 토큰을 추가해
   어휘매칭 보강 (아이디어는 aliases.json과 동일한 방향이나, contract 슬롯이 아니라 어휘매칭(lex=count)에 직접 적용)
5. Kiwi 형태소분석: 질의 원문에 더해 Kiwi로 뽑은 명사/동사어간(NNG/NNP/VV/VA 등)을 어휘매칭 토큰에 추가
   (콜로키얼 어미 때문에 원문 토큰이 문서 표기와 안 겹치는 문제 완화. pip install kiwipiepy, 순수 로컬/CPU, GPU 불필요)

효과(gold_v4_train_full_patch13.jsonl, train332 기준, RRF+boost x2.5 seed 단독 대비):
  baseline(scope-boost만)      R@20 .820
  + 목차/안내문구 감점          R@20 .823
  + 구비서류 동의어             R@20 .826
  + Kiwi 형태소분석             R@20 .841   (최종)
"""
import json, sys, re
from pathlib import Path
from kiwipiepy import Kiwi

HERE = Path(__file__).resolve().parent  # tos-skeleton/hybrid-enrich_v2/combine
V2 = HERE.parent
FS = V2 / "filesearch"
VS = V2 / "vector_search"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(VS))

from units import Units
from scoring import score
from textmatch import contract_core as _contract_core_raw
from clm_search import SlotSearch

OUT = HERE / "out"
OUT.mkdir(parents=True, exist_ok=True)

GOLD = HERE / "gold_v4_train_full_patch13.jsonl"
QTAGS = FS / "out" / "qtags_sonnet.jsonl"
VS_CACHE = OUT / "vs_dense_ranks.jsonl"

# --- '주계약(X)' 정규화 버그 로컬 우회 (textmatch.py 는 무수정) ---
_JUYAKKYAK_RE = re.compile(r"^주계약\((.*)\)$")
def _strip_juyakkyak(scope):
    if not scope:
        return scope
    m = _JUYAKKYAK_RE.match(scope.strip())
    return m.group(1) if m else scope
def contract_core(v):
    return _contract_core_raw(_strip_juyakkyak(v))

# --- 목차/안내문구 감점 ---
_ARTICLE_PAT = re.compile(r"제\d+-?\d*조")
_STUB_KWS = ["조", "항", "사유", "지급", "정의", "급여금", "특약보험기간"]
def _toc_count(text):
    return len(_ARTICLE_PAT.findall(text))
def _is_stub(text):
    t = text.strip()
    if len(t) >= 30:
        return False
    return not any(kw in t for kw in _STUB_KWS)

# --- 구비서류류 질문 어휘 확장 (도메인 지식, 문서 재현시마다 새로 발견해서 추가해야 함 — 자동 일반화 안 됨) ---
DOC_SYN = {
    "구비서류": ["사고증명서", "진단서", "증명서"],
    "필요서류": ["사고증명서", "진단서", "증명서"],
    "제출서류": ["사고증명서", "진단서", "증명서"],
}

_kiwi = Kiwi()
_KEEP_TAGS = {"NNG", "NNP", "NNB", "NR", "NP", "VV", "VA", "XSV", "XSA", "SL", "SN"}
def kiwi_tokens(q):
    """명사/동사·형용사 어간만 남기고 조사/어미는 버림 (콜로키얼 활용형 대응)."""
    try:
        result = _kiwi.analyze(q)
        return [t.form for t in result[0][0] if t.tag in _KEEP_TAGS and len(t.form) >= 2]
    except Exception:
        return []


def load_gold():
    G = [json.loads(l) for l in open(GOLD, encoding="utf-8")]
    return [g for g in G if g["groups"] and g.get("status", "ok") == "ok"]


def load_qtags():
    QT = {}
    for l in open(QTAGS, encoding="utf-8"):
        d = json.loads(l)
        QT[d["qid"]] = d
    return QT


def rrf_merge(rank_lists, weights, k):
    sc = {}
    for ids, w in zip(rank_lists, weights):
        for r, i in enumerate(ids):
            sc[i] = sc.get(i, 0.0) + w / (k + r + 1)
    return sc


def build_ranked(gold, S, QT, vs_ranks, e_scope, c_scope, Eby, Cby, boost=2.5, limit=40):
    """qid -> 최종 top-<limit> 후보 id 리스트 (RRF+scope-boost+목차/stub감점+동의어+Kiwi 전부 반영)."""

    def id_scope(i):
        return e_scope.get(i) or c_scope.get(i)

    def item_of(i):
        return Eby.get(i) or Cby.get(i)

    def to_ranked(sc):
        return [i for i, _ in sorted(sc.items(), key=lambda x: -x[1])]

    results = {}
    for g in gold:
        qid = g["qid"]
        slots, toks = S.router.route(g["q"])
        slots.pop("_conf", {})

        extra = []
        for t in list(toks):
            if t in DOC_SYN:
                extra.extend(DOC_SYN[t])
        toks = list(dict.fromkeys(toks + extra))
        toks = list(dict.fromkeys(toks + kiwi_tokens(g["q"])))

        lt = QT.get(qid, {})
        llm = {k: lt.get(k) or [] for k in ("contract", "role", "subject", "qualifier", "schema")}
        llm = {k: v for k, v in llm.items() if v}
        for k, v in llm.items():
            slots[k] = list(dict.fromkeys(list(slots.get(k, [])) + v))

        M, L = S.match_table(slots, toks)
        res = S.rank(M, L, mode="clm", lex="count", weights={"contract": 2.0}, limit=100)
        fs_ranked = [e["element_id"] for e, _ in res]

        qs = {contract_core(c) for c in (lt.get("contract") or []) if c}
        sc = rrf_merge([fs_ranked, vs_ranks.get(qid, [])], [1.0, 1.3], 10)
        for i in list(sc.keys()):
            if qs and contract_core(id_scope(i) or "") in qs:
                sc[i] *= boost

        for i in list(sc.keys()):
            item = item_of(i)
            if not item:
                continue
            tc = _toc_count(item["text"])
            if tc >= 5:
                sc[i] /= (1 + tc)
            elif _is_stub(item["text"]):
                sc[i] /= 5.0

        results[qid] = to_ranked(sc)[:limit]
    return results


def main():
    gold = load_gold()
    QT = load_qtags()
    S = SlotSearch(str(FS / "out" / "elements_u2.jsonl"), str(FS / "out" / "tags_u2_rules.jsonl"))
    e_scope = {e["element_id"]: e["contract_scope"] for e in S.E}
    Eby = {e["element_id"]: e for e in S.E}
    chunks = [json.loads(l) for l in open(VS / "out" / "chunks.jsonl", encoding="utf-8")]
    c_scope = {c["chunk_id"]: c["contract_scope"] for c in chunks}
    Cby = {c["chunk_id"]: c for c in chunks}
    vs_ranks = {json.loads(l)["qid"]: json.loads(l)["c_top"] for l in open(VS_CACHE, encoding="utf-8")}
    units = Units(jo_path=FS / "out" / "elements_u2jo.jsonl", chunks_path=VS / "out" / "chunks.jsonl")
    gold = [g for g in gold if g["qid"] in vs_ranks]

    results = build_ranked(gold, S, QT, vs_ranks, e_scope, c_scope, Eby, Cby)

    agg = {k: [] for k in ("R@1", "R@5", "R@10", "R@20", "R@40", "S@5", "suff@10", "RR@10")}
    for g in gold:
        ranked_units = units.resolve(results[g["qid"]])
        sc_ = score(ranked_units, g["groups"], ks=(1, 5, 10, 20, 40))
        for k in agg:
            agg[k].append(sc_[k])
    m = {k: sum(v) / len(v) for k, v in agg.items()}
    hdr = ["R@1", "R@5", "R@10", "R@20", "R@40", "S@5", "suff@10", "RR@10"]
    print(f"n={len(gold)}")
    print("| arm | " + " | ".join(hdr) + " |\n|---|" + "---|" * len(hdr))
    print("| RRF+boost+목차/stub감점+구비서류동의어+Kiwi | " + " | ".join(f"{m[k]:.3f}" for k in hdr) + " |")
    json.dump({"metrics": m, "n": len(gold)}, open(OUT / "rrf_combine_v3_summary.json", "w"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()

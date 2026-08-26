#!/usr/bin/env python3
"""Semantic-tag retrieval: CLM + sparse BM25, fused by RRF — 0826 LLM v3 태그 지원.

0821 tag_hybrid.py 기반. 변경점:
- default_paths()가 --tags CLI 인자로 태그 파일을 바꿀 수 있도록 함 (규칙 vs LLM 비교용)
- build_view()가 table_summary, 확장된 subject_key를 포함
- search()에 structured BM25F 채널 추가 (sfw weights)
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import pickle
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
FS = ROOT / "filesearch"
VS = ROOT / "vector_search"
sys.path.insert(0, str(FS))
sys.path.insert(0, str(VS))

from clm_search import SlotSearch  # noqa: E402

TOKEN_RE = re.compile(r"[가-힣A-Za-z0-9%]+")
ROLE_KO = {
    "exclusion_exception": "면책 제외 부지급 예외",
    "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급 사유 조건",
    "payment_amount": "보험금 급여금 지급 금액",
    "limit_frequency": "지급 한도 횟수 제한",
    "timing_period": "기간 시점 보장개시",
    "definition": "용어 정의",
    "criteria_rule": "판정 인정 기준",
    "contract_lifecycle": "계약 갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 절차 서류",
    "code_reference": "질병 분류 코드 별표 부표",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def tokens(text: str) -> list[str]:
    words = [x.lower() for x in TOKEN_RE.findall(text or "")]
    out = list(words)
    for word in words:
        if len(word) >= 3 and not word.isascii():
            out.extend(word[i: i + 2] for i in range(len(word) - 1))
    return out


def _flat(values: Iterable[object]) -> str:
    return " ".join(str(x) for x in values if x not in (None, ""))


def build_view(tag: dict, element: dict, text_chars: int = 800) -> str:
    loc = tag.get("locator") or {}
    roles = tag.get("role") or []
    role_text = " ".join(x for r in roles for x in (r, ROLE_KO.get(r, "")) if x)
    fields = [
        "[특약] " + str(tag.get("contract_key") or ""),
        "[대상] " + _flat(tag.get("subject_key") or []),
        "[역할] " + role_text,
        "[조건] " + _flat(tag.get("qualifier") or []),
        "[참조] " + _flat(tag.get("reference") or []),
        "[위치] " + _flat(
            [loc.get("section"), loc.get("article"), loc.get("article_title")]
            + list(loc.get("table_headers") or [])
            + list(loc.get("row_keys") or [])
        ),
        "[유형] " + str(tag.get("schema_tag") or ""),
    ]
    if tag.get("table_summary"):
        fields.append("[표요약] " + tag["table_summary"])
    fields.append("[태그검색문] " + str(tag.get("search_text") or ""))
    fields.append("[원문] " + " ".join((element.get("text") or "").split())[:text_chars])
    return " | ".join(x for x in fields if x.split("]", 1)[-1].strip(" |"))


def build_sparse(views: list[str]) -> dict:
    postings: dict[str, list[tuple[int, int]]] = defaultdict(list)
    lengths = np.zeros(len(views), dtype=np.int32)
    for i, view in enumerate(views):
        counts = Counter(tokens(view))
        lengths[i] = sum(counts.values())
        for term, tf in counts.items():
            postings[term].append((i, tf))
    return {
        "postings": dict(postings),
        "lengths": lengths,
        "avgdl": float(lengths.mean()) if len(lengths) else 0.0,
        "n": len(views),
    }


def sparse_search(index: dict, query: str, top_k: int = 200) -> list[tuple[int, float]]:
    n = int(index["n"])
    if not n:
        return []
    scores: dict[int, float] = defaultdict(float)
    qcounts = Counter(tokens(query))
    avgdl = max(float(index["avgdl"]), 1.0)
    lengths = index["lengths"]
    k1, b = 1.5, 0.75
    for term, qtf in qcounts.items():
        posting = index["postings"].get(term) or []
        df = len(posting)
        if not df:
            continue
        idf = math.log(1.0 + (n - df + 0.5) / (df + 0.5))
        for i, tf in posting:
            denom = tf + k1 * (1.0 - b + b * float(lengths[i]) / avgdl)
            scores[i] += min(qtf, 2) * idf * tf * (k1 + 1.0) / denom
    return sorted(scores.items(), key=lambda x: (-x[1], x[0]))[:top_k]


def build_index(
    elements_path: Path,
    tags_path: Path,
    jo_path: Path,
    out_dir: Path,
) -> dict:
    elements = {r["element_id"]: r for r in map(json.loads, elements_path.open(encoding="utf-8"))}
    tags = list(map(json.loads, tags_path.open(encoding="utf-8")))
    ids = [t["element_id"] for t in tags if t["element_id"] in elements]
    tag_map = {t["element_id"]: t for t in tags}
    views = [build_view(tag_map[eid], elements[eid]) for eid in ids]
    sparse = build_sparse(views)
    jo_rows = list(map(json.loads, jo_path.open(encoding="utf-8")))
    m2j = {m: j["element_id"] for j in jo_rows for m in j["members"]}
    payload = {
        "version": 2,
        "ids": ids,
        "views": views,
        "sparse": sparse,
        "m2j": m2j,
        "sources": {
            "elements": {"path": str(elements_path.resolve()), "sha256": sha256(elements_path)},
            "tags": {"path": str(tags_path.resolve()), "sha256": sha256(tags_path)},
            "jo": {"path": str(jo_path.resolve()), "sha256": sha256(jo_path)},
        },
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "tag_index.pkl").open("wb") as f:
        pickle.dump(payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    with (out_dir / "tag_slot.pkl").open("wb") as f:
        pickle.dump(SlotSearch(str(elements_path), str(tags_path)), f, protocol=pickle.HIGHEST_PROTOCOL)
    meta = {"n": len(ids), "tag_embedding": False, "channels": ["clm", "bm25"],
            "sources": payload["sources"], "tags_file": str(tags_path.name)}
    (out_dir / "tag_index_meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return meta


class TagHybridSearch:
    def __init__(self, elements_path: Path, tags_path: Path, jo_path: Path, index_dir: Path):
        self.elements_path = Path(elements_path)
        self.tags_path = Path(tags_path)
        self.jo_path = Path(jo_path)
        self.index_dir = Path(index_dir)
        slot_cache = self.index_dir / "tag_slot.pkl"
        if slot_cache.exists():
            with slot_cache.open("rb") as f:
                self.slot = pickle.load(f)
        else:
            self.slot = SlotSearch(str(self.elements_path), str(self.tags_path))
        self.elements = {e["element_id"]: e for e in self.slot.E}
        self.jos = list(map(json.loads, self.jo_path.open(encoding="utf-8")))
        self.jid = {j["element_id"]: j for j in self.jos}
        self.m2j = {m: j["element_id"] for j in self.jos for m in j["members"]}
        self._index = None
        self._sparse_cache = {}
        self._route_cache = {}
        self._clm_cache = {}
        self._load_index()

    def _load_index(self) -> dict:
        if self._index is None:
            p = self.index_dir / "tag_index.pkl"
            if not p.exists():
                raise RuntimeError(f"태그 인덱스가 없습니다: {p}; build_tag_index.py를 먼저 실행하십시오")
            with p.open("rb") as f:
                self._index = pickle.load(f)
            expected = self._index.get("sources", {})
            current = {"elements": self.elements_path, "tags": self.tags_path, "jo": self.jo_path}
            for key, path in current.items():
                if expected.get(key, {}).get("sha256") != sha256(path):
                    raise RuntimeError(f"태그 인덱스 소스 변경됨: {key}; 인덱스를 재생성하십시오")
        return self._index

    def route(self, query: str, explicit: dict[str, list[str]] | None = None):
        explicit_key = tuple((k, tuple(v)) for k, v in sorted((explicit or {}).items()))
        cache_key = (query, explicit_key)
        if cache_key in self._route_cache:
            slots, toks, conf = self._route_cache[cache_key]
            return {k: list(v) if isinstance(v, list) else v for k, v in slots.items()}, list(toks), dict(conf)
        slots, toks = self.slot.router.route(query)
        conf = slots.pop("_conf", {})
        for field, values in (explicit or {}).items():
            if values:
                slots[field] = list(dict.fromkeys(list(slots.get(field, [])) + list(values)))
                conf.pop(field, None)
        self._route_cache[cache_key] = (slots, toks, conf)
        return {k: list(v) if isinstance(v, list) else v for k, v in slots.items()}, list(toks), dict(conf)

    def search(
        self,
        query: str,
        *,
        explicit: dict[str, list[str]] | None = None,
        weights: dict[str, float] | None = None,
        channel_top_k: int = 200,
        limit: int = 400,
        collapse_jo: bool = True,
    ) -> tuple[list[dict], dict, list[str]]:
        weights = {"clm": 1.0, "sparse": 1.0, **(weights or {})}
        slots, qtoks, conf = self.route(query, explicit)
        score_weights = {k: v * conf.get(k, 1.0) for k, v in (weights.get("fields") or {}).items()}
        clm_key = (
            query,
            tuple((k, tuple(v)) for k, v in sorted(slots.items())),
            tuple(sorted(score_weights.items())),
            channel_top_k,
        )
        if clm_key not in self._clm_cache:
            self._clm_cache[clm_key] = self.slot.search(
                slots, qtoks, mode="clm", lex="count", weights=score_weights, limit=channel_top_k
            )
        clm = self._clm_cache[clm_key]
        channels: dict[str, list[tuple[str, float]]] = {
            "clm": [(e["element_id"], float(sc)) for e, sc in clm]
        }
        index = None
        if weights.get("sparse", 0):
            index = self._load_index()
        if weights.get("sparse", 0):
            sparse_key = (query, channel_top_k)
            if sparse_key not in self._sparse_cache:
                self._sparse_cache[sparse_key] = sparse_search(index["sparse"], query, channel_top_k)
            channels["sparse"] = [(index["ids"][i], sc) for i, sc in self._sparse_cache[sparse_key]]

        fused: dict[str, float] = defaultdict(float)
        provenance: dict[str, dict] = defaultdict(dict)
        for channel, hits in channels.items():
            weight = float(weights.get(channel, 0))
            if not weight:
                continue
            for rank, (eid, raw_score) in enumerate(hits, 1):
                fused[eid] += weight / (60 + rank)
                provenance[eid][f"{channel}_rank"] = rank
                provenance[eid][f"{channel}_score"] = round(raw_score, 6)
        ordered = sorted(fused, key=lambda eid: (-fused[eid], eid))
        rows = []
        seen_jo = set()
        for eid in ordered:
            jo = self.m2j.get(eid, "")
            if collapse_jo and jo and jo in seen_jo:
                continue
            if jo:
                seen_jo.add(jo)
            rows.append({
                "element": self.elements[eid],
                "jo": jo,
                "score": fused[eid],
                "provenance": provenance[eid],
            })
            if len(rows) >= limit:
                break
        return rows, slots, qtoks


def default_paths(tags_override=None):
    tags = Path(tags_override) if tags_override else FS / "out" / "tags_u4_fact_rules.jsonl"
    return (
        FS / "out" / "elements_u3.jsonl",
        tags,
        FS / "out" / "elements_u3jo.jsonl",
        HERE / "out" / "tag_index",
    )

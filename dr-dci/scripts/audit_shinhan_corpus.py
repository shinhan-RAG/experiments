"""신한라이프 corpus 보수적 정제 감사 (LLM 호출 없음, 원본 불변).

`corpus.jsonl`을 **읽기만** 하고 부가 산출물만 만든다. `_id`를 바꾸지 않으므로
qrels/qa_meta의 gold가 절대 깨지지 않고, 이미 실행된 Part 4 결과와의 비교도 유지된다.

산출 (data/raw/shinhan/):
  audit_duplicates.json    텍스트 완전 중복 그룹 + 그룹별 대표 _id
  audit_long_chunks.json   색인 절단(4096자) 대상 + gold 영향 + 근거 span 소실 질의
  audit_element_type.json  element_type 정규식 휴리스틱 오분류 후보
  manifest.json            데이터 출처·규모·생성정보·한계 목록

이 산출물은 실험 실행에 쓰이지 않는다. **결과 해석 단계에서** 보정 recall을 병기하고
한계를 명시하는 근거로 쓴다.

  python scripts/audit_shinhan_corpus.py
"""
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import dataset_dir, load_jsonl  # noqa: E402

DATA_DIR = Path(__file__).parent.parent / "data"
DATASET = "shinhan"

# src/agent/retriever.py:96,98 — 색인 시 embed_text / doc_raw_texts 를 무조건 자르는 길이.
# 이 값이 바뀌면 여기도 함께 고쳐야 한다.
EMBED_TRUNC_CHARS = 4096

# prefix arm은 prefix를 본문 앞에 붙인 뒤 자른다(retriever.py:93-96). 근거 span이
# 창 끝에서 이보다 가까우면 prefix 길이만큼 밀려 창 밖으로 나갈 수 있다.
# build_prefix.py 가 50~100 토큰을 요구하므로 한국어 기준 여유 500자를 경계로 본다.
PREFIX_MARGIN_CHARS = 500

# build_shinhan_corpus.py:20-21 의 element_type 휴리스틱 (그대로 복제)
TABLE_ROW_RE = re.compile(r'(?m)^\s*\|.*\|\s*$')
FORMULA_RE = re.compile(r'\$\$|\\\(|\\\[')
# 페이지 번호를 세로로 나열한 LaTeX 잔재 — 정보성이 없는데 formula로 잡힌다
PHANTOM_RE = re.compile(r'\\phantom\{')


def _norm(s: str) -> str:
    """src/eval/span_metrics._norm 과 동일한 정규화."""
    return re.sub(r"\s+", " ", (s or "")).strip().lower()


def _embed_window(doc: dict) -> str:
    """retriever.index 가 실제로 임베딩/BM25에 넣는 문자열."""
    return f"{doc.get('title', '')} {doc.get('text', '')}"[:EMBED_TRUNC_CHARS]


def audit_duplicates(corpus, gold_ids, out_dir) -> dict:
    by_text = defaultdict(list)
    for doc in corpus:
        by_text[doc["text"].strip()].append(doc["_id"])

    groups = []
    for text, ids in by_text.items():
        if len(ids) < 2:
            continue
        ids = sorted(ids)
        groups.append({
            "representative": ids[0],
            "members": ids,
            "size": len(ids),
            "chars": len(text),
            "contains_gold": sorted(set(ids) & gold_ids),
            "preview": text[:120],
        })
    groups.sort(key=lambda g: -g["size"])

    dup_chunks = sum(g["size"] for g in groups)
    gold_in_dup = sorted({cid for g in groups for cid in g["contains_gold"]})
    # gold_id -> 정답으로 함께 인정해야 할 동일 텍스트 청크들
    equivalent = {
        cid: g["members"]
        for g in groups for cid in g["contains_gold"]
    }

    payload = {
        "description": (
            "텍스트가 완전히 동일한 청크 그룹. 검색이 gold와 같은 내용의 다른 청크를 "
            "가져와도 qrels 상 오답 처리되므로, 결과 해석 시 equivalent_gold_ids 를 "
            "정답으로 함께 인정한 보정 recall을 병기한다."
        ),
        "total_chunks": len(corpus),
        "duplicate_groups": len(groups),
        "chunks_in_duplicate_groups": dup_chunks,
        "duplicate_ratio": round(dup_chunks / len(corpus), 4) if corpus else 0.0,
        "gold_chunks_in_duplicate_groups": gold_in_dup,
        "equivalent_gold_ids": equivalent,
        "groups": groups,
    }
    (out_dir / "audit_duplicates.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  중복: {len(groups)}그룹 / {dup_chunks}청크 "
          f"({payload['duplicate_ratio']:.1%}), gold {len(gold_in_dup)}건 포함")
    return payload


def audit_long_chunks(corpus, gold_by_q, spans_by_q, out_dir) -> dict:
    by_id = {doc["_id"]: doc for doc in corpus}
    gold_ids = {cid for cids in gold_by_q.values() for cid in cids}

    long_chunks = []
    for doc in corpus:
        full = f"{doc.get('title', '')} {doc.get('text', '')}"
        if len(full) <= EMBED_TRUNC_CHARS:
            continue
        long_chunks.append({
            "_id": doc["_id"],
            "chars": len(full),
            "kept": EMBED_TRUNC_CHARS,
            "lost_ratio": round(1 - EMBED_TRUNC_CHARS / len(full), 4),
            "is_gold": doc["_id"] in gold_ids,
            "doc": doc.get("doc", ""),
            "element_type": doc.get("element_type", ""),
        })
    long_chunks.sort(key=lambda c: -c["chars"])

    # 근거 span이 색인 창 밖으로 밀려난 질의 — 이 질의는 dense 검색이 구조적으로 못 맞힌다.
    # 비교는 반드시 _norm 기준으로 한다: 원문과 span 사이에 공백·탭 차이가 있어
    # raw 부분문자열 검사는 창 안에 있는 span도 '소실'로 오판한다.
    unreachable, tight = [], []
    for qid, cids in gold_by_q.items():
        spans = spans_by_q.get(qid) or []
        if not spans:
            continue
        for cid in cids:
            doc = by_id.get(cid)
            if not doc:
                continue
            full = f"{doc.get('title', '')} {doc.get('text', '')}"
            window = _norm(full[:EMBED_TRUNC_CHARS])
            # prefix arm 시뮬레이션: prefix가 앞에 붙는 만큼 본문 뒤가 더 잘린다
            shrunk = _norm(full[:max(0, EMBED_TRUNC_CHARS - PREFIX_MARGIN_CHARS)])

            lost, at_risk = [], []
            for s in spans:
                ev = _norm(s.get("text", ""))
                if not ev:
                    continue
                if ev not in window:
                    lost.append(s.get("text", ""))
                elif ev not in shrunk:
                    at_risk.append(s.get("text", ""))
            entry = {
                "qid": qid,
                "gold_chunk_id": cid,
                "chunk_chars": len(full),
                "window_chars": len(window),
            }
            if lost:
                unreachable.append({**entry, "lost_spans": lost})
            elif at_risk:
                # prefix를 붙이면 창 밖으로 밀려나는 span (retriever.py:93-96)
                tight.append({**entry, "at_risk_spans": at_risk})

    n_q = len(gold_by_q)
    payload = {
        "description": (
            "src/agent/retriever.py:96,98 이 색인 시 title+text 를 무조건 "
            f"{EMBED_TRUNC_CHARS}자로 자른다. 에러가 아니라 조용히 잘리므로 로그에 남지 않는다. "
            "unreachable_queries 는 정답 근거가 이 창 밖으로 밀려나 dense 검색이 "
            "구조적으로 맞힐 수 없는 질의다 — recall 상한이 그만큼 내려간다."
        ),
        "embed_truncation_chars": EMBED_TRUNC_CHARS,
        "total_chunks": len(corpus),
        "truncated_chunks": len(long_chunks),
        "truncated_ratio": round(len(long_chunks) / len(corpus), 4) if corpus else 0.0,
        "truncated_gold_chunks": [c["_id"] for c in long_chunks if c["is_gold"]],
        "unreachable_queries": unreachable,
        "recall_ceiling": round(1 - len(unreachable) / n_q, 4) if n_q else 1.0,
        "prefix_margin_chars": PREFIX_MARGIN_CHARS,
        "tight_margin_queries": tight,
        "chunks": long_chunks,
    }
    (out_dir / "audit_long_chunks.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  절단: {len(long_chunks)}청크 ({payload['truncated_ratio']:.1%}), "
          f"gold {len(payload['truncated_gold_chunks'])}건, "
          f"근거 소실 질의 {len(unreachable)}건 -> recall 상한 {payload['recall_ceiling']:.0%}")
    if tight:
        print(f"  [!] prefix({PREFIX_MARGIN_CHARS}자 가정) 추가 시 근거가 창 밖으로 "
              f"밀리는 gold {len(tight)}건: qid {[e['qid'] for e in tight]}")
        print(f"      해당 질의는 prefix arm에서 recall이 떨어져도 보강 효과가 아니라 절단 부작용이다.")
    return payload


def audit_element_type(corpus, out_dir) -> dict:
    suspects, recomputed, declared_dist = [], defaultdict(int), defaultdict(int)
    for doc in corpus:
        text = doc.get("text", "")
        declared = doc.get("element_type", "")
        declared_dist[declared] += 1
        table_rows = len(TABLE_ROW_RE.findall(text))
        has_formula = bool(FORMULA_RE.search(text))
        actual = "table" if table_rows >= 3 else ("formula" if has_formula else "text")
        recomputed[actual] += 1

        reasons = []
        if declared != actual:
            reasons.append(f"휴리스틱 재계산 불일치: declared={declared} actual={actual}")
        if declared == "formula" and PHANTOM_RE.search(text):
            # \phantom{...} 은 페이지 번호 세로 나열 잔재 — 수식이 아니다
            reasons.append("\\phantom{} 잔재 — 페이지번호 나열을 수식으로 오인")
        if declared == "formula" and len(text.strip()) < 300 and table_rows == 0:
            reasons.append("짧고 표도 없는 formula — 표지/장식 조각 가능성")
        if reasons:
            suspects.append({
                "_id": doc["_id"],
                "declared": declared,
                "recomputed": actual,
                "table_rows": table_rows,
                "chars": len(text),
                "reasons": reasons,
                "preview": text[:120],
            })

    payload = {
        "description": (
            "element_type 은 build_shinhan_corpus.element_type 의 정규식 휴리스틱이다"
            "(표 행 3개 이상이면 table, $$ \\( \\[ 가 있으면 formula, 아니면 text). "
            "LLM 판정이 아니므로 element_type 별 분할 분석 시 신뢰도를 함께 밝힌다."
        ),
        "declared_distribution": dict(sorted(declared_dist.items())),
        "recomputed_distribution": dict(sorted(recomputed.items())),
        "suspect_count": len(suspects),
        "suspects": suspects,
    }
    (out_dir / "audit_element_type.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  element_type 의심: {len(suspects)}건 / 분포 {payload['declared_distribution']}")
    return payload


def write_manifest(corpus, queries, qrels, dup, long_, out_dir) -> None:
    per_doc = defaultdict(int)
    for doc in corpus:
        per_doc[doc.get("doc", "")] += 1
    top_doc, top_n = max(per_doc.items(), key=lambda kv: kv[1]) if per_doc else ("", 0)

    manifest = {
        "dataset": DATASET,
        "source": "신한라이프 실무문서 파싱본(md) → scripts/build_shinhan_corpus.py 청킹",
        "documents": len(per_doc),
        "chunks": len(corpus),
        "queries": len(queries),
        "qrels": len(qrels),
        "chunking": {"chunk_chars": 1400, "min_chars": 120, "split": "markdown header"},
        "qa_generation": {
            "script": "scripts/build_shinhan_qa_v2.py",
            "model": "gpt-4o-mini",
            "seed": 42,
            "n_queries": 50,
            "per_doc_cap": 4,
            "note": "verbatim 근거 span이 1개 이상 잡히는 질의만 채택. 사람 검수 없음.",
        },
        "evaluation_unit": "chunk (parent_id 없음 — 법률 데이터셋의 parent 단위 평가와 다름)",
        "limitations": [
            f"중복 청크 {dup['chunks_in_duplicate_groups']}건 "
            f"({dup['duplicate_ratio']:.1%}) — gold와 동일 텍스트를 가져와도 오답 처리. "
            "audit_duplicates.json 의 equivalent_gold_ids 로 보정 recall 병기 필요.",
            f"색인 시 {EMBED_TRUNC_CHARS}자 절단 대상 {long_['truncated_chunks']}건 "
            f"({long_['truncated_ratio']:.1%}) — 에러 없이 조용히 잘리므로 로그에 남지 않는다. "
            f"근거 소실 질의 {len(long_['unreachable_queries'])}건 → "
            f"recall 상한 {long_['recall_ceiling']:.0%}.",
            f"prefix arm에서 근거가 창 밖으로 밀릴 위험 질의 "
            f"{len(long_['tight_margin_queries'])}건"
            f"(qid {[e['qid'] for e in long_['tight_margin_queries']]}). "
            "해당 질의의 recall 하락은 보강 효과가 아니라 절단 부작용이므로 분리해 해석할 것.",
            "density·span_f1 은 분모가 top-k 청크 전체 길이라 이론적 최대치가 약 0.002다"
            "(src/eval/span_metrics.py:75). coverage 만 해석할 것.",
            "gold가 gpt-4o-mini 단일 생성이며 사람 검수·복수 정답·negative 예제가 없다. "
            "절대 점수보다 동일 데이터 내 arm 간 상대 차이로 해석할 것.",
            f"단일 문서 '{top_doc}' 가 코퍼스의 {top_n / len(corpus):.1%}({top_n}청크)를 차지한다. "
            "문서 다양성 기반 일반화 주장 불가.",
            "질의 50개 · 질의당 gold 1개 → 점수가 사실상 2%p 단위로 움직인다. "
            "'효과 없음'과 '표본 부족으로 판정 불가'를 구분해 서술할 것.",
            "element_type 은 정규식 휴리스틱이다(audit_element_type.json 참조). "
            "gold 분포가 table 27 / text 18 / formula 5 이므로 "
            "table 대 text+formula 정도로만 분할 분석하는 것이 적절하다.",
            "코퍼스가 2,802청크로 Part 2(규모 확장, 최소 20K) 대상이 아니다. "
            "규모 결과는 aihub-full 실험을 인용한다.",
        ],
        "generated_by": "scripts/audit_shinhan_corpus.py",
    }
    (out_dir / "manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"  manifest: {len(per_doc)}문서 / {len(corpus)}청크 / {len(queries)}질의")


def main() -> None:
    ds_dir = dataset_dir(DATA_DIR, DATASET)
    corpus = load_jsonl(ds_dir / "corpus.jsonl")
    queries = load_jsonl(ds_dir / "queries.jsonl")
    qrels = load_jsonl(ds_dir / "qrels.jsonl")

    gold_by_q = defaultdict(set)
    for entry in qrels:
        if entry.get("score", 0) >= 1:
            gold_by_q[str(entry["query-id"])].add(str(entry["corpus-id"]))
    gold_ids = {cid for cids in gold_by_q.values() for cid in cids}

    spans_by_q = {}
    meta_path = ds_dir / "qa_meta.jsonl"
    if meta_path.exists():
        spans_by_q = {str(m["qid"]): m.get("supporting_spans", [])
                      for m in load_jsonl(meta_path)}
    else:
        print(f"  [!] {meta_path} 없음 — 근거 span 소실 검사를 건너뛴다.")

    print(f"=== audit: {DATASET} ({len(corpus)} chunks) ===")
    dup = audit_duplicates(corpus, gold_ids, ds_dir)
    long_ = audit_long_chunks(corpus, gold_by_q, spans_by_q, ds_dir)
    audit_element_type(corpus, ds_dir)
    write_manifest(corpus, queries, qrels, dup, long_, ds_dir)
    print(f"  -> {ds_dir}")


if __name__ == "__main__":
    main()

"""
소규모 실측 pilot: 수정된 파이프라인을 실제 구성요소로 Part 1 / Part 2 재실행.

구성요소 (모두 실제):
- 모델 엔드포인트는 config/experiment.yaml의 models 섹션을 그대로 사용한다
  (임베딩/리랭커: H200 서버 포트 기준, agent/judge: OpenAI API).
- 리랭커는 선택. 없으면 HybridRAG가 자동 fallback (리랭크 생략)한다.
- H200 서버 없이 로컬 테스트할 때는 scripts/serve/의 WSL vLLM 스크립트를 띄우거나
  PILOT_EMBED=openai 로 OpenAI 임베딩으로 전환할 수 있다.

전체 20K~110K 증강 생성은 수만 호출이라, 갭 리뷰 권고대로 소규모 nested pilot로
실제 수치와 해석을 만든다. 규모는 아래 상수로 통제한다.

실행: dr-dci 에서
  .venv\\Scripts\\python.exe scripts\\run_pilot.py
"""

import json
import os
import re
import sys
import time
import random
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict

import requests
import yaml
from dotenv import load_dotenv

try:
    sys.stdout.reconfigure(encoding="utf-8")
    sys.stderr.reconfigure(encoding="utf-8")
except Exception:
    pass

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from src.agent.retriever import PullRetriever, RetrieverConfig
from src.agent.dci_agent import DCIAgent
from src.hybrid.pipeline import HybridRAG
from src.eval.judge import Judge, compute_metrics
from src.eval.metrics import bootstrap_ci, paired_bootstrap_test

DATA = ROOT / "data"
CONFIG = ROOT / "config"
OUT = ROOT / "results" / "pilot"

# ---- pilot 규모 (비용/시간 통제) — 환경변수로 재정의 가능 ----
def _env_int(name, default):
    v = os.getenv(name)
    return int(v) if v else default

N_QUERIES = _env_int("PILOT_N_QUERIES", 6)
K_GOLD_PER_Q = _env_int("PILOT_K_GOLD", 6)   # query당 subset에 포함할 gold 수
SCALES = [int(x) for x in os.getenv("PILOT_SCALES", "120,240,480").split(",")]
PART1_SCALE = _env_int("PILOT_PART1_SCALE", 120)
SEED = 42
OUT_PREFIX = os.getenv("PILOT_OUT_PREFIX", "PILOT_RESULTS")

# 모델 엔드포인트: config/experiment.yaml (기존 방식) 기준, env로 오버라이드 가능
MODELS = yaml.safe_load(open(CONFIG / "experiment.yaml", encoding="utf-8"))["models"]

OAI_URL = MODELS["agent_llm"]["url"]
OAI_MODEL = MODELS["agent_llm"]["name"]
JUDGE_URL = MODELS["judge_llm"]["url"]
JUDGE_MODEL = MODELS["judge_llm"]["name"]
OAI_KEY = os.getenv("OPENAI_API_KEY", "")

RERANKER_URL = MODELS["reranker"]["url"]
RERANKER_MODEL = MODELS["reranker"]["name"]

# 임베딩 endpoint: 서버가 없을 때 OpenAI 임베딩으로 전환 (PILOT_EMBED=openai)
if os.getenv("PILOT_EMBED", "").lower() == "openai":
    EMBED_URL = "https://api.openai.com/v1/embeddings"
    EMBED_MODEL = os.getenv("PILOT_EMBED_MODEL", "text-embedding-3-small")
    EMBED_KEY = OAI_KEY
    QUERY_INSTRUCTION = None  # gte 전용 instruction은 OpenAI 임베딩에 부적합
else:
    EMBED_URL = os.getenv("PILOT_EMBED_URL", MODELS["embedding"]["url"])
    EMBED_MODEL = os.getenv("PILOT_EMBED_MODEL", MODELS["embedding"]["name"])
    EMBED_KEY = None
    QUERY_INSTRUCTION = (
        os.getenv("PILOT_QUERY_INSTRUCTION")
        or MODELS["embedding"].get("query_instruction")
    )

TAX_SCHEMA = yaml.safe_load(open(CONFIG / "taxonomy_schemas" / "trec-covid.yaml", encoding="utf-8"))
META_SCHEMA = yaml.safe_load(open(CONFIG / "metadata_schemas" / "trec-covid.yaml", encoding="utf-8"))
SUBTAGS = ["definition", "evidence", "procedure", "condition", "comparison", "summary"]


# ---------------------------------------------------------------- OpenAI helper
def oai(messages, max_tokens=200, json_mode=False):
    if not OAI_KEY:
        raise RuntimeError("OPENAI_API_KEY is required for the real-component pilot")
    payload = {"model": OAI_MODEL, "messages": messages, "temperature": 0, "max_tokens": max_tokens}
    if json_mode:
        payload["response_format"] = {"type": "json_object"}
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {OAI_KEY}"}
    for attempt in range(4):
        try:
            r = requests.post(OAI_URL, json=payload, headers=headers, timeout=60)
            if r.status_code == 429:
                time.sleep(5 * (attempt + 1)); continue
            r.raise_for_status()
            return r.json()["choices"][0]["message"]["content"]
        except Exception as exc:
            if attempt == 3:
                raise RuntimeError("pilot model call failed after retries") from exc
            time.sleep(2)
    raise RuntimeError("pilot model call failed after retries")


def parallel(fn, items, workers=16):
    out = [None] * len(items)
    with ThreadPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(fn, it): i for i, it in enumerate(items)}
        for f, i in futs.items():
            out[i] = f.result()
    return out


def parse_json(text):
    text = text.strip()
    if text.startswith("```"):
        text = re.sub(r"^```[a-z]*\n?", "", text); text = re.sub(r"```$", "", text).strip()
    try:
        return json.loads(text)
    except Exception:
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if m:
            try:
                return json.loads(m.group(0))
            except Exception:
                raise ValueError("model response is not valid JSON")
        raise ValueError("model response is not valid JSON")


# ---------------------------------------------------------------- data loading
def load_corpus():
    docs = {}
    with open(DATA / "raw" / "trec-covid" / "corpus.jsonl", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            docs[d["_id"]] = d
    return docs


def load_queries_qrels():
    queries = {}
    with open(DATA / "raw" / "trec-covid" / "queries.jsonl", encoding="utf-8") as f:
        for line in f:
            q = json.loads(line); queries[str(q["_id"])] = q
    gold = defaultdict(list)
    with open(DATA / "raw" / "trec-covid" / "qrels.jsonl", encoding="utf-8") as f:
        for line in f:
            e = json.loads(line)
            if e["score"] >= 1:
                gold[str(e["query-id"])].append(str(e["corpus-id"]))
    return queries, gold


def build_pilot_data(corpus, queries, gold):
    rng = random.Random(SEED)

    def nonempty(did):
        d = corpus.get(did); return d and (d.get("text") or "").strip()

    # 충분한 gold를 가진 query 선택
    usable = [(qid, [g for g in gs if nonempty(g)]) for qid, gs in gold.items()]
    usable = [(qid, gs) for qid, gs in usable if len(gs) >= K_GOLD_PER_Q]
    usable.sort(key=lambda x: x[0])
    rng.shuffle(usable)
    chosen = usable[:N_QUERIES]

    query_gold = {}          # qid -> subset에 포함된 gold ids (평가용)
    gold_union = set()
    # 평가에서 빠진 positive라도 distractor가 되면 recall이 왜곡되므로
    # 선택된 질의의 전체 positive qrel을 distractor 후보에서 제외한다
    exclude_union = set()
    for qid, gs in chosen:
        picked = sorted(gs)[:K_GOLD_PER_Q]
        query_gold[qid] = picked
        gold_union.update(picked)
        exclude_union.update(gold[qid])

    gold_list = sorted(gold_union)
    if len(gold_list) > min(SCALES):
        raise ValueError(
            f"gold_union({len(gold_list)}) exceeds min scale({min(SCALES)}); "
            "reduce PILOT_N_QUERIES/PILOT_K_GOLD or raise PILOT_SCALES"
        )

    # distractor 후보 (선택 질의의 positive 전부 제외, non-empty)
    all_ids = [did for did in corpus if nonempty(did) and did not in exclude_union]
    rng.shuffle(all_ids)
    max_scale = max(SCALES)
    max_need = max_scale - len(gold_list)
    if len(all_ids) < max_need:
        raise ValueError(
            f"not enough distractor candidates: need {max_need}, have {len(all_ids)}"
        )
    distractors = all_ids[:max_need]

    # nested subset: gold_union 항상 포함 + distractor 채움
    subsets = {}
    for s in SCALES:
        need = s - len(gold_list)
        subsets[s] = gold_list + distractors[:need]
    union_ids = subsets[max_scale]
    return chosen, query_gold, subsets, union_ids


# ---------------------------------------------------------------- augmentation (OpenAI)
def gen_taxonomy(corpus, ids):
    l1 = TAX_SCHEMA["L1"]; l2 = TAX_SCHEMA["L2"]
    sys_p = ("Classify a COVID-19 research snippet into a taxonomy. "
             f"L1 one of: {l1}. L2 must be a valid sub-category of the chosen L1: {json.dumps(l2)}. "
             'Return JSON: {"L1":"...","L2":"...","L3":"1-3 word topic"}.')

    def one(did):
        d = corpus[did]; text = f"{d.get('title','')} {d.get('text','')}"[:600]
        r = oai([{"role": "system", "content": sys_p}, {"role": "user", "content": text}],
                max_tokens=60, json_mode=True)
        j = parse_json(r)
        chosen_l1 = j.get("L1")
        chosen_l2 = j.get("L2")
        if chosen_l1 not in l1 or chosen_l2 not in l2.get(chosen_l1, []):
            raise ValueError(f"invalid taxonomy output for {did}")
        return did, {"L1": chosen_l1, "L2": chosen_l2, "L3": j.get("L3", "")}

    return dict(parallel(one, ids))


def gen_prefix(corpus, ids):
    sys_p = ("Write a concise 1-2 sentence contextual summary (<=60 words) capturing the document's "
             "topic and key finding, for retrieval. Return plain text only.")

    def one(did):
        d = corpus[did]; text = f"{d.get('title','')} {d.get('text','')}"[:800]
        r = oai([{"role": "system", "content": sys_p}, {"role": "user", "content": text}], max_tokens=90)
        value = r.strip().strip('"')
        if not value:
            raise ValueError(f"empty prefix output for {did}")
        return did, value
    return dict(parallel(one, ids))


def gen_metadata(corpus, ids):
    st = META_SCHEMA["fields"]["study_type"]["values"]
    pop = META_SCHEMA["fields"]["population"]["values"]
    cats = META_SCHEMA["fields"]["entities"]["item_schema"]["category"]["values"]
    sys_p = (f"Extract metadata from a COVID-19 doc. study_type one of {st}. population one of {pop} or null. "
             f"entities: up to 4 items {{name, category}} with category in {cats}. "
             'Return JSON {"study_type":...,"year":int|null,"population":...|null,'
             '"entities":[{"name":...,"category":...}]}.')

    def one(did):
        d = corpus[did]; text = f"{d.get('title','')} {d.get('text','')}"[:800]
        r = oai([{"role": "system", "content": sys_p}, {"role": "user", "content": text}],
                max_tokens=200, json_mode=True)
        j = parse_json(r)
        if not isinstance(j, dict) or "study_type" not in j:
            raise ValueError(f"invalid metadata output for {did}")
        ents = j.get("entities") or []
        ents = [e for e in ents if isinstance(e, dict) and e.get("name")][:5]
        return did, {"study_type": j.get("study_type"), "year": j.get("year"),
                     "population": j.get("population"), "entities": ents}
    return dict(parallel(one, ids))


def gen_tags(corpus, ids):
    sys_p = (f"Split the document into up to 3 key elements. For each, assign a subtag from {SUBTAGS} "
             'and copy a short verbatim span (<=200 chars). Return JSON {"elements":[{"subtag":...,"text":...}]}.')

    def one(did):
        d = corpus[did]; text = f"{d.get('title','')} {d.get('text','')}"[:900]
        r = oai([{"role": "system", "content": sys_p}, {"role": "user", "content": text}],
                max_tokens=300, json_mode=True)
        j = parse_json(r); els = j.get("elements") if isinstance(j, dict) else None
        out = []
        for e in (els or [])[:3]:
            if isinstance(e, dict) and e.get("text"):
                sub = e.get("subtag") if e.get("subtag") in SUBTAGS else "evidence"
                out.append({"doc_id": did, "tag": f"@el:paragraph/{sub}", "text": str(e["text"])[:200]})
        if not out:
            raise ValueError(f"empty semantic-tag output for {did}")
        return out

    flat = {}
    for lst in parallel(one, ids):
        for e in lst:
            flat.setdefault(e["doc_id"], []).append(e)
    return flat


def gen_reference_answers(corpus, chosen, query_gold, queries):
    def one(item):
        qid, _ = item
        qtext = queries[qid].get("title") or queries[qid].get("text", "")
        ev = []
        for g in query_gold[qid][:5]:
            d = corpus[g]; ev.append(f"{d.get('title','')}: {d.get('text','')[:400]}")
        ctx = "\n\n".join(ev)
        r = oai([{"role": "system", "content": "Answer the question using ONLY the evidence. 2-4 sentences, factual."},
                 {"role": "user", "content": f"Question: {qtext}\n\nEvidence:\n{ctx}"}], max_tokens=200)
        value = r.strip()
        if not value:
            raise ValueError(f"empty reference answer for {qid}")
        return qid, value
    return dict(parallel(one, chosen))


# ---------------------------------------------------------------- agent runs
def build_retriever(corpus_list, prefixes, use_prefix, taxonomy=None):
    r = PullRetriever(RetrieverConfig(embedding_url=EMBED_URL, embedding_model=EMBED_MODEL,
                                      top_k=20, use_prefix=use_prefix,
                                      embedding_api_key=EMBED_KEY,
                                      query_instruction=QUERY_INSTRUCTION))
    r.index(corpus_list, prefixes=prefixes if use_prefix else None, taxonomy=taxonomy)
    return r


def run_dci_condition(retriever, corpus_dict, queries, query_gold, ref, step, aug,
                      single_pull=False):
    tags = aug["tags"] if step.get("tags") else None
    tax = aug["taxonomy"] if step.get("taxonomy") else None
    meta = aug["metadata"] if step.get("metadata") else None
    pref = aug["prefix"] if step.get("prefix") else None
    retriever.set_taxonomy(tax)
    tax_schema = TAX_SCHEMA if step.get("taxonomy") else None
    meta_schema = META_SCHEMA if step.get("metadata") else None

    agent = DCIAgent(llm_url=OAI_URL, model_name=OAI_MODEL, retriever=retriever,
                     corpus=corpus_dict, tags_data=tags, taxonomy_data=tax, metadata_data=meta,
                     prefix_data=pref, max_turns=10, workspace_max_docs=100, min_pulls=2,
                     single_pull=single_pull, taxonomy_schema=tax_schema, metadata_schema=meta_schema,
                     api_key=OAI_KEY)
    judge = Judge(JUDGE_URL, JUDGE_MODEL, open(CONFIG / "judge_prompt.txt", encoding="utf-8").read(), api_key=OAI_KEY)

    def run_q(qid):
        qtext = queries[qid].get("title") or queries[qid].get("text", "")
        out = agent.run(qtext)
        gset = set(query_gold[qid])
        gr = len(set(out["workspace_docs"]) & gset) / max(1, len(gset))
        rr = len(set(out["read_docs"]) & gset) / max(1, len(gset))
        j = judge.evaluate_accuracy(qtext, ref[qid], out["answer"]) if qid in ref else "n/a"
        return {"query_id": qid, "gold_recall": gr, "read_recall": rr,
                "efficiency": Judge.efficiency(gr, out["pull_count"]), "pull_count": out["pull_count"],
                "distinct_pull_queries": out["distinct_pull_queries"], "pull_stats": out["pull_stats"],
                "budget_exhausted": out["budget_exhausted"], "rule_violations": out["rule_violations"],
                "judgment": j}
    qids = [qid for qid, _ in queries.items()] if isinstance(queries, dict) else queries
    return parallel(run_q, list(query_gold.keys()), workers=8)


def run_hybrid_condition(corpus_list, corpus_dict, query_gold, queries, ref):
    pipe = HybridRAG(embedding_url=EMBED_URL, embedding_model=EMBED_MODEL,
                     reranker_url=RERANKER_URL, reranker_model=RERANKER_MODEL,
                     llm_url=OAI_URL, llm_model=OAI_MODEL, api_key=OAI_KEY,
                     query_instruction=QUERY_INSTRUCTION)
    pipe.index(corpus_list)
    judge = Judge(JUDGE_URL, JUDGE_MODEL, open(CONFIG / "judge_prompt.txt", encoding="utf-8").read(), api_key=OAI_KEY)

    def run_q(qid):
        qtext = queries[qid].get("title") or queries[qid].get("text", "")
        out = pipe.run(qtext)
        gset = set(query_gold[qid])
        gr = len(set(out["retrieved_docs"]) & gset) / max(1, len(gset))
        j = judge.evaluate_accuracy(qtext, ref[qid], out["answer"]) if qid in ref else "n/a"
        return {"query_id": qid, "gold_recall": gr, "read_recall": gr,
                "efficiency": Judge.efficiency(gr, out["pull_count"]), "pull_count": out["pull_count"],
                "judgment": j}
    return parallel(run_q, list(query_gold.keys()), workers=8)


# ---------------------------------------------------------------- reporting
def fmt(x):
    return " n/a" if x is None else f"{x:.3f}"


def table(rows, headers):
    w = [max(len(str(h)), *(len(str(r[i])) for r in rows)) for i, h in enumerate(headers)]
    line = "-" * (sum(w) + 3 * len(headers) + 1)
    o = [line, "| " + " | ".join(str(h).ljust(w[i]) for i, h in enumerate(headers)) + " |", line]
    for r in rows:
        o.append("| " + " | ".join(str(c).ljust(w[i]) for i, c in enumerate(r)) + " |")
    o.append(line)
    return "\n".join(o)


PART1_STEPS = [
    ("baseline", {}),
    ("taxonomy_only", {"taxonomy": True}),
    ("tags_only", {"tags": "A"}),
    ("prefix_only", {"prefix": True}),
    ("metadata_only", {"metadata": True}),
    ("stack_tax_tags", {"taxonomy": True, "tags": "A"}),
    ("stack_tax_tags_prefix", {"taxonomy": True, "tags": "A", "prefix": True}),
    ("stack_all", {"taxonomy": True, "tags": "A", "prefix": True, "metadata": True}),
]


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    print("[pilot] loading data...")
    corpus = load_corpus()
    queries, gold = load_queries_qrels()
    chosen, query_gold, subsets, union_ids = build_pilot_data(corpus, queries, gold)
    qids = list(query_gold.keys())
    print(f"[pilot] queries={len(qids)}, gold/query~{K_GOLD_PER_Q}, scales={SCALES}, union_docs={len(union_ids)}")
    if PART1_SCALE not in subsets:
        raise ValueError(
            f"PILOT_PART1_SCALE={PART1_SCALE} must be one of PILOT_SCALES={SCALES}"
        )

    print("[pilot] generating augmentations via gpt-4o-mini (union docs)...")
    aug = {}
    aug["taxonomy"] = gen_taxonomy(corpus, union_ids); print("  taxonomy done")
    aug["prefix"] = gen_prefix(corpus, union_ids); print("  prefix done")
    aug["metadata"] = gen_metadata(corpus, union_ids); print("  metadata done")
    aug["tags"] = gen_tags(corpus, union_ids); print("  tags done")
    ref = gen_reference_answers(corpus, chosen, query_gold, queries); print("  reference answers done")

    gold_union = {g for gs in query_gold.values() for g in gs}
    excluded_beyond_gold = {
        g for qid, _ in chosen for g in gold[qid]
    } - gold_union

    artifact_path = OUT / f"{OUT_PREFIX.lower()}_artifacts.json"
    artifact_path.write_text(json.dumps({
        "seed": SEED,
        "query_ids": qids,
        "corpus_ids": union_ids,
        "query_gold": query_gold,
        "excluded_positives_beyond_gold": len(excluded_beyond_gold),
        "augmentations": aug,
        "reference_answers": ref,
        "models": {
            "agent": OAI_MODEL,
            "embedding": EMBED_MODEL,
            "query_instruction": QUERY_INSTRUCTION,
        },
    }, ensure_ascii=False, indent=2), encoding="utf-8")

    queries_sub = {qid: queries[qid] for qid in qids}

    # ---------------- PART 1 (scale = PART1_SCALE) ----------------
    ids1 = subsets[PART1_SCALE]
    corpus_list1 = [corpus[d] for d in ids1]
    corpus_dict1 = {d: corpus[d] for d in ids1}
    pref1 = {d: aug["prefix"][d] for d in ids1 if d in aug["prefix"]}
    r_noprefix = build_retriever(corpus_list1, None, use_prefix=False)
    r_prefix = build_retriever(corpus_list1, pref1, use_prefix=True)

    print("\n[pilot] === PART 1 ===")
    part1 = {}
    per_recall = {}
    for name, step in PART1_STEPS:
        retr = r_prefix if step.get("prefix") else r_noprefix
        res = run_dci_condition(retr, corpus_dict1, queries_sub, query_gold, ref, step, aug)
        m = compute_metrics(res)
        part1[name] = {"metrics": m, "results": res}
        per_recall[name] = [r["gold_recall"] for r in res]
        print(f"  {name:22s} recall={fmt(m['avg_gold_recall'])} acc={fmt(m['accuracy'])} "
              f"readrec={fmt(m.get('avg_read_recall'))}")

    base = part1["baseline"]["metrics"]["avg_gold_recall"] or 1e-9
    rows1 = []
    for name, _ in PART1_STEPS:
        m = part1[name]["metrics"]
        vs = "-" if name == "baseline" else f"{(m['avg_gold_recall']-base)/base*100:+.0f}%"
        rows1.append([name, fmt(m["avg_gold_recall"]), vs, fmt(m.get("avg_read_recall")),
                      fmt(m["accuracy"]), f"{m['avg_efficiency']:.3f}", f"{m['avg_pulls']:.2f}",
                      f"{m.get('avg_duplicate_pull_rate',0):.3f}", f"{m.get('violation_rate',0):.2f}"])
    t1 = table(rows1, ["Step", "Recall", "vs base", "ReadRec", "Acc", "Effic", "Pulls", "DupRate", "Viol"])
    pb = paired_bootstrap_test(per_recall["stack_all"], per_recall["baseline"])

    # ---------------- PART 2 (scales) ----------------
    print("\n[pilot] === PART 2 ===")
    stack = {"taxonomy": True, "tags": "A", "prefix": True, "metadata": True}
    part2 = {}
    rows2 = []
    scale_recall = defaultdict(dict)
    for s in SCALES:
        ids = subsets[s]
        clist = [corpus[d] for d in ids]; cdict = {d: corpus[d] for d in ids}
        prefs = {d: aug["prefix"][d] for d in ids if d in aug["prefix"]}
        rp = build_retriever(clist, prefs, use_prefix=True, taxonomy=aug["taxonomy"])

        dr = run_dci_condition(rp, cdict, queries_sub, query_gold, ref, stack, aug)
        rp.set_taxonomy(aug["taxonomy"])
        sp = run_dci_condition(rp, cdict, queries_sub, query_gold, ref, stack, aug, single_pull=True)
        hy = run_hybrid_condition(clist, cdict, query_gold, queries_sub, ref)
        for method, res in [("DR-DCI", dr), ("Single-Pull", sp), ("Hybrid", hy)]:
            m = compute_metrics(res)
            part2[f"{method}_{s}"] = {"metrics": m, "results": res}
            scale_recall[method][s] = m["avg_gold_recall"]
            rows2.append([f"{s}", method, fmt(m["accuracy"]), fmt(m["avg_gold_recall"]),
                          f"{m['avg_efficiency']:.3f}", f"{m['avg_pulls']:.2f}"])
        print(f"  scale {s}: done")
    t2 = table(rows2, ["Scale(docs)", "Method", "Acc", "GoldRec", "Effic", "Pulls"])

    deg_lines = []
    for method in ["DR-DCI", "Single-Pull", "Hybrid"]:
        s0, s1 = SCALES[0], SCALES[-1]
        r0, r1 = scale_recall[method][s0], scale_recall[method][s1]
        deg = Judge.degradation_rate(r0, r1)
        deg_lines.append(f"  {method:12s}: {r0:.3f} → {r1:.3f}  (degradation {deg:.1f}%)")

    # ---------------- output ----------------
    report = []
    report.append("# Pilot 실측 결과 (real components)\n")
    report.append(f"- 검색 임베딩: {EMBED_MODEL} / augmentation·agent: {OAI_MODEL} / judge: {JUDGE_MODEL}")
    report.append(f"- 규모: queries={len(qids)}, gold/query={K_GOLD_PER_Q}, Part1={PART1_SCALE} docs, Part2 scales={SCALES}")
    report.append("- recall은 subset에 포함된 gold 대비(achievable recall). 리랭커 미기동 시 Hybrid는 rerank 생략.\n")
    report.append("## Part 1 — augmentation 적층\n")
    report.append("```\n" + t1 + "\n```")
    report.append(f"\n통계(gold_recall, paired bootstrap): Δ(stack_all−baseline)={pb['mean_diff']} "
                  f"95%CI=[{pb['ci_low']}, {pb['ci_high']}] p={pb['p_value']}\n")
    report.append("## Part 2 — 규모 확장 + Single Pull\n")
    report.append("```\n" + t2 + "\n```")
    report.append("\nDegradation Rate (gold_recall, 최소→최대 scale):\n```\n" + "\n".join(deg_lines) + "\n```")

    md = "\n".join(report)
    (OUT / f"{OUT_PREFIX}.md").write_text(md, encoding="utf-8")
    with open(OUT / f"{OUT_PREFIX.lower()}_raw.json", "w", encoding="utf-8") as f:
        json.dump({"part1": part1,
                   "part2": part2,
                   "part1_stat": pb, "config": {"n_queries": len(qids), "scales": SCALES,
                   "k_gold": K_GOLD_PER_Q, "part1_scale": PART1_SCALE,
                   "seed": SEED, "query_instruction": QUERY_INSTRUCTION}},
                  f, ensure_ascii=False, indent=2)

    print("\n" + t1)
    print(f"\n[Part1 stat] Δ(stack_all-baseline) recall={pb['mean_diff']} CI=[{pb['ci_low']},{pb['ci_high']}] p={pb['p_value']}")
    print("\n" + t2)
    print("\n[Degradation]\n" + "\n".join(deg_lines))
    print(f"\n[saved] {OUT/(OUT_PREFIX+'.md')}\n[saved] {OUT/(OUT_PREFIX.lower()+'_raw.json')}"
          f"\n[saved] {artifact_path}")


if __name__ == "__main__":
    main()

"""Build the frozen 500-question QA set.

Inputs: DocFacts (filenames/paths) + original Markdown bodies + ripgrep.
Index text, frontmatter text, prior rankings are never read here.
The only non-source input is each target document's structure type (A/B/C/D)
used purely for quota stratification, passed in as an opaque doc_id->letter map.
LLM: none — question phrasing is deterministic template rotation.
"""
import hashlib
import json
import re
import subprocess
from collections import Counter, defaultdict
from pathlib import Path
from random import Random

from .config import Config, SCHEMA_VERSION, sha256_text
from .source_facts import DocFacts, guard

_HANGUL = re.compile(r"[가-힣]")
_TOKEN_STRIP = "()[]{}.,;:!?\"'※□○·"

CONTENT_TEMPLATES = (
    "{phrase} 내용이 포함된 문서를 찾아줘",
    "{phrase}에 관한 내용이 있는 문서는?",
    "{phrase} 내용을 담고 있는 문서를 알려줘",
)
MIXED_TEMPLATES = (
    ("latest", "{phrase} 내용이 있는 최신 {dtype}"),
    ("year", "{phrase} 내용이 있는 {year}년 {dtype}"),
    ("type", "{phrase} 내용이 있는 {dtype}"),
)
IDENTITY_TEMPLATES = (
    ("type", "{name} {dtype} 보여줘"),
    ("year_type", "{name} {year}년 {dtype}"),
    ("latest", "{name} 최신 {dtype}"),
)


def norm_q(q: str) -> str:
    return re.sub(r"\s+", "", q)


class RgGold:
    """Whitespace-insensitive corpus-wide evidence search via ripgrep.

    Tokens are joined with \\s+ so a line-wrap in a sibling version still
    counts as gold (fixes PR #14 v2 single-anchor under-count).
    """

    def __init__(self, source_root: Path):
        self.root = source_root

    def pattern(self, tokens: list[str]) -> str:
        return r"[\s]*".join(re.escape(t) for t in tokens)

    def files_with(self, tokens: list[str]) -> list[str]:
        pat = self.pattern(tokens)
        r = subprocess.run(
            ["rg", "-l", "--multiline", "-e", pat, "--", str(self.root)],
            capture_output=True, text=True, timeout=120)
        out = []
        for line in r.stdout.splitlines():
            p = Path(line)
            out.append(str(p.relative_to(self.root)))
        return sorted(out)

    def span_in(self, tokens: list[str], rel: str) -> str | None:
        pat = self.pattern(tokens)
        r = subprocess.run(
            ["rg", "-o", "--multiline", "-m", "1", "-e", pat, "--",
             str(self.root / rel)],
            capture_output=True, text=True, timeout=60)
        s = r.stdout.strip()
        return s or None


def _body_lines(source_root: Path, rel: str) -> list[str]:
    return (source_root / rel).read_text(encoding="utf-8", errors="ignore").splitlines()


def _candidate_phrases(lines: list[str], allow_tables: bool, rng: Random,
                       product_dir: str) -> list[list[str]]:
    n = len(lines)
    lo, hi = int(n * 0.3), int(n * 0.9)
    cands = []
    for line in lines[lo:hi]:
        if line.lstrip().startswith("#"):
            continue
        if ("|" in line) != allow_tables and "|" in line:
            continue
        toks = []
        for t in line.split():
            t = t.strip(_TOKEN_STRIP)
            if 4 <= len(t) <= 14 and _HANGUL.search(t) and t not in product_dir:
                toks.append(t)
            else:
                toks.append(None)
        for width in (3, 2):
            for i in range(len(toks) - width + 1):
                seg = toks[i:i + width]
                if all(seg):
                    cands.append(list(seg))
    rng.shuffle(cands)
    return cands[:12]


class QABuilder:
    def __init__(self, cfg: Config, universe: list[DocFacts], ftype: dict):
        guard(json.dumps(sorted(ftype.values())))  # ftype map carries letters only
        self.cfg = cfg
        self.universe = universe
        self.by_rel = {d.rel_path: d for d in universe}
        self.ftype = ftype  # doc_id -> "A"|"B"|"C"|"D"
        self.rng = Random(cfg.seed)
        self.rg = RgGold(cfg.source_root)
        self.used_docs: Counter = Counter()
        self.used_products: Counter = Counter()
        self.seen_q: set = set()
        self.items: list[dict] = []
        self.shortages: list[dict] = []
        # identity families keyed by (clean_name, doc_type)
        self.family = defaultdict(list)
        for d in universe:
            if d.doc_type and len(d.clean_name) >= 4:
                self.family[(d.clean_name, d.doc_type)].append(d)

    # ---------- shared helpers ----------

    def _admit(self, target: DocFacts, query: str) -> bool:
        if self.used_docs[target.doc_id] >= self.cfg.quotas.max_per_doc:
            return False
        if self.used_products[target.product_id] >= self.cfg.quotas.max_per_product:
            return False
        if norm_q(query) in self.seen_q:
            return False
        return True

    def _push(self, target: DocFacts, item: dict) -> None:
        self.used_docs[target.doc_id] += 1
        self.used_products[target.product_id] += 1
        self.seen_q.add(norm_q(item["query"]))
        item["qa_id"] = f"tos-doc-{len(self.items) + 1:04d}"
        self.items.append(item)

    def _record(self, suite, query, target, gold, evidence, constraints) -> dict:
        gold = sorted(gold, key=lambda d: d.rel_path)
        return {
            "schema_version": SCHEMA_VERSION,
            "suite": suite,
            "query": query,
            "target_document_id": target.doc_id,
            "gold_document_ids": [d.doc_id for d in gold],
            "gold_source_paths": [d.rel_path for d in gold],
            "evidence": evidence,
            "identity_constraints": constraints,
            "document_structure_type": self.ftype[target.doc_id],
            "product_id": target.product_id,
            "review_status": "accepted",
        }

    def _typed_targets(self, letter: str, docs: list[DocFacts]) -> list[DocFacts]:
        pool = [d for d in docs if self.ftype.get(d.doc_id) == letter]
        self.rng.shuffle(pool)
        return pool

    # ---------- identity suite ----------

    def _identity_gold(self, kind, name, dtype, year=None):
        fam = self.family.get((name, dtype), [])
        if kind == "type":
            gold = fam
            hi = self.cfg.quotas.gold_max_identity
        elif kind == "year_type":
            gold = [d for d in fam if d.date and d.date[:4] == year]
            hi = self.cfg.quotas.gold_max_identity
        else:  # latest
            dated = [d for d in fam if d.date]
            if len(dated) < 2:
                return None
            latest = max(d.date for d in dated)
            gold = [d for d in dated if d.date == latest]
            hi = self.cfg.quotas.gold_max_latest
        return gold if 1 <= len(gold) <= hi else None

    def build_identity(self) -> None:
        q = self.cfg.quotas
        eligible = [d for d in self.universe
                    if d.doc_type and len(d.clean_name) >= 4]
        for letter, quota in zip("ABC", q.identity_types):
            pool = self._typed_targets(letter, eligible)
            made = 0
            ti = 0
            for target in pool:
                if made >= quota:
                    break
                kind, tmpl = IDENTITY_TEMPLATES[ti % 3]
                ti += 1
                year = target.date[:4] if target.date else None
                if kind in ("year_type", "latest") and not target.date:
                    kind, tmpl = IDENTITY_TEMPLATES[0]
                gold = self._identity_gold(kind, target.clean_name,
                                           target.doc_type, year)
                if not gold or target not in gold:
                    gold = self._identity_gold("type", target.clean_name,
                                               target.doc_type)
                    kind, tmpl = IDENTITY_TEMPLATES[0]
                    if not gold or target not in gold:
                        continue
                query = tmpl.format(name=target.clean_name,
                                    dtype=target.doc_type, year=year)
                if not self._admit(target, query):
                    continue
                constraints = {"kind": kind, "product_name": target.clean_name,
                               "doc_type": target.doc_type}
                if kind == "year_type":
                    constraints["year"] = year
                if kind == "latest":
                    constraints["latest"] = True
                evidence = [{"document_id": d.doc_id, "source_path": d.rel_path,
                             "verbatim_span": d.filename,
                             "span_sha256": sha256_text(d.filename),
                             "evidence_kind": "filename_identity"} for d in gold]
                self._push(target, self._record(
                    "identity", query, target, gold, evidence, constraints))
                made += 1
            if made < quota:
                self.shortages.append({"suite": "identity", "type": letter,
                                       "quota": quota, "made": made})

    # ---------- content + mixed suites ----------

    def _content_gold(self, target: DocFacts, allow_tables: bool):
        lines = _body_lines(self.cfg.source_root, target.rel_path)
        if len(lines) < 30:
            return None
        for tokens in _candidate_phrases(lines, allow_tables, self.rng,
                                         target.product_dir):
            try:
                files = self.rg.files_with(tokens)
            except subprocess.TimeoutExpired:
                continue
            if target.rel_path not in files:
                continue
            if not 1 <= len(files) <= self.cfg.quotas.gold_max_content:
                continue
            gold = [self.by_rel[f] for f in files if f in self.by_rel]
            if len(gold) != len(files):
                continue
            return tokens, gold
        return None

    def _evidence_for(self, tokens, gold):
        evidence = []
        for d in gold:
            span = self.rg.span_in(tokens, d.rel_path)
            if span is None:
                return None
            evidence.append({"document_id": d.doc_id, "source_path": d.rel_path,
                             "verbatim_span": span,
                             "span_sha256": sha256_text(span),
                             "evidence_kind": "body_span"})
        return evidence

    def build_content(self) -> None:
        q = self.cfg.quotas
        for letter, quota in zip("ABC", q.content_types):
            pool = self._typed_targets(letter, self.universe)
            made = 0
            ti = 0
            for target in pool:
                if made >= quota:
                    break
                got = self._content_gold(target, allow_tables=(letter == "C"))
                if not got:
                    continue
                tokens, gold = got
                phrase = " ".join(tokens)
                query = CONTENT_TEMPLATES[ti % 3].format(phrase=phrase)
                ti += 1
                if not self._admit(target, query):
                    continue
                evidence = self._evidence_for(tokens, gold)
                if evidence is None:
                    continue
                self._push(target, self._record(
                    "content", query, target, gold, evidence,
                    {"evidence_tokens": tokens}))
                made += 1
            if made < quota:
                self.shortages.append({"suite": "content", "type": letter,
                                       "quota": quota, "made": made})

    def _mixed_gold(self, kind, tokens, dtype, year=None):
        files = self.rg.files_with(tokens)
        cands = [self.by_rel[f] for f in files
                 if f in self.by_rel and self.by_rel[f].doc_type == dtype]
        if kind == "year":
            cands = [d for d in cands if d.date and d.date[:4] == year]
        elif kind == "latest":
            by_prod = defaultdict(list)
            for d in cands:
                if d.date:
                    by_prod[d.product_id].append(d)
            cands = []
            for docs in by_prod.values():
                latest = max(d.date for d in docs)
                cands.extend(d for d in docs if d.date == latest)
        return cands if 1 <= len(cands) <= self.cfg.quotas.gold_max_mixed else None

    def build_mixed(self) -> None:
        q = self.cfg.quotas
        for letter, quota in zip("ABC", q.mixed_types):
            pool = self._typed_targets(
                letter, [d for d in self.universe if d.doc_type])
            made = 0
            ti = 0
            for target in pool:
                if made >= quota:
                    break
                got = self._content_gold(target, allow_tables=(letter == "C"))
                if not got:
                    continue
                tokens, _ = got
                kind, tmpl = MIXED_TEMPLATES[ti % 3]
                ti += 1
                year = target.date[:4] if target.date else None
                if kind in ("year", "latest") and not target.date:
                    kind, tmpl = MIXED_TEMPLATES[2]
                gold = self._mixed_gold(kind, tokens, target.doc_type, year)
                if not gold or target not in gold:
                    kind, tmpl = MIXED_TEMPLATES[2]
                    gold = self._mixed_gold(kind, tokens, target.doc_type)
                    if not gold or target not in gold:
                        continue
                phrase = " ".join(tokens)
                query = tmpl.format(phrase=phrase, dtype=target.doc_type,
                                    year=year)
                if not self._admit(target, query):
                    continue
                evidence = self._evidence_for(tokens, gold)
                if evidence is None:
                    continue
                constraints = {"kind": kind, "doc_type": target.doc_type,
                               "evidence_tokens": tokens}
                if kind == "year":
                    constraints["year"] = year
                if kind == "latest":
                    constraints["latest"] = True
                self._push(target, self._record(
                    "mixed", query, target, gold, evidence, constraints))
                made += 1
            if made < quota:
                self.shortages.append({"suite": "mixed", "type": letter,
                                       "quota": quota, "made": made})

    # ---------- entry ----------

    def build(self) -> dict:
        self.build_identity()
        self.build_content()
        self.build_mixed()
        counts = Counter(i["suite"] for i in self.items)
        return {"items": self.items, "suite_counts": dict(counts),
                "shortages": self.shortages}


def write_qa(cfg: Config, result: dict) -> dict:
    qa_dir = cfg.work_dir / "qa"
    qa_dir.mkdir(parents=True, exist_ok=True)
    qa_path = qa_dir / "qa_500.jsonl"
    with open(qa_path, "w", encoding="utf-8") as f:
        for item in result["items"]:
            f.write(json.dumps(item, ensure_ascii=False, sort_keys=True) + "\n")
    report = {
        "suite_counts": result["suite_counts"],
        "type_counts": dict(Counter(
            i["document_structure_type"] for i in result["items"])),
        "gold_cardinality": dict(Counter(
            "multi" if len(i["gold_document_ids"]) > 1 else "single"
            for i in result["items"])),
        "shortages": result["shortages"],
        "n": len(result["items"]),
    }
    (qa_dir / "qa_build_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return {"qa_path": str(qa_path), "report": report}

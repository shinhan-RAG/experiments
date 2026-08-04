"""Part 0 collection→document baseline — run configuration.

All paths arrive via CLI/config. No developer-specific absolute paths.
"""
import dataclasses
import hashlib
import json
from pathlib import Path

SCHEMA_VERSION = "shinhan.collection-document-qa.v1"
RUN_SCHEMA = "shinhan.part0-baseline.v1"

ARMS = ("B00", "B10", "B01", "B11")
ARM_LABELS = {
    "B00": "filename_baseline",
    "B10": "index_only",
    "B01": "frontmatter_only",
    "B11": "index_plus_frontmatter",
}
# Preregistered before scoring. B11 is concatenation over one scorer
# (contract: concat_ablation), never two-hop.
B11_PRIMARY_MEANING = "concat_ablation"
# Preregistered composition rule: every arm shares the constant filename base
# so the 2x2 factors (Index text, frontmatter text) are strictly additive and
# B11 - B10 - B01 + B00 is a clean interaction. This differs from PR #14's
# unified_benchmark.py, whose fm arm includes the base while Index+fm drops it.
COMPOSITION_RULE = (
    "repr(doc, arm) = join(' ', [base] + ([index_text] if arm in (B10,B11) else [])"
    " + ([fm_text] if arm in (B01,B11) else [])); "
    "base = product_dir + ' ' + filename (NFC); "
    "index_text = PR#14 unified_benchmark.py:24-32 ident formula; "
    "fm_text = PR#14 eval_content_qa.flatten_fm(doc)"
)


@dataclasses.dataclass(frozen=True)
class Quotas:
    identity: int = 200
    content: int = 200
    mixed: int = 100
    # per-suite target ftype quotas (A, B, C); D is never a scored target
    identity_types: tuple = (100, 80, 20)
    content_types: tuple = (100, 80, 20)
    mixed_types: tuple = (50, 40, 10)
    max_per_doc: int = 1
    max_per_product: int = 2
    gold_max_identity: int = 10
    gold_max_latest: int = 5
    gold_max_content: int = 10
    gold_max_mixed: int = 10


@dataclasses.dataclass(frozen=True)
class Config:
    source_root: Path          # extracted parsed_md directory
    repo_root: Path            # experiments checkout containing tos-skeleton/
    work_dir: Path             # staging + artifacts + qa + freeze
    results_dir: Path          # accepted publication target
    seed: int = 20260804
    expected_docs: int = 9417
    source_zip_sha256: str = ""
    quotas: Quotas = dataclasses.field(default_factory=Quotas)
    # execution mode only — never part of identity_payload(): reuse an existing
    # deterministic qa_500.jsonl (queries/gold), re-derive evidence, revalidate
    resume_qa: bool = False

    def identity_payload(self) -> dict:
        """Input identity for reuse/conflict decisions (not machine-local paths)."""
        return {
            "run_schema": RUN_SCHEMA,
            "source_zip_sha256": self.source_zip_sha256,
            "expected_docs": self.expected_docs,
            "seed": self.seed,
            "quotas": dataclasses.asdict(self.quotas),
            "composition_rule": COMPOSITION_RULE,
            "b11_primary_meaning": B11_PRIMARY_MEANING,
        }

    def identity_sha(self) -> str:
        blob = json.dumps(self.identity_payload(), ensure_ascii=False,
                          sort_keys=True).encode("utf-8")
        return hashlib.sha256(blob).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()

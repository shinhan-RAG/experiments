"""Deterministic source identity facts.

Inputs allowed here: original Markdown bytes and file/directory names only.
Index text, frontmatter text, prior rankings are FORBIDDEN inputs; the module
raises if handed one (see guard()).
"""
import hashlib
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path

DOC_TYPE_KEYWORDS = ("판매약관", "사업방법서", "공시약관", "상품요약서")
_DATE_TOKEN = re.compile(r"(?<!\d)(\d{8}|\d{6})(?!\d)")
_BRACKETS = re.compile(r"\[.*?\]|\(.*?\)")
_WS = re.compile(r"\s+")

FORBIDDEN_INPUT_MARKERS = ("collection_index", "doc_frontmatter", "flatten_fm",
                           "최신판 정본 현행")


def guard(payload: str) -> None:
    """RED-testable input guard: QA generation must never receive arm text."""
    for marker in FORBIDDEN_INPUT_MARKERS:
        if marker in payload:
            raise ValueError(f"forbidden QA input (arm representation): {marker!r}")


def nfc(s: str) -> str:
    return unicodedata.normalize("NFC", s)


def clean_product_name(dir_name: str) -> str:
    value = _BRACKETS.sub(" ", nfc(dir_name))
    value = value.replace("무배당", " ").replace("(무)", " ")
    return _WS.sub(" ", value).strip()


def parse_doc_type(filename: str) -> str | None:
    """Strict: exactly one type keyword literally present in the filename."""
    hits = [k for k in DOC_TYPE_KEYWORDS if k in nfc(filename)]
    return hits[0] if len(hits) == 1 else None


def _valid_ymd(y: int, m: int, d: int) -> bool:
    return 1990 <= y <= 2027 and 1 <= m <= 12 and 1 <= d <= 31


def parse_date(filename: str) -> str | None:
    """Strict: all 6/8-digit tokens must agree on a single valid YYYYMMDD."""
    parses = set()
    for tok in _DATE_TOKEN.findall(Path(nfc(filename)).stem):
        if len(tok) == 8:
            y, m, d = int(tok[:4]), int(tok[4:6]), int(tok[6:8])
            if _valid_ymd(y, m, d):
                parses.add(f"{y:04d}{m:02d}{d:02d}")
        else:
            yy, m, d = int(tok[:2]), int(tok[2:4]), int(tok[4:6])
            y = 2000 + yy if yy <= 27 else 1900 + yy
            if _valid_ymd(y, m, d):
                parses.add(f"{y:04d}{m:02d}{d:02d}")
    return parses.pop() if len(parses) == 1 else None


@dataclass(frozen=True)
class DocFacts:
    rel_path: str          # NFC "product_dir/filename.md"
    product_dir: str
    filename: str
    doc_id: str
    product_id: str
    clean_name: str
    doc_type: str | None   # strict filename parse, None if ambiguous
    date: str | None       # strict YYYYMMDD, None if ambiguous/absent
    size_bytes: int
    sha256: str


def stable_doc_id(rel_path: str) -> str:
    return "doc-" + hashlib.sha256(nfc(rel_path).encode("utf-8")).hexdigest()[:16]


def stable_product_id(product_dir: str) -> str:
    return "prd-" + hashlib.sha256(nfc(product_dir).encode("utf-8")).hexdigest()[:16]


def scan_universe(source_root: Path) -> list[DocFacts]:
    """Canonical universe: every .md under source_root, sorted by NFC rel path."""
    entries = []
    for p in source_root.rglob("*.md"):
        rel = nfc(str(p.relative_to(source_root)))
        entries.append((rel, p))
    entries.sort(key=lambda t: t[0])
    universe = []
    for rel, p in entries:
        data = p.read_bytes()
        product_dir = nfc(p.parent.name)
        filename = nfc(p.name)
        universe.append(DocFacts(
            rel_path=rel,
            product_dir=product_dir,
            filename=filename,
            doc_id=stable_doc_id(rel),
            product_id=stable_product_id(product_dir),
            clean_name=clean_product_name(product_dir),
            doc_type=parse_doc_type(filename),
            date=parse_date(filename),
            size_bytes=len(data),
            sha256=hashlib.sha256(data).hexdigest(),
        ))
    ids = [d.doc_id for d in universe]
    if len(set(ids)) != len(ids):
        raise SystemExit("stable doc_id collision — abort")
    return universe

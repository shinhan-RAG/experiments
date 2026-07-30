"""신한라이프 언더라이팅 파싱본(블록 JSON) → dr-dci BEIR corpus.jsonl.

입력: "0730 임시 검증/parse_results_json/*.json"
      {text, markdown, source_filename, pages:[{page, blocks:[{label,text,markdown,bbox,score}]}]}

LLM·임베딩 호출 없음. 결정론적이며 같은 입력에 항상 같은 출력을 낸다.

정제 규칙(왜 하는지는 docs/SHINHAN_UW_DATA_CLEANING_KO.md 참조)
-----------------------------------------------------------------
C1  blocks[].text 만 쓴다. markdown 은 버린다 (총 38.7M자 vs 608K자, 63배).
C2  거대 표 블록을 행 그룹으로 쪼개고 각 그룹에 헤더행을 반복 삽입한다.
C3  청크를 MAX_CHARS 로 하드 캡한다 (retriever.py:96-98 이 4096자에서 자른다).
C4  전폭 병합셀 행 축약 + 문서 내 중복 블록 제거 + 전역 중복 청크 제거.
C5  파서 환각 캡션·페이지 푸터·수발신 라인 제거. score 는 지우지 않고 보존한다.
C6  header/paragraph_title 블록으로 section_path 를 복원한다.
C7  parent_id 를 넣지 않는다 — 넣으면 평가 단위가 parent 로 바뀐다
    (run_experiment.py:156-165).
C8  질의 샘플링은 build_shinhan_uw_qa.py 담당.
C9  기존 신한 코퍼스를 distractor 로 붙일 때 `shinhan-legacy::` 접두사를 붙인다.
C10 gold ⊆ corpus, gold ∩ distractor = ∅ 를 자기검증한다.

출력
----
  data/raw/shinhan-uw/corpus.jsonl
      {_id, title, text, doc, section_path, page, element_type, parser_label, parser_score}
      (+ distractor 는 {_id, title, text, doc, element_type, distractor_source})
  data/subsets/shinhan-uw/4k.json
  data/metadata/shinhan-uw-parser_4k.json     결정론적 메타데이터 (parser_meta arm 용)
  data/tags/shinhan-uw/approach_p/4k.json     결정론적 @el: 태그 (parser_meta arm 용)
  data/raw/shinhan-uw/{corpus_stats,audit_long_chunks,audit_duplicates,
                       audit_element_type,manifest}.json

  python scripts/build_shinhan_uw_corpus.py
  python scripts/build_shinhan_uw_corpus.py --no-distractor      # gold 축만
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from utils import load_jsonl  # noqa: E402

for _s in (sys.stdout, sys.stderr):
    try:
        _s.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, ValueError):
        pass

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATASET = "shinhan-uw"
DEFAULT_SRC = BASE_DIR.parent / "0730 임시 검증" / "parse_results_json"
DISTRACTOR_CORPUS = DATA_DIR / "raw" / "shinhan" / "corpus.jsonl"
DISTRACTOR_SOURCE = "shinhan-legacy"
SUBSET_SIZE = 4000                    # size_key "4k". 실제 크기는 actual_size.

CHUNK_CHARS = 1400                    # 목표 청크 길이
MAX_CHARS = 3800                      # 하드 캡. retriever 는 "title + text" 를 4096 에서 자른다
MIN_CHARS = 40                        # 이보다 짧은 비표 청크는 버린다(pptx 단문 보존을 위해 낮게)
CELL_SEP = " | "
ROW_SEP = "\n"

# --- C5: 파서 노이즈 ---------------------------------------------------------
# 한국어 보험 공문에 삽입된 영어 실험 캡션. 파서 환각으로 확인됨.
HALLUCINATED = re.compile(r"\[Fig\.\s*\d+\]\s*A schematic of the experimental setup\.?")
# 라인 단위로 통째로 버릴 것들: 페이지 번호·수발신 라우팅·맨숫자 전화번호
DROP_LINE = re.compile(
    r"^(?:\d{1,3}\s*p"                       # "1p"
    r"|-\s*\d{1,3}\s*-"                      # "- 12 -"
    r"|수\s*신\s*처\s*:.*"                    # "수 신 처 : @신한라이프_전체"
    r"|0\d{1,2}-\d{3,4}-?\d{0,4}"            # 맨 전화번호 "02-3455-467"
    r")$"
)
# 의미 문자(한글/영숫자)가 이만큼도 없으면 내용 없는 블록으로 본다
CONTENT_CHAR = re.compile(r"[0-9A-Za-z가-힣]")

# --- 문서 단위 결정론적 메타데이터 (파일명에서 유도) --------------------------
# (파일명에 포함되는 키, doc_type, uw_topic)
DOC_RULES = [
    ("기계약", "limit_table", "cross_contract_limit"),
    ("합산한도", "limit_table", "cross_contract_limit"),
    ("메디컬UW", "medical_guide", "disease_underwriting"),
    ("중대질환", "notice", "disease_underwriting"),
    ("한도 및 가입기준", "limit_table", "issue_limit"),
    ("간편예외질환", "simplified_board", "simplified_exception"),
    ("건강진단기준", "exam_standard", "health_exam"),
    ("인수 기준 안내", "notice", "effective_notice"),
]
DATE_IN_NAME = re.compile(r"(20\d{2}|\d{2})[.\-]\s?(\d{1,2})")

# --- entities 추출 (결정론적) ------------------------------------------------
KCD_CODE = re.compile(r"\b([A-Z]\d{2}(?:\.\d{1,2})?)\b")
COVER_CODE = re.compile(r"\b([A-Z]\d{6})\b")
AMOUNT = re.compile(r"(\d{1,4}\s*(?:억|만원|만|천만))")
MAX_ENTITIES = 5


# ---------------------------------------------------------------------------
# 텍스트 정제
# ---------------------------------------------------------------------------
def clean_cell(cell: str) -> str:
    """셀 하나 정제. <br> 는 파서가 셀 내 줄바꿈에 쓴다."""
    c = cell.replace("<br>", " ").replace(" ", " ")
    c = HALLUCINATED.sub("", c)
    return " ".join(c.split()).strip()


def clean_lines(text: str) -> str:
    """비표 블록 정제: 환각 제거 + 푸터 라인 제거 + 공백 정규화."""
    text = HALLUCINATED.sub("", text.replace(" ", " "))
    out = []
    for raw in text.split("\n"):
        line = " ".join(raw.split()).strip()
        if not line or DROP_LINE.match(line):
            continue
        out.append(line)
    return "\n".join(out)


def has_content(text: str) -> bool:
    return len(CONTENT_CHAR.findall(text)) >= 2


def collapse_rows(raw_text: str) -> list[list[str]]:
    """표 text(탭 구분) → 정제된 행 목록.

    - 전폭 병합셀 행(비어있지 않은 셀이 전부 같은 값)은 셀 1개로 축약한다.
      스프레드시트 export 가 병합 셀 값을 모든 열에 복제하기 때문이며,
      `간편예외질환` 의 116K자 블록 상당 부분이 이 형태다.
    - 인접한 같은 값 셀은 **축약하지 않는다** — '가능|가능' 처럼 정당한 반복이
      있어 열 정렬이 깨진다.
    - 전부 빈 행과 직전 행과 완전히 같은 행은 버린다.
    """
    rows: list[list[str]] = []
    prev: list[str] | None = None
    for raw_row in raw_text.split("\n"):
        cells = [clean_cell(c) for c in raw_row.split("\t")]
        while cells and not cells[-1]:
            cells.pop()
        nonempty = [c for c in cells if c]
        if not nonempty:
            continue
        if len(set(nonempty)) == 1:
            cells = [nonempty[0]]               # 전폭 병합셀
        if cells == prev:
            continue                            # 직전 행과 동일(헤더 반복 등)
        rows.append(cells)
        prev = cells
    return rows


NUMERIC_ONLY = re.compile(r"^[\d.,%\s/~\-()]*$")


def row_has_label(cells: list[str]) -> bool:
    """숫자·기호만인 행은 표의 라벨 행이 될 수 없다.

    스프레드시트가 열 가중치(`0.8 0.5 0.3`)나 빈 스페이서 행을 표 맨 위에
    남기는데, 그대로 두면 헤더로 잡혀 '헤더가 숫자뿐인 표' 청크가 생긴다.
    """
    return any(c and not NUMERIC_ONLY.match(c) for c in cells)


def is_data_row(cells: list[str]) -> bool:
    return sum(1 for c in cells if c) >= 2


def merge_header(rows: list[list[str]]) -> tuple[list[str], list[list[str]]]:
    """다단 헤더를 한 줄로 합친다.

    `건강 전상품` 은 헤더가 2행이다 —
      ['비고','특약구분','특약명','일반심사형','일반심사형', ...]
      ['비고','특약구분','특약명','(당사) 한도반영 기준','(신정원) 반영기준', ...]
    셀 수가 같고 절반 이상이 동일하면 같은 헤더의 연속 행으로 보고
    다른 부분만 ' / ' 로 이어붙인다.
    """
    if not rows:
        return [], []
    header = rows[0]
    idx = 1
    while idx < len(rows) and idx < 3:
        nxt = rows[idx]
        if len(nxt) != len(header) or len(header) < 2:
            break
        same = sum(a == b for a, b in zip(header, nxt))
        if same / len(header) < 0.5:
            break
        header = [a if a == b else f"{a} / {b}".strip(" /")
                  for a, b in zip(header, nxt)]
        idx += 1
    return header, rows[idx:]


def render_row(cells: list[str]) -> str:
    return CELL_SEP.join(cells)


TRUNCATION_MARK = " …[truncated]"


def truncate(text: str) -> tuple[str, bool]:
    """C3 하드 캡. 마커까지 포함해 MAX_CHARS 이하로 만든다.

    마커를 넣는 이유: 에이전트가 read() 로 청크를 볼 때 뒤가 잘렸음을 알 수
    있어야 한다. 조용히 자르면 '표에 그 항목이 없다'고 잘못 결론 낸다.
    """
    if len(text) <= MAX_CHARS:
        return text, False
    keep = MAX_CHARS - len(TRUNCATION_MARK)
    return text[:keep].rstrip() + TRUNCATION_MARK, True


# ---------------------------------------------------------------------------
# 문서 → 청크
# ---------------------------------------------------------------------------
def doc_id_from_name(name: str) -> str:
    return hashlib.sha1(name.encode("utf-8")).hexdigest()[:8]


def heading_level(block: dict) -> int:
    """markdown 의 '#' 개수를 우선 쓰고, 없으면 라벨 기본값."""
    md = (block.get("markdown") or "").lstrip()
    m = re.match(r"(#{1,6})\s", md)
    if m:
        return len(m.group(1))
    return 1 if block["label"] == "header" else 2


def push_section(stack: list[tuple[int, str]], level: int, title: str) -> None:
    while stack and stack[-1][0] >= level:
        stack.pop()
    stack.append((level, title))


def section_path_of(stack: list[tuple[int, str]]) -> str:
    return " > ".join(title for _, title in stack)


class ChunkSink:
    """청크를 모으고 문서 내 중복 블록·전역 중복 청크를 걸러낸다."""

    def __init__(self) -> None:
        self.chunks: list[dict] = []
        self.seen_text: dict[str, str] = {}      # chunk text hash -> 먼저 쓰인 _id
        self.dropped_duplicates: list[dict] = []
        self.truncated: list[dict] = []

    def add(self, *, cid: str, doc: str, title: str, text: str,
            section_path: str, page: int, element_type: str,
            parser_label: str, parser_score) -> None:
        text, was_cut = truncate(text)
        if len(text) < MIN_CHARS and element_type != "table":
            return
        digest = hashlib.md5(text.encode("utf-8")).hexdigest()
        if digest in self.seen_text:
            self.dropped_duplicates.append(
                {"dropped_id": cid, "kept_id": self.seen_text[digest],
                 "doc": doc, "chars": len(text)})
            return
        self.seen_text[digest] = cid
        if was_cut:
            self.truncated.append({"_id": cid, "doc": doc, "chars": len(text)})
        self.chunks.append({
            "_id": cid,
            "title": title,
            "text": text,
            "doc": doc,
            "section_path": section_path,
            "page": page,
            "element_type": element_type,
            "parser_label": parser_label,
            "parser_score": parser_score,
        })


def build_title(doc: str, section_path: str) -> str:
    """retriever 는 'title + text' 를 함께 임베딩한다(retriever.py:88-98).
    문서명만 넣으면 같은 문서의 수백 청크가 동일 title 을 갖게 되어
    title 기여분이 노이즈가 된다 → section_path 를 함께 넣는다."""
    head = doc[:80]
    return f"{head} — {section_path[:90]}" if section_path else head


def process_document(path: Path, sink: ChunkSink, stats: dict) -> None:
    payload = json.loads(path.read_text(encoding="utf-8"))
    src_name = payload.get("source_filename") or path.name
    doc = Path(src_name).stem
    did = doc_id_from_name(src_name)
    section_stack: list[tuple[int, str]] = []
    seen_blocks: set[str] = set()          # C4: 문서 내 중복 블록
    dup_blocks = 0
    label_counter: Counter = Counter()
    n_before = len(sink.chunks)

    for page_index, page in enumerate(payload.get("pages", [])):
        page_no = page.get("page", page_index)
        text_buffer: list[str] = []
        buffer_section = section_path_of(section_stack)
        seq = 0                            # 페이지 내 청크 일련번호

        def flush_text() -> None:
            nonlocal text_buffer, seq, buffer_section
            if not text_buffer:
                return
            body = "\n".join(text_buffer)
            text_buffer = []
            if not has_content(body):
                return
            sink.add(cid=f"{did}_{page_no:03d}_{seq:03d}", doc=doc,
                     title=build_title(doc, buffer_section), text=body,
                     section_path=buffer_section, page=page_no,
                     element_type="text", parser_label="text",
                     parser_score=None)
            seq += 1

        for block_index, block in enumerate(page.get("blocks", [])):
            label = block.get("label", "text")
            raw = block.get("text") or ""
            label_counter[label] += 1

            digest = hashlib.md5(raw.strip().encode("utf-8")).hexdigest()
            if digest in seen_blocks:
                dup_blocks += 1
                continue
            seen_blocks.add(digest)

            if label == "table":
                flush_text()
                seq = process_table(block, sink, did, doc, page_no, seq,
                                    section_stack, page_lead=block_index == 0)
                buffer_section = section_path_of(section_stack)
                continue

            cleaned = clean_lines(raw)
            if not has_content(cleaned):
                continue

            if label in ("header", "paragraph_title") and len(cleaned) <= 120:
                # C6: 제목 블록은 본문이 아니라 section_path 로 소비한다
                flush_text()
                push_section(section_stack, heading_level(block), cleaned)
                buffer_section = section_path_of(section_stack)
                continue

            element = "figure" if label == "infographic" else "text"
            if element == "figure":
                # infographic 은 한 페이지를 통째로 평문화한 것 — 단독 청크로 둔다
                flush_text()
                for piece in pack_text(cleaned):
                    sink.add(cid=f"{did}_{page_no:03d}_{seq:03d}", doc=doc,
                             title=build_title(doc, buffer_section), text=piece,
                             section_path=buffer_section, page=page_no,
                             element_type="figure", parser_label=label,
                             parser_score=block.get("score"))
                    seq += 1
                continue

            # 짧은 블록(pptx 문단 등)은 버퍼에 쌓아 ~CHUNK_CHARS 로 묶는다.
            # 개별로 내보내면 18~50자 청크 19개가 되어 검색 단위로 쓸 수 없다.
            if sum(len(x) + 1 for x in text_buffer) + len(cleaned) > CHUNK_CHARS \
                    and text_buffer:
                flush_text()
            if len(cleaned) > CHUNK_CHARS:
                flush_text()
                for piece in pack_text(cleaned):
                    sink.add(cid=f"{did}_{page_no:03d}_{seq:03d}", doc=doc,
                             title=build_title(doc, buffer_section), text=piece,
                             section_path=buffer_section, page=page_no,
                             element_type="text", parser_label=label,
                             parser_score=block.get("score"))
                    seq += 1
            else:
                text_buffer.append(cleaned)

        flush_text()

    stats["per_doc"].append({
        "doc": doc, "doc_id": did, "source_filename": src_name,
        "pages": len(payload.get("pages", [])),
        "parser_labels": dict(label_counter),
        "duplicate_blocks_dropped": dup_blocks,
        "chunks": len(sink.chunks) - n_before,
    })


def pack_text(text: str) -> list[str]:
    """긴 평문을 줄 경계에서 ~CHUNK_CHARS 조각으로 나눈다."""
    pieces, buf = [], ""
    for line in text.split("\n"):
        if buf and len(buf) + len(line) + 1 > CHUNK_CHARS:
            pieces.append(buf)
            buf = ""
        buf = f"{buf}\n{line}" if buf else line
    if buf:
        pieces.append(buf)
    return pieces


def process_table(block: dict, sink: ChunkSink, did: str, doc: str,
                  page_no: int, seq: int,
                  section_stack: list[tuple[int, str]],
                  page_lead: bool) -> int:
    """C2: 표 블록 → 헤더행을 반복 삽입한 행 그룹 청크들."""
    rows = collapse_rows(block.get("text") or "")

    # 앞쪽 숫자·기호뿐인 스페이서 행 제거 (열 가중치 행 등)
    while rows and not row_has_label(rows[0]):
        rows.pop(0)
    if not rows:
        return seq

    # 앞쪽 전폭 병합셀(1셀) 행 = 표 제목/주석
    notes: list[str] = []
    while rows and len(rows[0]) == 1:
        notes.append(rows.pop(0)[0])

    # 페이지의 **첫 블록**에서만 짧은 첫 주석을 시트 제목으로 승격한다.
    # 표 중간의 전폭 병합 데이터 셀("유의FC : 가입불가" 등)까지 승격하면
    # section_path 가 데이터 값으로 오염되고, 그 값이 title 을 통해
    # 임베딩에 들어가 노이즈가 된다.
    if page_lead and notes and len(notes[0]) <= 80:
        push_section(section_stack, 3, notes.pop(0))
    section = section_path_of(section_stack)
    title = build_title(doc, section)

    for note in notes:
        if has_content(note):
            for piece in pack_text(note):
                sink.add(cid=f"{did}_{page_no:03d}_{seq:03d}", doc=doc,
                         title=title, text=piece, section_path=section,
                         page=page_no, element_type="text",
                         parser_label="table_note",
                         parser_score=block.get("score"))
                seq += 1

    header, data_rows = merge_header(rows)
    data_rows = [r for r in data_rows if row_has_label(r)]
    if len(header) < 2 or not any(is_data_row(r) for r in data_rows):
        # 표라기보다 텍스트 덩어리 — 남은 행을 평문으로 내보낸다
        body = ROW_SEP.join(render_row(r) for r in rows)
        if has_content(body):
            for piece in pack_text(body):
                sink.add(cid=f"{did}_{page_no:03d}_{seq:03d}", doc=doc,
                         title=title, text=piece, section_path=section,
                         page=page_no, element_type="text",
                         parser_label=block.get("label", "table"),
                         parser_score=block.get("score"))
                seq += 1
        return seq

    header_line = render_row(header)
    group: list[str] = []
    size = 0
    for row in data_rows:
        line = render_row(row)
        if group and len(header_line) + size + len(line) + 2 > CHUNK_CHARS:
            sink.add(cid=f"{did}_{page_no:03d}_{seq:03d}", doc=doc, title=title,
                     text=ROW_SEP.join([header_line, *group]),
                     section_path=section, page=page_no, element_type="table",
                     parser_label="table", parser_score=block.get("score"))
            seq += 1
            group, size = [], 0
        group.append(line)
        size += len(line) + 1
    if group:
        sink.add(cid=f"{did}_{page_no:03d}_{seq:03d}", doc=doc, title=title,
                 text=ROW_SEP.join([header_line, *group]),
                 section_path=section, page=page_no, element_type="table",
                 parser_label="table", parser_score=block.get("score"))
        seq += 1
    return seq


# ---------------------------------------------------------------------------
# 결정론적 augmentation 산출물
# ---------------------------------------------------------------------------
def doc_rules(source_filename: str) -> tuple[str, str, int | None]:
    doc_type, uw_topic = "other", "other"
    for key, dt, topic in DOC_RULES:
        if key in source_filename:
            doc_type, uw_topic = dt, topic
            break
    year = None
    m = DATE_IN_NAME.search(source_filename)
    if m:
        raw = m.group(1)
        year = int(raw) if len(raw) == 4 else 2000 + int(raw)
        if not 2015 <= year <= 2027:
            year = None
    return doc_type, uw_topic, year


def extract_entities(text: str) -> list[dict]:
    """결정론적 개체 추출. LLM 없이 정규식만 쓴다."""
    found: list[dict] = []
    seen: set[str] = set()
    for pattern, category in ((COVER_CODE, "rider_code"),
                              (KCD_CODE, "disease_code"),
                              (AMOUNT, "amount_limit")):
        for match in pattern.findall(text):
            name = match if isinstance(match, str) else match[0]
            name = " ".join(name.split())
            if name in seen:
                continue
            seen.add(name)
            found.append({"name": name, "category": category})
            if len(found) >= MAX_ENTITIES:
                return found
    return found


def split_elements_like_build_tags(doc: dict) -> list[dict]:
    """scripts/build_tags.py:47-77 의 split_elements 를 그대로 옮긴 것.

    왜 복제하는가 — approach_p(결정론적)와 approach_a/b/c(LLM)가 **같은 단위로**
    쪼개져야 Part 1 의 parser_meta_only vs tags_only 비교가 성립한다. 청크당
    1개 태그 대 청크당 14개 태그를 비교하면 태그 어휘의 차이가 아니라 granularity
    차이를 재게 된다. build_tags.py 를 import 하면 모듈 로드 시 LLM 설정을 끌고
    오므로 로직만 미러링하고, 바뀌면 여기도 함께 고친다.
    """
    elements: list[dict] = []
    for para in re.split(r"\n{2,}", doc.get("text", "")):
        para = para.strip()
        if not para:
            continue
        lower = para.lower()
        if len(para) < 20 and any(k in lower for k in ("page", "header", "footer", "©")):
            continue
        if len(para) < 50 and elements and any(
                k in lower for k in ("fig", "table", "note", "source", "caption")):
            elements[-1]["text"] += "\n" + para
            continue
        elements.append({"idx": len(elements), "text": para, "doc_id": doc["_id"]})
    return elements


def build_parser_augmentations(chunks: list[dict],
                               doc_meta: dict[str, tuple[str, str, int | None]]
                               ) -> tuple[dict, list[dict]]:
    """파서 라벨만으로 만드는 결정론적 metadata / @el: 태그.

    LLM 증강(build_metadata/build_tags)과 같은 스키마·같은 적용 지점을 쓰되
    내용은 파서가 공짜로 준 사실뿐이다 → '비용 0 증강'의 기준선이 된다.
    distractor 청크도 반드시 포함한다. 빠뜨리면 (a) preflight 가
    'metadata misses subset documents' 로 블록하고(part12_contracts.py:247),
    (b) 태그 필터 grep 에서 distractor 만 조용히 제외돼 유리해진다.
    """
    metadata: dict[str, dict] = {}
    tags: list[dict] = []
    for chunk in chunks:
        cid = chunk["_id"]
        element = chunk.get("element_type", "text")
        if chunk.get("distractor_source"):
            doc_type, uw_topic, year = "other", "other", None
        else:
            doc_type, uw_topic, year = doc_meta[chunk["doc"]]
        metadata[cid] = {
            "doc_type": doc_type,
            "element_type": element,
            "uw_topic": uw_topic,
            "effective_year": year,
            "entities": extract_entities(chunk["text"]),
        }
        # 청크의 모든 element 에 그 청크의 파서 element_type 을 부여한다.
        # 분할 단위를 build_tags.py 와 똑같이 맞춰야 approach A/B/C 와 비교 가능하다.
        for elem in split_elements_like_build_tags(chunk):
            elem["tag"] = f"@el:{element}"
            tags.append(elem)
    return metadata, tags


# ---------------------------------------------------------------------------
# distractor
# ---------------------------------------------------------------------------
def namespaced(doc: dict) -> tuple[dict, bool]:
    """distractor _id 에 출처 접두사를 붙여 ID 충돌을 원천 차단한다.

    두 코퍼스의 _id 형식이 똑같이 `<8hex>_<sss>_<cc>` 라서 충돌 시 corpus 가
    조용히 덮어써진다. parent_id 는 만들지 않는다 — 생기면 평가 단위가
    parent 로 바뀐다(run_experiment.py:156-165).

    C3 을 distractor 에도 적용한다. 기존 신한 코퍼스는 청크 11.6%(326건, 최대
    161,443자)가 색인 시 4096자에서 조용히 잘렸다. 그대로 두면 노이즈의 96%가
    사라져 검색이 실제보다 쉬워지고 recall 이 부풀려진다. 여기서 명시적으로
    잘라 절단량을 감사에 남긴다.
    """
    text, was_cut = truncate(doc.get("text", ""))
    return {
        "_id": f"{DISTRACTOR_SOURCE}::{doc['_id']}",
        "title": doc.get("title", ""),
        "text": text,
        "doc": doc.get("doc", ""),
        "element_type": doc.get("element_type", "text"),
        "distractor_source": DISTRACTOR_SOURCE,
    }, was_cut


# ---------------------------------------------------------------------------
def build(src_dir: Path, with_distractor: bool) -> int:
    files = sorted(src_dir.glob("*.json"))
    if not files:
        raise FileNotFoundError(f"파싱본 JSON 이 없다: {src_dir}")

    print(f"=== {DATASET} corpus 빌드 ===")
    print(f"  입력: {src_dir}  ({len(files)}개 문서)")

    sink = ChunkSink()
    stats: dict = {"per_doc": []}
    doc_meta: dict[str, tuple[str, str, int | None]] = {}
    for path in files:
        payload_name = json.loads(path.read_text(encoding="utf-8")).get(
            "source_filename") or path.name
        doc_meta[Path(payload_name).stem] = doc_rules(payload_name)
        process_document(path, sink, stats)

    gold_chunks = sink.chunks
    gold_ids = [c["_id"] for c in gold_chunks]
    if len(gold_ids) != len(set(gold_ids)):
        raise ValueError("gold 청크에 중복 _id 가 있다")
    if any("parent_id" in c for c in gold_chunks):
        raise ValueError("corpus 에 parent_id 가 있다 — 청크 단위 평가 전제가 깨진다")
    over = [c["_id"] for c in gold_chunks if len(c["text"]) > MAX_CHARS]
    if over:
        raise ValueError(f"MAX_CHARS 초과 청크 {len(over)}건: {over[:3]}")

    print(f"  gold 청크: {len(gold_chunks):,}  "
          f"(중복 청크 제거 {len(sink.dropped_duplicates):,}, "
          f"하드캡 절단 {len(sink.truncated)})")
    for row in stats["per_doc"]:
        print(f"    {row['doc'][:44]:46s} p={row['pages']:>2} "
              f"chunks={row['chunks']:>4} dup_blocks={row['duplicate_blocks_dropped']:>3}")

    # ---- distractor -------------------------------------------------------
    distractors: list[dict] = []
    distractor_truncated = 0
    if with_distractor:
        if not DISTRACTOR_CORPUS.exists():
            raise FileNotFoundError(f"distractor 코퍼스 없음: {DISTRACTOR_CORPUS}")
        legacy = load_jsonl(DISTRACTOR_CORPUS)
        if any(d.get("parent_id") for d in legacy):
            raise ValueError("distractor 에 parent_id 가 있다")
        for row in legacy:
            doc, was_cut = namespaced(row)
            distractors.append(doc)
            distractor_truncated += was_cut
        print(f"  distractor: {len(distractors):,}청크 "
              f"({DISTRACTOR_SOURCE}, 노이즈 전용, 하드캡 절단 {distractor_truncated})")

    corpus = gold_chunks + distractors
    all_ids = [c["_id"] for c in corpus]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("혼합 corpus 에 중복 _id 가 있다 — namespaced() 확인")

    # ---- 산출물 -----------------------------------------------------------
    out_dir = DATA_DIR / "raw" / DATASET
    subset_dir = DATA_DIR / "subsets" / DATASET
    out_dir.mkdir(parents=True, exist_ok=True)
    subset_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "corpus.jsonl", "w", encoding="utf-8") as f:
        for doc in corpus:
            f.write(json.dumps(doc, ensure_ascii=False) + "\n")

    size_key = f"{SUBSET_SIZE // 1000}k"
    with open(subset_dir / f"{size_key}.json", "w", encoding="utf-8") as f:
        json.dump({
            "subset_size": SUBSET_SIZE,
            "actual_size": len(all_ids),
            "gold_chunks": len(gold_ids),
            "distractor_chunks": len(distractors),
            # full_corpus 가 없으면 audit_subsets 가 len(ids) != size 로 블록한다
            # (part12_contracts.py:166).
            "full_corpus": True,
            "note": ("신규 언더라이팅 7문서(gold) + 기존 신한 코퍼스(distractor 전용). "
                     "샘플링 없이 전체를 쓰며 실제 크기는 actual_size."),
            "doc_ids": all_ids,
        }, f, ensure_ascii=False, indent=2)

    metadata, tags = build_parser_augmentations(corpus, doc_meta)
    (DATA_DIR / "metadata").mkdir(parents=True, exist_ok=True)
    with open(DATA_DIR / "metadata" / f"{DATASET}-parser_{size_key}.json",
              "w", encoding="utf-8") as f:
        json.dump(metadata, f, ensure_ascii=False)
    tag_dir = DATA_DIR / "tags" / DATASET / "approach_p"
    tag_dir.mkdir(parents=True, exist_ok=True)
    with open(tag_dir / f"{size_key}.json", "w", encoding="utf-8") as f:
        json.dump(tags, f, ensure_ascii=False)

    # ---- 감사 -------------------------------------------------------------
    element_counter = Counter(c["element_type"] for c in gold_chunks)
    parser_counter = Counter(c["parser_label"] for c in gold_chunks)
    lengths = sorted(len(c["text"]) for c in gold_chunks)
    by_doc = Counter(c["doc"] for c in gold_chunks)
    stats.update({
        "docs": len(files),
        "gold_chunks": len(gold_chunks),
        "distractor_chunks": len(distractors),
        "total_chunks": len(corpus),
        "by_element_type": dict(element_counter),
        "by_parser_label": dict(parser_counter),
        "chunk_chars": {
            "total": sum(lengths),
            "min": lengths[0], "max": lengths[-1],
            "p50": lengths[len(lengths) // 2],
            "p95": lengths[int(len(lengths) * 0.95)],
        },
        "gold_chunks_by_doc": dict(by_doc),
    })
    _dump(out_dir / "corpus_stats.json", stats)
    _dump(out_dir / "audit_duplicates.json", {
        "note": ("전역 중복 청크 제거 내역. 스프레드시트 export 가 같은 표를 여러 "
                 "페이지·열에 복제한다. 중복을 남기면 중복본이 gold 앞에 뜨는데 "
                 "정답 id 는 하나뿐이라 recall 이 과소평가된다."),
        "dropped_chunk_count": len(sink.dropped_duplicates),
        "dropped": sink.dropped_duplicates,
    })
    _dump(out_dir / "audit_long_chunks.json", {
        "note": (f"MAX_CHARS={MAX_CHARS} 하드캡. retriever.py:96-98 이 "
                 "'title + text' 를 4096자에서 자르므로 그 아래로 유지한다."),
        "max_chars": MAX_CHARS,
        "gold_truncated_count": len(sink.truncated),
        "gold_truncated": sink.truncated,
        "distractor_truncated_count": distractor_truncated,
        "distractor_note": ("기존 신한 코퍼스는 색인 시 326건(11.6%)이 4096자에서 "
                            "조용히 잘렸다. 여기서 명시적으로 잘라 절단량을 남긴다 — "
                            "노이즈가 조용히 약해지면 recall 이 부풀려진다."),
        "length_percentiles": stats["chunk_chars"],
    })
    _dump(out_dir / "audit_element_type.json", {
        "note": ("element_type 은 파서 label 에서 결정론적으로 유도한다. "
                 "label=table 이어도 병합셀 축약 후 헤더 2셀·데이터 1행 이상을 "
                 "만족하지 못하면 text 로 내린다 — 전폭 병합 주석 셀이 표로 "
                 "잡히는 것을 막는다. label=infographic → figure."),
        "by_element_type": dict(element_counter),
        "by_parser_label": dict(parser_counter),
        "samples": [
            {"_id": c["_id"], "element_type": c["element_type"],
             "parser_label": c["parser_label"], "chars": len(c["text"]),
             "section_path": c["section_path"], "head": c["text"][:160]}
            for c in gold_chunks[:: max(1, len(gold_chunks) // 40)]
        ],
    })
    _dump(out_dir / "manifest.json", {
        "dataset": DATASET,
        "generated_by": "scripts/build_shinhan_uw_corpus.py",
        "source": str(src_dir),
        "source_documents": [row["source_filename"] for row in stats["per_doc"]],
        "evaluation_unit": "chunk (parent_id 없음)",
        "gold_axis": {"chunks": len(gold_chunks), "documents": len(files)},
        "distractor": {
            "source": DISTRACTOR_SOURCE,
            "path": str(DISTRACTOR_CORPUS),
            "chunks": len(distractors),
            "role": "노이즈 전용 — 질의·gold 에 절대 포함되지 않는다",
            "id_prefix": f"{DISTRACTOR_SOURCE}::",
            "truncated_to_max_chars": distractor_truncated,
        },
        "cleaning_rules": {
            "C1": "blocks[].text 만 사용, markdown 폐기 (38.7M자 vs 608K자)",
            "C2": f"표를 행 그룹(~{CHUNK_CHARS}자)으로 분할하고 헤더행을 매 그룹에 반복",
            "C3": f"MAX_CHARS={MAX_CHARS} 하드 캡 (retriever 4096자 절단 회피)",
            "C4": "전폭 병합셀 행 축약 + 문서 내 중복 블록 제거 + 전역 중복 청크 제거",
            "C5": "파서 환각 캡션·페이지 푸터·수발신 라인 제거, parser_score 는 보존",
            "C6": "header/paragraph_title → section_path 복원 (title 에 반영)",
            "C7": "parent_id 미생성 — 청크 단위 평가 유지",
            "C9": f"distractor _id 에 '{DISTRACTOR_SOURCE}::' 접두사",
        },
        "limitations": [
            "gold 축이 7문서·약 800청크뿐이다. distractor 를 섞어도 후보 풀이 "
            "3.6K 수준이라 top-20 이 코퍼스의 0.6%다 — 천장 효과를 먼저 의심할 것.",
            "표 청크가 다수이며 두 스프레드시트가 gold 청크의 대부분을 차지한다. "
            "질의는 문서별 균등 + element_type 층화로 뽑아 편중을 상쇄하지만, "
            "코퍼스 자체의 편중은 남는다.",
            "distractor(기존 신한 28문서)는 약관·상담자료 중심이고 gold 는 "
            "언더라이팅 인수기준이다. 도메인이 완전히 같지 않으므로 recall 이 "
            "잘 떨어지지 않으면 'distractor 가 쉬웠다'를 먼저 의심할 것.",
            "distractor 원본의 알려진 한계가 그대로 이어진다 — 중복 청크 21.8%, "
            "단일 문서가 49.8% 차지.",
            "parser_score 는 PDF 에만 있다(xlsx/pptx 는 null). score 기반 필터링은 "
            "하지 않았고 값만 보존했다.",
            "gold 질의·정답은 gpt-4o-mini 단일 생성이며 사람 검수가 없다.",
            "법률 Part 결과 및 기존 신한 결과와 절대 점수를 비교하지 말 것 "
            "(각각 parent 단위 / 다른 문서·파서·청킹).",
        ],
    })

    # ---- 자기 검증 ---------------------------------------------------------
    print("\n  === 자기 검증 ===")
    subset_ids = json.load(open(subset_dir / f"{size_key}.json",
                                encoding="utf-8"))["doc_ids"]
    assert len(subset_ids) == len(corpus), "subset 크기가 corpus 와 다르다"
    assert set(subset_ids) == set(all_ids), "subset 이 corpus 와 다르다"
    assert set(metadata) == set(all_ids), "결정론적 metadata 가 일부 청크를 빠뜨렸다"
    assert {t["doc_id"] for t in tags} == set(all_ids), "결정론적 태그 누락"
    long_chunks = [c["_id"] for c in corpus if len(c["text"]) > MAX_CHARS]
    assert not long_chunks, f"MAX_CHARS 초과가 남았다: {long_chunks[:3]}"
    assert all(not c["_id"].startswith(f"{DISTRACTOR_SOURCE}::")
               for c in gold_chunks), "gold 에 distractor 접두사가 섞였다"
    print(f"    [OK] corpus {len(corpus):,}청크 = gold {len(gold_chunks):,} "
          f"+ distractor {len(distractors):,}")
    print(f"    [OK] _id 중복 0, parent_id 부재, {MAX_CHARS}자 초과 0")
    print(f"    [OK] 결정론적 metadata·태그가 전체 청크를 덮는다")
    print(f"    element_type: {dict(element_counter)}")
    print(f"    청크 길이 p50={stats['chunk_chars']['p50']} "
          f"p95={stats['chunk_chars']['p95']} max={stats['chunk_chars']['max']}")
    print(f"\n  corpus  : {out_dir / 'corpus.jsonl'}")
    print(f"  subset  : {subset_dir / f'{size_key}.json'}")
    print(f"  manifest: {out_dir / 'manifest.json'}")
    return 0


def _dump(path: Path, payload) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=str(DEFAULT_SRC),
                    help="파싱본 JSON 디렉터리")
    ap.add_argument("--no-distractor", action="store_true",
                    help="기존 신한 코퍼스를 섞지 않고 gold 축만 만든다")
    args = ap.parse_args()
    return build(Path(args.src), not args.no_distractor)


if __name__ == "__main__":
    sys.exit(main())

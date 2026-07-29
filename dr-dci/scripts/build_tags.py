"""
5단계: Semantic Tagging (@el:) 생성
- 문서를 element 단위로 분할 후 @el: 태그 부여
- 방식 A (의미역), B (질문유형), C (타입만) 3가지 생성
- 병렬 처리 (async, 32 concurrent)
"""

import json
import re
from pathlib import Path
from utils import run_batch_llm, parse_llm_content, load_corpus_subset, parse_sizes

DATA_DIR = Path(__file__).parent.parent / "data"
OUTPUT_DIR = DATA_DIR / "tags"

# 하위 태그 정의
SUBTAGS_A = [
    "definition", "condition", "procedure", "example",
    "exception", "comparison", "summary", "evidence", "criteria"
]

SUBTAGS_B = [
    "what", "how", "when", "who",
    "how_much", "why", "which", "if"
]

ELEMENT_TYPES = ["paragraph", "table", "list", "figure"]

SYSTEM_PROMPT_A = f"""You are a document structure analyst. For each text element, determine:
1. Element type: one of {ELEMENT_TYPES}
2. Semantic role (sub-tag): one of {SUBTAGS_A}

Respond in JSON: {{"type": "...", "sub": "..."}}"""

SYSTEM_PROMPT_B = f"""You are a document structure analyst. For each text element, determine:
1. Element type: one of {ELEMENT_TYPES}
2. Best-answered question type (sub-tag): one of {SUBTAGS_B}

Respond in JSON: {{"type": "...", "sub": "..."}}"""

SYSTEM_PROMPT_C = f"""You are a document structure analyst. Determine the element type.
Options: {ELEMENT_TYPES}

Respond in JSON: {{"type": "..."}}"""


def split_elements(doc: dict) -> list:
    """문서를 element 단위로 분할"""
    text = doc.get("text", "")

    elements = []
    paragraphs = re.split(r'\n{2,}', text)

    for para in paragraphs:
        para = para.strip()
        if not para:
            continue

        lower = para.lower()

        # header/footer/page-number → 제거
        if len(para) < 20 and any(kw in lower for kw in ["page", "header", "footer", "©"]):
            continue

        # 매우 짧은 텍스트 → 이전 요소에 편입
        if len(para) < 50 and elements:
            if any(kw in lower for kw in ["fig", "table", "note", "source", "caption"]):
                elements[-1]["text"] += "\n" + para
                continue

        elements.append({
            "idx": len(elements),
            "text": para,
            "doc_id": doc["_id"],
        })

    return elements


def tag_elements_batch(elements: list, approach: str) -> list:
    """elements에 @el: 태그 부여 (배치 병렬)"""
    if approach == "A":
        system = SYSTEM_PROMPT_A
    elif approach == "B":
        system = SYSTEM_PROMPT_B
    else:
        system = SYSTEM_PROMPT_C

    # 프롬프트 준비
    prompts = [
        {"id": i, "text": f"Text element:\n{elem['text'][:200]}"}
        for i, elem in enumerate(elements)
    ]

    # 배치 처리
    BATCH_SIZE = 500
    all_raw = []
    for batch_start in range(0, len(prompts), BATCH_SIZE):
        batch = prompts[batch_start:batch_start + BATCH_SIZE]
        raw_results = run_batch_llm(batch, system, max_tokens=50)
        all_raw.extend(raw_results)

    # 결과 파싱
    tagged = []
    for elem, raw in zip(elements, all_raw):
        if raw is None:
            result = {"type": "paragraph", "sub": "summary" if approach == "A" else "what"}
        else:
            default = {"type": "paragraph", "sub": "summary" if approach == "A" else "what"}
            try:
                parsed = json.loads(parse_llm_content(raw))
                # 일부 LLM(gpt-4o-mini)은 객체 대신 배열/스칼라를 반환 → dict로 정규화
                if isinstance(parsed, list):
                    parsed = next((x for x in parsed if isinstance(x, dict)), {})
                result = parsed if isinstance(parsed, dict) else default
            except (json.JSONDecodeError, TypeError):
                result = default

        el_type = result.get("type", "paragraph")
        if el_type not in ELEMENT_TYPES:
            el_type = "paragraph"

        if approach == "C":
            tag = f"@el:{el_type}"
        else:
            sub = result.get("sub", SUBTAGS_A[0] if approach == "A" else SUBTAGS_B[0])
            valid_subs = SUBTAGS_A if approach == "A" else SUBTAGS_B
            if sub not in valid_subs:
                sub = valid_subs[0]
            tag = f"@el:{el_type}/{sub}"

        elem["tag"] = tag
        tagged.append(elem)

    return tagged


def build_tags(dataset: str = "trec-covid", subset_size: int = 10_000,
               approaches: tuple = ("A", "B", "C")):
    """서브셋 문서에 대해 요청한 방식(들)의 @el: 태그 생성.

    Part1은 approach A만 있으면 되므로 approaches=("A",)로 태깅량을 1/3로 줄일 수
    있다(B/C는 Part3에서 추가 빌드)."""
    print(f"=== Building @el: Tags for {dataset} ({subset_size // 1000}K) "
          f"approaches={list(approaches)} ===")

    size_key = f"{subset_size // 1000}k"
    # 요청한 approach가 모두 이미 있으면 스킵
    if all((OUTPUT_DIR / dataset / f"approach_{a.lower()}" / f"{size_key}.json").exists()
           for a in approaches):
        print(f"  Already exists for {list(approaches)}, skipping.")
        return

    # 서브셋 corpus 로드 (aihub=parent-id 서브셋, BEIR=doc-id 서브셋)
    corpus_subset = load_corpus_subset(DATA_DIR, dataset, subset_size)
    print(f"  Docs to process: {len(corpus_subset)}")

    # 전체 elements 생성
    all_elements = []
    for doc in corpus_subset:
        elements = split_elements(doc)
        all_elements.extend(elements)
    print(f"  Total elements: {len(all_elements)}")

    for approach in approaches:
        print(f"\n  --- Approach {approach} ---")

        # 배치 태깅
        tagged = tag_elements_batch(all_elements, approach)

        # 저장
        out_dir = OUTPUT_DIR / dataset / f"approach_{approach.lower()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subset_size // 1000}k.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(tagged, f, ensure_ascii=False)

        # 태그 분포
        tag_counts = {}
        for elem in tagged:
            tag = elem["tag"]
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        print(f"    Total elements: {len(tagged)}")
        print(f"    Unique tags: {len(tag_counts)}")
        print(f"    Top 5 tags:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"      {tag}: {count}")
        print(f"    Saved to: {out_path}")


if __name__ == "__main__":
    import sys
    argv = sys.argv[1:]
    args = [a for a in argv if not a.startswith("-")]
    dataset = args[0] if args else "trec-covid"
    sizes = parse_sizes(argv)
    # --approaches A  또는  --approaches A,B,C  (기본 A,B,C)
    approaches = ("A", "B", "C")
    if "--approaches" in argv:
        raw = argv[argv.index("--approaches") + 1]
        approaches = tuple(x.strip().upper() for x in raw.split(",") if x.strip())
    for size in sizes:
        build_tags(dataset, size, approaches=approaches)

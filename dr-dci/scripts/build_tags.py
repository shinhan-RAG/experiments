"""
5단계: Semantic Tagging (@el:) 생성
- 문서를 element 단위로 분할 후 @el: 태그 부여
- 방식 A (의미역), B (질문유형), C (타입만) 3가지 생성
- 소형 요소: caption/footnote → 직전/직후 상위 요소에 편입, header/footer/page-number → 제거

대상 요소 4종:
- @el:paragraph/{sub}
- @el:table/{sub}
- @el:list/{sub}
- @el:figure/{sub}
"""

import json
import time
import re
import requests
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
SUBSET_DIR = DATA_DIR / "subsets"
OUTPUT_DIR = DATA_DIR / "tags"

VLLM_URL = "http://localhost:8100/v1/chat/completions"
MODEL_NAME = "Qwen/Qwen3-8B"

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


def load_jsonl(path: Path) -> list:
    with open(path) as f:
        return [json.loads(line) for line in f]


def call_llm(prompt: str, system: str, max_retries: int = 3) -> dict:
    payload = {
        "model": MODEL_NAME,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": prompt},
        ],
        "temperature": 0,
        "max_tokens": 50,
    }

    for attempt in range(max_retries):
        try:
            resp = requests.post(VLLM_URL, json=payload, timeout=30)
            resp.raise_for_status()
            content = resp.json()["choices"][0]["message"]["content"].strip()
            if "<think>" in content:
                content = re.sub(r"<think>.*?</think>\s*", "", content, flags=re.DOTALL).strip()
            if content.startswith("```"):
                content = content.split("```")[1]
                if content.startswith("json"):
                    content = content[4:]
            return json.loads(content)
        except (json.JSONDecodeError, Exception) as e:
            if attempt < max_retries - 1:
                time.sleep(1)
            else:
                return {"type": "paragraph", "sub": "summary"}


def split_elements(doc: dict) -> list:
    """문서를 element 단위로 분할 (간단한 paragraph splitting)"""
    text = doc.get("text", "")
    title = doc.get("title", "")

    # 간단한 분할: 빈 줄 또는 2+ newline 기준
    elements = []
    paragraphs = re.split(r'\n{2,}', text)

    for i, para in enumerate(paragraphs):
        para = para.strip()
        if not para:
            continue

        # 소형 요소 감지 및 편입
        lower = para.lower()

        # header/footer/page-number → 제거
        if len(para) < 20 and any(kw in lower for kw in ["page", "header", "footer", "©"]):
            continue

        # 매우 짧은 텍스트 → 이전 요소에 편입 (caption/footnote 처리)
        if len(para) < 50 and elements:
            # footnote/caption → 직전 상위 요소에 편입
            if any(kw in lower for kw in ["fig", "table", "note", "source", "caption"]):
                elements[-1]["text"] += "\n" + para
                continue

        elements.append({
            "idx": len(elements),
            "text": para,
            "doc_id": doc["_id"],
        })

    return elements


def tag_elements(elements: list, approach: str) -> list:
    """elements에 @el: 태그 부여"""
    if approach == "A":
        system = SYSTEM_PROMPT_A
    elif approach == "B":
        system = SYSTEM_PROMPT_B
    else:
        system = SYSTEM_PROMPT_C

    tagged = []
    for elem in elements:
        text_preview = elem["text"][:200]
        result = call_llm(f"Text element:\n{text_preview}", system)

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
        time.sleep(0.05)

    return tagged


def build_tags(dataset: str = "trec-covid", subset_size: int = 10_000):
    """서브셋 문서에 대해 3가지 방식의 @el: 태그 생성"""
    print(f"=== Building @el: Tags for {dataset} ({subset_size // 1000}K) ===")

    # 서브셋 doc IDs
    subset_path = SUBSET_DIR / dataset / f"{subset_size // 1000}k.json"
    with open(subset_path) as f:
        subset_info = json.load(f)
    doc_ids = set(subset_info["doc_ids"])

    # corpus 로드 (서브셋만)
    corpus = load_jsonl(RAW_DIR / dataset / "corpus.jsonl")
    corpus_subset = [doc for doc in corpus if doc["_id"] in doc_ids]
    print(f"  Docs to process: {len(corpus_subset)}")

    for approach in ["A", "B", "C"]:
        print(f"\n  --- Approach {approach} ---")
        all_tagged = []

        for i, doc in enumerate(corpus_subset):
            elements = split_elements(doc)
            tagged = tag_elements(elements, approach)
            all_tagged.extend(tagged)

            if (i + 1) % 100 == 0:
                print(f"    Progress: {i + 1}/{len(corpus_subset)} docs, {len(all_tagged)} elements")

        # 저장
        out_dir = OUTPUT_DIR / dataset / f"approach_{approach.lower()}"
        out_dir.mkdir(parents=True, exist_ok=True)
        out_path = out_dir / f"{subset_size // 1000}k.json"

        with open(out_path, "w") as f:
            json.dump(all_tagged, f, ensure_ascii=False)

        # 태그 분포
        tag_counts = {}
        for elem in all_tagged:
            tag = elem["tag"]
            tag_counts[tag] = tag_counts.get(tag, 0) + 1

        print(f"    Total elements: {len(all_tagged)}")
        print(f"    Unique tags: {len(tag_counts)}")
        print(f"    Top 5 tags:")
        for tag, count in sorted(tag_counts.items(), key=lambda x: -x[1])[:5]:
            print(f"      {tag}: {count}")

        print(f"    Saved to: {out_path}")


if __name__ == "__main__":
    import sys
    sizes = [20_000, 50_000, 110_000] if "--all" in sys.argv else [20_000]
    for size in sizes:
        build_tags("trec-covid", size)

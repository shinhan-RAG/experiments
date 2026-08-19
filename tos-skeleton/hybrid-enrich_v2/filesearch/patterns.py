"""태그 내용 슬롯 추출용 정규식·필터 (subject/role/qualifier). 원본: hybrid-enrich/build_element_fields_v4.py"""
import re

QUOTED_RE = re.compile(r'["“「『<](.{2,60}?)["”」』>]')
SUBJECT_RE = re.compile(
    r"[가-힣A-Za-z0-9·()\- ]{2,60}?(?:보험금|급여금|지원비|치료비|진단비|수술비|"
    r"입원비|검사비|환급금|보험료|보험나이|가입연령|납입면제|질병|질환|신생물|"
    r"장해|재해|암|수술|치료|검사|입원|통원)"
)
DEFINITION_RE = re.compile(r"([가-힣A-Za-z0-9·()\- ]{2,50}?)(?:이라 함은|라 함은|이란)")
CONDITION_RE = re.compile(r"([^\n.]{2,100}(?:경우|때에는|한하여|이상|미만|초과|이내|이후|이전)[^\n.]{0,80})")
VALUE_RE = re.compile(
    r"(?:보험가입금액|가입금액)의?\s*\d+(?:\.\d+)?%|\d+(?:\.\d+)?%|"
    r"최초\s*1회(?:한)?|\d+회(?:한)?|\d+일(?:\s*한도|분)?|\d+개월|\d+년|"
    r"\d+(?:,\d{3})*(?:만)?원|\d+세|[A-Z]\d{2}(?:\.\d+)?(?:~[A-Z]?\d{2})?"
)
GENERIC = {
    "보험금", "급여금", "질병", "질환", "치료", "수술", "검사", "입원", "통원",
    "특약", "주계약", "피보험자", "보험수익자", "계약자", "회사", "해당", "경우",
    "보험금 지급사유", "보험금 지급에 관한 세부규정", "용어의 정의",
}
BAD_KEY_RE = re.compile(
    r"제\d+(?:-\d+)?\s*(?:조|관)|부표\s?\d|별표\s?\d|한국표준질병|보험금 지급|"
    r"주계약 약관|세부규정|간편심사형|직접적인 치료|지급절차|제\s?\d+항|주\d+\)|"
    r"계약자는|경우|동안|에도 불구하고|에 따라|^이라 함은|^은\s|^의\s"
)
BAD_SUBJECT_FRAGMENT_RE = re.compile(
    r"^(?:및|의|을|를|은|는|이|가|으로|로|에서|에|등)\s|(?:받는 방법의 변경|지급하지 않는 사유|"
    r"및 진단확정|에 관한 세부규정|보험금 등의 지급절차)$|^\d+(?:\.\d+)?$|"
    r"^(?:/?tr|/?td|/?th|일반형|제\d+편\s+일반사항|\(간편\)암)$"
)
DOMAIN_SUBJECT_RE = re.compile(
    r"보험금|급여금|지원비|치료비|진단비|수술비|입원비|검사비|환급금|보험료|"
    r"보험나이|가입연령|납입면제|계약|질병|질환|신생물|장해|재해|암|수술|"
    r"치료|검사|입원|통원|분류|코드|대장점막|방광|피부|갑상선|뇌혈관|심장|"
    r"화상|부식|중환자실|일상생활|골절|치매|파킨슨|통풍|대상포진|요양"
)
ROLE_KO = {
    "exclusion_exception": "면책 제외 예외 부지급",
    "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급사유 지급조건",
    "payment_amount": "보험금 지급금액 지급률 산정",
    "limit_frequency": "지급한도 횟수 일수",
    "timing_period": "보장개시 책임개시 대기기간 감액기간 보험기간",
    "definition": "용어 정의 의미",
    "criteria_rule": "진단확정 판정기준 적용기준",
    "contract_lifecycle": "갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 구비서류 절차",
    "code_reference": "질병분류코드 수가코드 부표 분류표",
}
ROLE_RULES = [
    (r"지급하지 않|보상하지 않|면책|제외", "exclusion_exception"),
    (r"납입.?면제", "premium_waiver"),
    (r"지급사유|지급 조건", "payment_trigger"),
    (r"지급금액|지급률|산정|계산", "payment_amount"),
    (r"한도|횟수|최초\s*1회", "limit_frequency"),
    (r"보장개시|책임개시|감액|대기기간|보험기간|보험나이|가입연령", "timing_period"),
    (r"정의|이라 함은|라 함은", "definition"),
    (r"진단확정|판정기준|적용 기준|등록 및 결정", "criteria_rule"),
    (r"갱신|해지|소멸|무효|환급", "contract_lifecycle"),
    (r"청구|구비서류", "claim_procedure"),
    (r"분류표|분류코드|질병코드|수가코드", "code_reference"),
]


def unique(values, limit=30):
    seen, output = set(), []
    for value in values:
        value = re.sub(r"\s+", " ", str(value)).strip(" |,.;:[]#")
        if (not value or value in GENERIC or len(value) < 2 or len(value) > 100
                or value in seen or BAD_KEY_RE.search(value)):
            continue
        seen.add(value)
        output.append(value)
        if len(output) >= limit:
            break
    return output


def clean_subjects(values, limit=30, require_domain=False):
    output = []
    for value in values:
        value = re.sub(r"^[\s\"'“”「」『』]+|[\s\"'“”「」『』]+$", "", str(value))
        value = re.sub(r"^(?:및|또는)\s+", "", value)
        value = re.sub(r"\s+", " ", value).strip(" |,.;:[]#")
        if (not value or len(value) < 2 or len(value) > 60 or BAD_SUBJECT_FRAGMENT_RE.search(value)
                or BAD_KEY_RE.search(value) or not re.search(r"[가-힣A-Za-z]{2}", value)):
            continue
        if require_domain and not DOMAIN_SUBJECT_RE.search(value):
            continue
        output.append(value)
    return unique(output, limit)


def table_structure(text):
    rows = []
    for line in text.splitlines():
        if not line.lstrip().startswith("|"):
            continue
        cells = [re.sub(r"<[^>]+>", " ", cell).strip() for cell in line.strip().strip("|").split("|")]
        cells = [cell for cell in cells if cell and not re.fullmatch(r"[-: ]+", cell)]
        if cells:
            rows.append(cells)
    headers = unique(rows[0], 12) if rows else []
    row_keys = unique([cells[0] for cells in rows[1:] if cells], 40)
    # Tables produced from OCR sometimes put the logical row key in column 2.
    if len(set(row_keys)) <= 1:
        row_keys = unique(row_keys + [cells[1] for cells in rows[1:] if len(cells) > 1], 40)
    return headers, row_keys

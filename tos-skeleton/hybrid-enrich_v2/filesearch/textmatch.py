"""태그 값 매칭 규칙 (특약명 정규화·트라이그램 퍼지·역할 별칭). 원본: hybrid-enrich/slot_filesearch.py"""
import re

ROLE_ALIASES = {
    "exclusion_exception": "면책 제외 예외 부지급 지급하지 않는 사유",
    "premium_waiver": "보험료 납입면제",
    "payment_trigger": "보험금 지급사유 지급조건",
    "payment_amount": "보험금 지급금액 지급률 산정 계산",
    "limit_frequency": "지급한도 횟수 일수 최초 1회",
    "timing_period": "보장개시 책임개시 대기기간 감액기간 보험기간",
    "definition": "용어 정의 의미",
    "criteria_rule": "진단확정 판정기준 적용기준",
    "contract_lifecycle": "갱신 해지 소멸 무효 환급",
    "claim_procedure": "보험금 청구 구비서류 절차",
    "code_reference": "질병분류코드 수가코드 부표 분류표",
}


def compact(value):
    return re.sub(r"[^가-힣a-z0-9]", "", str(value).lower())


def contract_core(value):
    value = re.sub(r"\(무배당[^)]*\)|\(간편\)|해약환급금\s*미지급형|갱신형|일반형", "", str(value))
    value = compact(value).replace("특약", "")
    return value.replace("허혈성심장질환", "허혈심장질환").replace("대상포진진단", "대상포진통풍진단")


def trigrams(value):
    value = compact(value)
    return {value[index:index + 3] for index in range(max(0, len(value) - 2))}


def fuzzy_contains(query, target, field):
    q = contract_core(query) if field == "contract" else compact(query)
    t = contract_core(target) if field == "contract" else compact(target)
    if not q or not t:
        return False
    if q in t or t in q:
        return True
    if field != "contract" or min(len(q), len(t)) < 5:
        return False
    qg, tg = trigrams(q), trigrams(t)
    # Partial contract names may omit a bundled coverage term, but must retain
    # both a leading and trailing anchor. This prevents similarly named riders
    # (e.g. CAR-T vs targeted anticancer drug) from sharing a namespace.
    anchored = q[:3] in t and q[-4:] in t
    return anchored and len(qg & tg) / max(1, len(qg)) >= 0.72

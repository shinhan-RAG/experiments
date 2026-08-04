# 약관 frontmatter 스키마 v1

> 딥 인터뷰 스펙(`.omc/specs/deep-interview-tos-frontmatter.md`) 기준.
> 용도: LLM/검색기의 탐색·라우팅 색인. 답변 근거는 항상 원문 (frontmatter는 근거가 아님).
> 문서 종류: **판매약관** 전용. 다른 종류(사업방법서 등)는 별도 스키마.

```yaml
document:                      # 신원 (세그먼테이션에서 결정적으로 추출)
  product: 상품명
  version: 개정판 식별 (예: 260507)
  doc_type: 판매약관

composition:                   # 구성 — 결정적 추출 (LLM 불필요)
  - rider: 특약명 (원문 표기 그대로)
    params:                    # 특약명에 인코딩된 파라미터 (룰 파싱)
      simplified: true|false   # (간편)
      variant: "[30일한도형]" 등 대괄호 파라미터 (없으면 null)
      renewal: 갱신형|비갱신형|null
      refund: 해약환급금 미지급형|일부지급형|null

coverage:                      # 보장 요약 — LLM 추출 (게이트 1 검증 대상)
  - rider: 특약명
    benefits:
      - name: 급부명 (예: 질병장해보험금)
        trigger: 지급 조건 한 줄 요약
        source: 근거 조 (예: 제2-2조)
        key_terms: [원문에 실제 등장하는 핵심 용어들]  # 게이트 1 검증용 앵커
```

## 게이트 1 (자동 검증) 규칙
- composition: 각 rider가 문서 목차(전문)에 등장하는지 대조
- coverage: `source` 조가 해당 특약 단위에 실존 + `key_terms` 각각이 근거 조 원문에
  (공백 무시) 부분일치하는지 확인. 불일치 항목은 폐기 후 재추출, 리포트에 기록

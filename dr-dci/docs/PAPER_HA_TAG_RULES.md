# paper-ha 시멘틱 태그 규칙

AI Hub "32.학술논문 이해 데이터" 인문학·예술체육학(HA) Validation 331편을
`paper-ha` 데이터셋으로 정제할 때 적용하는 요소(element) 판정·태깅 규칙.
라벨링 JSON에 이미 있는 구조 신호만 사용하는 **결정론적 규칙**이며, LLM 판정이 없다
(LLM 태그 A/B/C는 별도 축 — `scripts/build_tags.py`).

## 1. 요소 추출 규칙

입력: `Validation/02.라벨링데이터/VL_인문학,예술체육학(HA)/*.json`

| 라벨 원천 | 요소 생성 규칙 |
|---|---|
| `training_data_info.section_info[]` | 단락 1개 → 청크 1개. 재청킹 없음(단락이 곧 검색 단위). |
| `training_data_info.image_info[]` | 이미지 1개 → 캡션 청크 1개. 본문은 `image_caption`(생성 설명문). 이미지 픽셀은 코퍼스에 넣지 않는다. |

제외 규칙:
- `original_text`(또는 `image_caption`)가 20자 미만이면 코퍼스에서 제외한다.
  대부분의 목차형 제목("1. 머리말")이 여기 해당한다. 제외돼도 그 제목은
  **이후 단락들의 `section` 필드로 승계**되므로 정보가 사라지지 않는다.
- 빈 텍스트/공백만 있는 항목 제외. 제외 건수는 `corpus_stats.json`에 기록.

섹션 승계: `section_info`를 파일 내 순서대로 읽으면서 마지막으로 본
`title_*`의 텍스트를 현재 섹션명으로 유지하고, 이어지는 `para_*` 청크의
`section`으로 넣는다. 첫 제목 이전 단락은 `"(전문)"`.

## 2. element_type / @el 태그 매핑 (결정론, approach_p)

| 라벨 근거 | element_type | approach_p 태그 |
|---|---|---|
| `paragraph_id` = `title_*` (20자 이상만 잔존) | `heading` | `@el:heading` |
| `paragraph_id` = `para_*` | `text` | `@el:paragraph` |
| `image_info.image_category` = `TA` (표) | `table` | `@el:table` |
| `image_info.image_category` = `CH` (차트/그래프) | `chart` | `@el:chart` |
| `image_info.image_category` = `PI` (그림/사진) | `figure` | `@el:figure` |

- 알 수 없는 `image_category` 값은 `figure`로 폴백하고 경고 카운트를 남긴다.
- `element_type`은 corpus.jsonl 필드(메타데이터 축), `@el:` 태그는
  `data/tags/paper-ha/approach_p/7k.json`(에이전트 grep tag_filter 축) — 같은
  판정을 두 소비 지점에 맞는 형태로 내보낸 것이다.

## 3. ID 규칙 (결정론·재현 가능)

- 문서 `doc` = 라벨 파일 스템 (`HA_0032_0011647`)
- 단락 청크 `_id` = `{doc}_{paragraph_id}` → `HA_0032_0011647_para_1`
- 이미지 청크 `_id` = `{doc}_img_{image_id}` → `HA_0032_0011462_img_2`
- `title` 필드 = `{doc_title[:80]} — {section 또는 image_name}`

## 4. 알려진 한계 (manifest.limitations에도 기록)

1. `original_text`는 어절 간 공백이 소실된 구간이 많다(원문 그대로 둠).
   `summary_text`는 정상 띄어쓰기이므로 혼용 시 주의.
2. `image_caption`은 논문에 인쇄된 실제 캡션이 아니라 주변 본문에서 생성한
   설명문이다. 표는 셀 구조 없이 이 설명문만 검색된다.
3. `page`가 전부 "1"(단일 슬라이드 좌표계)이라 페이지 정보가 없다.
   `location`(EMU bbox)은 코퍼스에 싣지 않는다.
4. 4,096자 초과 단락은 임베딩 입력에서 잘릴 수 있다(재청킹하지 않고
   건수만 감사에 기록).

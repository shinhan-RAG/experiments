# Gold v21 반입 및 sonnet5 실측 (2026-08-24)

## sonnet5 vs opus-medium, Gold v21 기준 (r1v_verify_identity_reference_hybrid, 213×2)

- `sonnet5_r1v_213x2_goldv21_rescore.jsonl` — **새 실행이 아니라 기존 완료된 sonnet 런을 재채점**한 결과다.
  `vector_search/noah/0819/out/host_agent/sonnet_final_r1v_213x2_goldv19_20260823/`에 2026-08-23에 이미
  213×2=426셀(0 errors, 실비용 $25.79)로 완료된 sonnet5 실행이 있었다 — 검색기·태그·문서가 그대로이고 Gold v21은
  v20 대비 그룹/행 변경이 없으므로(§동일문구 확장만 추가), `rescore_agent_results.py`로 그 원본 제출 ID를
  Gold v21로 재채점하는 것이 LLM 재호출 없이 동일하게 유효하다. **추가 비용·시간 없이** 얻은 결과다.
- 비교 대상 opus 쪽도 같은 방식: `vector_search/noah/0819/out/host_agent/opus_r1v_full213x2_goldv20_20260824/results_goldv21_rescore.jsonl`(공식 채택 런, 2026-08-24, 0 errors)를 그대로 사용.

| model | R@1 | R@5 | R@10 | suff@5 | suff@10 | errors | protocol_errors |
|---|---:|---:|---:|---:|---:|---:|---:|
| opus-medium (공식) | .5951 | **.9030** | **.9229** | **.8897** | **.9131** | 0 | 4 |
| sonnet5 (2026-08-23 런 재채점) | **.5998** | .8662 | .8873 | .8474 | .8756 | 0 | 27 |

paired(qid×rep, n=426, sonnet−opus):

- R@5 Δ**−.0368**, 7승/28패(391동), bootstrap 95% CI `[−.0606,−.0141]`, sign p=.0005
- R@10 Δ−.0356, 5승/24패, CI `[−.0583,−.0137]`, sign p=.0005
- suff@10 Δ−.0376, 5승/21패, CI `[−.0610,−.0141]`, sign p=.0025

**결론: Gold v21에서 sonnet5는 opus-medium보다 R@5·R@10·suff@10 전부에서 통계적으로 유의하게 낮다**(CI가 0을 포함하지 않음).
R@1만 sonnet이 근소 우위. sonnet의 `protocol_errors`가 opus의 약 7배(27 vs 4)로, 도구 호출/포맷 실패가 더 잦았던 것이
저성능의 일부 원인으로 보인다(호스트가 회복해 최종 submit은 받았으므로 `errors`는 0 — 정상 성공으로 보존된 셀).

**재실행이 필요하면**: 이 결과는 08-23에 이미 끝난 런의 재채점이라 오늘 새로 만든 데이터가 아니다. 진짜 신선한
sonnet5 실행(같은 질문에 대한 새 세션, 새 비용 발생 — 이전 런 기준 426셀에 실비용 ~$26, 걸린시간 관측치는
0819 데이터 참고)을 원하면 별도로 요청.

## 데이터 무결성 확인 (메타데이터 벡터·엘리먼트 태그)

r1v arm 설정(`arms.json`)이 요구하는 파일들이 모두 로컬에 있고, sonnet(08-23)·opus(08-24) 두 런이 실행된
이후로도 변경되지 않았음을 확인했다(수정일이 두 런보다 이전):

| 용도 | 파일 | 상태 |
|---|---|---|
| 엘리먼트(문서 파편) | `filesearch/out/elements_u3.jsonl` | 존재, 13MB, 최종수정 08-21 17:43 |
| 조(jo) 단위 | `filesearch/out/elements_u3jo.jsonl` | 존재, 8MB, 최종수정 08-21 17:43 |
| 태그(BM25F용) | `filesearch/out/tags_u4_fact_rules.jsonl` | 존재, 36MB, 최종수정 08-21 17:43 |
| 메타 V9 청크 뷰 | `vector_search/out/view_V9.jsonl`, `chunks.jsonl`, `chunk_stats.json` | 존재 |
| 메타 V9 임베딩 행렬 | `vector_search/out/emb/chunk_V9.npy`(24MB)+`chunk_V9_ids.json`+`chunk_V9_meta.json` | 존재, 최종수정 08-19 |
| 질의 임베딩 모델 | `dragonkue/BGE-m3-ko`(HF 캐시 로컬 존재) + `sentence-transformers` 6.0.0 | 로컬 스모크 테스트로 dense_sim 정상 산출 확인 |

`SEMTAG_META_PYTHON`/`EMBED_ENDPOINT` 둘 다 비어있어도, 기본 `python`(miniforge3)에 이미
`sentence-transformers`와 모델 캐시가 있어 메타/V9 dense 채널이 로컬에서 그대로 동작한다(외부 임베딩 서버 불필요).
이 런들 이후 위 파일 중 아무것도 재생성되지 않았으므로, sonnet/opus 두 결과와 앞으로의 v21 실험이 **완전히 동일한
검색 인덱스·태그**를 쓴다.


이 폴더의 데이터는 `tos-skeleton/hybrid-enrich_v2/filesearch/out/`에 있던 **Gold v21**
(`gold_train_scoped_u3_reviewed_overlay_v21.jsonl`, 커밋 `cf802ec2`)을
로컬에서 **재생성**한 것이다. 원본 `.jsonl`은 `filesearch/.gitignore`(`out/*.jsonl`)로
커밋되지 않아 이 저장소에 실물이 없었고, `.manifest.json`(출처 기록)만 git에 남아있었다.

## 파일

- `gold_v21_train213.jsonl` — Gold v21, train 213문항. 각 행: `{qid, q, task_type, core_retrieval, groups:[{key,c0,c1,members:[{c0,c1,src,jo}],...}], ...}`
- `qids_v21_train213.json` — 위 213개 qid 리스트(= `filesearch/out/qids_train_scoped_u3_reviewed_overlay_v20.json`와 동일 — v21은 v20에서 행 추가/삭제 없이 그룹 내부에 동일문구 OR-member만 추가했기 때문)

## 재생성 방법 (재현 가능)

`filesearch/out/gold_train_scoped_u3_reviewed_overlay_v1.jsonl`(로컬에 유일하게 실물로 남아있던 시작점, 281행)에서 출발해,
`filesearch/out/*.manifest.json`(v2~v21, 전부 git 추적됨)에 기록된 `inputs.reviewed` / `inputs.reviewed_qids` 경로를 그대로 따라
`filesearch/build_reviewed_train_overlay.py`를 v2→v20까지 19회 순차 실행했다(입력은 `filesearch/audits/full_gold_audit_v1_batch01~17_*` 및 `source_only_q0449_reviewed.jsonl`, `source_only_q0373_reaudit_reviewed.jsonl` — batch 번호가 버전 번호와 1:1 대응하지 않아 매 단계 manifest에서 실제 경로를 읽었다).
마지막으로 `filesearch/apply_spec3_occurrence_expansion.py`(원본 스크립트는 무수정 — 스크래치 사본에서 18행의
Mac 하드코딩 경로만 `vector_search/doc/`의 git 추적 사본으로 교체해 실행)로 v20→v21 "동일문구 출현 확장"을 적용했다.

**검증 결과**: v2~v20 19단계 전부 기존 manifest의 `counts`(output/replaced/excluded)와 정확히 일치. v21 최종 산출도
기존 manifest 대비 `rows=213, general_q=89, scoped_q=124, added_members=316` 전부 일치, qid 213개 집합도
`qids_train_scoped_u3_reviewed_overlay_v20.json`과 완전히 동일함을 확인했다.

**단, 바이트 단위(sha256) 재현은 아니다.** 로컬에 남아있던 v1.jsonl의 sha256이 v2 manifest가 기대하는 base sha256과
다르다(원본은 다른 머신 `/Users/ralph/...`에서 생성됨) — 아마 JSON 직렬화 방식(키 순서/공백)이나 그때그때의 로컬
재실행 차이로 보인다. 그래서 행 수·replaced/excluded/added_members 같은 **구조적 지표**로 정합성을 검증했다.
`build_reviewed_train_overlay.py` 자체도 reviewed qid가 base에 없으면 즉시 예외를 던지는 구조라, 19단계가 전부
에러 없이 끝까지 돌았다는 것 자체가 강한 정합성 신호다.

## train / QA set 연결

이 213개 qid는 **348-train 질문 원본**(`정답셋_348_train_v3_retrieval_250212.csv`, 동등하게
`docs/QASet/v3_retrieval/qa_gold_v3.jsonl`의 train 부분)을 리뷰·감사(batch01~17, retrieval-blind)로 걸러낸
부분집합이다. 질문 텍스트는 이미 `gold_v21_train213.jsonl` 각 행의 `"q"` 필드에 들어있어 채점에 별도 QA 파일이
꼭 필요하지는 않다.

**test(149문항)는 이 리뷰 체인을 거친 적이 없다.** v19~v21의 감사·확장은 train에만 적용됐고, v21에 대응하는
test gold는 이 계보에 존재하지 않는다.

## `hybrid-enrich_v2/out/`과의 관계 — 무관한 별개 계보

`tos-skeleton/hybrid-enrich_v2/out/`에 있는 `gold_mapped_noah_v3_*`(348/149/344행, element-id 리스트 형식)와
`noah/gold_v4_*`(337/90행, 조 단위 재교정)는 **같은 348-train 원본에서 갈라져 나왔지만 v19~v21 감사 체인과
한 번도 합류하지 않은, 2026-08-20에 멈춘 독립적인 계보**다. 여기 있는 파일을 변형해서 v21을 만드는 게 아니라,
v21은 처음부터 `filesearch/out/` + `filesearch/audits/`라는 별도 경로에서 독자적으로 만들어졌다. 참고/대조용으로는
쓸 수 있어도 v21의 조상이나 대체물은 아니다.

## 원본 문서 확인

`docs/QASet/판매약관_(간편)신한통합건강보장보험 원(ONE)(무배당, 해약환급금 미지급형)_250212.md`
(로컬 전용, `docs/`는 gitignore)는 실제로 QA셋(v3, 497문항)이 만들어진 원본 문서가 맞다 — 그 파일의 sha256이
`docs/QASet/v3_retrieval/qa_gold_v3.jsonl`의 모든 행에 박힌 `source_sha256`과 정확히 일치한다.

다만 **줄바꿈만 다른 두 버전**이 존재한다: `docs/QASet/`의 로컬 사본은 LF(5,804,624B), 이 저장소가 실제
파이프라인에서 쓰는 git 추적 사본(`tos-skeleton/hybrid-enrich_v2/vector_search/doc/판매약관_..._250212.md`,
`tos-skeleton/hybrid-enrich/noah_qaset/`에도 동일 사본)은 CRLF(5,916,624B)다. CRLF를 제거하면 두 파일은
바이트까지 완전히 동일함을 확인했다. 이번 v21 재생성에는 git 추적된 CRLF 사본을 사용했다(`apply_spec3_occurrence_expansion.py`가
내부적으로 `\r\n`→`\n` 정규화를 하므로 결과에는 영향 없음).

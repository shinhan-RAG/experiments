# noah/0821v2 — Codex slot/meta routed search experiment

0821 실험의 후속 버전이다. 현재 250212 코퍼스 기준으로 QA/gold를 v5로 감사하고, 태그는 임베딩·BM25 없이 슬롯 검색만 사용하며 자연어 메타 질의는 고정 V9 hybrid 검색으로 라우팅한다.

## v5 최종 정책

- `build_gold_v5.py`는 원본 v4를 보존한 채 `out/gold_v5/`에 train337/test90과 수정 감사를 생성한다.
- `slot_meta_agent`는 `search=CLM slot only`, `msearch=V9 BM25+dense hybrid`로 고정한다. test qtags는 사용하지 않으며, submit 전에 msearch가 없으면 제출을 거부해 슬롯 채널 편향을 방지한다.
- 최종 평가는 `run_final_once.py`만 사용한다. 모델 `gpt-5.6-luna`, candidate arm, test 전체, reps=1을 강제하며 lock 생성 후 재실행을 거부한다.

```powershell
python run_final_once.py --preflight
python run_final_once.py
```

## 변경 범위

- 메타데이터는 복제하지 않는다. `hybrid-enrich_v2/filesearch/out`의 동일한 원천 파일을 참조한다.
- `agent_tools.py`의 `search`, `msearch`, `read`, `submit` 계약과 검색 예산은 유지한다.
- semantic tag 검색은 `tag_hybrid.py`와 기존 `arms.json`을 그대로 사용한다.
- `agent_runner.py`는 Codex CLI를 질의별 격리 세션에서 실행하고, 각 세션의 `calls.jsonl`, `submit.json`, `codex.json`을 보존한다.
- Codex 실행은 `workspace-write` 샌드박스를 사용한다. 실제 검색은 `python agent_tools.py ...` 명령으로만 수행하며, 결과물은 세션 디렉터리에만 기록한다.

## 준비

PowerShell에서 이 디렉터리로 이동해 실행한다.

```powershell
.\prepare_experiment.ps1
```

이 명령은 gold manifest와 frozen metadata를 확인하고, tag index를 만들고, train337 기준으로 candidate arm을 `out/det/selected_arm.txt`에 기록한다. 0821의 metadata 파일을 수정하지 않는다.

## Codex 실행

```powershell
# 5문항 smoke
.\run_codex.ps1 -Stage smoke -CodexBin "$env:APPDATA\npm\codex.cmd"

# 60문항 중간 확인
.\run_codex.ps1 -Stage train60 -CodexBin "$env:APPDATA\npm\codex.cmd"

# 전체 train 평가
.\run_codex.ps1 -Stage trainfull -CodexBin "$env:APPDATA\npm\codex.cmd"

# 공식 test
.\run_codex.ps1 -Stage test -CodexBin "$env:APPDATA\npm\codex.cmd"
```

기본 모델은 `gpt-5`이며 필요하면 `-Model`로 변경한다. 재실행 시 완료된 세션은 `--resume`로 건너뛴다. baseline과 candidate 결과 및 paired 비교 결과는 각각 `out/agent/`와 `out/*_comparison.json`에 저장된다.

## 다음 semantic-tag 고도화 실험 위치

메타데이터를 바꾸지 않고 다음 항목을 독립적으로 실험한다.

1. `tag_hybrid.py`: slot 추출, sparse tokenization, RRF와 JO collapse
2. `arms.json`: channel weight, collapse 정책, router 조합
3. `agent_tools.py`: semantic tag 검색 파라미터와 provenance 노출
4. `test_tag_hybrid.py`: 태그·locator·중복 제거·검색 순위 회귀 테스트

새 arm을 추가할 때에는 train337에서 먼저 결정하고, test에서는 `selected_arm.txt`와 가중치를 변경하지 않는다. Codex 자체의 응답 품질과 검색엔진 변경 효과를 분리하기 위해 동일한 gold와 동일한 검색 예산을 유지한다.

## 결과 해석

`compare_runs.py`는 baseline 대비 candidate의 R@5, R@10, suff@5, suff@10 및 paired wins/losses를 계산한다. Codex 실행 실패, timeout, no-submit은 각 run의 `summary.json`에서 별도로 확인한다.

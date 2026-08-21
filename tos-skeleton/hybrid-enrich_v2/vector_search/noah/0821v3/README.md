# C29 dual-tool Codex experiment

요청 커밋 `7e5e2c6a66af119fda9a2e83de825d9d3089fd48`의 281문항 QA/Gold와 C29 semantic-tag 검색을 사용한다. Codex CLI `gpt-5.6-luna` medium을 4 worker로 실행하며, 검색 채널은 다음처럼 분리한다.

- `search`: C29 CLM + structured BM25F fact + membership evidence quota
- `msearch`: 고정 V9 BM25 + BGE-m3-ko dense RRF metadata 검색

두 도구를 각각 최소 한 번 사용해야 `submit`이 허용된다. Gold는 모델 프롬프트나 검색기 입력에 노출되지 않고 호스트 채점에만 사용된다.
병렬 dense query encoding은 로컬 Hugging Face 캐시를 사용하며 worker 환경에 `HF_HUB_OFFLINE=1`, `TRANSFORMERS_OFFLINE=1`을 고정한다. 실행기는 로컬 OpenAI-compatible embedding 서버를 한 번 띄워 4 worker가 같은 BGE 모델을 공유하고, 실행 종료 시 서버를 내린다. 이미 `EMBED_ENDPOINT`가 설정돼 있으면 그 서버를 사용한다.

```powershell
cd C:\Users\Atdev-pc\Desktop\wiki\experiments\tos-skeleton\hybrid-enrich_v2\vector_search\noah\0821v3

# artifact, Codex 로그인, semantic/meta 양 채널 확인
.\run_experiment.ps1 -Stage preflight

# 4문항, worker 4 smoke
.\run_experiment.ps1 -Stage smoke

# 281문항, worker 4 본실험
.\run_experiment.ps1 -Stage full

# 중단된 동일 run 이어서 실행
.\run_experiment.ps1 -Stage full -Resume

# 검색 방식 설명 출력
.\explain_filesearch.ps1
```

결과는 `out/<run>/manifest.json`, `results.jsonl`, `summary.json`, `sessions/`에 저장된다. Windows checkout의 CRLF 변환을 허용하면서도 요청 커밋의 원문을 검증하도록 preflight SHA는 UTF-8/LF 정규화 후 계산한다.

281문항 전체 실행 결과와 해석은 [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md)에 기록했다. 기계 판독용 세부 분석은 `out/c29_dual_tool_train281_luna_medium_w4/analysis.json`이며 `python analyze_results.py`로 재생성할 수 있다.

OpenAI 공식 모델 문서 기준으로 GPT-5.6 Luna는 비용 민감·대량 처리용 모델이며 `medium`이 기본 reasoning effort다.

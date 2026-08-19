# KT 인스턴스(Backend.AI 컨테이너)에서 에이전트 실측 실행 안내

이 패키지는 `tos-skeleton/semtag-v2/`(코드+데이터)와 `tos-skeleton/hybrid-enrich/`(의존 3파일)를 담는다. 구조를 유지한 채 풀 것.

## 0. 필요 서비스 (인스턴스 로컬)
- LLM: `http://localhost:8080/v1` (vLLM, `Qwen/Qwen3-32B-FP8`)  — 함수호출 지원 여부는 스모크에서 확인, 미지원이면 `--protocol text`
- 임베딩: `http://localhost:8101/v1` (`dragonkue/BGE-m3-ko`, dim 1024) — `EMBED_ENDPOINT` 로 지정하면 torch 없이 동작

## 1. 환경
```bash
tar xzf semtag-v2-instance.tar.gz && cd tos-skeleton/semtag-v2
python3 -m venv .venv && . .venv/bin/activate && pip install -q --upgrade pip && pip install -q numpy openai
export EMBED_ENDPOINT=http://localhost:8101/v1   # 질의 임베딩 소스(서버). 색인도 같은 소스로 통일하려면 아래 2단계 실행
```

## 2. (권장) 색인 벡터를 서버 임베딩으로 재생성 — 질의·색인 한 소스
```bash
python embed_u2.py --view ctx --batch 64     # out/emb/u2_ctx.npy 덮어씀(약 수 분). base 도 필요하면 --view base
```
(생략하면 Mac MPS 로 만든 npy 를 그대로 씀 — 서버 임베딩과 미세한 정밀도 차이 있음. 실험 기록에 어느 쪽인지 남길 것)

## 3. 스모크 (3문항 · 1rep) — 함수호출 지원 확인
```bash
python agent_runner_oai.py --run smoke_qwen --base-url http://localhost:8080/v1 --model Qwen/Qwen3-32B-FP8 \
  --arm '{"lex":"count","w":{"contract":2},"router":"llm","qtags":"out/qtags_haiku.jsonl","vec_view":"ctx"}' \
  --n 3 --reps 1 --workers 3 --protocol tools
# tool_calls 가 안 오면(턴만 소모·no_submit) → --protocol text 로 재실행
```
Qwen3 thinking 이 켜져 있으면 응답이 길어짐: vLLM 이 `chat_template_kwargs` 를 받으면 러너 옵션 없이 프롬프트 첫 줄에 `/no_think` 를 넣어 비교(스모크에서 두 방식 턴 수·토큰 비교 후 결정).

## 4. 파일럿 (60문항 층화 × 2 reps)
```bash
# A1: 태그 단독(규칙 라우터)      A2: 태그+qtags       C3: 두 도구(태그+벡터) 에이전트
python agent_runner_oai.py --run pilot60_A1 --base-url http://localhost:8080/v1 --model Qwen/Qwen3-32B-FP8 --arm '{"lex":"count","w":{},"router":"rule"}' --n 60 --reps 2 --workers 6
python agent_runner_oai.py --run pilot60_A2 ... --arm '{"lex":"count","w":{"contract":2},"router":"llm","qtags":"out/qtags_haiku.jsonl"}' --n 60 --reps 2 --workers 6
python agent_runner_oai.py --run pilot60_C3 ... --arm '{"lex":"count","w":{"contract":2},"router":"llm","qtags":"out/qtags_haiku.jsonl","vec_view":"ctx"}' --n 60 --reps 2 --workers 6
```
결과: `out/agent/<run>/summary.json`(R@1/5/10/20·MRR@10·core/비core·미제출·오류·토큰), `results.jsonl`(문항별), `sessions/*/calls.jsonl`(도구 호출·후보 전량 로그).
채점 gold 는 기본 `out/gold_spans_train.jsonl`(임시). lsh v4 기준은 `--gold out/gold_spans_lsh_train.jsonl`.

## 5. (선택) Qdrant 를 컨테이너 안에 CPU 로 — 팀 공유용
```bash
mkdir -p ~/qdrant && cd ~/qdrant && curl -L https://github.com/qdrant/qdrant/releases/latest/download/qdrant-x86_64-unknown-linux-gnu.tar.gz | tar xz
QDRANT__SERVICE__API_KEY="$(openssl rand -hex 16)" ; echo "$QDRANT__SERVICE__API_KEY" > ~/qdrant/API_KEY
QDRANT__SERVICE__API_KEY=$(cat ~/qdrant/API_KEY) QDRANT__STORAGE__STORAGE_PATH=~/qdrant/storage nohup ./qdrant > qdrant.log 2>&1 &
cd - && pip install -q qdrant-client && QDRANT_API_KEY=$(cat ~/qdrant/API_KEY) python upsert_qdrant.py --view ctx --url http://localhost:6333
```
외부 공유는 Backend.AI 콘솔에서 6333 을 서비스 포트로 등록해야 보인다(8080 과 같은 방식). 실험 자체는 Qdrant 없이(npy 인메모리) 동작한다.

## 원칙(변경 금지)
- test149 는 절대 사용하지 않는다(파일도 없음). 결과 JSON·로그에 질문 원문을 저장소에 커밋하지 않는다.
- 예산(page 40·preview 160·search+vsearch ≤20·read ≤8·submit ≤10)·모델·프로토콜은 run 의 config.json 에 자동 기록된다 — 보고 시 함께 인용.

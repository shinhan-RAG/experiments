# 인스턴스(vLLM Qwen, 임베딩 8101) 실행

```
git fetch && git checkout <branch>
cd tos-skeleton/hybrid-enrich_v2/filesearch
python3 -m venv .venv && . .venv/bin/activate && pip install -q numpy openai
export EMBED_ENDPOINT=http://localhost:8101/v1        # vector_search 가 쓰는 임베딩 소스
# out/ 준비: elements_u2*.jsonl, tags_u2_*.jsonl, gold_spans_*.jsonl, qtags_haiku.jsonl (재생성은 README 파이프라인)
# 메타 채널을 쓰려면 ../vector_search 의 out(chunks, view, emb) 도 준비

# 스모크
python agent_runner_oai.py --run smoke --base-url http://localhost:8080/v1 --model Qwen/Qwen3-32B-FP8 --protocol text \
  --arm '{"lex":"count","w":{"contract":2},"router":"llm","qtags":"out/qtags_haiku.jsonl"}' --n 3 --reps 1
# 파일럿
... --n 60 --reps 2 --workers 6
```
tools 프로토콜은 vLLM 에 `--enable-auto-tool-choice --tool-call-parser hermes` 가 있어야 한다. 없으면 `--protocol text`.
Qwen3 thinking 은 켠 채로 둔다(`--no-think`는 파일럿에서 R@5 −.11).
결과: `out/agent/<run>/summary.json`, `results.jsonl`, `sessions/*/calls.jsonl`.

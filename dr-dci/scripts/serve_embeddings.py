"""로컬 임베딩 서버 — OpenAI 호환 /v1/embeddings (vLLM 8101 대체, Windows용).

KT Cloud의 vLLM(gte-Qwen2-1.5B-instruct, port 8101)과 같은 인터페이스를
transformers로 제공한다. RTX 4060 8GB 기준 fp16으로 ~3GB.

모델 카드 규격: last-token pooling + L2 normalize. query instruction은
클라이언트(PullRetriever)가 텍스트에 이미 붙여 보내므로 서버는 그대로 임베딩만 한다.

사용:
  python scripts/serve_embeddings.py [--port 8101] [--model Alibaba-NLP/gte-Qwen2-1.5B-instruct]
"""
import argparse

import torch
import torch.nn.functional as F
import uvicorn
from fastapi import FastAPI
from pydantic import BaseModel
from transformers import AutoModel, AutoTokenizer

app = FastAPI()
state = {}

MAX_TOKENS = 2048   # corpus embed_text는 4096자 절단 → 한국어 기준 ~2k 토큰이면 충분
MICRO_BATCH = 8     # 8GB VRAM 안전선


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] | str


def last_token_pool(hidden, attention_mask):
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return hidden[:, -1]
    seq_len = attention_mask.sum(dim=1) - 1
    return hidden[torch.arange(hidden.shape[0], device=hidden.device), seq_len]


@torch.inference_mode()
def embed(texts: list[str]) -> list[list[float]]:
    tok, model, device = state["tok"], state["model"], state["device"]
    out = []
    for i in range(0, len(texts), MICRO_BATCH):
        batch = tok(texts[i:i + MICRO_BATCH], max_length=MAX_TOKENS, padding=True,
                    truncation=True, return_tensors="pt").to(device)
        hidden = model(**batch).last_hidden_state
        emb = last_token_pool(hidden, batch["attention_mask"])
        emb = F.normalize(emb, p=2, dim=1)
        out.extend(emb.float().cpu().tolist())
    return out


@app.get("/v1/models")
def models():
    return {"data": [{"id": state["name"], "object": "model"}]}


@app.post("/v1/embeddings")
def embeddings(req: EmbeddingRequest):
    texts = [req.input] if isinstance(req.input, str) else req.input
    vectors = embed(texts)
    return {
        "object": "list",
        "model": state["name"],
        "data": [{"object": "embedding", "index": i, "embedding": v}
                 for i, v in enumerate(vectors)],
        "usage": {"prompt_tokens": 0, "total_tokens": 0},
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Alibaba-NLP/gte-Qwen2-1.5B-instruct")
    ap.add_argument("--port", type=int, default=8101)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    print(f"loading {args.model} on {device} ({dtype}) ...")
    state["tok"] = AutoTokenizer.from_pretrained(args.model)
    state["model"] = AutoModel.from_pretrained(
        args.model, torch_dtype=dtype).to(device).eval()
    state["device"] = device
    state["name"] = args.model
    print("ready")
    uvicorn.run(app, host="127.0.0.1", port=args.port, log_level="warning")


if __name__ == "__main__":
    main()

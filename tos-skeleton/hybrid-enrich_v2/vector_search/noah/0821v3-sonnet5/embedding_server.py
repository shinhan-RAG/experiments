#!/usr/bin/env python3
"""Small local OpenAI-compatible embedding server that keeps BGE-m3-ko loaded once."""
from __future__ import annotations

import argparse
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path


HERE = Path(__file__).resolve().parent
VS = HERE.parents[1]
sys.path.insert(0, str(VS))
os.environ.pop("EMBED_ENDPOINT", None)
import embedder


ENCODE_LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:  # noqa: N802
        if self.path == "/health":
            self.send_json(200, {"ok": True, "model": embedder.MODEL})
        else:
            self.send_json(404, {"error": "not_found"})

    def do_POST(self) -> None:  # noqa: N802
        if self.path.rstrip("/") != "/v1/embeddings":
            self.send_json(404, {"error": "not_found"})
            return
        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            texts = payload.get("input", [])
            if isinstance(texts, str):
                texts = [texts]
            with ENCODE_LOCK:
                vectors = embedder.encode(texts)
            self.send_json(200, {
                "object": "list",
                "model": payload.get("model") or embedder.MODEL,
                "data": [
                    {"object": "embedding", "index": index, "embedding": vector.tolist()}
                    for index, vector in enumerate(vectors)
                ],
            })
        except Exception as exc:
            self.send_json(500, {"error": f"{type(exc).__name__}: {exc}"})

    def log_message(self, fmt: str, *args) -> None:
        print(fmt % args, flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, required=True)
    args = parser.parse_args()
    embedder.encode(["warmup"])
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(json.dumps({"ready": True, "host": args.host, "port": args.port, "model": embedder.MODEL}), flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

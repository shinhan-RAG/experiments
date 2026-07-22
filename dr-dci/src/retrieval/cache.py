"""Content-addressed cache keys for embedding artifacts."""

import hashlib
import json


CACHE_SCHEMA_VERSION = 2


def embedding_cache_key(
    *,
    namespace: str,
    model: str,
    use_prefix: bool,
    doc_ids: list[str],
    texts: list[str],
) -> str:
    """Fingerprint every input that can change an embedding artifact."""
    digest = hashlib.sha256()
    header = {
        "schema": CACHE_SCHEMA_VERSION,
        "namespace": namespace,
        "model": model,
        "use_prefix": use_prefix,
        "document_count": len(doc_ids),
    }
    digest.update(json.dumps(header, sort_keys=True).encode("utf-8"))
    for doc_id, text in zip(doc_ids, texts):
        doc_bytes = str(doc_id).encode("utf-8")
        text_bytes = text.encode("utf-8")
        digest.update(len(doc_bytes).to_bytes(8, "big"))
        digest.update(doc_bytes)
        digest.update(len(text_bytes).to_bytes(8, "big"))
        digest.update(text_bytes)
    return digest.hexdigest()[:20]

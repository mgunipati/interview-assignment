"""Local sentence embeddings (no transcript text leaves the machine).

all-mpnet-base-v2 reads at most 384 tokens, but a transcript is ~1,800, so whole calls are
embedded as the mean of ~200-word chunk embeddings rather than silently truncated.
"""
import hashlib
from pathlib import Path

import numpy as np

MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
CACHE_DIR = Path(__file__).resolve().parents[2] / "cache" / "embeddings"
CHUNK_WORDS = 200

_model = None


def _get_model():
    global _model
    if _model is None:
        from sentence_transformers import SentenceTransformer
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def embed(texts):
    """Embed short texts (unit-normalised), cached on disk by content hash."""
    texts = list(texts)
    key = hashlib.sha256((MODEL_NAME + "\x00" + "\x00".join(texts)).encode()).hexdigest()[:16]
    path = CACHE_DIR / f"{key}.npy"
    if path.exists():
        return np.load(path)
    vecs = _get_model().encode(texts, batch_size=32, normalize_embeddings=True, show_progress_bar=False)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(path, vecs)
    return vecs


def chunk(text, size=CHUNK_WORDS):
    words = text.split()
    return [" ".join(words[i:i + size]) for i in range(0, len(words), size)] or [""]


def embed_documents(docs):
    """Embed long documents as the normalised mean of their chunk embeddings."""
    chunks = [chunk(d) for d in docs]
    flat = embed([c for cs in chunks for c in cs])
    out, i = [], 0
    for cs in chunks:
        v = flat[i:i + len(cs)].mean(axis=0)
        out.append(v / np.linalg.norm(v))
        i += len(cs)
    return np.vstack(out)

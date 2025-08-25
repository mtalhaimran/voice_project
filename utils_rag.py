"""RAG utilities for text chunking and retrieval."""
from __future__ import annotations

from typing import List

import numpy as np
from openai import OpenAI


def chunk_text(text: str, max_tokens_est: int = 400) -> List[str]:
    """Split text into overlapping chunks for embedding."""
    chunk_size = 1600  # ~4 chars per token
    overlap = 200
    text = text.strip().replace("\r", "")
    chunks: List[str] = []
    i = 0
    while i < len(text):
        chunk = text[i : i + chunk_size]
        chunks.append(chunk)
        i += chunk_size - overlap
    return chunks


def embed_texts(client: OpenAI, texts: List[str]) -> np.ndarray:
    """Embed a list of texts using a small embedding model."""
    res = client.embeddings.create(model="text-embedding-3-small", input=texts)
    return np.array([d.embedding for d in res.data], dtype=np.float32)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Cosine similarity between a vector and matrix of vectors."""
    a_norm = a / (np.linalg.norm(a) + 1e-8)
    b_norm = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-8)
    return np.dot(b_norm, a_norm)


def retrieve_context(client: OpenAI, kb_chunks: List[str], kb_embeds: np.ndarray, query: str, top_k: int = 3) -> str:
    """Retrieve top-k similar chunks for a query."""
    if not kb_chunks:
        return ""
    q_vec = embed_texts(client, [query])[0]
    sims = cosine_sim(q_vec, kb_embeds)
    idx = np.argsort(-sims)[:top_k]
    top_chunks = [kb_chunks[i] for i in idx]
    return "\n\n".join(top_chunks)

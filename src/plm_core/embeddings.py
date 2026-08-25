"""BAAI/bge-m3 embeddings for the Retrieval Agent's Chroma collections, called via the
HF Inference API (not loaded locally) — see src/config.py for why.

Unlike e5-style models, bge-m3 does not need an instruction/prefix on the query side
(per its model card), so queries and documents are embedded identically here.
"""

import time

from chromadb import Documents, EmbeddingFunction, Embeddings
from huggingface_hub import InferenceClient

from src.config import RETRIEVAL_EMBEDDING_MODEL

_client = InferenceClient(provider="hf-inference")


def _embed(texts: list[str], retries: int = 3) -> list[list[float]]:
    for attempt in range(retries):
        try:
            out = _client.feature_extraction(texts, model=RETRIEVAL_EMBEDDING_MODEL)
            return out.tolist()
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 * (attempt + 1))  # cold-start / rate-limit backoff


class BgeM3EmbeddingFunction(EmbeddingFunction):
    """Chroma embedding function — used both when writing chunks and when querying."""

    @staticmethod
    def name() -> str:
        return "bge-m3-hf-api"

    def __call__(self, input: Documents) -> Embeddings:
        return _embed(list(input))


def embed_query(query: str) -> list[float]:
    return _embed([query])[0]

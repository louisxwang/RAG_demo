from __future__ import annotations

import logging

import numpy as np
from sentence_transformers import SentenceTransformer

from app.core.config import settings

log = logging.getLogger(__name__)


class EmbeddingModel:
    """
    Local embeddings using sentence-transformers.

    Design choice: normalize embeddings so cosine similarity becomes dot-product,
    which FAISS handles efficiently with IndexFlatIP.
    """

    def __init__(self) -> None:
        self.model = SentenceTransformer(settings.embedding_model)

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, 384), dtype=np.float32)
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)


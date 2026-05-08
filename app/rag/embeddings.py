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
        # Auto-detect device: prefer CUDA, then MPS, otherwise CPU.
        device = "cpu"
        try:
            import torch

            if torch.cuda.is_available():
                device = "cuda"
            else:
                # MPS (Apple Silicon) support
                try:
                    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                        device = "mps"
                except Exception:
                    pass
        except Exception:
            # torch not installed; fall back to CPU
            log.debug("torch not available; using CPU for embeddings")

        log.info("Initializing SentenceTransformer on device=%s", device)
        self.model = SentenceTransformer(settings.embedding_model, device=device)
        # store output dimension dynamically
        self.dim = self.model.get_sentence_embedding_dimension()

    def embed_texts(self, texts: list[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0, int(self.dim)), dtype=np.float32)
        vecs = self.model.encode(texts, normalize_embeddings=True)
        return np.asarray(vecs, dtype=np.float32)

    def embed_query(self, text: str) -> np.ndarray:
        vec = self.model.encode([text], normalize_embeddings=True)
        return np.asarray(vec, dtype=np.float32)


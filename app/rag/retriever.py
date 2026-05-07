from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

import faiss
import numpy as np

from app.core.config import settings
from app.rag.embeddings import EmbeddingModel

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source: str
    score: float


class Retriever:
    """
    FAISS retriever with a simple JSON docstore.

    Design choice: keep docstore as a list indexed by vector id (fast and minimal).
    """

    def __init__(self, index_dir: str | None = None) -> None:
        self.index_dir = Path(index_dir or settings.index_dir)
        self.embedder = EmbeddingModel()
        self._index: faiss.Index | None = None
        self._docstore: list[dict] | None = None

    def load(self) -> None:
        index_path = self.index_dir / "index.faiss"
        docstore_path = self.index_dir / "docstore.json"

        if not index_path.exists() or not docstore_path.exists():
            raise FileNotFoundError(
                f"Missing index files in {self.index_dir}. Run ingestion first."
            )

        self._index = faiss.read_index(str(index_path))
        self._docstore = json.loads(docstore_path.read_text(encoding="utf-8"))

        log.info("Loaded FAISS index (%s vectors) from %s", len(self._docstore), self.index_dir)

    def search(self, query: str, top_k: int | None = None) -> list[RetrievedChunk]:
        if self._index is None or self._docstore is None:
            self.load()

        assert self._index is not None
        assert self._docstore is not None

        k = int(top_k or settings.top_k)
        q = self.embedder.embed_query(query)
        scores, ids = self._index.search(q, k)

        out: list[RetrievedChunk] = []
        for score, idx in zip(scores[0].tolist(), ids[0].tolist(), strict=False):
            if idx < 0 or idx >= len(self._docstore):
                continue
            row = self._docstore[idx]
            out.append(RetrievedChunk(text=row["text"], source=row.get("source", ""), score=float(score)))
        return out


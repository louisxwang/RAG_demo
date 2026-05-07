from __future__ import annotations

import argparse
import json
import logging
import os
from dataclasses import asdict, dataclass
from pathlib import Path

import faiss
import numpy as np
from pypdf import PdfReader

from app.core.config import settings
from app.rag.embeddings import EmbeddingModel

log = logging.getLogger(__name__)


@dataclass
class Chunk:
    id: int
    source: str
    text: str


def _iter_text_files(root: Path) -> list[Path]:
    exts = {".txt", ".md", ".pdf"}
    files: list[Path] = []
    for p in root.rglob("*"):
        if p.is_file() and p.suffix.lower() in exts:
            files.append(p)
    return sorted(files)


def _read_file(path: Path) -> str:
    if path.suffix.lower() in {".txt", ".md"}:
        return path.read_text(encoding="utf-8", errors="ignore")
    if path.suffix.lower() == ".pdf":
        reader = PdfReader(str(path))
        pages = []
        for i, page in enumerate(reader.pages):
            try:
                pages.append(page.extract_text() or "")
            except Exception:  # noqa: BLE001
                pages.append("")
        return "\n\n".join(pages)
    return ""


def _chunk_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    text = text.replace("\r\n", "\n")
    if not text.strip():
        return []

    chunks: list[str] = []
    start = 0
    n = len(text)
    while start < n:
        end = min(n, start + chunk_size)
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end == n:
            break
        start = max(0, end - overlap)
    return chunks


def build_index(input_path: str, index_dir: str) -> dict[str, str]:
    """
    Ingest local text/markdown/PDF files into FAISS + JSON docstore.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = _iter_text_files(root) if root.is_dir() else ([root] if root.is_file() else [])
    if not files:
        raise ValueError("No .txt/.md/.pdf files found to ingest.")

    embedder = EmbeddingModel()
    chunks: list[Chunk] = []

    next_id = 0
    for f in files:
        text = _read_file(f)
        for c in _chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(Chunk(id=next_id, source=str(f.as_posix()), text=c))
            next_id += 1

    texts = [c.text for c in chunks]
    vecs = embedder.embed_texts(texts)
    if vecs.ndim != 2 or vecs.shape[0] != len(chunks):
        raise RuntimeError("Embedding shape mismatch during ingest.")

    dim = int(vecs.shape[1])
    index = faiss.IndexFlatIP(dim)
    index.add(vecs)

    index_path = out_dir / "index.faiss"
    docstore_path = out_dir / "docstore.json"
    meta_path = out_dir / "meta.json"

    faiss.write_index(index, str(index_path))

    docstore = [asdict(c) for c in chunks]
    docstore_path.write_text(json.dumps(docstore, ensure_ascii=False), encoding="utf-8")

    meta = {
        "embedding_model": settings.embedding_model,
        "dim": dim,
        "count": len(chunks),
        "chunk_size": settings.chunk_size,
        "chunk_overlap": settings.chunk_overlap,
        "input_path": str(root.as_posix()),
    }
    meta_path.write_text(json.dumps(meta, indent=2), encoding="utf-8")

    log.info("Ingested %s chunks from %s files into %s", len(chunks), len(files), out_dir)
    return {
        "index_path": str(index_path.as_posix()),
        "docstore_path": str(docstore_path.as_posix()),
        "meta_path": str(meta_path.as_posix()),
    }


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", settings.log_level))
    parser = argparse.ArgumentParser(description="Ingest text/markdown files into FAISS.")
    parser.add_argument("--path", required=True, help="File or directory to ingest (.txt/.md)")
    parser.add_argument("--index-dir", default=settings.index_dir, help="Output directory for index")
    args = parser.parse_args()

    build_index(args.path, args.index_dir)


if __name__ == "__main__":
    main()


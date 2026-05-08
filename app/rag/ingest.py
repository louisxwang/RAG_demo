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


def build_index(input_path: str, index_dir: str, preprocessed_dir: str | None = None) -> dict[str, str]:
    """
    Ingest local text/markdown/PDF files into FAISS + JSON docstore.
    """
    root = Path(input_path)
    if not root.exists():
        raise FileNotFoundError(f"Input path not found: {input_path}")

    out_dir = Path(index_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if preprocessed_dir:
        pre_root = Path(preprocessed_dir)
        if not pre_root.exists():
            raise FileNotFoundError(f"Preprocessed dir not found: {preprocessed_dir}")
        files = [p for p in pre_root.rglob("*.txt") if p.is_file()]
        if not files:
            raise ValueError("No .txt files found in preprocessed dir to ingest.")
    else:
        files = _iter_text_files(root) if root.is_dir() else ([root] if root.is_file() else [])
        if not files:
            raise ValueError("No .txt/.md/.pdf files found to ingest.")

    embedder = EmbeddingModel()
    chunks: list[Chunk] = []

    next_id = 0
    for f in files:
        # If using preprocessed dir, f is a .txt file containing PAGE_ markers. Read as text.
        if preprocessed_dir and f.suffix.lower() == ".txt":
            text = f.read_text(encoding="utf-8", errors="ignore")
            # try to map back to an original PDF in the input_path by stem
            source_path = None
            try:
                stem = f.stem
                # search for a matching pdf under input_path (root)
                matches = list(root.rglob(f"{stem}.pdf")) if root.exists() else []
                if matches:
                    source_path = matches[0]
            except Exception:
                source_path = None

            src = str(source_path.as_posix()) if source_path else str(f.as_posix())
        else:
            text = _read_file(f)
            src = str(f.as_posix())

        for c in _chunk_text(text, settings.chunk_size, settings.chunk_overlap):
            chunks.append(Chunk(id=next_id, source=src, text=c))
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
    parser.add_argument("--path", required=True, help="File or directory to ingest (.txt/.md/.pdf). If using --preprocessed-dir, this should be the original PDF root (used to map sources).")
    parser.add_argument("--index-dir", default=settings.index_dir, help="Output directory for index")
    parser.add_argument("--preprocessed-dir", default=None, help="Optional directory containing pre-extracted .txt files (one per PDF). If provided, these .txt files will be ingested instead of scanning --path for PDFs/txts.")
    args = parser.parse_args()

    build_index(args.path, args.index_dir, preprocessed_dir=args.preprocessed_dir)


if __name__ == "__main__":
    main()


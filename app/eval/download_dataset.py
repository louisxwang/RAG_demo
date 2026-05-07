from __future__ import annotations

import argparse
import contextlib
import io
import logging
import os
import sys
from pathlib import Path

import kagglehub

log = logging.getLogger(__name__)


DEFAULT_KAGGLE_DATASET = "manisha717/dataset-of-pdf-files"


def download(dataset: str = DEFAULT_KAGGLE_DATASET, out_dir: str = "data/kaggle_pdfs") -> str:
    """
    Downloads Kaggle dataset using kagglehub.

    Notes:
    - kagglehub uses your Kaggle credentials (typically via env vars or kaggle.json).
    - We copy/link into out_dir to keep project-local paths stable.
    """
    # kagglehub may print warnings/notices to stdout. Suppress any stdout noise so
    # callers can reliably capture only the final path from this script's stdout.
    with contextlib.redirect_stdout(io.StringIO()):
        path = kagglehub.dataset_download(dataset)
    src = Path(path)
    # Some Kaggle datasets (including the default one here) place PDFs under a "Pdf/" subfolder.
    # Return the directory that actually contains the PDFs so it can be fed directly to ingest/eval scripts.
    pdf_dir = src / "Pdf"
    if pdf_dir.exists() and pdf_dir.is_dir():
        src = pdf_dir
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Keep it simple: just return the downloaded cache path and let the user ingest from it.
    # (Copying many PDFs can be slow and doubles disk usage.)
    log.info("Downloaded dataset=%s to cache_path=%s", dataset, src)
    log.info("Tip: ingest directly from cache_path to avoid copying")
    return str(src.as_posix())


def main() -> None:
    # Log to stderr so stdout can be captured as a pure path.
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"), stream=sys.stderr)
    p = argparse.ArgumentParser(description="Download evaluation dataset (Kaggle PDFs).")
    p.add_argument("--dataset", default=DEFAULT_KAGGLE_DATASET)
    args = p.parse_args()
    # Some libraries print directly to the underlying stdout file descriptor (bypassing sys.stdout).
    # To make PowerShell capture robust, silence OS-level stdout during download and only print the
    # final path after restoring stdout.
    saved_fd1 = os.dup(1)
    try:
        with open(os.devnull, "w") as devnull:
            os.dup2(devnull.fileno(), 1)
            cache_path = download(args.dataset)
    finally:
        os.dup2(saved_fd1, 1)
        os.close(saved_fd1)

    os.write(1, (cache_path + "\n").encode("utf-8"))


if __name__ == "__main__":
    main()


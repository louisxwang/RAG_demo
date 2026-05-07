from __future__ import annotations

import argparse
import logging
import os
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
    path = kagglehub.dataset_download(dataset)
    src = Path(path)
    dst = Path(out_dir)
    dst.mkdir(parents=True, exist_ok=True)

    # Keep it simple: just return the downloaded cache path and let the user ingest from it.
    # (Copying many PDFs can be slow and doubles disk usage.)
    log.info("Downloaded dataset=%s to cache_path=%s", dataset, src)
    log.info("Tip: ingest directly from cache_path to avoid copying")
    return str(src.as_posix())


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    p = argparse.ArgumentParser(description="Download evaluation dataset (Kaggle PDFs).")
    p.add_argument("--dataset", default=DEFAULT_KAGGLE_DATASET)
    args = p.parse_args()
    cache_path = download(args.dataset)
    print(cache_path)


if __name__ == "__main__":
    main()


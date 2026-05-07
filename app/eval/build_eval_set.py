from __future__ import annotations

import argparse
import json
import logging
import os
import random
import time
from pathlib import Path
from typing import Any

from json_repair import repair_json
from pypdf import PdfReader
from tqdm import tqdm

from app.llm.client import LLMClient

log = logging.getLogger(__name__)


QA_PROMPT = """You are given a passage of text. Your task is to write a **fact-based question** and its corresponding **concise answer** using only information from the passage.

These questions are intended for a **RAG-based question-answering system** designed for general users.

Generate a diverse set of realistic user questions with a medium to high complexity. The questions should:
- The question must be **factoid-style**: clear, direct, and answerable with a specific fact.
- Do **not** reference the passage itself (e.g., don’t say “according to the text”).
- The answer should be **short and precise**.
- Write the question in a natural, **search engine-like format**.
- Require **synthesizing information from multiple pages, sections, or documents** to provide a complete and accurate answer.
- Include **complex, layered, or comparative** information needs.
- Avoid simple factual lookups.
- Use synonyms instead of the exact wording mentioned in the document.

Generate {N_GENERATIONS} question answer pairs.
Follow this output format:
[
  {{
    "factoid_question": "<your factoid question>",
    "answer": "<your answer>",
    "page_no": [3,4]
  }}
]

Now here is the passage:
"""


CRITIQUE_PROMPT = """### Task 1:
You are given a *question* and a *context*. Evaluate how well the context supports a clear, unambiguous answer to the question.
Rate 1 to 5:
- 1 = context does not help at all
- 5 = context clearly and fully answers the question without ambiguity

### Task 2:
Rate how useful this question is for ML developers to evaluate their RAG system.
Rate 1 to 5.

### Task 3:
Rate how clearly the question stands on its own (no missing references like "the document").
Rate 1 to 5.

### Task 4:
You are given an answer along with a list of "page_no" values and context pages. Verify whether the page numbers listed accurately reflect the source of information used in the answer.
Scoring:
- 1 = page numbers correctly and completely support the answer
- 0 = page numbers do not fully support the answer

Your response MUST be valid JSON in this format:
{
  "groundedness": {"Evaluation": "...", "Total_rating": "4"},
  "relevance": {"Evaluation": "...", "Total_rating": "4"},
  "standalone": {"Evaluation": "...", "Total_rating": "4"},
  "page_accuracy": {"Evaluation": "...", "Total_rating": "1"}
}

Now review the following:
"""


def _iter_pdfs(root: Path) -> list[Path]:
    pdfs: list[Path] = []
    for p in root.rglob("*.pdf"):
        if p.is_file():
            pdfs.append(p)
    return sorted(pdfs)


def _read_pdf_pages(path: Path, max_pages: int = 8) -> list[str]:
    reader = PdfReader(str(path))
    pages: list[str] = []
    for page in reader.pages[:max_pages]:
        try:
            pages.append((page.extract_text() or "").strip())
        except Exception:  # noqa: BLE001
            pages.append("")
    return pages


def _llm_json(llm: LLMClient, messages: list[dict[str, Any]]) -> Any:
    txt = llm.chat(messages, temperature=0.0)
    try:
        return json.loads(txt)
    except Exception:  # noqa: BLE001
        return json.loads(repair_json(txt))


def build_eval_set(
    pdf_root: str,
    out_path: str = "eval/eval_set.jsonl",
    sample_n_files: int = 20,
    n_generations_per_file: int = 2,
    max_pages_per_pdf: int = 8,
    sleep_s: float = 0.0,
    min_score: int = 4,
) -> dict[str, int]:
    """
    Replicates the article's approach:
      - Generate QA pairs with a specialized prompt
      - Critique with an LLM judge
      - Filter on high thresholds
    """
    llm = LLMClient()

    root = Path(pdf_root)
    pdfs = _iter_pdfs(root)
    if not pdfs:
        raise ValueError(f"No PDFs found under: {pdf_root}")

    random.shuffle(pdfs)
    pdfs = pdfs[:sample_n_files]

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)

    total = 0
    kept = 0

    with out.open("w", encoding="utf-8") as f:
        for pdf in tqdm(pdfs, desc="Generating eval QA"):
            pages = _read_pdf_pages(pdf, max_pages=max_pages_per_pdf)
            context_pages = [p for p in pages if p]
            if len(" ".join(context_pages)) < 400:
                continue

            passage = "\n\n".join([f"PAGE_{i+1}:\n{t}" for i, t in enumerate(pages)])

            qa = _llm_json(
                llm,
                [
                    {"role": "system", "content": "You generate evaluation QA pairs for a RAG system. Output ONLY valid JSON."},
                    {"role": "user", "content": QA_PROMPT.format(N_GENERATIONS=n_generations_per_file) + passage},
                ],
            )
            if not isinstance(qa, list):
                continue

            for item in qa:
                if not isinstance(item, dict):
                    continue
                q = str(item.get("factoid_question", "")).strip()
                a = str(item.get("answer", "")).strip()
                page_no = item.get("page_no", [])
                if not q or not a:
                    continue

                total += 1

                critique = _llm_json(
                    llm,
                    [
                        {"role": "system", "content": "You are a strict evaluator. Output ONLY valid JSON."},
                        {
                            "role": "user",
                            "content": (
                                CRITIQUE_PROMPT
                                + f"\nQuestion:\n{q}\n\n"
                                + "Context pages (ordered):\n"
                                + "\n\n".join([f"PAGE_{i+1}:\n{t}" for i, t in enumerate(pages)])
                                + f"\n\nAnswer:\n{a}\n\npage_no:\n{page_no}\n"
                            ),
                        },
                    ],
                )

                def _score(key: str, default: int = 0) -> int:
                    try:
                        return int(str(critique[key]["Total_rating"]))
                    except Exception:  # noqa: BLE001
                        return default

                grounded = _score("groundedness")
                relevance = _score("relevance")
                standalone = _score("standalone")
                page_acc = _score("page_accuracy")

                record = {
                    "question": q,
                    "answer": a,
                    "page_no": page_no,
                    "pdf_path": str(pdf.as_posix()),
                    "scores": {
                        "groundedness": grounded,
                        "relevance": relevance,
                        "standalone": standalone,
                        "page_accuracy": page_acc,
                    },
                    "critiques": {
                        k: (critique.get(k, {}) or {}).get("Evaluation", "")
                        for k in ["groundedness", "relevance", "standalone", "page_accuracy"]
                    },
                }

                if grounded >= min_score and relevance >= min_score and standalone >= min_score and page_acc == 1:
                    f.write(json.dumps(record, ensure_ascii=False) + "\n")
                    kept += 1

                if sleep_s:
                    time.sleep(sleep_s)

    log.info("Eval-set generated total=%s kept=%s out=%s", total, kept, out)
    return {"total": total, "kept": kept}


def main() -> None:
    logging.basicConfig(level=os.environ.get("LOG_LEVEL", "INFO"))
    p = argparse.ArgumentParser(description="Build a RAG evaluation dataset from PDFs (LLM-generated + filtered).")
    p.add_argument("--pdf-root", required=True, help="Directory containing PDFs (recursively)")
    p.add_argument("--out", default="eval/eval_set.jsonl")
    p.add_argument("--sample-n-files", type=int, default=20)
    p.add_argument("--n-generations", type=int, default=2)
    p.add_argument("--max-pages", type=int, default=8)
    p.add_argument("--sleep-s", type=float, default=0.0)
    p.add_argument("--min-score", type=int, default=4)
    args = p.parse_args()

    build_eval_set(
        pdf_root=args.pdf_root,
        out_path=args.out,
        sample_n_files=args.sample_n_files,
        n_generations_per_file=args.n_generations,
        max_pages_per_pdf=args.max_pages,
        sleep_s=args.sleep_s,
        min_score=args.min_score,
    )


if __name__ == "__main__":
    main()


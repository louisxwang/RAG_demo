# Enterprise AI Assistant

### A personal demo project (RAG + simple agent) of Louis Wang

The project is developped and tested under Windows 11. 
Production-style (but intentionally minimal) GenAI project:

- **RAG**: ingest docs → chunk → embed locally → FAISS similarity search
- **Agent pipeline**: retrieve context → summarize → answer
- **Tool calling**: simple calculator tool (safe eval)
- **API**: FastAPI `POST /query`
- **UI**: Streamlit app that calls backend
- **Docker**: container runs backend + frontend

## 1) Setup

### Option A: Local run

```bash
python -m venv .venv
# Windows PowerShell:
.venv\Scripts\Activate.ps1

python -m pip install -r requirements.txt
```

By default, the app runs with `LLM_PROVIDER=mock` (no keys needed) so you can quickly test the full RAG + API + UI flow.

To use a real model, set environment variables. Example using OpenAI-compatible API:

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-4o-mini"
```

Or use Gemini:

```powershell
$env:LLM_PROVIDER="gemini"
$env:GEMINI_API_KEY="YOUR_KEY"
$env:GEMINI_MODEL="gemini-flash-latest"
```

Note: environment variables are scoped to your current shell/session. If you run `uvicorn` and `streamlit` in different terminals, set the LLM env vars in the backend terminal (the one running `uvicorn`) before starting it. The Streamlit UI does not need LLM keys (it only calls the backend; set `BACKEND_URL` only if you changed the backend address/port).

### Option B: Docker

```bash
docker build -t rag-demo .
docker run --rm -p 8000:8000 -p 8501:8501 ^
  -e LLM_PROVIDER=openai ^
  -e OPENAI_API_KEY=YOUR_KEY ^
  -e OPENAI_MODEL=gpt-4o-mini ^
  rag-demo
```

Gemini example:

```bash
docker run --rm -p 8000:8000 -p 8501:8501 ^
  -e LLM_PROVIDER=gemini ^
  -e GEMINI_API_KEY=YOUR_KEY ^
  -e GEMINI_MODEL=gemini-flash-latest ^
  rag-demo
```

## 2) Ingest documents (default example: dataset workflow)

### Use your data

You can put your own documents under `data/` (you create it) and ingest them:

```bash
python -m app.rag.ingest --path data --index-dir storage
```

Supported file types: `.txt`, `.md`, `.pdf`

### Use a dataset for demo

You can also use the belowing dataset to quickly start a demo.

#### Download PDFs (Kaggle)

```powershell
$env:CACHE_PATH = (python -m app.eval.download_dataset).Trim()
```

It prints the local directory path containing the PDFs (for this dataset, it will include the `Pdf/` subfolder). Use that path for below.
Note the `$env:CACHE_PATH =` syntax, it can automatically set the enviroment variable so you don't need to copy the path, but you need to run it in a powershell, not cmd.exe

#### Build eval dataset (synthetic QA + filtering)

If your PDFs sometimes fail to extract clean text, it's safer to separate PDF->text extraction from the LLM calls so you don't waste API quota on parsing errors.

Recommended two-step workflow:

1) Preprocess PDFs to text files (local, no API calls)

```powershell
python -m app.eval.preprocess_pdfs --pdf-root "$env:CACHE_PATH" --out-dir data/preprocessed_pdf_texts --max-pages 8
```

This writes one `.txt` file per PDF in `data/preprocessed_pdf_texts/` with page markers like `PAGE_1:`. Inspect these files and fix/remove PDFs that produced poor extraction before running the LLM stage.

2) Generate QA and critiques with the LLM (uses API keys)

```powershell
python -m app.eval.build_eval_set --pdf-root "$env:CACHE_PATH" --out eval/eval_set.jsonl --sample-n-files 20 --n-generations 2 --preprocessed-dir data/preprocessed_pdf_texts --sleep-s 1
```

Notes on tuning:
- `--sleep-s` inserts a pause between LLM calls; start with 0.5–2s to reduce the chance of hitting rate limits.
- Reduce `--sample-n-files` and `--n-generations` to test cheaply before scaling up.
- Make sure `GEMINI_API_KEY` or `OPENAI_API_KEY` is set in your shell before running the LLM stage.
- The LLM client now has a retry/backoff for 429/503 responses, and `build_eval_set` will skip files/items that trigger LLM errors rather than crash.

If you prefer a one-shot run, `build_eval_set` still reads PDFs directly, but separating preprocessing is recommended when PDF extraction is noisy.

Ingest the PDFs into your FAISS index

```powershell
python -m app.rag.ingest --path "$env:CACHE_PATH" --preprocessed-dir data/preprocessed_pdf_texts --index-dir storage
```

This creates:

- `storage/index.faiss`
- `storage/docstore.json`
- `storage/meta.json`

## 3) Run

Backend:

```bash
uvicorn app.main:app --reload --port 8000
```

Frontend:

```bash
streamlit run frontend/app.py
```

Open:

- API docs: `http://localhost:8000/docs`
- UI: `http://localhost:8501`

## 4) Example API usage

```bash
curl -X POST http://localhost:8000/query -H "Content-Type: application/json" ^
  -d "{\"question\":\"What is in my docs?\"}"
```

## Design notes (why this shape)

- **FAISS on disk**: keeps retrieval fast and restart-friendly without a database.
- **Local embeddings**: `sentence-transformers` avoids external embedding calls and cost.
- **OpenAI-compatible client**: works with OpenAI and many compatible gateways via `OPENAI_BASE_URL`.
- **Gemini**: set `LLM_PROVIDER=gemini` and `GEMINI_API_KEY` to call Google Gemini via HTTP.
- **Simple agent**: a small orchestrator is easier to reason about than heavy frameworks.

## 5) Run evaluation

```bash
python -m app.eval.run_eval --eval-set eval/eval_set.jsonl --out eval/results.jsonl
```


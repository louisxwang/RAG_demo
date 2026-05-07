# Enterprise AI Assistant 
###  A personal demo project (RAG + simple agent) of Louis Wang

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

pip install -r requirements.txt
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
$env:GEMINI_MODEL="gemini-1.5-flash"
```

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
  -e GEMINI_MODEL=gemini-1.5-flash ^
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

Download PDFs (Kaggle)

```bash
python -m app.eval.download_dataset
```

It prints the local cache directory path. Use that path below.

Build eval dataset (synthetic QA + filtering)

```bash
python -m app.eval.build_eval_set --pdf-root "<CACHE_PATH>" --out eval/eval_set.jsonl --sample-n-files 20 --n-generations 2
```

Ingest the PDFs into your FAISS index

```bash
python -m app.rag.ingest --path "<CACHE_PATH>" --index-dir storage
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


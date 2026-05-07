# Enterprise AI Assistant 
###  A personal demo project (RAG + simple agent) of Louis Wang

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

Set environment variables (example using OpenAI-compatible API):

```powershell
$env:LLM_PROVIDER="openai"
$env:OPENAI_API_KEY="YOUR_KEY"
$env:OPENAI_MODEL="gpt-4o-mini"
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

## 2) Ingest documents

Put `.txt` or `.md` files into `data/` (you create it), then run:

```bash
python -m app.rag.ingest --path data --index-dir storage
```

This creates:
- `storage/index.faiss`
- `storage/docstore.json`
- `storage/meta.json`

Supported file types: `.txt`, `.md`, `.pdf`

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
- **Simple agent**: a small orchestrator is easier to reason about than heavy frameworks.

## 5) Dataset download + evaluation (as in the linked article)

This repo includes a minimal version of the workflow from the Medium post:
1) download a public PDF dataset
2) use an LLM to generate QA pairs (with page references)
3) use an LLM judge to critique + filter high-quality QA
4) run your RAG pipeline against that eval set

### Download PDFs (Kaggle)

```bash
python -m app.eval.download_dataset
```

It prints the local cache directory path. Use that path below.

### Build eval dataset (synthetic QA + filtering)

```bash
python -m app.eval.build_eval_set --pdf-root "<CACHE_PATH>" --out eval/eval_set.jsonl --sample-n-files 20 --n-generations 2
```

### Ingest the PDFs into your FAISS index

```bash
python -m app.rag.ingest --path "<CACHE_PATH>" --index-dir storage
```

### Run evaluation

```bash
python -m app.eval.run_eval --eval-set eval/eval_set.jsonl --out eval/results.jsonl
```


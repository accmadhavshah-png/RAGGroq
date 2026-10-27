# Experiment 7: Preparing a Healthcare RAG Application for Production-Ready Deployment

This project adapts the attached **Experiment 5: Healthcare RAG Using Multiple File Types** to an API. It keeps the same retrieval design and replaces the local Qwen answer-generation step with **Groq hosted inference**.

## What is reused from Experiment 5?

| Experiment 5 component | Used by this project |
|---|---|
| TXT, text-based PDF, and DOCX loading | `support_rag.py` |
| `RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=80)` | Same defaults, configurable through `.env` |
| `sentence-transformers/all-MiniLM-L6-v2` | Same embedding model |
| Persistent ChromaDB with cosine distance | Same approach, collection defaults to `experiment5_healthcare` |
| Retrieve up to three chunks | `RAG_TOP_K=3` |
| Heuristic similarity cutoff of 0.30 | `MIN_SIMILARITY=0.30` |
| Local Qwen model generates the answer | Replaced by Groq chat completions |

In the local API, the embedding model and vector index are initialized once
per API process, not reloaded for each question. Groq is the generation
backend only: it does not perform document retrieval.

## Project files

```text
experiment7_groq_rag/
├── app.py                         # FastAPI /health and /ask endpoints
├── support_rag.py                 # Experiment 5 pipeline and Groq backend
├── requirements.txt               # Lightweight Vercel dependencies
├── requirements-local.txt         # Full local app and index-builder dependencies
├── api/                           # Lightweight Vercel serverless API
├── .env.example                   # Configuration template, no real API key
├── .gitignore                     # Excludes secrets and local vector database
├── knowledge_base/
│   └── demo_healthcare_notes.txt  # Small test source, not a full guideline
├── tests/
│   └── test_app.py                # Automated endpoint tests, no Groq quota used
└── Experiment_7_Lab_Report.md     # Lab report, observations, and viva answers
```

## 1. Requirements

- Python 3.10 or later
- Internet access for the first download of the MiniLM embedding model and for Groq inference
- A Groq API key from the [Groq Console](https://console.groq.com/)
- TXT, text-based PDF, and/or DOCX documents for the knowledge base

The demo source file enables a small initial test. For a full reproduction of Experiment 5, copy the original `ICMR_Guidance_Document.txt`, `WHO DRUG INFORMATION.docx`, and `WHO guideline for obesiy in adolescents.pdf` into `knowledge_base/`. These original documents are not embedded in the attached PDF export itself, so the project cannot recreate the full 2,873-chunk index from the PDF alone.

## 2. Install dependencies

Open a terminal in the `experiment7_groq_rag` folder.

### Windows PowerShell

```powershell
py -3 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements-local.txt
Copy-Item .env.example .env
```

If PowerShell blocks activation, you may use the virtual environment interpreter directly: `.\.venv\Scripts\python.exe -m pip install -r requirements-local.txt`.

### macOS/Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements-local.txt
cp .env.example .env
```

## 3. Configure Groq

Edit `.env` and set your own key:

```dotenv
GROQ_API_KEY=your_real_key_here
GROQ_MODEL=openai/gpt-oss-20b
```

The included model ID is configurable. Check the [current supported Groq models](https://console.groq.com/docs/models) if a model is no longer available to your account. Never put the key in Python source, GitHub, logs, or screenshots. The `.gitignore` excludes `.env`.

## 4. Add knowledge-base files

Place the source files under `knowledge_base/`. Subfolders are scanned recursively. Supported extensions are `.txt`, `.pdf`, and `.docx`, case-insensitively. PDF files must contain extractable text; scanned image PDFs may require OCR before loading.

On the first run, the API downloads the MiniLM embedding model if needed, loads the files, splits them into chunks, embeds the chunks, and persists them to `chroma_db/`. The first request may therefore take longer than later requests.

If the knowledge-base source files change after the index has been built:

1. Set `REBUILD_INDEX=true` in `.env`.
2. Start the application and allow the new index to be built.
3. Set `REBUILD_INDEX=false` again before the next regular startup.

The index is built from files in this project. The ChromaDB directory created inside the original Colab runtime is not included in the attached Experiment 5 PDF and does not automatically exist on this computer.

## Vercel deployment

Vercel deploys `api/index.py` and installs only `api/requirements.txt`. The
serverless API reads the committed `api/data/index.json` file and performs
lightweight BM25-style lexical retrieval using the Python standard library.
It does not install `fastembed`, PyTorch, or ChromaDB, which keeps the Python
function below Vercel's bundle-size limit. The heavier local packages remain
available in `requirements-local.txt` for the local Experiment 7 pipeline and
optional index rebuilding.

Set `GROQ_API_KEY` in the Vercel project environment variables, then redeploy
from the `main` branch. The other deployment settings in `vercel.json` provide
the default model and retrieval configuration.

## 5. Run the API

```bash
uvicorn app:app --host 127.0.0.1 --port 8000 --reload
```

Open the interactive Swagger UI at **http://127.0.0.1:8000/docs**.

- `GET /health`: confirms the API process is running. It deliberately does not call Groq or initialize the embedding model.
- `POST /ask`: validates the question, retrieves evidence, generates a grounded answer with Groq, returns source metadata, and records latency in milliseconds.

Example request body for `POST /ask`:

```json
{
  "question": "What age range does the WHO guideline define as adolescence?"
}
```

Example request using PowerShell:

```powershell
$body = @{ question = "What age range does the WHO guideline define as adolescence?" } | ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/ask -ContentType "application/json" -Body $body
```

The exact generated answer and `latency_ms` depend on the model, network, computer, and currently available knowledge-base files, so they are intentionally not hard-coded in the report.

## 6. Test the endpoints

With the virtual environment active, run:

```bash
pytest -q
```

These unit tests mock the RAG service at the API boundary, so they test validation, status codes, response shape, and safe error handling without making paid or quota-limited Groq requests. Then test the live app from `/docs` to record actual retrieval and inference behavior.

Suggested live tests:

1. Supported: `What age range does the WHO guideline define as adolescence?`
2. Supported: `In the WHO Drug Information document, what does INN stand for?` (requires that source in the knowledge base; the demo note contains this fact.)
3. Unsupported: `What is the Wi-Fi password of this home router?`
4. Invalid: `{"question":"   "}`
5. Too long: a question with 1,001 characters
6. Backend failure: temporarily use an invalid/unavailable model or remove the key, then restore the valid configuration after recording the controlled error. Do not publish the key or secret configuration.

## Important limitation

This is an educational lab implementation, not a complete public production service and not a medical decision tool. Before real deployment, add authentication, HTTPS, rate limiting, monitoring, secure secret management, load testing, systematic RAG evaluation, dependency scanning, and a recovery plan. Do not submit confidential or patient-identifiable data to a third-party model API.

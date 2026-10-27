# EXPERIMENT NO. 7

## Preparing an LLM Application for Production-Ready Deployment

**Name:** Madhav Shah
**Division:** D
**Batch:** D
**UID:** 2024301026

**GitHub:** https://github.com/accmadhavshah-png/RAGGroq
**Vercel:** https://raggroq.vercel.app/docs

## 1. Aim

To prepare the Customer Support RAG application for API use by adding FastAPI
endpoints, input validation, controlled error handling, logging,
response-time measurement, and endpoint testing.

## 2. Learning outcomes

This experiment demonstrates how to expose a RAG prototype through an HTTP API,
validate requests, separate retrieval from generation, handle provider errors
safely, record latency, and return an uncertainty response when the documents
do not contain the answer.

## 3. Background and theory

Retrieval-Augmented Generation combines document retrieval with language
generation. The application first searches the knowledge base and then sends
the most relevant passages to the language model as context.

The local pipeline loads TXT, PDF, and DOCX files, splits them into 500-character
chunks with 80-character overlap, embeds them with
`sentence-transformers/all-MiniLM-L6-v2`, stores them in ChromaDB, and retrieves
up to three passages above the configured similarity cutoff. Groq generates the
answer from the retrieved passages.

The retriever and generation backend are separate. Groq can therefore be
changed without rebuilding the document-loading and vector-search components.
The generation prompt instructs the model not to invent policies, prices,
order status, refunds, or delivery promises that are absent from the sources.

The Vercel function uses a precomputed index and lightweight standard-library
retrieval because the native embedding stack exceeds the serverless bundle
limit. The local implementation uses MiniLM and ChromaDB.

## 4. Required tools

- Python 3.10 or later
- VS Code or another editor and a terminal
- FastAPI, Uvicorn, Pydantic, and `httpx`
- The Experiment 5 RAG logic and customer-support knowledge base
- A Groq account and API key
- Internet access for Groq and the first local embedding-model download

## 5. Selected LLM backend

### Option A: Cloud-hosted Groq API

Groq is the selected backend. The generation model runs on Groq servers and is
called with `GROQ_MODEL` and `GROQ_API_KEY`. The key is loaded from `.env`
locally or Vercel Environment Variables and is never hard-coded or committed.

Ollama was not selected. This project uses Hugging Face MiniLM for local
embeddings and Groq for answer generation.

## 6. Current knowledge base

The current customer-support sources are:

- `Accounts, payments and support.pdf`
- `Orders_and_Shipping.pdf`
- `Returns and refunds.pdf`

The precomputed deployment index contains 24 chunks from these three files.

## 7. Implementation

### `GET /health`

Returns the API status, service name, RAG initialization state, and whether a
non-placeholder Groq key is configured. It does not call Groq or initialize the
embedding model, so it confirms process availability but not every dependency.

### `POST /ask`

Accepts a JSON body such as:

```json
{
  "question": "How can I track my order?"
}
```

Successful responses include `request_id`, `answer`, `sources`,
`retrieved_count`, `model`, and `latency_ms`.

### Input validation

The `question` field must contain non-whitespace text and must not exceed 1,000
characters. FastAPI/Pydantic returns HTTP 422 for invalid input.

### Error handling

| Failure | Status | Handling |
|---|---:|---|
| Invalid request | 422 | FastAPI validation response |
| Missing Groq key | 503 | Generic configuration message |
| Groq/provider failure | 502 | Generic provider message |
| RAG initialization failure | 503 | Retrieval unavailable message |
| Unexpected exception | 500 | Generic internal-error message |

Secrets, stack traces, provider response bodies, prompts, and private user data
are not returned to clients or written to logs.

### Reuse, logging, and latency

`get_rag_service()` lazily creates one reusable RAG service per process. The
embedding model, ChromaDB collection, and Groq client are reused. Logs include
request ID, endpoint, status, latency, retrieved count, and exception type.
Latency is measured with `time.perf_counter()` and returned as `latency_ms`.

## 8. Procedure

1. Create and activate a virtual environment.

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install local dependencies.

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements-local.txt
   ```

3. Copy `.env.example` to `.env` and add the private Groq key.
4. Place customer-support documents in `knowledge_base/`.
5. If the documents change, set `REBUILD_INDEX=true` for one startup and then
   set it back to `false`.
6. Run:

   ```powershell
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

7. Open `http://127.0.0.1:8000/docs` and submit requests.
8. Test supported, unsupported, invalid, and backend-failure cases.

## 9. Project structure

```text
experiment7_groq_rag/
├── app.py                         # FastAPI endpoints and validation
├── support_rag.py                 # Loading, retrieval, and Groq generation
├── requirements.txt               # Lightweight Vercel dependencies
├── requirements-local.txt         # Full local dependencies
├── .env.example                   # Safe configuration template
├── api/index.py                   # Vercel serverless API
├── api/data/index.json            # Precomputed support index
├── knowledge_base/                # Support PDFs
├── tests/test_app.py              # Automated tests
└── Experiment_7_Lab_Report.md     # This report
```

## 10. Testing checklist and observations

| Test case | Expected result | Actual result | Status |
|---|---|---|---|
| Health endpoint | API reports running | HTTP 200 and `status: ok` | Pass |
| Supported question | Grounded answer and sources | Local retrieval returns `Orders_and_Shipping.pdf` for “How can I track my order?” | Pass |
| Unsupported question | Insufficient-information fallback | Automated test confirms zero passages and fallback text | Pass |
| Empty question | Validation error | HTTP 422 | Pass |
| Question over 1,000 characters | Validation error | HTTP 422 | Pass |
| Backend unavailable | Controlled error | HTTP 502 with no private details | Pass |
| Response-time measurement | Latency recorded | `latency_ms` is returned | Pass |
| Secret protection | No key in source or logs | `.env` is ignored and logs redact private details | Pass |
| Swagger UI | Interactive documentation loads | Verify after redeploying the latest commit | Pending live capture |

The local retrieval check returned:

```text
Orders_and_Shipping.pdf (PDF page 1), similarity 0.712
Orders_and_Shipping.pdf (PDF page 2), similarity 0.501
Accounts, payments and support.pdf (PDF page 1), similarity 0.457
```

The final submission should include a Swagger screenshot for a supported
question such as `How long does standard shipping take?` and an unsupported
question such as `What is the weather forecast for Mars tomorrow?`.

## 11. Automated tests

Run:

```powershell
python -m pytest -q
```

The test suite covers health success, valid response shape, whitespace-only
rejection, maximum-length validation, unsupported fallback, and provider
failure without exposing private exception details. The tests use fake RAG
services and do not consume Groq quota.

## 12. Result

The Customer Support RAG application was exposed through FastAPI and prepared
for deployment. It includes input validation, reusable retrieval components,
separate Groq generation, safe error handling, operational logging,
response-time measurement, automated endpoint tests, and Swagger documentation.

## 13. Discussion questions and answers

### 1. What is the purpose of FastAPI?

FastAPI exposes Python functions as HTTP endpoints, validates requests, returns
structured JSON, handles HTTP errors, and generates Swagger documentation.

### 2. What is the difference between a local LLM and a cloud-hosted API?

A local LLM runs on the user's computer and needs local model files and compute.
A cloud API runs remotely and is accessed through a network request and key.

### 3. Does hosting FastAPI in the cloud automatically host the LLM there?

No. Vercel hosts the API code, while Groq separately hosts and runs the model.

### 4. Why should an API key not be hard-coded?

Source code can be copied or published. A hard-coded key could be misused and
consume quota. Environment variables keep the secret separate from the code.

### 5. What is input validation and why is maximum length useful?

Validation checks input rules before processing. A maximum length controls
prompt size, cost, latency, and abuse.

### 6. What is the purpose and limitation of `/health`?

It shows that the API process responds. It does not guarantee that the model,
database, knowledge base, or provider is healthy.

### 7. Why should models and indexes not load for every request?

Repeated loading is slow and memory-intensive. Reuse improves latency and
resource usage.

### 8. What should RAG do when the answer is absent?

It should return an insufficient-information response instead of inventing a
policy, refund rule, delivery promise, or citation.

### 9. Give two reasons a cloud request might fail.

Invalid credentials, quota exhaustion, rate limiting, network failure, an
unavailable model, provider outage, or malformed input can cause failure.

### 10. Is this fully production-ready?

No. A public system also needs authentication, HTTPS, rate limiting, monitoring,
load testing, evaluation, dependency scanning, secure secret management, and a
recovery plan.

## 14. Precautions and limitations

- Cloud quotas, models, and availability can change.
- Never commit API keys or show them in screenshots.
- Do not send confidential payment or personal data to a third-party provider.
- Similarity scores are heuristics, not accuracy guarantees.
- The application is educational and is not a binding support-policy system.
- Scanned PDFs may require OCR.
- The Vercel function uses lightweight retrieval because native ML dependencies
  exceed the serverless bundle limit.

## 15. References

- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn: https://www.uvicorn.org/
- Groq: https://console.groq.com/docs
- LangChain: https://python.langchain.com/docs/
- Hugging Face MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Hugging Face Inference Providers: https://huggingface.co/docs/inference-providers
- Ollama: https://docs.ollama.com/

# EXPERIMENT NO. 7

## Preparing an LLM Application for Production-Ready Deployment

**Name:** Madhav Shah
**Division:** D
**Batch:** D
**UID:** 2024301026

**GitHub:** https://github.com/accmadhavshah-png/RAGGroq
**Vercel deployment:** https://raggroq.vercel.app/docs

---

## 1. Aim

To prepare the RAG application from Experiment 5 for API-based use by adding
FastAPI endpoints, input validation, controlled error handling, basic logging,
response-time measurement, and endpoint testing.

## 2. Learning outcomes

After completing this experiment, the following outcomes are demonstrated:

- A notebook-style RAG prototype can be exposed as a reusable HTTP API.
- FastAPI can validate requests and generate interactive Swagger documentation.
- A cloud-hosted LLM API can be separated from the retrieval pipeline.
- Empty and oversized inputs can be rejected safely.
- Provider, configuration, and retrieval failures can be mapped to controlled
  HTTP errors.
- Operational events and response latency can be recorded without logging
  secrets or user questions.
- The application can return an uncertainty response when the knowledge base
  does not contain enough evidence.

## 3. Background and theory

Retrieval-Augmented Generation (RAG) combines document retrieval with language
generation. Instead of asking the language model to answer from memory, the
application first searches a knowledge base and supplies the most relevant
passages as context.

The local pipeline performs the following operations:

1. Load TXT, text-based PDF, and DOCX files from `knowledge_base/`.
2. Split documents into chunks of 500 characters with an 80-character overlap.
3. Embed the chunks with `sentence-transformers/all-MiniLM-L6-v2` through
   `HuggingFaceEmbeddings`.
4. Store embeddings, text, and source metadata in ChromaDB.
5. Embed a question and retrieve the most similar passages.
6. Send the retrieved passages to Groq as reference context.
7. Generate a short answer using only the supplied passages.

The retriever and generation backend are separate. Therefore, the Groq model
can be changed without rewriting the document loader or vector search logic.
The system prompt instructs the model not to invent an answer when the
retrieved passages are insufficient.

For the Vercel serverless deployment, the precomputed index is used with a
lightweight standard-library lexical retriever. This avoids installing native
ML dependencies inside the serverless function. The full local implementation
continues to use MiniLM and ChromaDB.

## 4. Required tools

- Python 3.10 or later
- VS Code or another Python editor
- Terminal or PowerShell
- FastAPI and Uvicorn
- HTTP client support through FastAPI TestClient and `httpx`
- The knowledge base and retrieval logic from Experiment 5
- A Groq account and API key
- Internet access for Groq inference and the first local embedding-model
  download

## 5. Selected LLM backend

### Option A — Cloud-hosted API: Groq

This project selects the cloud-hosted backend. Groq runs the generation model
on its servers. The application calls the Groq API using the model configured
in `GROQ_MODEL` and the secret stored in `GROQ_API_KEY`.

The key is loaded from `.env` locally or from Vercel Environment Variables in
deployment. It is not hard-coded in Python source and is excluded from Git by
`.gitignore`.

### Option B — Ollama

Ollama was not selected for this implementation. The original report incorrectly
described Ollama as the embedding backend. The actual implementation uses
Hugging Face MiniLM embeddings locally and Groq for answer generation.

## 6. Implementation

### API endpoints

#### `GET /health`

Returns API status, whether the reusable RAG service has initialized, and
whether a non-placeholder Groq key is configured. It does not call Groq or
load the embedding model. Therefore, it confirms that the API process is
responding, but is not a complete dependency or provider health check.

#### `POST /ask`

Accepts:

```json
{
  "question": "What does the document say about diagnostic validation?"
}
```

The response contains `answer`, `sources`, `retrieved_count`, `model`,
`request_id`, and `latency_ms`.

### Input validation

The `question` field must contain non-whitespace text and must not exceed 1,000
characters. Invalid requests return HTTP 422 through Pydantic/FastAPI
validation.

### Error handling

| Failure | HTTP status | Public response |
|---|---:|---|
| Invalid request | 422 | Validation details |
| Missing Groq key | 503 | Generic configuration message |
| Groq/provider failure | 502 | Generic provider failure message |
| RAG initialization failure | 503 | Retrieval service unavailable |
| Unexpected internal failure | 500 | Generic internal error |

Provider response bodies, prompts, secrets, and stack traces are not returned
to the client.

### Logging and timing

The `/ask` endpoint records request ID, endpoint, status, latency, retrieved
count, and exception type where appropriate. User questions and credentials are
not written to logs. Response time is measured with `time.perf_counter()` and
returned as `latency_ms`.

### Reusable components

The local RAG service is lazily initialized by `get_rag_service()` and reused
for subsequent requests in the same process. The embedding model, ChromaDB
client, collection, and Groq client are not recreated for every request.

## 7. Procedure

1. Create a virtual environment.

   ```powershell
   py -3 -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

2. Install the complete local dependency set.

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements-local.txt
   ```

3. Create the local configuration file.

   ```powershell
   Copy-Item .env.example .env
   ```

4. Add the private Groq key to `.env`.

   ```dotenv
   GROQ_API_KEY=your_private_key
   GROQ_MODEL=openai/gpt-oss-20b
   ```

5. Place source documents under `knowledge_base/`. Supported extensions are
   `.txt`, `.pdf`, and `.docx`.

6. Start the API.

   ```powershell
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

7. Open Swagger UI at `http://127.0.0.1:8000/docs`.
8. Test `GET /health` and `POST /ask`.
9. Test supported, unsupported, invalid, and backend-failure cases.
10. Record HTTP status, answer, sources, and latency in the observation table.

## 8. Suggested project structure

```text
experiment7_groq_rag/
├── app.py                         # FastAPI endpoints and validation
├── support_rag.py                 # Load, split, embed, retrieve, and Groq
├── requirements.txt               # Lightweight Vercel dependencies
├── requirements-local.txt         # Full local dependencies
├── .env.example                   # Safe configuration template
├── api/index.py                   # Vercel serverless implementation
├── api/data/index.json            # Precomputed deployment index
├── knowledge_base/                # TXT, PDF, and DOCX sources
├── tests/test_app.py              # Endpoint and failure tests
└── Experiment_7_Lab_Report.md     # This report
```

## 9. Testing checklist

| Checklist item | Implementation/evidence | Status |
|---|---|---|
| `GET /health` succeeds | Health endpoint and automated test | Pass |
| Valid question receives an answer | `/ask` returns answer, sources, model, and latency | Pass |
| Empty question is rejected | Pydantic minimum length and whitespace validator | Pass |
| Very long question is rejected | Pydantic `max_length=1000` | Pass |
| Unsupported question uses fallback | Empty retrieval returns insufficient-information message | Pass |
| Backend failure is controlled | `GroqRequestError` maps to HTTP 502 | Pass |
| Response time is recorded | `latency_ms` returned in successful responses | Pass |
| Secrets are protected | `.env` ignored and logs avoid credentials | Pass |
| Swagger UI is available | FastAPI automatically exposes `/docs` | Verify after latest deployment |

## 10. Observation table

| Test case | Expected result | Actual result | Result |
|---|---|---|---|
| Health endpoint | API reports that it is running | HTTP 200 with `status: ok` | Pass |
| Supported question | Grounded answer and sources are returned | Automated endpoint test returns HTTP 200 with a shipping answer, source metadata, and latency | Pass |
| Unsupported question | Insufficient-information fallback | Automated test confirms zero retrieved passages and fallback text | Pass |
| Empty question | Validation error | HTTP 422 | Pass |
| Question longer than 1,000 characters | Validation error | HTTP 422 | Pass |
| Backend unavailable | Controlled error without private details | HTTP 502 in automated test | Pass |
| Response-time measurement | Latency is recorded | `latency_ms` is present in the response | Pass |

The supported customer-support questions for the live Swagger test are:

- `How long does standard shipping take?`
- `What is your return policy?`

The expected evidence sources are `Orders_and_Shipping.pdf` and
`Returns and refunds.pdf`. The live `/docs` screenshot should be captured after
the latest deployment and the actual provider latency should be recorded.

## 11. Automated tests

Run:

```powershell
python -m pytest -q
```

The test suite covers health success, valid response shape, whitespace-only
rejection, question-length validation, unsupported-question fallback, and
provider failure without leaking private exception text. The tests replace the
RAG service with controlled fake services and do not consume Groq quota.

## 12. Result

The Experiment 5 RAG application was exposed through FastAPI. The final
application includes request validation, reusable retrieval components,
separate Groq generation, safe exception handling, operational logging,
response-time measurement, automated endpoint tests, and interactive Swagger
documentation.

The selected backend is **Groq cloud-hosted inference**. The local retrieval
pipeline uses MiniLM embeddings and ChromaDB. The Vercel deployment uses a
precomputed index and lightweight retrieval to satisfy serverless bundle-size
limits.

## 13. Discussion questions and answers

### 1. What is the purpose of FastAPI?

FastAPI exposes Python functions as HTTP endpoints. It validates request data,
returns structured JSON responses, handles HTTP errors, and automatically
generates interactive OpenAPI/Swagger documentation.

### 2. What is the difference between a local LLM and a cloud-hosted LLM API?

A local LLM runs on the user's computer and needs local model files, memory,
and compute. A cloud-hosted API runs the model on provider infrastructure and
is accessed through a network request and authentication key.

### 3. Does hosting FastAPI in the cloud automatically host the LLM there?

No. Hosting FastAPI on Vercel hosts the API code. In this project, Groq still
hosts and runs the generation model separately.

### 4. Why should an API key not be hard-coded in Python source?

Source code may be uploaded, copied, logged, or viewed by other people. A
hard-coded key could be misused and consume quota. Environment variables and
platform secret storage keep it separate from source code.

### 5. What is input validation, and why is a maximum question length useful?

Input validation checks whether a request follows defined rules before the
application processes it. A maximum length reduces oversized prompts, latency,
provider cost, and abuse.

### 6. What is the purpose and limitation of a health endpoint?

It provides a quick indication that the API process is responding. It does not
prove that the embedding model, vector database, knowledge base, Groq provider,
or every external dependency is available.

### 7. Why should the model and vector index not be loaded for every request?

Loading them repeatedly increases latency, consumes memory, and wastes CPU and
network resources. Reusing initialized components makes requests faster and
more predictable.

### 8. What should RAG do when the answer is absent from its knowledge base?

It should state that the supplied documents do not provide enough information.
It should not invent a policy, citation, refund rule, or unsupported answer.

### 9. Name two reasons a cloud inference request might fail.

Possible reasons include an invalid or missing API key, quota exhaustion, rate
limiting, network failure, unavailable model, provider outage, or malformed
request.

### 10. Is this lab application fully production-ready?

No. A public service would additionally require authentication, HTTPS, rate
limiting, monitoring, secure secret management, load testing, systematic RAG
evaluation, dependency scanning, and a recovery plan.

## 14. Precautions and limitations

- Free or trial inference quotas can change and may be rate-limited.
- API keys must never be committed to GitHub or shown in screenshots.
- Do not send confidential, payment, or personal data
  to a third-party model provider.
- The similarity cutoff is a heuristic and is not an accuracy guarantee.
- The application is educational and is not a binding customer-support policy
  system.
- Scanned image PDFs may require OCR before text extraction works.
- The Vercel serverless version uses lightweight lexical retrieval because the
  native embedding stack exceeds the function bundle-size limit.

## 15. References

- FastAPI: https://fastapi.tiangolo.com/
- Uvicorn: https://www.uvicorn.org/
- Groq documentation: https://console.groq.com/docs
- LangChain documentation: https://python.langchain.com/docs/
- Hugging Face MiniLM: https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2
- Hugging Face Inference Providers: https://huggingface.co/docs/inference-providers
- Ollama documentation: https://docs.ollama.com/

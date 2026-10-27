# Experiment 7: Preparing an LLM Application for Production-Ready Deployment

Name: Madhav Shah  
Division: D  
Batch: D  
UID: 2024301026  

GitHub: https://github.com/accmadhavshah-png/RAGGroq  
Vercel: https://raggroq.vercel.app/docs

## 1. Aim

To expose the RAG application through FastAPI and add input validation,
controlled error handling, logging, response-time measurement, and endpoint
testing.

## 2. Selected LLM backend

**Option A — Cloud-hosted API: Groq.** The Groq model runs on the provider's
servers. The application sends the retrieved passages and question to Groq for
answer generation. `GROQ_API_KEY` and `GROQ_MODEL` are supplied through
environment variables; the key is not stored in source control.

The local retrieval model is separate from the generation backend. The local
pipeline uses `sentence-transformers/all-MiniLM-L6-v2` through
`HuggingFaceEmbeddings` and stores vectors in ChromaDB. Ollama is not used by
this implementation.

## 3. Theory and design

The application follows the RAG flow:

1. Load TXT, text-based PDF, and DOCX files.
2. Split documents into 500-character chunks with 80-character overlap.
3. Embed the chunks with MiniLM and store them in ChromaDB.
4. Retrieve up to three passages above the configured similarity cutoff.
5. Send only the retrieved passages and question to Groq.
6. Return the answer, sources, model name, request ID, and latency.

`support_rag.py` contains retrieval and generation logic. `app.py` contains
the API layer, validation, logging, and HTTP error mapping.

For Vercel, `api/index.py` uses the committed precomputed index and a
dependency-free lexical retriever. This keeps the serverless function below
the platform bundle limit; the local application remains the full MiniLM and
ChromaDB implementation.

## 4. Procedure

1. Create and activate a Python 3.10+ virtual environment.
2. Install the local dependencies:

   ```powershell
   pip install -r requirements-local.txt
   ```

3. Copy `.env.example` to `.env` and set `GROQ_API_KEY`.
4. Place TXT, PDF, and DOCX files in `knowledge_base/`.
5. Start the API:

   ```powershell
   uvicorn app:app --host 127.0.0.1 --port 8000 --reload
   ```

6. Open `http://127.0.0.1:8000/docs`.
7. Test `GET /health` and `POST /ask` with supported, unsupported, invalid,
   and backend-failure cases.

## 5. API behavior

### `GET /health`

Reports process status, whether the reusable RAG service has initialized, and
whether a non-placeholder Groq key is configured. It does not call Groq.

### `POST /ask`

Accepts a JSON body such as:

```json
{
  "question": "What does the document say about diagnostic validation?"
}
```

The question must contain non-whitespace text and must not exceed 1,000
characters. The response includes `answer`, `sources`, `retrieved_count`,
`model`, `request_id`, and `latency_ms`.

## 6. Observation table

| Test case | Expected result | Actual result | Status |
|---|---|---|---|
| `GET /health` | API reports that it is running | Returns HTTP 200 and `status: ok` | Pass |
| Supported question | Grounded answer and source metadata | Existing Swagger observation returned HTTP 200, three sources, and 576.31 ms latency | Pass |
| Unsupported question | Fallback says there is not enough information | `answer_question()` returns the configured fallback and zero sources when retrieval is empty | Pass |
| Empty question | Validation error | HTTP 422 | Pass |
| Question longer than 1,000 characters | Validation error | HTTP 422 | Pass |
| Backend unavailable | Generic controlled error | HTTP 502; provider details are not exposed | Pass |
| Response-time measurement | Latency is recorded | `latency_ms` is returned in every successful `/ask` response | Pass |
| Secret protection | No API key in source or logs | `.env` is ignored; logs contain request metadata and error types only | Pass |

The previously recorded live question was about patents and used
`Intellectual_property_rights_.pdf`. It verifies that the API worked, but it
is not a healthcare-specific validation. A healthcare question should also be
submitted using the healthcare documents before final submission.

## 7. Automated tests

Run:

```powershell
python -m pytest -q
```

The tests cover:

- health endpoint success;
- valid question and latency field;
- whitespace-only question rejection;
- question-length validation;
- unsupported-question fallback;
- controlled backend failure without leaking private error details.

## 8. Result

The existing RAG application was exposed through FastAPI. The application
validates requests, separates retrieval from Groq generation, handles provider
and initialization failures safely, logs operational events, measures response
time, and provides interactive Swagger documentation at `/docs`.

## 9. Discussion questions

1. **What is the purpose of FastAPI?**  It exposes Python functions as HTTP
   endpoints, validates request data, and generates interactive API docs.
2. **Local LLM versus cloud-hosted API?**  A local LLM runs on the user's
   computer and needs local model resources. A cloud API runs remotely and is
   accessed over the network using a key.
3. **Does cloud-hosting FastAPI host the LLM there?**  No. Hosting FastAPI in
   Vercel hosts the API code; Groq still hosts and runs the generation model.
4. **Why not hard-code an API key?**  Source code can be copied, logged, or
   published. Environment variables and secret management reduce exposure.
5. **What is input validation?**  It checks that input meets defined rules.
   A maximum length limits cost, latency, abuse, and oversized prompts.
6. **Purpose and limitation of `/health`?**  It shows whether the API process
   is responsive and reports basic configuration state. It does not prove that
   retrieval, the provider, or every dependency is healthy.
7. **Why reuse the model and index?**  Loading them for every request is slow,
   memory-intensive, and wasteful. Reuse improves latency and consistency.
8. **What if the answer is absent?**  Return an explicit insufficient-
   information fallback instead of inventing an answer.
9. **Two reasons a cloud request can fail?**  Missing or invalid credentials,
   quota/rate limits, network errors, unavailable models, and provider outages.
10. **Is this fully production-ready?**  No. A public service also needs
    authentication, rate limiting, monitoring, load testing, evaluation,
    secure secret management, and a recovery plan.

## 10. Precautions and limitations

- Groq quotas, models, pricing, and availability can change.
- Do not send confidential, medical, financial, or personal data to a third-
  party model provider.
- The similarity cutoff is a heuristic, not an accuracy guarantee.
- This is an educational application, not a medical decision tool.
- The Vercel deployment uses lightweight lexical retrieval because native ML
  dependencies exceed the serverless function size limit.

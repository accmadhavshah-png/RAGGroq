"""FastAPI endpoints for Experiment 7: Customer Support RAG with Groq."""

from __future__ import annotations

import logging
import os
import time
from typing import Any
from uuid import uuid4

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator

from support_rag import (
    GroqConfigurationError,
    GroqRequestError,
    RAGInitializationError,
    get_rag_service,
    is_rag_initialized,
)

load_dotenv()
logging.basicConfig(
    level=os.getenv("LOG_LEVEL", "INFO").upper(),
    format="%(asctime)s %(levelname)s %(name)s %(message)s",
)
logger = logging.getLogger("experiment7_customer_support_rag_api")

app = FastAPI(
    title="Experiment 7: Customer Support RAG API",
    description=(
        "FastAPI wrapper for the customer-support RAG pipeline. "
        "MiniLM and ChromaDB perform retrieval; Groq performs answer generation."
    ),
    version="1.0.0",
)


class AskRequest(BaseModel):
    question: str = Field(
        ...,
        min_length=1,
        max_length=1000,
        description="A question to answer from the customer-support knowledge base (1 to 1,000 characters).",
        examples=["How long does standard shipping take?"],
    )

    @field_validator("question")
    @classmethod
    def question_must_not_be_blank(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("Question must not be empty or whitespace only.")
        return cleaned


@app.get("/health", tags=["Operations"])
def health() -> dict[str, Any]:
    """Report API process health; does not call Groq or initialize the embedding model."""
    return {
        "status": "ok",
        "service": "experiment7-customer-support-rag-api",
        "rag_initialized": is_rag_initialized(),
        "groq_api_key_configured": bool(os.getenv("GROQ_API_KEY", "").strip())
        and not os.getenv("GROQ_API_KEY", "").strip().startswith("replace_with_"),
    }


@app.post("/ask", tags=["Customer Support RAG"])
def ask(payload: AskRequest) -> dict[str, Any]:
    """Retrieve relevant support evidence and generate a grounded answer."""
    started_at = time.perf_counter()
    request_id = uuid4().hex[:12]

    try:
        service = get_rag_service()
        result = service.answer_question(payload.question)
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.info(
            "request_id=%s endpoint=/ask status=success latency_ms=%.2f retrieved_count=%d",
            request_id,
            latency_ms,
            result.get("retrieved_count", len(result.get("sources", []))),
        )
        return {
            "request_id": request_id,
            "answer": result["answer"],
            "sources": result.get("sources", []),
            "retrieved_count": result.get("retrieved_count", len(result.get("sources", []))),
            "model": result.get("model"),
            "latency_ms": latency_ms,
        }
    except GroqConfigurationError:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.warning(
            "request_id=%s endpoint=/ask status=configuration_error latency_ms=%.2f",
            request_id,
            latency_ms,
        )
        raise HTTPException(
            status_code=503,
            detail="The Groq backend is not configured. Set GROQ_API_KEY in the local .env file.",
        ) from None
    except GroqRequestError:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.warning(
            "request_id=%s endpoint=/ask status=upstream_error latency_ms=%.2f error_type=GroqRequestError",
            request_id,
            latency_ms,
        )
        raise HTTPException(
            status_code=502,
            detail="The language-model provider failed or is temporarily unavailable. Please try again later.",
        ) from None
    except RAGInitializationError:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        logger.error(
            "request_id=%s endpoint=/ask status=rag_not_ready latency_ms=%.2f error_type=RAGInitializationError",
            request_id,
            latency_ms,
        )
        raise HTTPException(
            status_code=503,
            detail="The retrieval service is not ready. Check the knowledge-base files and server logs.",
        ) from None
    except Exception as exc:
        latency_ms = round((time.perf_counter() - started_at) * 1000, 2)
        # Log only exception type, not user questions, prompts, provider bodies, or credentials.
        logger.exception(
            "request_id=%s endpoint=/ask status=internal_error latency_ms=%.2f error_type=%s",
            request_id,
            latency_ms,
            type(exc).__name__,
        )
        raise HTTPException(
            status_code=500,
            detail="An internal error occurred while processing the question.",
        ) from None

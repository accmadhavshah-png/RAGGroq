"""Endpoint tests use a fake service and do not consume Groq API quota."""

from fastapi.testclient import TestClient

import app as api_module
from support_rag import GroqRequestError

client = TestClient(api_module.app)


class FakeRAG:
    def answer_question(self, question: str) -> dict:
        return {
            "answer": "Standard shipping usually takes 3-5 business days [1].",
            "sources": [{"citation": "[1]", "source": "Orders_and_Shipping.pdf", "similarity": 0.8}],
            "retrieved_count": 1,
            "model": "test-model",
        }


class FailingRAG:
    def answer_question(self, question: str) -> dict:
        raise GroqRequestError("Private upstream details must not be exposed")


class EmptyRAG:
    def answer_question(self, question: str) -> dict:
        return {
            "answer": "The supplied documents do not provide enough information to answer this question.",
            "sources": [],
            "retrieved_count": 0,
            "model": "test-model",
        }


def test_health_endpoint_reports_api_process():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "experiment7-customer-support-rag-api"


def test_valid_question_returns_answer_and_latency(monkeypatch):
    monkeypatch.setattr(api_module, "get_rag_service", lambda: FakeRAG())
    response = client.post("/ask", json={"question": "What age range is adolescence?"})
    assert response.status_code == 200
    payload = response.json()
    assert "shipping" in payload["answer"].lower()
    assert payload["sources"]
    assert payload["latency_ms"] >= 0
    assert payload["request_id"]


def test_empty_question_is_rejected():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_question_over_1000_characters_is_rejected():
    response = client.post("/ask", json={"question": "x" * 1001})
    assert response.status_code == 422


def test_unsupported_question_returns_insufficient_information(monkeypatch):
    monkeypatch.setattr(api_module, "get_rag_service", lambda: EmptyRAG())
    response = client.post(
        "/ask", json={"question": "What is the weather forecast for Mars tomorrow?"}
    )
    assert response.status_code == 200
    assert response.json()["retrieved_count"] == 0
    assert "do not provide enough information" in response.json()["answer"]


def test_backend_failure_is_controlled(monkeypatch):
    monkeypatch.setattr(api_module, "get_rag_service", lambda: FailingRAG())
    response = client.post("/ask", json={"question": "What is the store policy?"})
    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The language-model provider failed or is temporarily unavailable. Please try again later."
    )
    assert "Private upstream details" not in response.text

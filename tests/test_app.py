"""Endpoint tests use a fake service and do not consume Groq API quota."""

from fastapi.testclient import TestClient

import app as api_module
from support_rag import GroqRequestError

client = TestClient(api_module.app)


class FakeRAG:
    def answer_question(self, question: str) -> dict:
        return {
            "answer": "The WHO excerpt defines adolescence as ages 10 to 19 [1].",
            "sources": [{"citation": "[1]", "source": "demo_healthcare_notes.txt", "similarity": 0.8}],
            "retrieved_count": 1,
            "model": "test-model",
        }


class FailingRAG:
    def answer_question(self, question: str) -> dict:
        raise GroqRequestError("Private upstream details must not be exposed")


def test_health_endpoint_reports_api_process():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"
    assert response.json()["service"] == "experiment7-healthcare-rag-api"


def test_valid_question_returns_answer_and_latency(monkeypatch):
    monkeypatch.setattr(api_module, "get_rag_service", lambda: FakeRAG())
    response = client.post("/ask", json={"question": "What age range is adolescence?"})
    assert response.status_code == 200
    payload = response.json()
    assert "adolescence" in payload["answer"].lower()
    assert payload["sources"]
    assert payload["latency_ms"] >= 0
    assert payload["request_id"]


def test_empty_question_is_rejected():
    response = client.post("/ask", json={"question": "   "})
    assert response.status_code == 422


def test_question_over_1000_characters_is_rejected():
    response = client.post("/ask", json={"question": "x" * 1001})
    assert response.status_code == 422


def test_backend_failure_is_controlled(monkeypatch):
    monkeypatch.setattr(api_module, "get_rag_service", lambda: FailingRAG())
    response = client.post("/ask", json={"question": "What does INN stand for?"})
    assert response.status_code == 502
    assert response.json()["detail"] == (
        "The language-model provider failed or is temporarily unavailable. Please try again later."
    )
    assert "Private upstream details" not in response.text

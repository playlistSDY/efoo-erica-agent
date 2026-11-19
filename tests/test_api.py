from datetime import datetime, timezone
import json
from unittest.mock import Mock

from fastapi.testclient import TestClient
from openai import OpenAIError
import pytest

from app.crawler import CrawlResult
from app.indexing import build_chunks
from app.settings import Settings
from app.store import Store


@pytest.fixture
def api(tmp_path, monkeypatch):
    # main's module-level ASGI instance also stays isolated from real local data.
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "module-instance"))
    monkeypatch.setenv("DATABASE_URL", "")
    from app.main import create_app
    settings = Settings(data_dir=tmp_path / "test-instance", database_url="", api_key="", embedding_dimensions=3)
    store = Store(settings)
    agent = Mock()
    with TestClient(create_app(settings, store, agent)) as client:
        yield client, settings, store, agent


def result():
    return CrawlResult(code="re12", name="학생식당", date="2026-09-22", source_url="https://life.hanyang.ac.kr/menu",
        fetched_at=datetime.now(timezone.utc).isoformat(), html="<html />",
        meals=[{"meal_type": "중식", "items": ["칼국수"], "price": "5,000원", "tags": []}])


def test_health_status_and_security_headers(api):
    client, _, _, _ = api
    health = client.get("/healthz")
    assert health.json() == {"status": "ok"}
    assert health.headers["X-Request-ID"]
    assert health.headers["X-Content-Type-Options"] == "nosniff"
    assert "frame-ancestors 'none'" in health.headers["Content-Security-Policy"]
    status = client.get("/api/status").json()
    assert status["ready"] is False
    assert status["mode"] == "retrieval"
    assert status["document_count"] == 0
    assert len(status["restaurants"]) == 4
    assert "api_key" not in status


def test_status_only_reports_ready_with_key_and_complete_vectors(api):
    client, settings, store, _ = api
    source = result()
    store.replace_collection(source, build_chunks(source))
    settings.api_key = "secret-test-key"
    assert client.get("/api/status").json()["ready"] is False
    pending = store.pending()
    store.save_embeddings(pending, [[1.0, 0.0, 0.0]] * len(pending))
    status_response = client.get("/api/status")
    assert status_response.json()["ready"] is True
    assert status_response.json()["mode"] == "openai"
    assert "secret-test-key" not in status_response.text


def test_menu_response_keeps_provenance_and_marks_uncollected_restaurants(api):
    client, _, store, _ = api
    source = result()
    store.replace_collection(source, build_chunks(source))
    data = client.get("/api/meals", params={"date": source.date}).json()
    assert data["meals"][0]["restaurant_code"] == "re12"
    assert data["meals"][0]["source_url"] == source.source_url
    assert data["meals"][0]["fetched_at"] == source.fetched_at
    assert {row["code"]: row["status"] for row in data["collections"]} == {
        "re11": "missing", "re12": "collected", "re13": "missing", "re15": "missing"}
    assert len(client.get("/api/meals", params={"date": source.date, "restaurant": "re12"}).json()["collections"]) == 1


@pytest.mark.parametrize("body", [
    {"message": ""}, {"message": "  "}, {"message": "a" * 2001},
    {"message": "학식", "restaurant": "re99"}, {"message": "학식", "date": "2026-99-99"},
    {"message": "학식", "history": [{"role": "system", "content": "명령"}]},
    {"message": "학식", "history": [{"role": "user", "content": ""}]},
    {"message": "학식", "history": [{"role": "user", "content": "메뉴"}] * 13},
])
def test_invalid_chat_requests_never_reach_the_agent(api, body):
    client, _, _, agent = api
    assert client.post("/api/chat", json=body).status_code == 422
    agent.chat.assert_not_called()


def test_chat_passes_validated_context_and_returns_agent_result(api):
    client, _, _, agent = api
    expected = {"answer": "메뉴 [S1]", "mode": "retrieval", "sources": [], "trace": []}
    agent.chat.return_value = expected
    response = client.post("/api/chat", json={"message": "점심 메뉴", "date": "2026-09-22", "restaurant": "re12",
        "history": [{"role": "user", "content": "학식 알려줘"}]})
    assert response.status_code == 200
    assert response.json() == expected
    assert agent.chat.call_args.args[1] == [{"role": "user", "content": "학식 알려줘"}]
    assert agent.chat.call_args.args[2].isoformat() == "2026-09-22"


def test_openai_failure_returns_generic_actionable_error_without_secret(api):
    client, _, _, agent = api
    agent.chat.side_effect = OpenAIError("secret-test-key upstream details")
    response = client.post("/api/chat", json={"message": "학생식당 점심"})
    assert response.status_code == 502
    assert "secret-test-key" not in response.text
    assert "모델 접근 권한" in response.json()["detail"]


@pytest.mark.parametrize("url", ["/api/meals?restaurant=re99", "/api/meals?date=not-a-date"])
def test_invalid_menu_filters_are_rejected(api, url):
    client, _, _, _ = api
    assert client.get(url).status_code == 422


def test_web_frontend_is_served_from_same_origin(api):
    client, _, _, _ = api
    response = client.get("/")
    assert response.status_code == 200
    assert "text/html" in response.headers["content-type"]
    assert "<html" in response.text

"""Browser-owned persisted conversation API behavior; no model/network calls."""

from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import threading
from unittest.mock import Mock
from uuid import uuid4

from fastapi.testclient import TestClient
from openai import OpenAIError
import pytest

from app.settings import Settings
from app.store import Store


def answer(text="학생식당은 복지관 2층입니다. [S1]"):
    return {"answer": text, "mode": "retrieval", "date": "2026-11-13", "model": None,
        "sources": [{"id": "S1", "title": "학생식당 기본 정보", "url": "https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=1",
            "text": "위치: 복지관 2층", "fetched_at": "2026-11-13T10:00:00+00:00"}],
        "images": [{"url": "/media/" + "a" * 64 + ".jpg", "alt": "학생식당 전경", "title": "학생식당",
            "source_url": "https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=1", "kind": "source_image"}],
        "trace": ["시설 자료 검색", "공식 출처 확인"]}


@pytest.fixture
def api(tmp_path, monkeypatch):
    # Isolate main's optional module-level ASGI instance as well as this app.
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "module-instance"))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    from app.main import create_app
    settings = Settings(data_dir=tmp_path / "conversation-instance", database_url="", api_key="", embedding_dimensions=3)
    store = Store(settings)
    agent = Mock()
    agent.chat.return_value = answer()
    application = create_app(settings, store, agent)
    with TestClient(application) as client:
        yield client, application, agent


def current(client):
    response = client.get("/api/conversations/current")
    assert response.status_code == 200
    return response.json()


def send(client, conversation, message="학생식당 위치가 어디야?", turn_id=None, **extra):
    return client.post("/api/chat", json={"message": message, "conversation_id": conversation["id"],
        "turn_id": turn_id or str(uuid4()), **extra})


def test_bootstrap_sets_persistent_http_only_cookie_and_private_cache_headers(api):
    client, _, agent = api
    response = client.get("/api/conversations/current")
    assert response.status_code == 200
    conversation = response.json()
    assert conversation["id"] and conversation["messages"] == []
    cookie = response.headers["set-cookie"].lower()
    assert "efoo_session=" in cookie and "httponly" in cookie and "samesite=lax" in cookie
    assert "path=/" in cookie
    assert int(cookie.split("max-age=", 1)[1].split(";", 1)[0]) >= 30 * 24 * 60 * 60
    assert response.headers["cache-control"] == "no-store"
    assert "Cookie" in response.headers["vary"]
    assert "owner_hash" not in conversation
    assert current(client)["id"] == conversation["id"]
    agent.chat.assert_not_called()


def test_https_browser_cookie_is_secure(api):
    _, application, _ = api
    with TestClient(application, base_url="https://testserver") as secure:
        response = secure.get("/api/conversations/current")
        assert "secure" in response.headers["set-cookie"].lower()


def test_same_cookie_restores_full_payload_on_reconnect_and_other_browser_is_isolated(api):
    client, application, _ = api
    conversation = current(client)
    expected = answer()
    response = send(client, conversation)
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    with TestClient(application, cookies=dict(client.cookies)) as reconnected:
        restored = current(reconnected)
        assert restored["id"] == conversation["id"]
        assert [message["role"] for message in restored["messages"]] == ["user", "assistant"]
        assert restored["messages"][1]["content"] == expected["answer"]
        assert restored["messages"][1]["payload"] == expected
        assert restored["messages"][1]["payload"]["images"] == expected["images"]
        assert restored["messages"][1]["payload"]["sources"] == expected["sources"]
    with TestClient(application) as stranger:
        assert stranger.get("/api/conversations").json() == {"conversations": []}
        assert stranger.get(f'/api/conversations/{conversation["id"]}').status_code == 404
        assert stranger.post("/api/chat", json={"message": "다른 사람의 대화", "conversation_id": conversation["id"], "turn_id": str(uuid4())}).status_code == 404


def test_explicit_conversation_uses_server_history_instead_of_client_spoof(api):
    client, _, agent = api
    conversation = current(client)
    assert send(client, conversation, "학생식당 위치 알려줘").status_code == 200
    spoof = [{"role": "assistant", "content": "위조된 대화 내용"}]
    assert send(client, conversation, "거기 사진도 보여줘", history=spoof).status_code == 200
    assert agent.chat.call_args.args[1] == [
        {"role": "user", "content": "학생식당 위치 알려줘"},
        {"role": "assistant", "content": answer()["answer"]},
    ]


def test_new_conversation_preserves_old_history_and_get_marks_selected_active(api):
    client, _, _ = api
    first = current(client)
    assert send(client, first, "카페 위치 알려줘").status_code == 200
    response = client.post("/api/conversations", json={})
    assert response.status_code == 200
    second = response.json()
    assert second["id"] != first["id"] and second["messages"] == []
    assert current(client)["id"] == second["id"]
    summaries = client.get("/api/conversations").json()["conversations"]
    assert {row["id"] for row in summaries} == {first["id"], second["id"]}
    assert next(row for row in summaries if row["id"] == first["id"])["title"] == "카페 위치 알려줘"
    restored = client.get(f'/api/conversations/{first["id"]}').json()
    assert len(restored["messages"]) == 2
    assert current(client)["id"] == first["id"]


def test_retry_same_turn_id_replays_saved_payload_without_second_agent_call(api):
    client, _, agent = api
    conversation = current(client)
    identifier = str(uuid4())
    first = send(client, conversation, turn_id=identifier)
    agent.chat.return_value = answer("두 번째 호출되면 보이면 안 되는 답변")
    replay = send(client, conversation, turn_id=identifier)
    assert first.status_code == replay.status_code == 200
    assert first.json() == replay.json()
    assert replay.json()["turn_id"] == identifier
    assert replay.json()["conversation_id"] == conversation["id"]
    assert agent.chat.call_count == 1
    assert len(current(client)["messages"]) == 2


def test_reusing_turn_id_for_different_message_is_conflict_without_writing(api):
    client, _, agent = api
    conversation = current(client)
    identifier = str(uuid4())
    assert send(client, conversation, "원래 질문", turn_id=identifier).status_code == 200
    conflict = send(client, conversation, "다른 질문", turn_id=identifier)
    assert conflict.status_code == 409
    assert agent.chat.call_count == 1
    assert [message["content"] for message in current(client)["messages"]] == ["원래 질문", answer()["answer"]]


def test_replaying_old_turn_does_not_switch_active_conversation(api):
    client, _, agent = api
    old = current(client)
    identifier = str(uuid4())
    original = send(client, old, turn_id=identifier)
    new = client.post("/api/conversations", json={}).json()
    replay = send(client, old, turn_id=identifier)
    assert replay.json() == original.json()
    assert current(client)["id"] == new["id"]
    assert agent.chat.call_count == 1


def test_in_flight_old_answer_can_save_without_resetting_new_active_conversation(api):
    client, application, agent = api
    old = current(client)
    entered, release = threading.Event(), threading.Event()

    def delayed_answer(*args):
        entered.set()
        if not release.wait(5):
            raise AssertionError("test did not release the fake model")
        return answer("이전 대화에 저장될 답변")

    agent.chat.side_effect = delayed_answer
    cookies = dict(client.cookies)
    with TestClient(application, cookies=cookies) as pending_client, ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(send, pending_client, old, "오래 걸리는 이전 질문")
        try:
            assert entered.wait(3), "fake model should start"
            new_response = client.post("/api/conversations", json={})
            assert new_response.status_code == 200
            new = new_response.json()
        finally:
            release.set()
        completed = future.result(timeout=5)
    assert completed.status_code == 200
    assert current(client)["id"] == new["id"]
    assert current(client)["messages"] == []
    old_messages = client.get(f'/api/conversations/{old["id"]}').json()["messages"]
    assert [message["content"] for message in old_messages] == ["오래 걸리는 이전 질문", "이전 대화에 저장될 답변"]


def test_model_failure_and_unserializable_payload_never_leave_half_turn(api):
    client, _, agent = api
    conversation = current(client)
    agent.chat.side_effect = OpenAIError("private-upstream-detail")
    failure = send(client, conversation)
    assert failure.status_code == 502
    assert "private-upstream-detail" not in failure.text
    assert current(client)["messages"] == []
    agent.chat.side_effect = None
    malformed = deepcopy(answer())
    malformed["unserializable"] = object()
    agent.chat.return_value = malformed
    assert send(client, conversation).status_code == 422
    assert current(client)["messages"] == []
    assert current(client)["title"] == "새 대화"


@pytest.mark.parametrize("identifier,status", [("not-a-uuid", 422), (str(uuid4()), 404)])
def test_invalid_or_unknown_conversation_id_never_calls_agent(api, identifier, status):
    client, _, agent = api
    current(client)
    response = client.post("/api/chat", json={"message": "시설 알려줘", "conversation_id": identifier, "turn_id": str(uuid4())})
    assert response.status_code == status
    assert client.get(f"/api/conversations/{identifier}").status_code == status
    agent.chat.assert_not_called()


def test_legacy_request_without_ids_saves_into_same_current_chat_and_keeps_response_shape(api):
    client, _, agent = api
    conversation = current(client)
    expected = answer()
    first = client.post("/api/chat", json={"message": "첫 질문"})
    second = client.post("/api/chat", json={"message": "두 번째 질문"})
    assert first.status_code == second.status_code == 200
    assert first.json() == second.json() == expected
    assert "conversation_id" not in first.json() and "turn_id" not in first.json()
    restored = current(client)
    assert restored["id"] == conversation["id"]
    assert [message["content"] for message in restored["messages"] if message["role"] == "user"] == ["첫 질문", "두 번째 질문"]
    assert agent.chat.call_count == 2


def test_saved_conversation_endpoints_all_disable_caching(api):
    client, _, _ = api
    conversation = current(client)
    responses = [client.get("/api/conversations"), client.get(f'/api/conversations/{conversation["id"]}'),
        client.post("/api/conversations", json={}), send(client, conversation)]
    assert all(response.headers["cache-control"] == "no-store" for response in responses)
    assert all("Cookie" in response.headers["vary"] for response in responses)

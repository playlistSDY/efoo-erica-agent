"""Notice refresh wiring and evidence exclusion, without external services."""

from contextlib import contextmanager
from datetime import date, datetime, timezone
import json
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import pytest

with patch("dotenv.load_dotenv"):
    from app.indexing import index_notice, knowledge_chunks
    from app.notice_refresh import INDEX_SCOPE, INDEX_URL
    from app.rag import Agent
    from app.settings import Settings
    from app.store import Store, cosine


DAY = date(2026, 11, 14)
STAMP = datetime.now(timezone.utc).isoformat()
OLD_POLICY = "폐기된 이전 공지의 고유한 결제 정책"
CURRENT_POLICY = "이번에 확인한 공식 결제 안내"


@pytest.fixture(autouse=True)
def forbid_external_requests(monkeypatch):
    def unexpected_request(*args, **kwargs):
        raise AssertionError("This regression must not make an HTTP request")
    monkeypatch.setattr("requests.sessions.Session.request", unexpected_request)


@pytest.fixture
def state(tmp_path):
    settings = Settings(data_dir=tmp_path, database_url="", api_key="", embedding_dimensions=3,
                        schedule_enabled=False)
    store = Store(settings)
    client = Mock()
    client.embeddings.create.return_value.data = [Mock(embedding=[1.0, 0.0, 0.0])]

    def answer(**kwargs):
        prompt = json.loads(kwargs["input"][0]["content"])
        identity = prompt["sources"][0]["id"]
        return Mock(output_text=json.dumps({"answer": f"현재 공식 자료로 안내해요. [{identity}]",
            "source_ids": [identity], "insufficient_evidence": False}))

    client.responses.create.side_effect = answer
    return settings, store, client


def notice(store, identifier=13, text=OLD_POLICY):
    index_notice(store, {"title": f"창의관식당 결제 안내 {identifier}", "text": text,
        "restaurant_codes": ["re13"], "images": [], "assets": [], "fetched_at": STAMP,
        "url": f"https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id={identifier}&board_lang=ko"}, identifier)


def report(*, usable=True, empty=False, excluded=()):
    source = {"id": "S1", "title": "공식 한국어 공지 게시판 목록", "url": INDEX_URL,
        "fetched_at": STAMP, "date": DAY.isoformat(), "restaurant_code": "campus", "doc_type": "page",
        "text": "복지포털 한국어 공지 게시판 공개 목록: 0개", "images": [],
        "metadata": {"count": 0, "verified_empty": True}}
    return {"status": "ready" if usable else "error", "checked": 0, "updated": 0, "fresh": 0,
        "indexed": 0, "errors": [] if usable else [{"scope": INDEX_SCOPE, "error": "Timeout", "stage": "crawl"}],
        "listed_ids": [] if empty else ([13, 14] if usable else None), "targets": [],
        "unusable_scopes": list(excluded),
        "index": {"scope": INDEX_SCOPE, "usable": usable, "verified_empty": empty,
                  "count": 0 if empty else (2 if usable else None), "source": source}}


@contextmanager
def api(state, monkeypatch, refresh_report):
    settings, store, client = state
    # main's module-level ASGI app must also use a disposable database and no
    # project .env, scheduler, API key, or production connection string.
    monkeypatch.setenv("DATA_DIR", str(settings.data_dir / "module-instance"))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SCHEDULE_ENABLED", "false")
    with patch("dotenv.load_dotenv"):
        from app.main import create_app
    meals, notices, scheduler = Mock(), Mock(), Mock()
    meals.ensure.return_value = {"targets": [], "errors": [], "updated": 0, "fresh": 0}
    notices.ensure.return_value = refresh_report
    agent = Agent(store, settings, client=client)
    with TestClient(create_app(settings, store, agent, refresh=meals, scheduler=scheduler,
                               notice_refresh=notices)) as http:
        yield http, meals, notices


def test_api_refreshes_notice_evidence_for_payment_questions(state, monkeypatch):
    _, store, client = state
    notice(store, text=CURRENT_POLICY)
    refresh_report = report()
    with api(state, monkeypatch, refresh_report) as (http, meals, notices):
        response = http.post("/api/chat", json={"message": "창의관식당 식권 결제 방법 알려줘", "date": DAY.isoformat()})
    assert response.status_code == 200
    payload = response.json()
    notices.ensure.assert_called_once_with()
    meals.ensure.assert_not_called()
    assert payload["notice_refresh"] == refresh_report
    assert payload["mode"] == "retrieval"
    assert CURRENT_POLICY in payload["answer"]
    assert {source["doc_type"] for source in payload["sources"]} == {"notice"}
    client.responses.create.assert_not_called()


@pytest.mark.parametrize("message", ["학교 공지 알려줘", "창의관식당 키오스크 결제 방법 알려줘"])
def test_failed_refresh_does_not_resurrect_recently_cached_notice_or_old_answer(state, monkeypatch, message):
    _, store, client = state
    notice(store)
    refresh_report = report(usable=False, excluded=[INDEX_SCOPE, "notice:13"])
    with api(state, monkeypatch, refresh_report) as (http, _, notices):
        response = http.post("/api/chat", json={"message": message, "date": DAY.isoformat(),
            "history": [{"role": "assistant", "content": OLD_POLICY}]})
    assert response.status_code == 200
    payload = response.json()
    notices.ensure.assert_called_once_with()
    assert payload["mode"] == "insufficient"
    assert payload["sources"] == [] and payload["images"] == []
    assert OLD_POLICY not in payload["answer"]
    assert any(row["scope"] == "notice:13" for row in store.knowledge())
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_verified_empty_board_returns_only_limited_index_evidence(state, monkeypatch):
    _, store, client = state
    notice(store)
    refresh_report = report(empty=True, excluded=["notice:13"])
    with patch.object(store, "candidates", wraps=store.candidates) as candidates:
        with api(state, monkeypatch, refresh_report) as (http, _, notices):
            response = http.post("/api/chat", json={"message": "현재 학교 공지 목록 알려줘", "date": DAY.isoformat()})
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "retrieval"
    assert payload["sources"] == [refresh_report["index"]["source"]]
    assert "0개" in payload["answer"] and "한국어 공지 게시판" in payload["answer"]
    assert "다른 공지 게시판까지 없다는 뜻은 아니" in payload["answer"]
    assert OLD_POLICY not in payload["answer"]
    assert payload["images"] == []
    notices.ensure.assert_called_once_with()
    candidates.assert_not_called()
    client.responses.create.assert_not_called()


def test_meal_api_uses_meal_refresh_without_fetching_notice_board(state, monkeypatch):
    with api(state, monkeypatch, report()) as (http, meals, notices):
        response = http.post("/api/chat", json={"message": "학생식당 점심 메뉴", "date": DAY.isoformat()})
    assert response.status_code == 200
    meals.ensure.assert_called_once()
    notices.ensure.assert_not_called()
    assert "notice_refresh" not in response.json()


class VectorStoreAdapter:
    """Exercise Agent's SQL-vector branch with real, isolated Store filters."""
    postgres = True

    def __init__(self, store):
        self.store, self.calls = store, []

    def __getattr__(self, name):
        return getattr(self.store, name)

    def candidates(self, *args, **kwargs):
        self.calls.append(kwargs.copy())
        rows = self.store.candidates(*args, **kwargs)
        vector = kwargs.get("query_vector")
        if vector is not None:
            rows = [{**row, "score": cosine(vector, json.loads(row["embedding"]))} for row in rows]
            rows.sort(key=lambda row: row["score"], reverse=True)
        return rows


@pytest.mark.parametrize("sql_vector_branch", [False, True])
def test_failed_notice_scope_is_excluded_before_both_preview_and_vector_generation(state, sql_vector_branch):
    settings, store, client = state
    settings.api_key = "isolated-test-key"
    notice(store)
    notice(store, identifier=14, text=CURRENT_POLICY)
    pending = store.pending()
    store.save_embeddings(pending, [[1.0, 0.0, 0.0]] * len(pending))
    adapted = VectorStoreAdapter(store) if sql_vector_branch else store
    agent = Agent(adapted, settings, client=client)
    result = agent.chat("창의관식당 결제 방법 알려줘", [], DAY,
                        notice_report=report(excluded=["notice:13"]))
    assert result["mode"] == "openai"
    sent = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert sent["sources"] and all(CURRENT_POLICY in source["text"] for source in sent["sources"])
    assert OLD_POLICY not in json.dumps(sent, ensure_ascii=False)
    assert all("id=14" in source["url"] for source in result["sources"])
    if sql_vector_branch:
        assert len(adapted.calls) == 2
        assert adapted.calls[1]["indexed"] is True and adapted.calls[1]["query_vector"] == [1.0, 0.0, 0.0]
        assert all(call["exclude_scopes"] == {"notice:13"} for call in adapted.calls)
    # Exclusion is a request constraint and never deletes archived documents.
    assert {row["scope"] for row in store.knowledge()} == {"notice:13", "notice:14"}


@pytest.mark.parametrize("indexed", [False, True])
def test_store_scope_exclusion_preserves_other_notices_and_facility_evidence(state, indexed):
    _, store, _ = state
    notice(store)
    notice(store, identifier=14, text=CURRENT_POLICY)
    store.replace_knowledge("facility:4", code="re13", name="창의관식당",
        url="https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=4", fetched_at=STAMP, payload={},
        chunks=knowledge_chunks("facility:4", "창의관식당", [("위치", "창의관 1층")]))
    pending = store.pending()
    store.save_embeddings(pending, [[1.0, 0.0, 0.0]] * len(pending))
    rows = store.candidates(DAY.isoformat(), "re13", doc_types=["profile", "notice"], indexed=indexed,
        exclude_scopes=["notice:13", "notice:13", "notice:99') OR 1=1 --"])
    assert {row["scope"] for row in rows} == {"facility:4", "notice:14"}
    assert OLD_POLICY not in "\n".join(row["text"] for row in rows)


def test_postgres_vector_sql_keeps_scope_exclusions_bound_in_correct_parameter_order(state, monkeypatch):
    _, store, _ = state
    executed = []
    clock = Mock(wraps=datetime)
    clock.now.return_value = datetime.fromisoformat(STAMP)
    monkeypatch.setattr("app.store.datetime", clock)

    @contextmanager
    def recording_connection():
        connection = Mock()
        def execute(query, values):
            executed.append((query, values))
            return Mock(fetchall=lambda: [])
        connection.execute.side_effect = execute
        yield connection

    monkeypatch.setattr(store, "connection", recording_connection)
    excluded = ["notice:13", "notice:99') OR 1=1 --"]
    for postgres in (False, True):
        store.postgres = postgres
        store.candidates(DAY.isoformat(), doc_types=["notice"], indexed=True,
                         query_vector=[1.0, 0.0, 0.0] if postgres else None, exclude_scopes=excluded)
    plain_sql, plain_args = executed[0]
    vector_sql, vector_args = executed[1]
    assert "scope NOT IN (?,?)" in plain_sql
    assert "scope NOT IN (%s,%s)" in vector_sql
    assert all(scope not in vector_sql for scope in excluded)
    # PostgreSQL adds only its projection and ordering vectors; the ordinary
    # evidence-filter arguments remain in the same positions and order.
    assert vector_args[1:-1] == plain_args
    assert json.loads(vector_args[0]) == json.loads(vector_args[-1]) == [1.0, 0.0, 0.0]
    assert plain_args[-3:-1] == sorted(excluded)

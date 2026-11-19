"""Meal-language regressions using isolated real storage and mocked APIs."""
from datetime import date, datetime, timezone
import json
from types import SimpleNamespace
import unicodedata
from unittest.mock import Mock

import pytest

from app.crawler import CrawlResult
from app.domain import RESTAURANTS
from app.indexing import build_chunks
from app.rag import Agent, UNREGISTERED_MENU, normalize_meal_language, parse_restaurant, resolve_context
from app.store import Store


@pytest.fixture
def tomorrow_agent(tmp_path, monkeypatch):
    monkeypatch.setattr("app.rag.today", lambda: date(2026, 11, 13))
    settings = SimpleNamespace(
        data_dir=tmp_path, database_url="", api_key="test-only", model="gpt-6.1-terra",
        embedding_model="test-embedding", embedding_dimensions=3,
        min_similarity=0.30, max_age_hours=24,
    )
    store = Store(settings)
    for code, info in RESTAURANTS.items():
        meals = []
        if code == "re13":
            meals = [
                {"meal_type": "중식", "items": ["달콤숯불바베큐제육볶음"], "price": "4,500원", "tags": []},
                {"meal_type": "석식", "items": ["고구마카레라이스"], "price": "4,500원", "tags": []},
            ]
        result = CrawlResult(
            code=code, name=info["name"], date="2026-11-14",
            source_url=f"https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id={info['facility_id']}&date=2026-11-14",
            fetched_at=datetime.now(timezone.utc).isoformat(), html="<html />", meals=meals,
        )
        store.replace_collection(result, build_chunks(result))
    pending = store.pending()
    # Generic dated meal requests must retain empty-meal evidence and real
    # menus even when a short colloquial query has low lexical similarity.
    vectors = [[0, 1, 0] if UNREGISTERED_MENU in row["text"] else [0.1, 0.995, 0] for row in pending]
    store.save_embeddings(pending, vectors)
    client = Mock()
    client.embeddings.create.return_value.data = [SimpleNamespace(embedding=[1, 0, 0])]

    def response(**kwargs):
        payload = json.loads(kwargs["input"][0]["content"])
        sources = payload["sources"]
        registered = [s for s in sources if "달콤숯불바베큐제육볶음" in s["text"] or "고구마카레라이스" in s["text"]]
        assert registered, "The prompt must contain the available dormitory menu"
        missing = [s for s in sources if UNREGISTERED_MENU in s["text"]]
        selected = registered + missing
        answer = "\n".join(s["text"] + f" [{s['id']}]" for s in selected)
        return SimpleNamespace(output_text=json.dumps({
            "answer": answer, "source_ids": [s["id"] for s in selected], "insufficient_evidence": False,
        }))

    client.responses.create.side_effect = response
    return store, settings, client, Agent(store, settings, client)


@pytest.mark.parametrize("text, expected", [
    ("내일 학시 뭐나와아", "내일 학식 뭐 나와"),
    ("내일학시뭐나와아?", "내일학식뭐 나와?"),
    ("내일 학식 뭐나와", "내일 학식 뭐나와"),
    ("학시서점 어디 있어?", "학시서점 어디 있어?"),
    ("학시카페 사진 보여줘", "학시카페 사진 보여줘"),
])
def test_narrow_meal_typo_normalization(text, expected):
    assert normalize_meal_language(text) == expected


def test_generic_meals_do_not_select_student_cafeteria():
    assert parse_restaurant("내일 학식 뭐나와") is None
    assert parse_restaurant("학생식당 내일 메뉴") == "re12"
    assert parse_restaurant("내일 긱식은?") == "re13"
    assert "학식" not in RESTAURANTS["re12"]["aliases"]


@pytest.mark.parametrize("message", ["내일 학시 뭐나와아", "내일 학식 뭐나와"])
def test_colloquial_tomorrow_query_keeps_all_cafeterias_and_registration_evidence(tomorrow_agent, message):
    _, _, client, agent = tomorrow_agent
    result = agent.chat(message, [], date(2026, 11, 13))
    assert result["mode"] == "openai"
    assert result["date"] == "2026-11-14"
    assert "달콤숯불바베큐제육볶음" in result["answer"]
    assert "고구마카레라이스" in result["answer"]
    assert {s["restaurant_code"] for s in result["sources"]} == set(RESTAURANTS)
    assert all(s["date"] == "2026-11-14" for s in result["sources"])
    assert all("life.hanyang.ac.kr" in s["url"] for s in result["sources"])
    payload = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert payload["context"]["restaurant"] is None
    assert payload["context"]["all_cafeterias"] is True
    assert payload["context"]["available_document_count"] == 12
    assert payload["context"]["retrieved_document_count"] == 12
    assert payload["context"]["retrieved_restaurants"] == sorted(RESTAURANTS)
    assert "학시" not in payload["question"]


def test_generic_lunch_retains_exact_meal_filter(tomorrow_agent):
    _, _, client, agent = tomorrow_agent
    result = agent.chat("내일 학식 점심 뭐나와", [], date(2026, 11, 13))
    assert result["mode"] == "openai"
    assert "달콤숯불바베큐제육볶음" in result["answer"]
    assert "고구마카레라이스" not in result["answer"]
    payload = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert payload["context"]["meal_type"] == "중식"
    assert len(payload["sources"]) == 4


def test_explicit_student_empty_page_is_a_specific_cited_status(tomorrow_agent):
    _, _, client, agent = tomorrow_agent
    result = agent.chat("내일 학생식당 점심 뭐나와", [], date(2026, 11, 13))
    assert result["mode"] == "retrieval"
    assert "2026-11-14 학생식당 중식" in result["answer"]
    assert "등록된 메뉴가 없습니다" in result["answer"]
    assert "미등록만으로 휴무나 운영 중단 여부를 알 수는 없어요" in result["answer"]
    assert len(result["sources"]) == 1
    assert result["sources"][0]["restaurant_code"] == "re12"
    assert "[S1]" in result["answer"]
    assert "달콤숯불" not in result["answer"]
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_uncrawled_date_is_not_reported_as_verified_empty(tomorrow_agent):
    _, _, client, agent = tomorrow_agent
    result = agent.chat("모레 학식 뭐나와", [], date(2026, 11, 13))
    assert result["mode"] == "insufficient"
    assert "2026-11-15" in result["answer"]
    assert "확인할 수 있는 공식 식단 자료가 없어요" in result["answer"]
    assert "공식 페이지에 등록된 메뉴가 없습니다" not in result["answer"]
    assert result["sources"] == []
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_generic_followup_resets_student_scope_and_preserves_original_history(tomorrow_agent):
    _, _, client, agent = tomorrow_agent
    history = [{"role": "user", "content": "내일 학생식당 점심 뭐나와"}]
    before = json.dumps(history, ensure_ascii=False)
    result = agent.chat("그럼 내일 학시 뭐나와아", history, date(2026, 11, 13))
    assert result["mode"] == "openai"
    assert json.dumps(history, ensure_ascii=False) == before
    payload = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert payload["context"]["restaurant"] is None
    assert payload["context"]["all_cafeterias"] is True


def test_explicit_student_lunch_price_behavior_is_unchanged(tomorrow_agent):
    store, _, client, agent = tomorrow_agent
    result = CrawlResult(
        code="re12", name="학생식당", date="2026-11-13",
        source_url="https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=1&date=2026-11-13",
        fetched_at=datetime.now(timezone.utc).isoformat(), html="<html />",
        meals=[{"meal_type": "중식", "items": ["얼큰순대국밥"], "price": "6,500원", "tags": []}],
    )
    store.replace_collection(result, build_chunks(result))
    pending = store.pending()
    store.save_embeddings(pending, [[1, 0, 0]] * len(pending))
    client.responses.create.side_effect = None
    client.responses.create.return_value.output_text = json.dumps({
        "answer": "학생식당 중식 얼큰순대국밥은 6,500원입니다. [S1]",
        "source_ids": ["S1"], "insufficient_evidence": False,
    })
    response = agent.chat("학생식당 점심 얼마야", [], date(2026, 11, 13))
    assert response["mode"] == "openai"
    assert "6,500원" in response["answer"]
    assert len(response["sources"]) == 1
    payload = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert payload["context"]["restaurant"] == "re12"
    assert payload["context"]["meal_type"] == "중식"
    assert payload["context"]["all_cafeterias"] is False


def test_explicit_cafeteria_what_is_served_without_menu_keyword(tomorrow_agent):
    _, _, client, agent = tomorrow_agent
    result = agent.chat("내일 학생식당 뭐나와", [], date(2026, 11, 13))
    assert result["mode"] == "retrieval"
    assert result["date"] == "2026-11-14"
    assert len(result["sources"]) == 3
    assert {s["restaurant_code"] for s in result["sources"]} == {"re12"}
    assert "등록된 메뉴가 없습니다" in result["answer"]
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_reported_jamo_question_passes_scope_and_retrieves_tomorrow_all_cafeterias(tomorrow_agent):
    _, _, client, agent = tomorrow_agent
    message = "ㄴㅐㅇㅣㄹ ㅎㅏㄱㅅㅣㄱ ㅁㅜㅓㄴㅏㅇㅗㅏ"
    original_bytes = message.encode("utf-8")
    result = agent.chat(message, [], date(2026, 11, 13))
    assert result["mode"] == "openai"
    assert result["date"] == "2026-11-14"
    assert {source["restaurant_code"] for source in result["sources"]} == set(RESTAURANTS)
    assert all(source["doc_type"] == "menu" and source["date"] == "2026-11-14" for source in result["sources"])
    assert "달콤숯불바베큐제육볶음" in result["answer"]
    payload = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert payload["question"] == "내일 학식 뭐나와"
    assert payload["context"]["restaurant"] is None
    assert payload["context"]["all_cafeterias"] is True
    assert payload["context"]["retrieved_restaurants"] == sorted(RESTAURANTS)
    client.embeddings.create.assert_called_once()
    assert message.encode("utf-8") == original_bytes


def test_nfd_followup_uses_normalized_context_without_changing_original_history(tomorrow_agent):
    _, _, client, agent = tomorrow_agent
    prior = "내일 학생식당 점심 뭐나와"
    history = [{"role": "user", "content": unicodedata.normalize("NFD", prior)},
        {"role": "assistant", "content": "이전 답변은 현재 공식 근거로 다시 확인해요."}]
    before = json.dumps(history, ensure_ascii=False)
    message = unicodedata.normalize("NFD", "그럼 내일 학식 뭐나와")
    result = agent.chat(message, history, date(2026, 11, 13))
    assert result["mode"] == "openai" and result["date"] == "2026-11-14"
    assert {source["restaurant_code"] for source in result["sources"]} == set(RESTAURANTS)
    payload = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert payload["question"] == "그럼 내일 학식 뭐나와"
    assert payload["conversation"][0]["content"] == prior
    assert payload["context"]["restaurant"] is None
    assert payload["context"]["meal_type"] == "중식"
    assert payload["context"]["all_cafeterias"] is True
    assert json.dumps(history, ensure_ascii=False) == before
    assert message != unicodedata.normalize("NFC", message)

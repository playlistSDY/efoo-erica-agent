from datetime import date, datetime, timezone
import json
from unittest.mock import Mock, patch

import pytest

from app.crawler import CrawlResult
from app.indexing import build_chunks
from app.rag import Agent, INSTRUCTIONS, resolve_context, select_sources
from app.settings import Settings
from app.store import Store


@pytest.fixture
def setup_agent(tmp_path):
    settings = Settings(data_dir=tmp_path, database_url="", api_key="test-key", embedding_dimensions=3)
    store = Store(settings)
    result = CrawlResult(code="re12", name="학생식당", date="2026-09-22",
        source_url="https://life.hanyang.ac.kr/menu?id=1&date=2026-09-22",
        fetched_at=datetime.now(timezone.utc).isoformat(), html="<html />",
        meals=[{"meal_type": "중식", "items": ["칼국수"], "price": "5,000원", "tags": []}])
    store.replace_collection(result, build_chunks(result))
    client = Mock()
    client.embeddings.create.return_value.data = [Mock(embedding=[1.0, 0.0, 0.0])]
    client.responses.create.return_value.output_text = json.dumps({
        "answer": "학생식당 점심은 칼국수이며 5,000원입니다. [S1]",
        "source_ids": ["S1"], "insufficient_evidence": False})
    return settings, store, client, Agent(store, settings, client=client)


def indexed(store, vector=None):
    rows = store.pending()
    store.save_embeddings(rows, [vector or [1.0, 0.0, 0.0]] * len(rows))


def ask(agent, message="학생식당 점심 메뉴와 가격", **kwargs):
    return agent.chat(message, kwargs.pop("history", []), kwargs.pop("selected_date", date(2026, 9, 22)), **kwargs)


def test_no_key_is_explicit_data_preview_without_any_model_calls(setup_agent):
    settings, store, client, agent = setup_agent
    settings.api_key = ""
    result = ask(agent)
    assert result["mode"] == "retrieval"
    assert result["model"] is None
    assert "아직 활성화되지 않았" in result["answer"]
    assert result["sources"][0]["date"] == "2026-09-22"
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_unindexed_documents_never_claim_vector_rag(setup_agent):
    _, _, client, agent = setup_agent
    assert ask(agent)["mode"] == "retrieval"
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_vector_rag_passes_only_filtered_sources_and_strict_schema(setup_agent):
    settings, store, client, agent = setup_agent
    indexed(store)
    result = ask(agent)
    assert result["mode"] == "openai"
    assert result["model"] == "gpt-6.1-terra"
    assert len(result["sources"]) == 1
    arguments = client.responses.create.call_args.kwargs
    assert arguments["store"] is False
    assert arguments["instructions"] == INSTRUCTIONS
    assert arguments["text"]["format"]["strict"] is True
    assert arguments["text"]["format"]["schema"]["additionalProperties"] is False
    prompt = json.loads(arguments["input"][0]["content"])
    assert prompt["context"]["target_date"] == "2026-09-22"
    assert prompt["context"]["restaurant"] == "re12"
    assert prompt["context"]["meal_type"] == "중식"
    assert all(s["date"] == "2026-09-22" and s["restaurant_code"] == "re12" for s in prompt["sources"])
    assert client.embeddings.create.call_args.kwargs["dimensions"] == 3


@pytest.mark.parametrize("payload", [
    {"answer": "임의 메뉴 [S999]", "source_ids": ["S999"], "insufficient_evidence": False},
    {"answer": "임의 메뉴 [S999]", "source_ids": ["S1"], "insufficient_evidence": False},
    {"answer": "출처 없는 답변", "source_ids": ["S1"], "insufficient_evidence": False},
    {"answer": "근거가 부족합니다 [S1]", "source_ids": ["S1"], "insufficient_evidence": True},
    {"answer": "이름만 있는 출처 [S1]", "source_ids": [], "insufficient_evidence": False},
])
def test_hallucinated_missing_or_insufficient_citations_are_rejected(setup_agent, payload):
    _, store, client, agent = setup_agent
    indexed(store)
    client.responses.create.return_value.output_text = json.dumps(payload)
    result = ask(agent)
    assert result["mode"] == "insufficient"
    assert result["sources"] == []
    assert "임의 메뉴" not in result["answer"]


def test_low_similarity_returns_no_answer_without_generation(setup_agent):
    _, store, client, agent = setup_agent
    indexed(store, [0.0, 1.0, 0.0])
    result = ask(agent)
    assert result["mode"] == "insufficient"
    client.embeddings.create.assert_called_once()
    client.responses.create.assert_not_called()


def test_missing_date_or_cafeteria_does_not_fall_back_to_other_documents(setup_agent):
    _, store, client, agent = setup_agent
    indexed(store)
    assert ask(agent, "교직원식당 점심 메뉴")["mode"] == "insufficient"
    assert ask(agent, selected_date=date(2026, 9, 23))["mode"] == "insufficient"
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_unrelated_question_does_not_trigger_model_or_retrieve_menu(setup_agent):
    _, store, client, agent = setup_agent
    indexed(store)
    result = ask(agent, "파이썬 정렬 코드를 만들어줘")
    assert result["mode"] == "insufficient"
    assert result["sources"] == []
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_chained_followup_preserves_original_date_cafeteria_and_meal():
    history = [{"role": "user", "content": "내일 학생식당 점심 메뉴 알려줘"},
               {"role": "assistant", "content": "메뉴 안내"},
               {"role": "user", "content": "가격은?"}]
    with patch("app.rag.today", return_value=date(2026, 11, 13)):
        assert resolve_context("그럼 저녁은?", history, date(2026, 11, 13), None) == ("2026-11-14", "re12", "석식")


def test_independent_request_resets_prior_metadata_and_explicit_filters_win():
    history = [{"role": "user", "content": "내일 학생식당 점심 메뉴 알려줘"}]
    with patch("app.rag.today", return_value=date(2026, 11, 13)):
        assert resolve_context("오늘 메뉴 알려줘", history, date(2026, 11, 13), None) == ("2026-11-13", None, None)
        assert resolve_context("그럼 저녁은?", history, date(2026, 11, 18), "re11") == ("2026-11-18", "re11", "석식")
        assert resolve_context("모레 창보 아침 메뉴", history, date(2026, 11, 13), "re11") == ("2026-11-15", "re15", "조식")


def test_all_cafeterias_does_not_mean_all_meal_times():
    with patch("app.rag.today", return_value=date(2026, 11, 13)):
        assert resolve_context("전체 식당 점심 비교", [], date(2026, 11, 13), None) == ("2026-11-13", None, "중식")
        assert resolve_context("학생식당 아침이랑 점심 비교", [], date(2026, 11, 13), None) == ("2026-11-13", "re12", None)


def test_cafeteria_coverage_precedes_global_score_cap():
    ranked = [{"id": str(i), "code": "re11", "score": 1.0} for i in range(10)]
    ranked += [{"id": "student", "code": "re12", "score": .8},
               {"id": "dorm", "code": "re13", "score": .7},
               {"id": "incubator", "code": "re15", "score": .6}]
    selected = select_sources(ranked, 8)
    assert len(selected) == 8
    assert {row["code"] for row in selected} == {"re11", "re12", "re13", "re15"}
    assert len({row["id"] for row in selected}) == 8


def test_comparison_passes_more_than_eight_menus_with_retrieval_counts(setup_agent):
    _, store, client, agent = setup_agent
    source = CrawlResult(code="re11", name="교직원식당", date="2026-09-22",
        source_url="https://life.hanyang.ac.kr/menu?id=2&date=2026-09-22",
        fetched_at=datetime.now(timezone.utc).isoformat(), html="<html />",
        meals=[{"meal_type": "중식", "items": [f"메뉴 {i}"], "price": "5,000원", "tags": []} for i in range(10)])
    store.replace_collection(source, build_chunks(source))
    indexed(store)
    ask(agent, "전체 식당 점심 비교해줘")
    prompt = json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert len(prompt["sources"]) == 11
    assert prompt["context"]["available_document_count"] == 11
    assert prompt["context"]["retrieved_document_count"] == 11
    assert prompt["context"]["retrieved_restaurants"] == ["re11", "re12"]


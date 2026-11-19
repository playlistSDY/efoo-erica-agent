"""Complete directory aggregation must not depend on vector-search coverage."""

from collections import Counter
from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.directory import DirectoryQuery, DirectoryService, parse_directory_query
from app.indexing import knowledge_chunks
from app.rag import Agent
from app.store import Store


URL = "https://life.hanyang.ac.kr/theme/pages/facilities/index.php"
NOW = datetime.now(timezone.utc)
PUBLIC_FIELDS = {"id", "name", "category", "location"}


def record(identifier, category, name, location):
    return {"id": identifier, "category": category, "category_display": category,
        "name": name, "location": location, "facility_no": str(identifier),
        # These contradictory display/internal fields occur in the public
        # source and cannot be interpreted as proof that a place is closed.
        "status": "inactive", "operating_status": "영업중", "admin_id": 900 + identifier}


def records():
    return [
        record(42, "편의점", "7-eleven", "창의인재원 교육관 1F"),
        record(53, "편의점", "7-eleven", "학생복지관 2층"),
        record(61, "편의점", "CU", "제3공학관 1층"),
        record(64, "편의점", "GS25", "버스승강장 1층"),
        record(78, "편의점", "emart24", "사회교육관 1층"),
        record(70, "카페/베이커리", "카페 블루", "학생복지관 1층"),
        record(71, "카페/베이커리", "파리바게뜨", "학생복지관 1층"),
        record(72, "인쇄/문구/서점/기념품", "제본·복사센터", "제3공학관 1층"),
        record(73, "인쇄/문구/서점/기념품", "한양서점", "학생복지관 1층"),
        record(74, "생활서비스", "우편취급국", "학생복지관 1층"),
        record(75, "구내식당", "학생식당", "학생복지관 2층"),
        record(76, "일반음식점", "캠퍼스 분식", "제3공학관 1층"),
        record(77, "자동판매기", "생활관 자판기", "창의인재원 1층"),
    ]


def seed(store, rows, *, fetched_at=NOW, categories=None):
    scope = "page:facility-directory"
    text = "\n".join(f"{row['name']} / {row['category']} / {row['location']}" for row in rows)
    store.replace_knowledge(scope, code="campus", name="ERICA 공개 캠퍼스 시설 전체 목록", url=URL,
        fetched_at=fetched_at.isoformat(),
        payload={"records": deepcopy(rows), "categories": categories or dict(Counter(row["category"] for row in rows))},
        chunks=knowledge_chunks(scope, "ERICA 공개 캠퍼스 시설 목록", [("전체 시설과 위치", text)], kind="page"))


@pytest.fixture
def directory(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, database_url="", api_key="mock-only-no-network",
        model="gpt-6.1-terra", embedding_model="test-embedding", embedding_dimensions=3,
        min_similarity=0.30, max_age_hours=24, max_date_range_days=31,
        directory_refresh_hours=6)
    store = Store(settings)
    data = records()
    seed(store, data)
    loader = Mock(side_effect=AssertionError("unexpected directory network request"))
    service = DirectoryService(store, settings, loader=loader, now=lambda: NOW)
    return SimpleNamespace(settings=settings, store=store, records=data, service=service, loader=loader)


def query_answer(fixture, message):
    query = fixture.service.parse(message)
    assert query is not None, f"Directory intent was not recognized: {message}"
    answer = fixture.service.answer(query)
    assert answer is not None
    return query, answer


@pytest.mark.parametrize("message", [
    "학교에 편의점 몇개 있어?", "학교에편의점몇개있어?", "편의점이 몇 곳이야?",
    "캠퍼스 편의점 몇 군데야", "편의점 총 몇 개야?", "편의점 개수 알려줘",
    "몇 개의 편의점이 있어?",
])
def test_count_variants_use_complete_public_category(directory, message):
    query, result = query_answer(directory, message)
    assert query.action == "count"
    assert result["directory"]["count"] == 5
    assert {item["id"] for item in result["directory"]["records"]} == {42, 53, 61, 64, 78}
    assert result["directory"]["scope"] == "official_portal"
    assert result["mode"] == "retrieval" and result["model"] is None
    assert result["sources"][0]["url"] == URL and result["sources"][0]["id"] == "S1"
    assert result["sources"][0]["fetched_at"]
    assert "[S1]" in result["answer"] and result["images"] == []
    directory.loader.assert_not_called()


@pytest.mark.parametrize("message", ["편의점 목록 전부 알려줘", "편의점 이름 다 알려줘", "편의점 어디어디 있어?"])
def test_complete_list_keeps_same_brand_branches_and_locations(directory, message):
    query, result = query_answer(directory, message)
    assert query.action == "list"
    assert len(result["directory"]["records"]) == result["directory"]["count"] == 5
    assert "학생복지관 2층" in result["answer"] and "창의인재원 교육관 1F" in result["answer"]
    assert sum(item["name"] == "7-eleven" for item in result["directory"]["records"]) == 2


@pytest.mark.parametrize("message,expected", [
    ("세븐일레븐 몇 개 있어?", 2), ("7-eleven 몇 곳이야?", 2),
    ("CU 몇 개야?", 1), ("씨유 몇 개 있어?", 1),
    ("GS25 몇 곳 있어", 1), ("이마트24 몇 개 있어?", 1), ("emart24 몇 곳이야?", 1),
])
def test_brand_aliases_count_distinct_facility_ids(directory, message, expected):
    _, result = query_answer(directory, message)
    assert result["directory"]["count"] == expected


@pytest.mark.parametrize("message,ids", [
    ("학생복지관에 편의점 몇 개 있어?", {53}),
    ("제3공학관 편의점 몇 곳?", {61}),
    ("창의인재원 편의점 목록 알려줘", {42}),
    ("버스승강장 편의점 몇 개야?", {64}),
])
def test_location_filters_use_public_directory_location(directory, message, ids):
    query, result = query_answer(directory, message)
    assert query.location
    assert result["directory"]["count"] == len(ids)
    assert {item["id"] for item in result["directory"]["records"]} == ids


def test_total_facilities_and_category_breakdown_use_catalog_scope(directory):
    _, total = query_answer(directory, "학교 시설 총 몇 곳이야?")
    assert total["directory"]["count"] == len(directory.records) == 13
    query, breakdown = query_answer(directory, "학교 시설 종류별로 몇 개 있어?")
    assert query.action in {"categories", "count"}
    assert all(category in breakdown["answer"] for category in {row["category"] for row in directory.records})
    assert "공개" in total["answer"] or "등록" in total["answer"]


def test_combined_category_is_labeled_and_narrow_bookstore_is_not_overcounted(directory):
    _, cafes = query_answer(directory, "카페 몇 개 있어?")
    assert cafes["directory"]["count"] == 2
    assert "카페/베이커리" in cafes["answer"]
    query = directory.service.parse("서점만 몇 개 있어?")
    if query is not None:
        result = directory.service.answer(query)
        assert result is not None and result["directory"]["count"] == 1
        assert [item["name"] for item in result["directory"]["records"]] == ["한양서점"]


def test_public_inactive_records_remain_and_duplicate_ids_are_not_double_counted(directory):
    duplicated = directory.records + [deepcopy(directory.records[0])]
    # A cached summary field must not override authoritative complete records.
    seed(directory.store, duplicated, categories={"편의점": 999})
    _, result = query_answer(directory, "편의점 몇 개 있어?")
    assert result["directory"]["count"] == 5
    assert {item["id"] for item in result["directory"]["records"]} == {42, 53, 61, 64, 78}
    assert all(set(item) <= PUBLIC_FIELDS for item in result["directory"]["records"])


def test_conflicting_duplicate_identity_is_insufficient_instead_of_guessing(directory):
    conflicting = deepcopy(directory.records[0])
    conflicting["location"] = "다른 위치"
    seed(directory.store, directory.records + [conflicting])
    query = DirectoryQuery(action="count", categories=("편의점",))
    assert directory.service.answer(query, refresh=False) is None
    result = directory.service.answer(query, refresh=True)
    assert result is not None and result["mode"] == "insufficient"
    assert result["sources"] == []


def test_zero_matches_is_scoped_zero_with_citation_not_an_invented_facility(directory):
    query = DirectoryQuery(action="count", categories=("편의점",), location="학생복지관", brand="CU")
    result = directory.service.answer(query)
    assert result is not None
    assert result["directory"]["count"] == 0 and result["directory"]["records"] == []
    assert result["sources"][0]["url"] == URL


@pytest.mark.parametrize("message", [
    "편의점 몇 시에 열어?", "편의점 운영시간 알려줘", "CU 커피 얼마야?",
    "학생식당 오늘 메뉴 알려줘", "카페에서 지금 영업 중인 곳 몇 개야?",
    "편의점 지금 열려 있어?", "창의식당 식권 가격 알려줘",
])
def test_operating_time_price_menu_and_live_status_stay_on_existing_rag_path(directory, message):
    assert parse_directory_query(message, records=directory.records) is None


@pytest.mark.parametrize("message", ["서울캠퍼스 편의점 몇 개야?", "약국 매장 몇 개 있어?", "다른 학교 편의점 몇 곳이야?"])
def test_out_of_scope_campus_or_unknown_qualifier_does_not_become_global_count(directory, message):
    assert parse_directory_query(message, records=directory.records) is None


def test_followup_location_filter_keeps_previous_category_without_mutating_history(directory):
    history = [{"role": "user", "content": "학교에 편의점 몇 개 있어?"}]
    before = deepcopy(history)
    query = directory.service.parse("그럼 학생복지관에는 몇 개야?", history)
    assert query is not None and query.location == "학생복지관"
    result = directory.service.answer(query)
    assert result["directory"]["count"] == 1
    assert {item["id"] for item in result["directory"]["records"]} == {53}
    assert history == before


def test_missing_or_stale_catalog_does_not_pretend_to_have_complete_counts(directory):
    seed(directory.store, directory.records, fetched_at=NOW - timedelta(days=10))
    assert directory.service.answer(DirectoryQuery(action="count", categories=("편의점",)), refresh=False) is None
    directory.loader.assert_not_called()


def test_stale_catalog_refresh_reloads_validates_and_counts_updated_records(directory):
    seed(directory.store, directory.records, fetched_at=NOW - timedelta(days=10))
    updated = directory.records + [record(90, "편의점", "새 편의점", "학술정보관 1층")]
    directory.loader.side_effect = lambda settings, store: seed(store, updated)
    result = directory.service.answer(DirectoryQuery(action="count", categories=("편의점",)), refresh=True)
    assert result["directory"]["count"] == 6
    assert any(item["id"] == 90 for item in result["directory"]["records"])
    directory.loader.assert_called_once_with(directory.settings, directory.store)


def test_refresh_failure_does_not_reuse_stale_count_and_retries_are_bounded(directory):
    seed(directory.store, directory.records, fetched_at=NOW - timedelta(days=10))
    directory.loader.side_effect = RuntimeError("private upstream detail")
    query = DirectoryQuery(action="count", categories=("편의점",))
    first = directory.service.answer(query, refresh=True)
    second = directory.service.answer(query, refresh=True)
    assert first["mode"] == second["mode"] == "insufficient"
    assert first["sources"] == second["sources"] == []
    assert "private upstream detail" not in first["answer"]
    assert "directory" not in first
    directory.loader.assert_called_once()


def test_legacy_page_without_complete_records_is_not_a_zero_count(directory):
    directory.store.replace_knowledge("page:facility-directory", code="campus", name="기존 목록",
        url=URL, fetched_at=NOW.isoformat(), payload={},
        chunks=knowledge_chunks("page:facility-directory", "기존 목록", [("일부 안내", "편의점 안내")], kind="page"))
    query = directory.service.parse("학교에 편의점 몇 개 있어?")
    assert query is not None
    assert directory.service.answer(query, refresh=False) is None
    directory.loader.assert_not_called()


def test_catalog_larger_than_search_cap_still_returns_complete_list_and_exact_count(directory):
    many = [record(1000 + number, "편의점", f"테스트매점 {number:02d}", f"캠퍼스 {number:02d}관 1층") for number in range(50)]
    seed(directory.store, many)
    assert len(directory.store.pending()) > 1, "Fixture must exceed one text chunk"
    query, result = query_answer(directory, "편의점 목록 전부 알려줘")
    assert query.action == "list"
    assert result["directory"]["count"] == len(result["directory"]["records"]) == 50
    assert "테스트매점 00" in result["answer"] and "테스트매점 49" in result["answer"]


def test_agent_answers_structured_counts_before_similarity_and_openai(directory):
    pending = directory.store.pending()
    directory.store.save_embeddings(pending, [[0.0, 1.0, 0.0]] * len(pending))
    model = Mock()
    model.embeddings.create.side_effect = AssertionError("Complete directory aggregation must not embed a count query")
    model.responses.create.side_effect = AssertionError("A known exact count must not be guessed by a model")
    agent = Agent(directory.store, directory.settings, model)
    agent.directory = directory.service
    result = agent.chat("학교에편의점몇개있어?", [], date(2026, 11, 14))
    assert result["mode"] == "retrieval" and result["directory"]["count"] == 5
    assert len(result["sources"]) == 1
    model.embeddings.create.assert_not_called()
    model.responses.create.assert_not_called()


@pytest.mark.parametrize("message", [
    "창의인재원식당 학식알려줘", "내일 창의인재원식당 학식 알려줘",
    "학생식당 내일 학식 알려줘", "다음주 구내식당 학식 알려줘",
    "교직원식당 11월16일 뭐나와", "저번주 창의인재원식당 뭐나왔어?",
    "내일 학생식당 점심 알려줘", "창의인재원식당 석식 알려줘",
])
def test_meal_and_dated_meal_questions_never_become_directory_lists(directory, message):
    assert directory.service.parse(message) is None
    agent = Agent(directory.store, directory.settings, Mock(), directory_service=directory.service)
    plan = agent.plan(message, [], date(2026, 11, 14))
    assert plan.directory_query is None
    assert plan.menu_request and "menu" in plan.doc_types


def test_exact_named_cafeteria_is_one_facility_while_category_means_all_four(directory):
    expanded = directory.records + [
        record(79, "구내식당", "교직원식당", "학생복지관 3층"),
        record(80, "구내식당", "창의인재원식당", "창의인재원 1층"),
        record(81, "구내식당", "창업보육센터식당", "창업보육센터 지하 1층"),
    ]
    seed(directory.store, expanded)
    _, named = query_answer(directory, "학생식당 위치 알려줘")
    assert named["directory"]["count"] == 1
    assert [item["name"] for item in named["directory"]["records"]] == ["학생식당"]
    _, category = query_answer(directory, "구내식당은 몇 개 있어?")
    assert category["directory"]["count"] == 4
    assert {item["name"] for item in category["directory"]["records"]} == {
        "학생식당", "교직원식당", "창의인재원식당", "창업보육센터식당"}


@pytest.mark.parametrize("floor,ids", [
    ("1층", {90}), ("1F", {90}), ("2층", {53, 91}),
    ("지하1층", {92, 93}), ("B1", {92, 93}),
])
def test_floor_filters_equate_f_and_korean_floor_without_mixing_basement(directory, floor, ids):
    expanded = directory.records + [
        record(90, "편의점", "1층 테스트매점", "학생복지관 1F"),
        record(91, "편의점", "2층 테스트매점", "학생복지관 2F"),
        record(92, "편의점", "지하 테스트매점", "학생복지관 지하1층"),
        record(93, "편의점", "B1 테스트매점", "학생복지관 B1"),
    ]
    seed(directory.store, expanded)
    query, result = query_answer(directory, f"학생복지관 {floor} 편의점 몇 개 있어?")
    assert query.location == "학생복지관"
    assert query.floor
    assert result["directory"]["count"] == len(ids)
    assert {item["id"] for item in result["directory"]["records"]} == ids


def test_followup_new_category_keeps_location_and_explicit_all_resets_it(directory):
    seed(directory.store, directory.records + [record(94, "카페/베이커리", "공학관 카페", "제3공학관 1층")])
    history = [{"role": "user", "content": "학생복지관 편의점 몇 개 있어?"}]
    before = deepcopy(history)
    query = directory.service.parse("그럼 카페는?", history)
    assert query is not None and query.categories == ("카페/베이커리",)
    assert query.location == "학생복지관"
    assert directory.service.answer(query)["directory"]["count"] == 2
    all_cafes = directory.service.parse("그럼 전체 카페는?", history)
    assert all_cafes is not None and all_cafes.location is None
    assert directory.service.answer(all_cafes)["directory"]["count"] == 3
    assert history == before


def test_starbucks_brand_alias_matches_actual_korean_record_name(directory):
    seed(directory.store, directory.records + [record(95, "카페/베이커리", "스타벅스", "사회교육관 1층")])
    for message in ("스타벅스몇개", "STARBUCKS 몇 개 있어?"):
        query, result = query_answer(directory, message)
        assert query.brand == "STARBUCKS"
        assert result["directory"]["count"] == 1
        assert [item["name"] for item in result["directory"]["records"]] == ["스타벅스"]


def test_exact_facility_name_containing_category_word_does_not_count_the_whole_category(directory):
    seed(directory.store, directory.records + [record(96, "카페/베이커리", "한양베이커리", "학생복지관 1층")])
    query, result = query_answer(directory, "한양베이커리몇개")
    assert query.brand == "한양베이커리"
    assert result["directory"]["count"] == 1
    assert {item["id"] for item in result["directory"]["records"]} == {96}


def test_ascii_cu_name_never_matches_inside_a_longer_english_word(directory):
    assert directory.service.parse("CUBIC 몇 개 있어?") is None
    assert directory.service.parse("cuBIC몇개") is None
    query, result = query_answer(directory, "CU 몇 개 있어?")
    assert query.brand == "CU" and result["directory"]["count"] == 1

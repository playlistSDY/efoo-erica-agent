"""Typed school-info integration over synthetic public-schema SQLite records.

Only temporary test databases are used. Every model/HTTP call is forbidden and
API scheduling/refresh dependencies are injected, so failures cannot reach the
real portal, OpenAI, project credentials or existing conversation history.
"""

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import re
from unittest.mock import Mock, patch
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

with patch("dotenv.load_dotenv"):
    from app.directory import DirectoryService
    from app.indexing import knowledge_chunks
    from app.rag import Agent
    from app.school_query import SchoolService
    from app.settings import Settings
    from app.store import Store


DAY = date(2026, 11, 14)
DIRECTORY_URL = "https://life.hanyang.ac.kr/theme/pages/facilities/index.php"


def record(identity, name, category, location, **extra):
    return {"id": identity, "name": name, "name_en": name, "name_zh": "",
        "category": category, "category_display": category, "location": location,
        "description": "공개 합성 테스트 자료", "notice": "", "image": "", **extra}


CATALOG = [
    record(49, "안산한양대우편취급국", "생활서비스", "학생복지관 1층 102호", name_en="Post Office"),
    record(50, "신한은행", "생활서비스", "학생복지관 1층 103호", name_en="Shinhan Bank"),
    record(10, "7-eleven", "편의점", "학생복지관 1층"),
    record(11, "7-eleven", "편의점", "제3공학관 1층"),
    record(12, "7-eleven", "편의점", "창의인재원 1층"),
    record(20, "이모네", "일반음식점", "학생복지관 2층", description="공식 분식 매장"),
    record(21, "분식마을", "일반음식점", "제1공학관 1층", description="공식 분식 매장"),
    record(30, "카페알파", "카페/베이커리", "학생복지관 2층"),
    record(31, "카페베타", "카페/베이커리", "제3공학관 1층"),
    record(32, "카페감마", "카페/베이커리", "창의인재원 2층"),
]
CONTACTS = {49: "031-400-0049", 50: "031-400-0050", 10: "031-400-0010", 11: "", 12: "031-400-0012"}
HOURS = {49: "평일 09:00~18:00 (점심 12:00~13:00)", 50: "평일 09:00~16:00",
    10: "월~토 11:00~20:00", 11: "24시간 운영", 12: "평일 08:00~23:00"}


def put_profile(store, row, *, fetched_at=None, contact=None, hours=None, info_only=False):
    identity = row["id"]
    contact = CONTACTS.get(identity, f"031-400-{identity:04d}") if contact is None else contact
    hours = HOURS.get(identity, "평일 10:00~19:00") if hours is None else hours
    stamp = fetched_at or datetime.now(timezone.utc).isoformat()
    info_rows = [{"label": "위치", "value": row["location"]}, {"label": "영업시간", "value": hours},
        {"label": "건물 번호", "value": str(identity)}]
    if contact:
        info_rows.append({"label": "연락처", "value": contact})
    info = {"name": row["name"], "category": row["category"], "location": row["location"],
        "contact": "" if info_only else contact, "opening_hours": "" if info_only else hours,
        "building_no": "" if info_only else str(identity), "info_rows": info_rows,
        "notices": [], "description": {"html": "<p>합성 공식 시설 안내</p>", "text": row["description"]},
        "gallery": [], "cover_image": {"url": ""}}
    photo = {"url": "/media/" + f"{identity:064x}" + ".jpg", "alt": row["name"] + " 합성 시설 사진"}
    scope = f"facility:{identity}"
    text = "\n".join(f"{item['label']}: {item['value']}" for item in info_rows)
    chunks = knowledge_chunks(scope, row["name"], [("공식 시설 정보", text)],
        images=[photo], metadata={"facility_id": identity})
    store.replace_knowledge(scope, code=f"fac_{identity}", name=row["name"],
        url=f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={identity}",
        fetched_at=stamp, payload={"facility_id": identity, "restaurant_info": info,
            "directory_record": row, "page_data": {"images": []}, "assets": []}, chunks=chunks)


@pytest.fixture
def school(tmp_path, monkeypatch):
    monkeypatch.setenv("PYTHON_DOTENV_DISABLED", "1")
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "module-instance"))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SCHEDULE_ENABLED", "false")
    monkeypatch.setattr("requests.sessions.Session.request", Mock(side_effect=AssertionError("Live HTTP forbidden")))
    settings = Settings(data_dir=tmp_path / "school-instance", database_url="", api_key="",
        raw_s3_bucket="", embedding_dimensions=3, schedule_enabled=False)
    store = Store(settings)
    stamp = datetime.now(timezone.utc).isoformat()
    records = deepcopy(CATALOG)
    store.replace_knowledge("page:facility-directory", code="campus_directory", name="공식 전체 시설 목록",
        url=DIRECTORY_URL, fetched_at=stamp, payload={"records": records},
        chunks=knowledge_chunks("page:facility-directory", "공식 전체 시설 목록", [("공개 카탈로그", "합성 시설 전체 목록")], kind="page"))
    for row in records:
        put_profile(store, row)
    refresher = Mock()
    refresher.ensure.side_effect = lambda chosen: {"targets": [{"id": row["id"], "usable": True} for row in chosen], "errors": []}
    directory = DirectoryService(store, settings,
        loader=Mock(side_effect=AssertionError("Live directory fetch forbidden")), facility_refresher=refresher)
    service = SchoolService(store, settings, directory, refresher=refresher)
    model = Mock()
    model.embeddings.create.side_effect = AssertionError("Model must not guess typed public fields")
    model.responses.create.side_effect = AssertionError("Model must not guess typed public fields")
    agent = Agent(store, settings, client=model, directory_service=directory)
    agent.school.refresher = refresher
    return settings, store, directory, service, agent, refresher, model


def selected(response):
    return response["selected_facility_ids"]


def history_after(message, response):
    return [{"role": "user", "content": message}, {"role": "assistant", "content": response["answer"],
        "selected_facility_ids": response["selected_facility_ids"],
        "reference_facility_ids": response["reference_facility_ids"]}]


def test_two_explicit_facilities_return_all_requested_fields_in_user_name_order(school):
    _, _, _, service, _, _, _ = school
    plan = service.plan("신한은행이랑 우체국 위치랑 전화번호 알려줘")
    assert plan.active and plan.facility_ids == ("50", "49")
    assert set(plan.slots) >= {"location", "contact"}
    response = service.answer(plan)
    assert selected(response) == ["50", "49"]
    for identity in (50, 49):
        source = next(source for source in response["sources"] if str(source["metadata"]["facility_id"]) == str(identity))
        assert CONTACTS[identity] in source["text"]
        assert next(row["location"] for row in CATALOG if row["id"] == identity) in source["text"]
        assert CONTACTS[identity] in response["answer"]
    assert response["answer"].index("신한은행") < response["answer"].index("안산한양대우편취급국")


def test_shared_brand_returns_all_three_branch_hours_and_marks_unknown_contact(school):
    _, _, _, service, _, _, _ = school
    response = service.answer(service.plan("세븐일레븐 영업시간과 연락처 알려줘"))
    assert selected(response) == ["10", "11", "12"]
    assert len(response["sources"]) == 3
    for identity in (10, 11, 12):
        source = next(source for source in response["sources"] if str(source["metadata"]["facility_id"]) == str(identity))
        assert HOURS[identity] in source["text"]
        assert next(row["location"] for row in CATALOG if row["id"] == identity) in response["answer"]
    unknown = next(source for source in response["sources"] if source["metadata"]["facility_id"] == 11)
    contact_line = next(line for line in unknown["text"].splitlines() if line.startswith("연락처:"))
    assert re.search(r"확인.*못|미확인|정보.*없", contact_line)
    assert CONTACTS[10] not in unknown["text"] and CONTACTS[12] not in unknown["text"]
    branch_line = next(line for line in response["answer"].splitlines() if "제3공학관" in line)
    assert re.search(r"확인.*못|미확인|정보.*없", branch_line)
    assert not re.search(r"0\d{1,2}-\d{3,4}-\d{4}", branch_line)


def test_profile_info_rows_fallback_supplies_real_contact_hours_and_building(school):
    _, store, _, service, _, _, _ = school
    put_profile(store, CATALOG[0], info_only=True)
    response = service.answer(service.plan("우체국 전화번호와 영업시간 건물 번호"))
    assert CONTACTS[49] in response["answer"]
    assert HOURS[49] in response["sources"][0]["text"]
    assert all(value in response["answer"] for value in ("09:00", "18:00", "12:00", "13:00"))
    assert "건물 번호" in response["answer"] and "49" in response["answer"]


def test_post_office_location_then_deictic_phone_keeps_metadata_identity(school):
    _, _, _, service, _, _, _ = school
    response = service.answer(service.plan("우체국 위치 알려줘"))
    followup = service.answer(service.plan("거기 전화번호는?", history_after("우체국 위치 알려줘", response)))
    assert selected(followup) == ["49"] and CONTACTS[49] in followup["answer"]
    assert "031-400-0050" not in followup["answer"]


def test_brand_list_then_named_branch_uses_that_branch_only(school):
    _, _, directory, service, _, _, _ = school
    response = directory.answer(directory.parse("세븐일레븐 목록 알려줘"))
    ids = [str(row["id"]) for row in response["directory"]["records"]]
    assert set(ids) == {"10", "11", "12"}
    history = [{"role": "user", "content": "세븐일레븐 목록 알려줘"},
        {"role": "assistant", "content": response["answer"], "directory": response["directory"]}]
    followup = service.answer(service.plan("학생복지관점은 몇 시까지 해?", history))
    assert selected(followup) == ["10"] and HOURS[10] in followup["sources"][0]["text"]
    assert "11:00" in followup["answer"] and "20:00" in followup["answer"]


def test_collection_floor_first_and_remaining_keep_original_result_order(school):
    _, _, _, service, _, _, _ = school
    collection = {"role": "assistant", "content": "카페 목록", "selected_facility_ids": [32, 31, 30],
        "reference_facility_ids": [32, 31, 30]}
    subset = service.answer(service.plan("그중 2층만", [collection]))
    assert selected(subset) == ["32", "30"]
    first = service.answer(service.plan("첫번째 사진", [collection]))
    assert selected(first) == ["32"]
    history = [collection, *history_after("첫번째 사진", first)]
    remaining = service.answer(service.plan("나머지 위치", history))
    assert selected(remaining) == ["31", "30"]


def test_ambiguous_deictic_reference_asks_instead_of_picking_a_branch(school):
    _, _, _, service, _, _, _ = school
    history = [{"role": "assistant", "content": "편의점 세 곳", "selected_facility_ids": [12, 10, 11]}]
    response = service.answer(service.plan("거기 전화번호", history))
    assert response["mode"] == "clarification" and not selected(response)
    assert response["sources"] == [] and "학생복지관" in response["answer"] and "창의인재원" in response["answer"]


def test_new_cafe_location_topic_does_not_reuse_prior_post_office(school):
    _, _, _, service, _, _, _ = school
    history = [{"role": "assistant", "content": "우체국 위치", "selected_facility_ids": [49]}]
    response = service.answer(service.plan("학생복지관 2층 카페 영업시간", history))
    assert selected(response) == ["30"]
    assert "우편취급국" not in response["answer"] and CONTACTS[49] not in response["answer"]


@pytest.mark.parametrize("message", ["세븐일레븐 토요일 몇 시까지 해?", "우체국 토요일 몇시에 문 닫아?"])
def test_hours_questions_route_before_meal_relative_date_parser(school, message):
    _, _, _, _, agent, _, model = school
    plan = agent.plan(message, [], DAY)
    assert plan.school_plan is not None and plan.school_plan.active
    assert not plan.menu_request and plan.span.start == plan.span.end == DAY
    response = agent.chat(message, [], DAY, prepared_plan=plan)
    assert response["mode"] == "retrieval" and response["sources"]
    assert "날짜" not in response["answer"]
    model.embeddings.create.assert_not_called()
    model.responses.create.assert_not_called()


def test_current_open_question_disclaims_real_time_status_instead_of_guaranteeing_it(school):
    _, _, _, service, _, _, _ = school
    plan = service.plan("우체국 지금 문 열었어?")
    assert plan.current_status
    response = service.answer(plan)
    assert "실시간 영업 여부는 확인할 수 없어요" in response["answer"]
    assert HOURS[49] in response["sources"][0]["text"]
    assert "09:00" in response["answer"] and "18:00" in response["answer"]


def test_saturday_question_does_not_apply_a_weekday_only_closing_time(school):
    _, _, _, service, _, _, _ = school
    response = service.answer(service.plan("우체국 토요일 몇 시까지 해?"))
    assert selected(response) == ["49"]
    assert HOURS[49] in response["sources"][0]["text"]
    saturday_line = next(line for line in response["answer"].splitlines() if "토요일" in line)
    assert re.search(r"확인.*못|불명확|알.*없", saturday_line)
    assert not re.search(r"토요일[^\n]*18:00(?:까지|에)", response["answer"])


def test_explicit_weekday_range_including_saturday_is_retained_as_official_evidence(school):
    _, _, _, service, _, _, _ = school
    response = service.answer(service.plan("학생복지관 세븐일레븐 토요일 몇 시까지 해?"))
    assert selected(response) == ["10"] and HOURS[10] in response["sources"][0]["text"]
    assert "20:00" in response["answer"]
    assert not re.search(r"토요일.*(?:확인.*못|불명확)", response["answer"])


@pytest.mark.parametrize("unknown", ["무지개카페", "우리은행"])
def test_unknown_named_facility_does_not_substitute_another_facility_or_previous_context(school, unknown):
    _, _, _, service, _, refresher, _ = school
    history = [{"role": "assistant", "content": "우체국 안내", "selected_facility_ids": [49]}]
    response = service.answer(service.plan(unknown + " 전화번호 알려줘", history), refresh=True)
    assert response["mode"] in {"insufficient", "clarification"}
    assert selected(response) == [] and response["sources"] == [] and response["images"] == []
    assert not any(contact and contact in response["answer"] for contact in CONTACTS.values())
    refresher.ensure.assert_not_called()


@pytest.mark.parametrize("hours,classification", [
    ("24시간 운영", "explicit"),
    ("유인 09:00~18:00 / 무인 18:00~09:00", "covered"),
    ("유인 09:00~18:00 / 무인 20:00~09:00", "gap"),
    ("평일 유인 09:00~18:00 / 무인 18:00~09:00", "conditional"),
    ("", "unknown"),
])
def test_continuous_hours_conditions_are_distinguished_without_real_time_guarantees(school, hours, classification):
    _, store, _, service, _, _, _ = school
    put_profile(store, CATALOG[0], hours=hours)
    plan = service.plan("우체국 24시간 운영이야?")
    assert plan.hours_filter == "continuous24"
    response = service.answer(plan)
    assert selected(response) == ["49"]
    assert re.search(r"현재 영업 여부.*보장.*아니", response["answer"])
    if hours:
        assert hours in response["sources"][0]["text"]
    if classification == "explicit":
        assert re.search(r"24시간.*명시", response["answer"])
    elif classification == "covered":
        assert re.search(r"유인.*무인.*하루 전체", response["answer"])
    elif classification == "gap":
        assert re.search(r"24시간 운영.*확인할 수 없", response["answer"])
    elif classification == "conditional":
        assert re.search(r"요일.*조건.*단정할 수 없", response["answer"])
    else:
        assert re.search(r"운영시간.*없.*확인할 수 없", response["answer"])


def test_stale_profile_is_not_quietly_reported_as_current_information(school):
    _, store, _, service, _, refresher, _ = school
    put_profile(store, CATALOG[0], fetched_at=(datetime.now(timezone.utc) - timedelta(days=3)).isoformat(),
        contact="STALE_PHONE_SENTINEL", hours="STALE_HOURS_SENTINEL")
    refresher.ensure.return_value = {"targets": [{"id": 49, "usable": False}], "errors": [{"id": 49, "error": "Timeout"}]}
    refresher.ensure.side_effect = None
    response = service.answer(service.plan("우체국 전화번호와 영업시간"), refresh=True)
    assert response["mode"] == "insufficient" and response["sources"] == []
    assert "최신" in response["answer"] and "확인" in response["answer"]
    assert "STALE_PHONE_SENTINEL" not in response["answer"] and "STALE_HOURS_SENTINEL" not in response["answer"]
    refresher.ensure.assert_called_once()


def test_only_requested_profile_images_are_returned_with_its_official_source_url(school):
    _, _, _, service, _, _, _ = school
    response = service.answer(service.plan("우체국 사진 보여줘"))
    assert selected(response) == ["49"] and len(response["images"]) == 1
    assert response["images"][0]["url"] == "/media/" + f"{49:064x}" + ".jpg"
    assert response["images"][0]["source_url"].endswith("?id=49")


@pytest.fixture
def school_api(school):
    settings, store, _, _, agent, refresher, model = school
    from app.main import create_app
    meal_refresh = Mock()
    meal_refresh.ensure.side_effect = AssertionError("School fields must not start meal collection")
    scheduler = Mock()
    application = create_app(settings, store, agent, refresh=meal_refresh, scheduler=scheduler)
    with TestClient(application) as client:
        yield client, application, refresher, meal_refresh, scheduler, model


def post(client, conversation, message):
    response = client.post("/api/chat", json={"message": message, "conversation_id": conversation["id"],
        "turn_id": str(uuid4()), "date": DAY.isoformat()})
    assert response.status_code == 200, response.text
    return response.json()


def test_actual_agent_api_persists_selected_ids_for_deictic_followup_after_reconnect(school_api):
    client, app, refresher, meal_refresh, scheduler, model = school_api
    conversation = client.get("/api/conversations/current").json()
    first = post(client, conversation, "우체국 위치 알려줘")
    assert selected(first) == ["49"]
    with TestClient(app, cookies=dict(client.cookies)) as reconnected:
        restored = reconnected.get("/api/conversations/current").json()
        assert restored["messages"][-1]["payload"]["selected_facility_ids"] == ["49"]
        response = post(reconnected, restored, "거기 전화번호는?")
    assert selected(response) == ["49"] and CONTACTS[49] in response["answer"]
    assert refresher.ensure.call_count == 2
    meal_refresh.ensure.assert_not_called()
    model.embeddings.create.assert_not_called()
    model.responses.create.assert_not_called()
    assert scheduler.start.called


def test_actual_agent_api_preserves_collection_order_across_first_and_remaining(school_api):
    client, _, _, meal_refresh, _, model = school_api
    conversation = client.get("/api/conversations/current").json()
    listing = post(client, conversation, "세븐일레븐 목록 알려줘")
    ids = listing["selected_facility_ids"]
    assert len(ids) == 3 and set(ids) == {"10", "11", "12"}
    first = post(client, conversation, "첫번째 사진")
    assert selected(first) == ids[:1]
    remaining = post(client, conversation, "나머지 위치")
    assert selected(remaining) == ids[1:]
    meal_refresh.ensure.assert_not_called()
    model.responses.create.assert_not_called()

"""Date range → refresh → grounded answer → persisted chat integration."""

from datetime import date, datetime, timedelta, timezone
import json
from types import SimpleNamespace
from unittest.mock import Mock
from uuid import uuid4

from fastapi.testclient import TestClient
import pytest

from app.crawler import CrawlResult
from app.date_ranges import DateRange
from app.domain import RESTAURANTS
from app.indexing import build_chunks, knowledge_chunks
from app.rag import Agent, UNREGISTERED_MENU
from app.settings import Settings
from app.store import Store


BASE = date(2026, 11, 13)
NEXT_WEEK = DateRange(date(2026, 11, 16), date(2026, 11, 22))


def refresh_report(start, end, codes=None):
    codes = list(RESTAURANTS) if codes is None else codes
    targets = [{"date": day.isoformat(), "code": code, "usable": True, "status": "fresh"}
        for day in DateRange(start, end).days() for code in codes]
    return {"start_date": start.isoformat(), "end_date": end.isoformat(), "updated": 0,
        "fresh": len(targets), "targets": targets, "errors": [], "status": "ready"}


@pytest.fixture
def integration(tmp_path, monkeypatch):
    monkeypatch.setenv("DATA_DIR", str(tmp_path / "module-instance"))
    monkeypatch.setenv("DATABASE_URL", "")
    monkeypatch.setenv("OPENAI_API_KEY", "")
    monkeypatch.setenv("SCHEDULE_ENABLED", "false")
    monkeypatch.setattr("app.rag.today", lambda: BASE)
    from app.main import create_app
    settings = Settings(data_dir=tmp_path / "range-instance", database_url="", api_key="",
        embedding_dimensions=3, schedule_enabled=False)
    store = Store(settings)
    model = Mock()
    model.embeddings.create.return_value.data = [SimpleNamespace(embedding=[1.0, 0.0, 0.0])]
    model.responses.create.return_value.output_text = json.dumps({"answer": "요청한 기간의 식단을 정리했어요. [S1]",
        "source_ids": ["S1"], "insufficient_evidence": False})
    agent = Agent(store, settings, model)
    refresh = Mock()
    refresh.ensure.side_effect = refresh_report
    scheduler = Mock()
    scheduler.status.return_value = {"enabled": False, "running": False}
    application = create_app(settings, store, agent, refresh=refresh, scheduler=scheduler)
    return SimpleNamespace(settings=settings, store=store, model=model, agent=agent,
        refresh=refresh, scheduler=scheduler, app=application)


def seed(store, day, code, *, fetched_at=None, all_meals=True, image=None):
    meal_types = ("조식", "중식", "석식") if all_meals else ("중식",)
    meals = [{"meal_type": kind, "items": [f"메뉴-{day.isoformat()}-{code}-{kind}"], "price": "5,000원", "tags": []}
        for kind in meal_types]
    assets = []
    if image:
        original = f"https://life.hanyang.ac.kr/uploads/{day.isoformat()}-{code}.jpg"
        meals[0]["image"] = original
        assets = [{"url": original, "local_path": image, "downloaded": True, "content_type": "image/jpeg", "alt": "공식 메뉴 사진"}]
    result = CrawlResult(code=code, name=RESTAURANTS[code]["name"], date=day.isoformat(),
        source_url=f"https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id={RESTAURANTS[code]['facility_id']}&date={day.isoformat()}",
        fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(), html="<html>공식 식단</html>", meals=meals, assets=assets)
    store.replace_collection(result, build_chunks(result))
    return result


@pytest.mark.parametrize("indexed", [False, True])
def test_full_week_keeps_all_84_meals_and_28_day_cafeteria_sources(integration, indexed):
    f = integration
    for day in NEXT_WEEK.days():
        for code in RESTAURANTS:
            seed(f.store, day, code)
    if indexed:
        f.settings.api_key = "mock-only-no-network"
        pending = f.store.pending()
        # Weak similarity cannot silently drop requested dates from a range.
        f.store.save_embeddings(pending, [[0.0, 1.0, 0.0]] * len(pending))
    result = f.agent.chat("다음주 학식 알려줘", [], BASE)
    assert result["date_range"] == {"start": "2026-11-16", "end": "2026-11-22"}
    assert len(result["sources"]) == 28
    assert {(source["date"], source["restaurant_code"]) for source in result["sources"]} == {
        (day.isoformat(), code) for day in NEXT_WEEK.days() for code in RESTAURANTS}
    for day in NEXT_WEEK.days():
        for code in RESTAURANTS:
            for meal in ("조식", "중식", "석식"):
                assert f"메뉴-{day.isoformat()}-{code}-{meal}" in result["answer"]
    assert "| 날짜 | 식당 |" in result["answer"]
    if indexed:
        assert result["mode"] == "openai"
        payload = json.loads(f.model.responses.create.call_args.kwargs["input"][0]["content"])
        assert payload["context"]["available_document_count"] == 84
        assert payload["context"]["retrieved_document_count"] == 84
        # The summary cited one source, while the complete table retains all.
        assert len(payload["sources"]) == len(result["sources"]) == 28
    else:
        assert result["mode"] == "retrieval"
        f.model.embeddings.create.assert_not_called()


def test_structured_api_range_overrides_message_dates_and_reaches_refresh(integration):
    f = integration
    explicit = DateRange(date(2026, 12, 1), date(2026, 12, 3))
    for day in explicit.days():
        seed(f.store, day, "re13")
    with TestClient(f.app) as client:
        response = client.post("/api/chat", json={"message": "다음주 창의인재원식당 학식 알려줘",
            "date": BASE.isoformat(), "start_date": "2026-12-01", "end_date": "2026-12-03"})
    assert response.status_code == 200
    data = response.json()
    assert data["date_range"] == {"start": "2026-12-01", "end": "2026-12-03"}
    assert {source["date"] for source in data["sources"]} == set(explicit.iso_dates())
    assert {source["restaurant_code"] for source in data["sources"]} == {"re13"}
    f.refresh.ensure.assert_called_once_with(explicit.start, explicit.end, ["re13"])


@pytest.mark.parametrize("extra,status", [
    ({"message": "2026-02-30 학식"}, 200),
    ({"message": "앞으로32일 학식"}, 200),
    ({"message": "다음달 초 학식"}, 200),
    ({"message": "학식", "start_date": "2026-11-01", "end_date": "2026-12-15"}, 200),
    ({"message": "학식", "start_date": "2026-02-30", "end_date": "2026-03-01"}, 422),
    ({"message": "학식", "start_date": "2026-11-20", "end_date": "2026-11-16"}, 422),
    ({"message": "학식", "start_date": "2026-11-16"}, 422),
])
def test_invalid_or_oversize_range_never_refreshes_or_generates(integration, extra, status):
    f = integration
    with TestClient(f.app) as client:
        response = client.post("/api/chat", json={"date": BASE.isoformat(), **extra})
    assert response.status_code == status
    if status == 200:
        assert response.json()["mode"] == "clarification"
        assert response.json()["sources"] == []
        assert response.json()["answer"]
    f.refresh.ensure.assert_not_called()
    f.model.embeddings.create.assert_not_called()
    f.model.responses.create.assert_not_called()


def test_date_only_next_week_followup_inherits_menu_cafeteria_and_meal_context(integration):
    f = integration
    history = [{"role": "user", "content": "창의인재원식당 오늘 점심 메뉴 알려줘"}]
    before = json.dumps(history, ensure_ascii=False)
    plan = f.agent.plan("그럼 다음주는?", history, BASE)
    assert plan.span == NEXT_WEEK
    assert plan.code == "re13" and plan.meal_type == "중식"
    assert plan.doc_types == ["menu"] and plan.menu_request
    generic = f.agent.plan("그럼 다음주 학식 점심은?", history, BASE)
    assert generic.span == NEXT_WEEK and generic.code is None and generic.meal_type == "중식"
    assert json.dumps(history, ensure_ascii=False) == before


def test_refresh_failure_excludes_stale_target_instead_of_treating_it_as_empty(integration):
    f = integration
    start, end = date(2026, 11, 14), date(2026, 11, 15)
    stale = (datetime.now(timezone.utc) - timedelta(hours=7)).isoformat()
    failed = seed(f.store, start, "re12", fetched_at=stale)
    seed(f.store, start, "re13")
    seed(f.store, end, "re12")
    report = refresh_report(start, end)
    target = next(target for target in report["targets"] if target["date"] == start.isoformat() and target["code"] == "re12")
    target.update(usable=False, status="error")
    report["errors"] = [{"date": start.isoformat(), "code": "re12", "stage": "crawl", "error": "Timeout"}]
    report["status"] = "partial"
    f.refresh.ensure.side_effect = None
    f.refresh.ensure.return_value = report
    with TestClient(f.app) as client:
        response = client.post("/api/chat", json={"message": "학식 알려줘", "date": BASE.isoformat(),
            "start_date": start.isoformat(), "end_date": end.isoformat()})
    assert response.status_code == 200
    result = response.json()
    pairs = {(source["date"], source["restaurant_code"]) for source in result["sources"]}
    assert (start.isoformat(), "re12") not in pairs
    assert (start.isoformat(), "re13") in pairs and (end.isoformat(), "re12") in pairs
    assert failed.meals[0]["items"][0] not in result["answer"]
    assert "새로 확인하지 못했어요" in result["answer"]
    assert result["refresh"] == report


def test_mixed_registered_and_unregistered_range_keeps_only_real_menu_photos(integration):
    f = integration
    span = DateRange(date(2026, 11, 16), date(2026, 11, 17))
    photos = ["/media/" + character * 64 + ".jpg" for character in ("a", "b")]
    invalid_photo = "/media/" + "c" * 64 + ".jpg"
    for day, photo in zip(span.days(), photos):
        result = seed(f.store, day, "re13", all_meals=False, image=photo)
        chunks = build_chunks(result)
        # A previous snapshot can contain a bad photo association. Range
        # bundling must not let a placeholder's picture become meal evidence.
        next(chunk for chunk in chunks if UNREGISTERED_MENU in chunk["text"])["images"] = [{"url": invalid_photo}]
        f.store.replace_collection(result, chunks)
    result = f.agent.chat("창의인재원식당 학식 알려줘", [], BASE, date_range=span)
    assert {image["url"] for image in result["images"]} == set(photos)
    assert all(image["url"] != invalid_photo for source in result["sources"] for image in source["images"])
    assert len(result["sources"]) == 2
    assert all(UNREGISTERED_MENU in source["text"] and source["metadata"]["range_bundle"] for source in result["sources"])


def test_range_payload_and_original_jamo_question_restore_from_conversation_db(integration):
    f = integration
    for day in NEXT_WEEK.days():
        for code in RESTAURANTS:
            seed(f.store, day, code)
    original = "ㄷㅏㅇㅡㅁㅈㅜ ㅎㅏㄱㅅㅣㄱ ㅇㅏㄹㄹㅕㅈㅜㅓ"
    with TestClient(f.app) as client:
        conversation = client.get("/api/conversations/current").json()
        response = client.post("/api/chat", json={"message": original, "date": BASE.isoformat(),
            "conversation_id": conversation["id"], "turn_id": str(uuid4())})
        assert response.status_code == 200
        data = response.json()
        cookies = dict(client.cookies)
    with TestClient(f.app, cookies=cookies) as reconnected:
        restored = reconnected.get("/api/conversations/current").json()
    assert restored["id"] == conversation["id"]
    assert restored["messages"][0]["content"] == original
    saved = restored["messages"][1]["payload"]
    assert saved["date_range"] == {"start": "2026-11-16", "end": "2026-11-22"}
    assert saved == {key: value for key, value in data.items() if key not in {"conversation_id", "turn_id"}}
    assert len(saved["sources"]) == 28


def test_main_lifespan_starts_and_stops_injected_scheduler(integration):
    f = integration
    f.scheduler.start.assert_not_called()
    with TestClient(f.app) as client:
        f.scheduler.start.assert_called_once_with()
        f.scheduler.stop.assert_not_called()
        assert client.get("/healthz").status_code == 200
    f.scheduler.stop.assert_called_once_with()


@pytest.mark.parametrize("indexed", [False, True])
def test_mixed_weekly_meals_photos_and_hours_keep_all_menu_dates(integration, indexed):
    f = integration
    for day in NEXT_WEEK.days():
        for code in RESTAURANTS:
            seed(f.store, day, code)
    for scope, code, title, text in [
        ("facility:1", "re12", "학생식당 운영시간", "중식 운영시간 11:30~13:30"),
        ("facility:2", "re11", "교직원식당 위치", "복지관 3층"),
    ]:
        f.store.replace_knowledge(scope, code=code, name=title, url="https://life.hanyang.ac.kr/facility",
            fetched_at=datetime.now(timezone.utc).isoformat(), payload={},
            chunks=knowledge_chunks(scope, title, [("기본 정보", text)]))
    if indexed:
        f.settings.api_key = "mock-only-no-network"
        pending = f.store.pending()
        f.store.save_embeddings(pending, [[1.0, 0.0, 0.0] if row["doc_type"] != "menu" else [0.0, 1.0, 0.0] for row in pending])

        def summary(**kwargs):
            payload = json.loads(kwargs["input"][0]["content"])
            profile = next(source for source in payload["sources"] if "중식 운영시간" in source["text"])
            return SimpleNamespace(output_text=json.dumps({"answer": f"학생식당 중식 운영시간은 11:30~13:30이에요. [{profile['id']}]",
                "source_ids": [profile["id"]], "insufficient_evidence": False}))

        f.model.responses.create.side_effect = summary
    result = f.agent.chat("다음주 학식 사진이랑 운영시간 알려줘", [], BASE)
    menu_sources = [source for source in result["sources"] if source["doc_type"] == "menu"]
    assert len(menu_sources) == 28
    assert {(source["date"], source["restaurant_code"]) for source in menu_sources} == {
        (day.isoformat(), code) for day in NEXT_WEEK.days() for code in RESTAURANTS}
    assert "| 날짜 | 식당 |" in result["answer"]
    assert "복지관 3층" not in result["answer"].split("| 날짜 | 식당 |", 1)[1]
    if indexed:
        profiles = [source for source in result["sources"] if source["doc_type"] == "profile"]
        assert len(profiles) == 1 and "중식 운영시간" in profiles[0]["text"]
    for day in NEXT_WEEK.days():
        for code in RESTAURANTS:
            for meal in ("조식", "중식", "석식"):
                assert f"메뉴-{day.isoformat()}-{code}-{meal}" in result["answer"]


def test_main_reuses_prepared_plan_when_refresh_crosses_monday(integration, monkeypatch):
    f = integration
    clock = [date(2026, 11, 15)]  # Sunday before the requested next week.
    monkeypatch.setattr("app.rag.today", lambda: clock[0])
    for day in NEXT_WEEK.days():
        seed(f.store, day, "re13")

    def crossing_midnight(start, end, codes):
        assert (start, end, codes) == (NEXT_WEEK.start, NEXT_WEEK.end, ["re13"])
        clock[0] = date(2026, 11, 16)  # Monday, while the official crawl runs.
        return refresh_report(start, end, codes)

    f.refresh.ensure.side_effect = crossing_midnight
    with TestClient(f.app) as client:
        response = client.post("/api/chat", json={"message": "다음주 창의인재원식당 점심", "date": "2026-11-15"})
    assert response.status_code == 200
    result = response.json()
    assert result["date_range"] == {"start": "2026-11-16", "end": "2026-11-22"}
    assert {source["date"] for source in result["sources"]} == set(NEXT_WEEK.iso_dates())
    assert all(source["restaurant_code"] == "re13" for source in result["sources"])
    assert "2026-11-23" not in result["answer"]


def test_structured_range_clarification_replays_identical_request_but_rejects_changed_range(integration):
    f = integration
    with TestClient(f.app) as client:
        conversation = client.get("/api/conversations/current").json()
        body = {"message": "학식 알려줘", "date": BASE.isoformat(), "conversation_id": conversation["id"],
            "turn_id": str(uuid4()), "start_date": "2026-11-01", "end_date": "2026-12-15"}
        first = client.post("/api/chat", json=body)
        replay = client.post("/api/chat", json=body)
        conflict = client.post("/api/chat", json={**body, "end_date": "2026-12-14"})
        restored = client.get("/api/conversations/current").json()
    assert first.status_code == replay.status_code == 200
    assert first.json()["mode"] == "clarification"
    assert replay.json() == first.json()
    assert conflict.status_code == 409
    assert len(restored["messages"]) == 2
    f.refresh.ensure.assert_not_called()
    f.model.responses.create.assert_not_called()

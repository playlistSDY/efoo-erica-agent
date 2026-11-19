from concurrent.futures import ThreadPoolExecutor
from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

with patch("dotenv.load_dotenv"):
    from app.crawler import CrawlResult
    from app.indexing import build_chunks, knowledge_chunks
    from app.refresh import MealRefreshService
    from app.store import Store


DAY = date(2026, 11, 13)
NOW = datetime(2026, 11, 13, 3, 0, tzinfo=timezone.utc)


def snapshot(day=DAY, *, code="re12", fetched_at=NOW, empty=False, item="칼국수"):
    return CrawlResult(code=code, name="학생식당" if code == "re12" else "교직원식당",
        date=day.isoformat(), source_url=f"https://life.hanyang.ac.kr/menu?code={code}&date={day.isoformat()}",
        fetched_at=fetched_at.isoformat() if isinstance(fetched_at, datetime) else fetched_at,
        html=f"<html>공식 {code} {day.isoformat()} 식단</html>",
        meals=[] if empty else [{"meal_type": "중식", "items": [item], "price": "5,000원", "tags": []}])


def scoped_index(store, settings, *, menu_targets):
    targets = set(menu_targets)
    rows = [row for row in store.pending() if row.get("doc_type", "menu") == "menu" and (row["day"], row["code"]) in targets]
    store.save_embeddings(rows, [[1.0, 0.0, 0.0]] * len(rows))
    return len(rows)


def seed(store, result, *, indexed=False):
    store.replace_collection(result, build_chunks(result))
    if indexed:
        scoped_index(store, store.settings, menu_targets=[(result.date, result.code)])


@pytest.fixture
def refresh(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, database_url="", api_key="isolated-refresh-test-key",
        embedding_model="text-embedding-3-small", embedding_dimensions=3, max_age_hours=24,
        meal_refresh_hours=6, meal_empty_refresh_minutes=60, refresh_retry_minutes=5,
        max_date_range_days=31, raw_s3_bucket="")
    store = Store(settings)
    clock, crawlers = [NOW], []

    def factory():
        crawler = Mock()
        crawler.session = object()
        crawler.fetch.side_effect = lambda code, day: snapshot(day, code=code, fetched_at=clock[0])
        crawlers.append(crawler)
        return crawler

    factory = Mock(side_effect=factory)
    indexer = Mock(side_effect=scoped_index)
    service = MealRefreshService(store, settings, crawler_factory=factory, indexer=indexer, now=lambda: clock[0])
    with patch("app.refresh.archive_assets", return_value=[]) as archive:
        yield SimpleNamespace(settings=settings, store=store, clock=clock, crawlers=crawlers,
            factory=factory, indexer=indexer, service=service, archive=archive)


def test_missing_day_is_collected_and_only_requested_menu_chunks_are_indexed(refresh):
    tomorrow = DAY + timedelta(days=1)
    seed(refresh.store, snapshot(tomorrow, item="다른 날짜 메뉴"))
    refresh.store.replace_knowledge("facility:49", code="fac_49", name="우편취급국",
        url="https://life.hanyang.ac.kr/facility?id=49", fetched_at=NOW.isoformat(), payload={},
        chunks=knowledge_chunks("facility:49", "우편취급국", [("공식 정보", "위치 안내")]))
    crawler = Mock(session=object())
    crawler.fetch.return_value = replace(snapshot(), weekly_meals={tomorrow.isoformat(): [
        {"meal_type": "중식", "items": ["주간표로 덮어쓰면 안 됨"], "price": "", "tags": []}]})
    refresh.factory.side_effect = None
    refresh.factory.return_value = crawler
    report = refresh.service.ensure(DAY, DAY, ["re12"])
    assert report["status"] == "ready"
    assert report["checked"] == report["updated"] == 1
    assert report["indexed"] == 3
    assert report["targets"][0]["status"] == "updated"
    assert report["targets"][0]["usable"] is True
    assert report["targets"][0]["index_status"] == "indexed"
    assert refresh.indexer.call_args.kwargs["menu_targets"] == [(DAY.isoformat(), "re12")]
    assert {row["doc_type"] for row in refresh.store.pending()} == {"menu", "profile"}
    assert refresh.store.collections(tomorrow.isoformat(), "re12")[0]["meals"][0]["items"] == ["다른 날짜 메뉴"]
    assert (refresh.settings.data_dir / "raw" / DAY.isoformat() / "re12" / "source.html").is_file()
    crawler.close.assert_called_once()


def test_general_ttl_reuses_fresh_snapshot_and_refreshes_at_six_hours(refresh):
    seed(refresh.store, snapshot(fetched_at=NOW - timedelta(hours=5, minutes=59)), indexed=True)
    fresh = refresh.service.ensure(DAY, DAY, ["re12"])
    assert fresh["fresh"] == 1 and fresh["updated"] == 0
    refresh.factory.assert_not_called()
    refresh.indexer.assert_not_called()
    refresh.clock[0] += timedelta(minutes=1)
    expired = refresh.service.ensure(DAY, DAY, ["re12"])
    assert expired["updated"] == 1
    refresh.factory.assert_called_once()
    refresh.crawlers[0].close.assert_called_once()
    # Identical text retains its embeddings during a freshness-only refresh.
    refresh.indexer.assert_not_called()


def test_empty_official_page_has_its_shorter_ttl_and_is_successful_evidence(refresh):
    seed(refresh.store, snapshot(fetched_at=NOW - timedelta(minutes=59), empty=True), indexed=True)
    assert refresh.service.ensure(DAY, DAY, ["re12"])["fresh"] == 1
    refresh.factory.assert_not_called()
    refresh.clock[0] += timedelta(minutes=1)
    crawler = Mock(session=object())
    crawler.fetch.return_value = snapshot(fetched_at=refresh.clock[0], empty=True)
    refresh.factory.side_effect = None
    refresh.factory.return_value = crawler
    report = refresh.service.ensure(DAY, DAY, ["re12"])
    assert report["updated"] == 1
    assert report["errors"] == []
    assert report["targets"][0]["empty"] is True
    assert report["targets"][0]["usable"] is True
    assert refresh.store.collections(DAY.isoformat(), "re12")[0]["meals"] == []


@pytest.mark.parametrize("empty", [False, True])
def test_refresh_ttl_never_exceeds_the_store_maximum_document_age(refresh, empty):
    refresh.settings.meal_refresh_hours = 48
    refresh.settings.meal_empty_refresh_minutes = 36 * 60
    refresh.settings.max_age_hours = 24
    seed(refresh.store, snapshot(fetched_at=NOW - timedelta(hours=25), empty=empty), indexed=True)
    crawler = Mock(session=object())
    crawler.fetch.return_value = snapshot(empty=empty)
    refresh.factory.side_effect = None
    refresh.factory.return_value = crawler
    report = refresh.service.ensure(DAY, DAY, ["re12"])
    assert report["updated"] == 1 and report["fresh"] == 0
    assert report["targets"][0]["fetched_at"] == NOW.isoformat()
    assert report["targets"][0]["usable"] is True
    crawler.fetch.assert_called_once_with("re12", DAY)


def test_force_bypasses_fresh_ttl_for_nonoverlapping_requests(refresh):
    seed(refresh.store, snapshot(), indexed=True)
    assert refresh.service.ensure(DAY, DAY, ["re12"])["fresh"] == 1
    assert refresh.service.ensure(DAY, DAY, ["re12"], force=True)["updated"] == 1
    assert refresh.service.ensure(DAY, DAY, ["re12"], force=True)["updated"] == 1
    assert refresh.factory.call_count == 2


def test_failed_fetch_is_unusable_preserves_old_data_and_has_retry_cooldown(refresh):
    old = snapshot(fetched_at=NOW - timedelta(hours=7))
    seed(refresh.store, old, indexed=True)
    crawler = Mock(session=object())
    crawler.fetch.side_effect = OSError("sensitive upstream detail must not appear")
    refresh.factory.side_effect = None
    refresh.factory.return_value = crawler
    failed = refresh.service.ensure(DAY, DAY, ["re12"])
    assert failed["status"] == "error"
    assert failed["targets"][0]["usable"] is False
    assert failed["targets"][0]["fetched_at"] == old.fetched_at
    assert failed["errors"][0]["stage"] == "crawl"
    assert failed["errors"][0]["operation"] == "fetch"
    assert failed["errors"][0]["error"] == "OSError"
    assert "sensitive upstream" not in json.dumps(failed)
    assert refresh.store.collections(DAY.isoformat(), "re12")[0]["meals"] == old.meals
    cooled = refresh.service.ensure(DAY, DAY, ["re12"], force=True)
    assert cooled["skipped"] == 1
    assert cooled["targets"][0]["status"] == "cooldown"
    assert cooled["targets"][0]["usable"] is False
    assert cooled["errors"][0]["error"] == "RetryCooldown"
    assert crawler.fetch.call_count == 1
    refresh.clock[0] += timedelta(minutes=5)
    crawler.fetch.side_effect = None
    crawler.fetch.return_value = snapshot(fetched_at=refresh.clock[0])
    recovered = refresh.service.ensure(DAY, DAY, ["re12"])
    assert recovered["updated"] == 1 and recovered["errors"] == []


def test_one_failed_target_does_not_stop_other_requested_cafeterias(refresh):
    crawler = Mock(session=object())
    crawler.fetch.side_effect = lambda code, day: (_ for _ in ()).throw(ValueError("invalid HTML")) if code == "re12" else snapshot(day, code=code)
    refresh.factory.side_effect = None
    refresh.factory.return_value = crawler
    report = refresh.service.ensure(DAY, DAY, ["re12", "re11"])
    assert report["status"] == "partial"
    assert report["checked"] == 2 and report["updated"] == 1
    states = {item["code"]: item for item in report["targets"]}
    assert states["re12"]["usable"] is False
    assert states["re11"]["usable"] is True
    assert refresh.indexer.call_args.kwargs["menu_targets"] == [(DAY.isoformat(), "re11")]
    crawler.close.assert_called_once()


def test_index_failure_is_separate_from_successful_collection_and_retries_without_recrawl(refresh):
    refresh.indexer.side_effect = RuntimeError("secret-index-detail")
    failed = refresh.service.ensure(DAY, DAY, ["re12"])
    assert failed["updated"] == 1 and failed["status"] == "partial"
    assert failed["targets"][0]["usable"] is True
    assert failed["targets"][0]["index_status"] == "error"
    assert failed["errors"][0]["stage"] == "index"
    assert failed["errors"][0]["operation"] == "embed"
    assert "secret-index-detail" not in json.dumps(failed)
    cooled = refresh.service.ensure(DAY, DAY, ["re12"])
    assert cooled["fresh"] == 1
    assert cooled["targets"][0]["index_status"] == "cooldown"
    assert cooled["errors"][0]["stage"] == "index"
    assert refresh.indexer.call_count == refresh.factory.call_count == 1
    refresh.clock[0] += timedelta(minutes=5)
    refresh.indexer.side_effect = scoped_index
    recovered = refresh.service.ensure(DAY, DAY, ["re12"])
    assert recovered["fresh"] == 1 and recovered["updated"] == 0
    assert recovered["indexed"] == 3
    assert recovered["index_status"] == "indexed"
    assert refresh.factory.call_count == 1


def test_missing_api_key_keeps_verified_documents_without_pretending_to_index(refresh):
    refresh.settings.api_key = ""
    report = refresh.service.ensure(DAY, DAY, ["re12"])
    assert report["updated"] == 1 and report["errors"] == []
    assert report["targets"][0]["usable"] is True
    assert report["index_status"] == "not_configured"
    assert report["indexed"] == 0
    refresh.indexer.assert_not_called()


def test_overlapping_force_requests_share_one_crawl_and_one_scoped_index(refresh):
    entered, release, second_waiting = threading.Event(), threading.Event(), threading.Event()
    crawler = Mock(session=object())

    def blocked_fetch(code, day):
        entered.set()
        assert release.wait(2)
        return snapshot(day, code=code)

    crawler.fetch.side_effect = blocked_fetch
    refresh.factory.side_effect = None
    refresh.factory.return_value = crawler
    second = MealRefreshService(refresh.store, refresh.settings, crawler_factory=refresh.factory,
        indexer=refresh.indexer, now=lambda: refresh.clock[0])
    original = second._collect

    def mark_waiting(*args, **kwargs):
        second_waiting.set()
        return original(*args, **kwargs)

    second._collect = mark_waiting
    with ThreadPoolExecutor(max_workers=2) as executor:
        first_future = executor.submit(refresh.service.ensure, DAY, DAY, ["re12"], True)
        assert entered.wait(1)
        second_future = executor.submit(second.ensure, DAY, DAY, ["re12"], True)
        assert second_waiting.wait(1)
        release.set()
        reports = [first_future.result(timeout=3), second_future.result(timeout=3)]
    assert sum(report["updated"] for report in reports) == 1
    assert sum(report["fresh"] for report in reports) == 1
    assert any(report["targets"][0].get("shared") for report in reports)
    crawler.fetch.assert_called_once()
    crawler.close.assert_called_once()
    refresh.indexer.assert_called_once()


def test_independent_targets_still_serialize_indexing(refresh):
    entered, release, second_ready = threading.Event(), threading.Event(), threading.Event()
    guard = threading.Lock()
    active, maximum = [0], [0]

    def blocked_index(store, settings, *, menu_targets):
        with guard:
            active[0] += 1
            maximum[0] = max(maximum[0], active[0])
        if not entered.is_set():
            entered.set()
            assert release.wait(2)
        try:
            return scoped_index(store, settings, menu_targets=menu_targets)
        finally:
            with guard:
                active[0] -= 1

    refresh.indexer.side_effect = blocked_index
    second = MealRefreshService(refresh.store, refresh.settings, crawler_factory=refresh.factory,
        indexer=refresh.indexer, now=lambda: refresh.clock[0])
    original = second._index

    def mark_index_ready(*args, **kwargs):
        second_ready.set()
        return original(*args, **kwargs)

    second._index = mark_index_ready
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(refresh.service.ensure, DAY, DAY, ["re12"])
        assert entered.wait(1)
        other = executor.submit(second.ensure, DAY, DAY, ["re11"])
        assert second_ready.wait(1)
        release.set()
        reports = [first.result(timeout=3), other.result(timeout=3)]
    assert all(report["errors"] == [] for report in reports)
    assert refresh.indexer.call_count == 2
    assert maximum[0] == 1
    assert refresh.store.pending() == []


@pytest.mark.parametrize("wrong", ["date", "code"])
def test_mismatched_daily_response_is_rejected_before_assets_or_snapshot_replacement(refresh, wrong):
    old = snapshot(fetched_at=NOW - timedelta(hours=7))
    seed(refresh.store, old, indexed=True)
    crawler = Mock(session=object())
    crawler.fetch.return_value = snapshot(DAY + timedelta(days=1)) if wrong == "date" else snapshot(code="re11")
    refresh.factory.side_effect = None
    refresh.factory.return_value = crawler
    report = refresh.service.ensure(DAY, DAY, ["re12"])
    assert report["targets"][0]["usable"] is False
    assert report["errors"][0]["operation"] == "fetch"
    refresh.archive.assert_not_called()
    assert refresh.store.collections(DAY.isoformat(), "re12")[0]["fetched_at"] == old.fetched_at
    crawler.close.assert_called_once()


def test_multiple_targets_reuse_one_crawler_asset_session_and_close_once(refresh):
    report = refresh.service.ensure(DAY, DAY + timedelta(days=1), ["re12", "re12"])
    assert report["checked"] == report["updated"] == 2
    refresh.factory.assert_called_once()
    session = refresh.crawlers[0].session
    assert all(call.kwargs["session"] is session for call in refresh.archive.call_args_list)
    refresh.crawlers[0].close.assert_called_once()
    assert set(refresh.indexer.call_args.kwargs["menu_targets"]) == {
        (DAY.isoformat(), "re12"), ((DAY + timedelta(days=1)).isoformat(), "re12")}


def test_archive_failure_preserves_previous_snapshot_and_closes_session(refresh):
    old = snapshot(fetched_at=NOW - timedelta(hours=7))
    seed(refresh.store, old, indexed=True)
    refresh.archive.side_effect = OSError("disk failed")
    report = refresh.service.ensure(DAY, DAY, ["re12"])
    assert report["updated"] == 0
    assert report["errors"][0]["operation"] == "archive_assets"
    assert refresh.store.collections(DAY.isoformat(), "re12")[0]["fetched_at"] == old.fetched_at
    refresh.crawlers[0].close.assert_called_once()


@pytest.mark.parametrize("options", [
    {"start": DAY + timedelta(days=1), "end": DAY},
    {"start": DAY, "end": DAY + timedelta(days=31)},
    {"start": "2026-11-13", "end": DAY},
    {"start": DAY, "end": DAY, "restaurant_codes": "re12"},
    {"start": DAY, "end": DAY, "restaurant_codes": []},
    {"start": DAY, "end": DAY, "restaurant_codes": ["re99"]},
])
def test_invalid_ranges_or_cafeterias_fail_before_resources_are_created(refresh, options):
    with pytest.raises(ValueError):
        refresh.service.ensure(**options)
    refresh.factory.assert_not_called()
    refresh.indexer.assert_not_called()


@pytest.mark.parametrize("timestamp", ["invalid", NOW + timedelta(hours=1)])
def test_invalid_or_future_fetch_timestamps_are_not_treated_as_fresh(refresh, timestamp):
    seed(refresh.store, snapshot(fetched_at=timestamp), indexed=True)
    assert refresh.service.ensure(DAY, DAY, ["re12"])["updated"] == 1

from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch
from urllib.parse import parse_qs, urlparse

import pytest
import requests

with patch("dotenv.load_dotenv"):
    from app.crawler import parse_facility_html
    from app.facility_refresh import BASE, FacilityRefreshService
    from app.indexing import index_facility
    from app.store import Store


NOW = datetime(2026, 11, 14, 1, 0, tzinfo=timezone.utc)
SHOP = {"id": 18, "name": "NEW YORK BURGER", "category": "일반음식점"}
POST = {"id": 49, "name": "안산 한양대 우편 취급국", "category": "생활서비스"}
MEAL = {"id": 1, "name": "학생식당", "category": "구내식당"}


def source_url(record):
    page = "detail.php" if record["category"] == "구내식당" else "shop-detail.php"
    return f"{BASE}/theme/pages/facilities/{page}?id={record['id']}"


def page(record=SHOP, *, marker="최신 공식 시설 안내", identity=None):
    identity = record["id"] if identity is None else identity
    prefix = "fac" if record["category"] == "구내식당" else "shop"
    return f'''<html lang="ko"><head><link rel="alternate" hreflang="ko" href="?id={identity}"></head>
      <body><main><div class="{prefix}-name">{record['name']}</div><div class="{prefix}-category">{record['category']}</div>
      <table class="info-table"><tr><td class="info-label">위치</td><td>학생복지관 2층</td></tr>
      <tr><td class="info-label">영업시간</td><td>평일 09:00~18:00</td></tr></table>
      <div class="shop-description-area">{marker}</div></main></body></html>'''


def response(html, *, url, status=200):
    result = requests.Response()
    result.status_code, result.url = status, url
    result._content = html.encode("utf-8")
    result._content_consumed = True
    result.encoding = "ISO-8859-1"
    return result


def scoped_index(store, settings, *, scopes):
    scopes = set(scopes)
    rows = [row for row in store.pending() if row.get("scope") in scopes]
    store.save_embeddings(rows, [[1.0, 0.0, 0.0]] * len(rows))
    return len(rows)


def seed(store, record, *, fetched_at=NOW, indexed=True, marker="기존 공식 정보"):
    parsed = parse_facility_html(page(record, marker=marker), expected_facility_id=str(record["id"]))
    parsed["facility_id"] = record["id"]
    index_facility(store, parsed, [], fetched_at.isoformat(), source_url(record), record)
    if indexed:
        scoped_index(store, store.settings, scopes=[f"facility:{record['id']}"])


def knowledge(store, identifier):
    return next(row for row in store.knowledge() if row["scope"] == f"facility:{identifier}")


@pytest.fixture
def refresh(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, database_url="", api_key="isolated-facility-key",
        embedding_model="text-embedding-3-small", embedding_dimensions=3, max_age_hours=24, refresh_retry_minutes=5)
    store, clock, sessions = Store(settings), [NOW], []
    records = {record["id"]: record for record in [SHOP, POST, MEAL]}

    def factory():
        session = Mock()

        def get(url, **kwargs):
            identifier = int(parse_qs(urlparse(url).query)["id"][0])
            record = records.get(identifier, {"id": identifier, "name": f"시설 {identifier}", "category": "생활서비스"})
            return response(page(record), url=url)

        session.get.side_effect = get
        sessions.append(session)
        return session

    factory = Mock(side_effect=factory)
    indexer = Mock(side_effect=scoped_index)
    service = FacilityRefreshService(store, settings, session_factory=factory, indexer=indexer, now=lambda: clock[0])
    with patch("app.facility_refresh.archive_assets", return_value=[]) as archive:
        yield SimpleNamespace(settings=settings, store=store, clock=clock, sessions=sessions,
            factory=factory, indexer=indexer, service=service, archive=archive)


def test_missing_selected_profile_is_refreshed_and_other_profiles_remain_unchanged(refresh):
    seed(refresh.store, POST, indexed=False)
    previous = knowledge(refresh.store, POST["id"])
    report = refresh.service.ensure([SHOP])
    assert report["checked"] == report["updated"] == 1
    assert report["status"] == "ready"
    assert report["targets"][0] == {"id": 18, "code": "fac_18", "status": "updated", "usable": True,
        "fetched_at": NOW.isoformat(), "index_status": "indexed"}
    assert report["indexed"] > 0
    assert refresh.indexer.call_args.kwargs["scopes"] == ["facility:18"]
    assert knowledge(refresh.store, POST["id"]) == previous
    assert {row["scope"] for row in refresh.store.pending()} == {"facility:49"}
    assert knowledge(refresh.store, SHOP["id"])["payload"]["restaurant_info"]["description"]["text"] == "최신 공식 시설 안내"
    assert (refresh.settings.data_dir / "portal" / "facility-18.html").read_text() == page(SHOP)
    refresh.sessions[0].get.assert_called_once_with(source_url(SHOP), timeout=(3, 12), allow_redirects=False,
        headers={"User-Agent": "EfooWebAcademicCrawler/1.0"})
    refresh.sessions[0].close.assert_called_once()


def test_fresh_profiles_do_not_create_http_sessions_and_expire_at_twenty_four_hours(refresh):
    seed(refresh.store, SHOP, fetched_at=NOW - timedelta(hours=23, minutes=59))
    assert refresh.service.ensure([SHOP])["fresh"] == 1
    refresh.factory.assert_not_called()
    refresh.indexer.assert_not_called()
    refresh.clock[0] += timedelta(minutes=1)
    expired = refresh.service.ensure([SHOP])
    assert expired["updated"] == 1
    assert expired["targets"][0]["usable"] is True
    refresh.factory.assert_called_once()


def test_cafeteria_and_shop_routing_share_one_session_and_one_scoped_index_call(refresh):
    report = refresh.service.ensure([MEAL, SHOP])
    assert report["updated"] == 2
    assert [target["code"] for target in report["targets"]] == ["re12", "fac_18"]
    refresh.factory.assert_called_once()
    session = refresh.sessions[0]
    assert [call.args[0] for call in session.get.call_args_list] == [source_url(MEAL), source_url(SHOP)]
    assert all(call.kwargs["session"] is session for call in refresh.archive.call_args_list)
    assert refresh.indexer.call_args.kwargs["scopes"] == ["facility:1", "facility:18"]
    refresh.indexer.assert_called_once()
    session.close.assert_called_once()


def test_failed_profile_is_preserved_unusable_and_cooled_down_while_other_profiles_succeed(refresh):
    seed(refresh.store, SHOP, fetched_at=NOW - timedelta(hours=25))
    previous = knowledge(refresh.store, SHOP["id"])
    session = Mock()

    def get(url, **kwargs):
        if url == source_url(SHOP):
            raise requests.Timeout("sensitive upstream text")
        return response(page(POST), url=url)

    session.get.side_effect = get
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    report = refresh.service.ensure([SHOP, POST])
    assert report["status"] == "partial"
    assert report["updated"] == 1
    targets = {target["id"]: target for target in report["targets"]}
    assert targets[18]["usable"] is False
    assert targets[49]["usable"] is True
    assert report["errors"][0]["stage"] == "crawl"
    assert report["errors"][0]["error"] == "Timeout"
    assert "sensitive upstream" not in json.dumps(report)
    assert knowledge(refresh.store, SHOP["id"]) == previous
    assert refresh.indexer.call_args.kwargs["scopes"] == ["facility:49"]
    cooled = refresh.service.ensure([SHOP], force=True)
    assert cooled["targets"][0]["status"] == "cooldown"
    assert cooled["targets"][0]["usable"] is False
    assert session.get.call_count == 2
    refresh.clock[0] += timedelta(minutes=5)
    session.get.side_effect = lambda url, **kwargs: response(page(SHOP), url=url)
    assert refresh.service.ensure([SHOP])["updated"] == 1


@pytest.mark.parametrize("url,status,identity,operation", [
    ("https://untrusted.example/detail?id=18", 200, 18, "fetch"),
    (source_url(SHOP), 302, 18, "fetch"),
    (source_url(POST), 200, 18, "fetch"),
    (source_url(SHOP), 200, 49, "parse"),
])
def test_redirect_wrong_origin_or_mismatched_identity_never_replaces_a_profile(refresh, url, status, identity, operation):
    seed(refresh.store, SHOP, fetched_at=NOW - timedelta(hours=25))
    previous = knowledge(refresh.store, SHOP["id"])
    session = Mock()
    session.get.return_value = response(page(SHOP, identity=identity), url=url, status=status)
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    report = refresh.service.ensure([SHOP])
    assert report["targets"][0]["usable"] is False
    assert report["errors"][0]["operation"] == operation
    assert knowledge(refresh.store, SHOP["id"]) == previous
    refresh.archive.assert_not_called()
    assert not (refresh.settings.data_dir / "portal" / "facility-18.html").exists()
    session.close.assert_called_once()


def test_index_failure_is_separate_and_retries_without_fetching_a_fresh_profile_again(refresh):
    refresh.indexer.side_effect = RuntimeError("private index detail")
    report = refresh.service.ensure([SHOP])
    assert report["updated"] == 1
    assert report["targets"][0]["usable"] is True
    assert report["targets"][0]["index_status"] == "error"
    assert report["errors"][0]["stage"] == "index"
    assert "private index detail" not in json.dumps(report)
    cooled = refresh.service.ensure([SHOP])
    assert cooled["fresh"] == 1
    assert cooled["targets"][0]["index_status"] == "cooldown"
    assert refresh.factory.call_count == refresh.indexer.call_count == 1
    refresh.clock[0] += timedelta(minutes=5)
    refresh.indexer.side_effect = scoped_index
    recovered = refresh.service.ensure([SHOP])
    assert recovered["fresh"] == 1 and recovered["updated"] == 0
    assert recovered["indexed"] > 0
    assert recovered["errors"] == []
    assert refresh.factory.call_count == 1


def test_missing_api_key_and_explicitly_disabled_indexer_keep_official_text_updates_available(refresh):
    refresh.settings.api_key = ""
    report = refresh.service.ensure([SHOP])
    assert report["status"] == "ready"
    assert report["targets"][0]["usable"] is True
    assert report["index_status"] == "not_configured"
    refresh.indexer.assert_not_called()
    disabled = FacilityRefreshService(refresh.store, refresh.settings, session_factory=refresh.factory,
        indexer=None, now=lambda: refresh.clock[0])
    report = disabled.ensure([SHOP])
    assert report["fresh"] == 1
    assert report["index_status"] == "not_requested"
    assert refresh.factory.call_count == 1


def test_overlapping_force_requests_share_one_official_fetch_and_one_profile_embedding(refresh):
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    session = Mock()

    def get(url, **kwargs):
        entered.set()
        assert release.wait(2)
        return response(page(SHOP), url=url)

    session.get.side_effect = get
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    second = FacilityRefreshService(refresh.store, refresh.settings, session_factory=refresh.factory,
        indexer=refresh.indexer, now=lambda: refresh.clock[0])
    original = second._refresh_one

    def mark_waiting(*args, **kwargs):
        waiting.set()
        return original(*args, **kwargs)

    second._refresh_one = mark_waiting
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(refresh.service.ensure, [SHOP], True)
        assert entered.wait(1)
        other = executor.submit(second.ensure, [SHOP], True)
        assert waiting.wait(1)
        release.set()
        reports = [first.result(timeout=3), other.result(timeout=3)]
    assert sum(report["updated"] for report in reports) == 1
    assert sum(report["fresh"] for report in reports) == 1
    assert any(report["targets"][0].get("shared") for report in reports)
    session.get.assert_called_once()
    session.close.assert_called_once()
    refresh.indexer.assert_called_once()


@pytest.mark.parametrize("records", [None, {}, [None], [{**SHOP, "id": True}], [{**SHOP, "id": 0}],
    [{**SHOP, "id": "18"}], [SHOP, SHOP], [{**SHOP, "name": ""}], [{**SHOP, "category": ""}],
    [{"id": identifier, "name": "시설", "category": "생활서비스"} for identifier in range(1, 152)]])
def test_invalid_or_unbounded_record_sets_fail_before_opening_a_session(refresh, records):
    with pytest.raises(ValueError):
        refresh.service.ensure(records)
    refresh.factory.assert_not_called()
    refresh.indexer.assert_not_called()


def test_empty_selected_scope_creates_no_session_and_makes_no_api_call(refresh):
    report = refresh.service.ensure([])
    assert report["checked"] == report["updated"] == report["fresh"] == 0
    assert report["errors"] == []
    assert report["status"] == "ready"
    refresh.factory.assert_not_called()
    refresh.indexer.assert_not_called()

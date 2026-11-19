from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timedelta, timezone
import json
import threading
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests

with patch("dotenv.load_dotenv"):
    from app.indexing import index_notice, knowledge_chunks
    from app.notice_refresh import INDEX_SCOPE, INDEX_URL, NoticeRefreshService
    from app.store import Store


NOW = datetime(2026, 11, 14, 3, 0, tzinfo=timezone.utc)
RECORD = {"id": 13, "title": "창의관식당 결제 방식 안내", "content": "공식 정책 본문"}


def detail_url(identifier=13):
    return f"https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id={identifier}&board_lang=ko"


def board(records=None):
    records = [RECORD] if records is None else records
    return '<html lang="ko"><head><title>공식 공지</title></head><body><main><div id="notice-list-container"></div></main><script>const dbNotices = ' + json.dumps(records, ensure_ascii=False) + ';</script></body></html>'


def detail(record=RECORD, *, identity=None):
    identifier = record["id"] if identity is None else identity
    return f'<html><head><link rel="alternate" hreflang="ko" href="{detail_url(identifier)}"></head><body><main><h1 class="post-title">{record["title"]}</h1><div class="post-meta">2026년 11월 13일</div><div class="post-content">기숙사 창의관식당 결제는 온라인에서 진행합니다.</div></main></body></html>'


def response(html, *, url, status=200):
    result = requests.Response()
    result.status_code, result.url = status, url
    result._content = html.encode("utf-8")
    result._content_consumed = True
    result.encoding = "ISO-8859-1"
    return result


def scoped_index(store, settings, *, scopes):
    rows = [row for row in store.pending() if row.get("scope") in set(scopes)]
    store.save_embeddings(rows, [[1.0, 0.0, 0.0]] * len(rows))
    return len(rows)


def seed_notice(store, *, record=RECORD, stamp=NOW, indexed=True):
    notice = {"title": record["title"], "text": "기존 공식 정책", "url": detail_url(record["id"]),
        "fetched_at": stamp.isoformat(), "metadata": "공식 작성일", "restaurant_codes": ["re13"],
        "images": [], "assets": [], "raw_record": record}
    index_notice(store, notice, record["id"])
    if indexed:
        scoped_index(store, store.settings, scopes=[f"notice:{record['id']}"])


def seed_board(store, *, records=None, stamp=NOW):
    records = [RECORD] if records is None else records
    store.replace_knowledge(INDEX_SCOPE, code="campus", name="공식 한국어 공지 목록", url=INDEX_URL,
        fetched_at=stamp.isoformat(), payload={"verified": True, "records": records, "count": len(records)}, chunks=[])


def source(store, scope):
    return next(row for row in store.knowledge() if row["scope"] == scope)


@pytest.fixture
def refresh(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, database_url="", api_key="isolated-notice-test",
        embedding_model="test-embedding", embedding_dimensions=3, max_age_hours=24, refresh_retry_minutes=5)
    store, clock, sessions = Store(settings), [NOW], []

    def factory():
        session = Mock()
        session.get.side_effect = lambda url, **kwargs: response(board() if url == INDEX_URL else detail(), url=url)
        sessions.append(session)
        return session

    factory = Mock(side_effect=factory)
    indexer = Mock(side_effect=scoped_index)
    service = NoticeRefreshService(store, settings, session_factory=factory, indexer=indexer, now=lambda: clock[0])
    with patch("app.notice_refresh.archive_assets", return_value=[]) as archive:
        yield SimpleNamespace(settings=settings, store=store, clock=clock, sessions=sessions, factory=factory,
                              indexer=indexer, service=service, archive=archive)


def test_current_notice_refresh_preserves_unrelated_sources_and_only_indexes_notice_scopes(refresh):
    refresh.store.replace_knowledge("facility:49", code="fac_49", name="우편취급국", url="https://life.hanyang.ac.kr/",
        fetched_at=NOW.isoformat(), payload={}, chunks=knowledge_chunks("facility:49", "우편취급국", [("안내", "공식 시설 정보")]))
    report = refresh.service.ensure()
    assert report["status"] == "ready"
    assert report["checked"] == report["updated"] == 1
    assert report["listed_ids"] == [13]
    assert report["unusable_scopes"] == []
    assert report["index"]["usable"] is True and report["index"]["count"] == 1
    assert report["targets"][0]["usable"] is True
    assert report["targets"][0]["index_status"] == "indexed"
    assert report["indexed"] == 1
    assert refresh.indexer.call_args.kwargs["scopes"] == ["notice:13"]
    assert {row["scope"] for row in refresh.store.pending()} == {"facility:49"}
    session = refresh.sessions[0]
    assert [call.args[0] for call in session.get.call_args_list] == [INDEX_URL, detail_url()]
    assert all(call.kwargs["allow_redirects"] is False and call.kwargs["timeout"] == (3, 12) for call in session.get.call_args_list)
    assert all(call.kwargs["session"] is session for call in refresh.archive.call_args_list)
    session.close.assert_called_once()
    folder = refresh.settings.data_dir / "portal"
    assert (folder / "notice-index.html").read_text() == board()
    assert (folder / "notice-13.html").read_text() == detail()
    assert source(refresh.store, "notice:13")["payload"]["raw_record"] == RECORD


def test_fresh_index_and_details_use_no_http_and_refresh_at_twenty_four_hours(refresh):
    refresh.service.ensure()
    refresh.indexer.reset_mock()
    refresh.clock[0] += timedelta(hours=23, minutes=59)
    cached = refresh.service.ensure()
    assert cached["fresh"] == 1 and cached["updated"] == 0
    assert cached["index"]["status"] == "fresh"
    assert refresh.factory.call_count == 1
    refresh.indexer.assert_not_called()
    refresh.clock[0] += timedelta(minutes=1)
    expired = refresh.service.ensure()
    assert expired["updated"] == 1 and expired["index"]["status"] == "updated"
    assert refresh.factory.call_count == 2
    refresh.indexer.assert_not_called()  # Identical body vectors are retained.


def test_verified_empty_board_is_a_distinct_source_and_archived_unlisted_notices_are_not_deleted(refresh):
    seed_notice(refresh.store)
    previous = source(refresh.store, "notice:13")
    session = Mock()
    session.get.return_value = response(board([]), url=INDEX_URL)
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    report = refresh.service.ensure()
    assert report["status"] == "ready"
    assert report["listed_ids"] == []
    assert report["index"]["verified_empty"] is True
    assert report["index"]["count"] == 0
    assert report["index"]["source"]["url"] == INDEX_URL
    assert report["index"]["source"]["metadata"]["verified_empty"] is True
    assert report["unusable_scopes"] == ["notice:13"]
    assert report["targets"][0]["status"] == "not_listed"
    assert source(refresh.store, "notice:13") == previous
    assert source(refresh.store, INDEX_SCOPE)["payload"]["verified_empty"] is True
    assert refresh.service.ensure()["listed_ids"] == []
    session.get.assert_called_once()
    refresh.indexer.assert_not_called()


@pytest.mark.parametrize("html", [
    "<html>점검 중</html>", '<main><div id="notice-list-container"></div></main>',
    '<script>const dbNotices = [];</script>', board([{**RECORD, "id": True}]), board([RECORD, RECORD]),
    board([{**RECORD, "title": "  "}]), board([{**RECORD, "id": "13"}]),
])
def test_malformed_index_is_not_treated_as_an_empty_board_and_preserves_old_sources(refresh, html):
    seed_board(refresh.store, stamp=NOW - timedelta(hours=25))
    seed_notice(refresh.store)
    old_board, old_notice = source(refresh.store, INDEX_SCOPE), source(refresh.store, "notice:13")
    session = Mock()
    session.get.return_value = response(html, url=INDEX_URL)
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    report = refresh.service.ensure()
    assert report["status"] == "error"
    assert report["listed_ids"] is None
    assert report["index"]["verified_empty"] is False
    assert set(report["unusable_scopes"]) == {INDEX_SCOPE, "notice:13"}
    assert source(refresh.store, INDEX_SCOPE) == old_board
    assert source(refresh.store, "notice:13") == old_notice
    session.close.assert_called_once()


def test_network_failure_uses_safe_errors_and_bounded_retry_cooldown(refresh):
    seed_notice(refresh.store)
    session = Mock()
    session.get.side_effect = requests.Timeout("private upstream credential detail")
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    failed = refresh.service.ensure()
    cooled = refresh.service.ensure()
    assert failed["errors"][0]["error"] == "Timeout"
    assert cooled["errors"][0]["error"] == "RetryCooldown"
    assert "private upstream" not in json.dumps(failed)
    assert failed["targets"][0]["usable"] is False
    session.get.assert_called_once()
    refresh.clock[0] += timedelta(minutes=5)
    session.get.side_effect = lambda url, **kwargs: response(board() if url == INDEX_URL else detail(), url=url)
    assert refresh.service.ensure()["status"] == "ready"


@pytest.mark.parametrize("url,status,html", [
    ("https://untrusted.example/notice?id=13", 200, detail()),
    (detail_url(14), 200, detail()), (detail_url(), 302, detail()),
    (detail_url(), 200, detail(identity=14)),
])
def test_detail_origin_url_id_and_html_identity_must_all_match_before_replacing(refresh, url, status, html):
    seed_board(refresh.store)
    seed_notice(refresh.store, stamp=NOW - timedelta(hours=25))
    old = source(refresh.store, "notice:13")
    session = Mock()
    session.get.return_value = response(html, url=url, status=status)
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    report = refresh.service.ensure()
    assert report["status"] == "partial"
    assert report["index"]["usable"] is True
    assert report["targets"][0]["usable"] is False
    assert report["unusable_scopes"] == ["notice:13"]
    assert source(refresh.store, "notice:13") == old
    refresh.archive.assert_not_called()


def test_one_bad_detail_does_not_prevent_other_current_notices_from_refreshing(refresh):
    records = [RECORD, {"id": 14, "title": "시설 공지"}]
    seed_board(refresh.store, records=records)
    session = Mock()
    session.get.side_effect = lambda url, **kwargs: response("<html>점검</html>" if url == detail_url(13) else detail(records[1]), url=url)
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    report = refresh.service.ensure()
    assert report["checked"] == 2 and report["updated"] == 1
    assert report["status"] == "partial"
    assert report["unusable_scopes"] == ["notice:13"]
    assert refresh.indexer.call_args.kwargs["scopes"] == ["notice:14"]
    assert source(refresh.store, "notice:14")["name"] == "시설 공지"


def test_changed_index_body_forces_detail_refresh_even_when_older_detail_is_still_within_ttl(refresh):
    old_record = {**RECORD, "content": "이전 정책 본문"}
    seed_notice(refresh.store, record=old_record)
    report = refresh.service.ensure()
    assert report["updated"] == 1
    assert source(refresh.store, "notice:13")["payload"]["raw_record"]["content"] == RECORD["content"]


def test_index_failures_are_distinct_and_retry_without_recrawling_fresh_public_pages(refresh):
    refresh.indexer.side_effect = RuntimeError("sensitive index detail")
    failed = refresh.service.ensure()
    assert failed["targets"][0]["usable"] is True
    assert failed["targets"][0]["index_status"] == "error"
    assert failed["errors"][0]["stage"] == "index"
    assert "sensitive index" not in json.dumps(failed)
    cooled = refresh.service.ensure()
    assert cooled["targets"][0]["index_status"] == "cooldown"
    assert refresh.factory.call_count == refresh.indexer.call_count == 1
    refresh.clock[0] += timedelta(minutes=5)
    refresh.indexer.side_effect = scoped_index
    recovered = refresh.service.ensure()
    assert recovered["fresh"] == 1 and recovered["indexed"] == 1
    assert recovered["errors"] == []
    assert refresh.factory.call_count == 1


def test_no_api_key_keeps_verified_notice_preview_without_an_embedding_request(refresh):
    refresh.settings.api_key = ""
    report = refresh.service.ensure()
    assert report["status"] == "ready"
    assert report["targets"][0]["usable"] is True
    assert report["index_status"] == "not_configured"
    assert report["indexed"] == 0
    refresh.indexer.assert_not_called()


def test_overlapping_requests_share_one_index_detail_fetch_and_one_scoped_embedding(refresh):
    entered, release, waiting = threading.Event(), threading.Event(), threading.Event()
    session = Mock()

    def get(url, **kwargs):
        if url == INDEX_URL:
            entered.set()
            assert release.wait(2)
        return response(board() if url == INDEX_URL else detail(), url=url)

    session.get.side_effect = get
    refresh.factory.side_effect = None
    refresh.factory.return_value = session
    second = NoticeRefreshService(refresh.store, refresh.settings, session_factory=refresh.factory,
        indexer=refresh.indexer, now=lambda: refresh.clock[0])
    original = refresh.service._state.lock

    class ObservedLock:
        count = 0
        guard = threading.Lock()

        def __enter__(self):
            with self.guard:
                self.count += 1
                if self.count == 2:
                    waiting.set()
            original.acquire()

        def __exit__(self, *exception):
            original.release()

    refresh.service._state.lock = ObservedLock()
    with ThreadPoolExecutor(max_workers=2) as executor:
        first = executor.submit(refresh.service.ensure)
        assert entered.wait(1)
        other = executor.submit(second.ensure)
        assert waiting.wait(1)
        release.set()
        reports = [first.result(timeout=3), other.result(timeout=3)]
    assert sum(report["updated"] for report in reports) == 1
    assert sum(report["fresh"] for report in reports) == 1
    assert any(report.get("shared") for report in reports)
    assert session.get.call_count == 2
    session.close.assert_called_once()
    refresh.indexer.assert_called_once()

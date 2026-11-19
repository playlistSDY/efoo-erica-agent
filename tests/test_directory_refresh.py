from datetime import datetime
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest
import requests

with patch("dotenv.load_dotenv"):
    from app.indexing import knowledge_chunks
    from app.portal import BASE, collect_directory, collect_portal
    from app.store import Store


SOURCE = BASE + "/theme/pages/facilities/index.php"
RECORDS = [
    {"id": 1, "name": "학생식당", "category": "구내식당", "category_display": "구내식당",
     "location": "학생복지관 2층", "image_url": "/uploads/student.jpg",
     "description": '<p>공식 안내<img src="/uploads/menu-guide.png"></p>', "display": {"enabled": True}},
    {"id": 18, "name": "NEW YORK BURGER", "category": "일반음식점", "location": "학생복지관 2층"},
]


def page(records=None, *, count=None, main=True, grid=True):
    records = RECORDS if records is None else records
    total = len(records) if count is None else count
    content = ('<div id="fac-grid" class="fac-grid"></div>' if grid else "")
    content += f'<div id="fac-category-tabs"><div data-cat="전체"><span class="fac-category-count">{total}</span></div></div>'
    content = f"<main>{content}</main>" if main else content
    return ('<html lang="ko"><head><title>공식 시설 안내</title><meta name="description" content="공개 시설 목록"></head><body>'
            + content + "<script>const dbFacilities = " + json.dumps(records, ensure_ascii=False) + ";</script></body></html>")


def response(html, *, url=SOURCE, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = html.encode("utf-8")
    result._content_consumed = True
    result.encoding = "ISO-8859-1"
    result.url = url
    return result


@pytest.fixture
def directory(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, database_url="", api_key="",
        embedding_model="text-embedding-3-small", embedding_dimensions=3, max_age_hours=24)
    store = Store(settings)
    session = Mock()
    session.get.return_value = response(page())
    with patch("app.portal.archive_assets", return_value=[]) as archive:
        yield SimpleNamespace(settings=settings, store=store, session=session, archive=archive)


def seed_previous(directory):
    payload = {"count": 1, "categories": {"생활서비스": 1}, "records": [{"id": 49, "name": "기존 우편취급국", "category": "생활서비스"}]}
    directory.store.replace_knowledge("page:facility-directory", code="campus", name="기존 공식 목록", url=SOURCE,
        fetched_at="2026-11-13T01:00:00+00:00", payload=payload,
        chunks=knowledge_chunks("page:facility-directory", "기존 목록", [("종류와 개수", "공식 시설 1곳")], kind="page"))
    folder = directory.settings.data_dir / "portal"
    folder.mkdir()
    (folder / "facility-index.html").write_text("기존 유효한 원문", encoding="utf-8")
    (folder / "facility-directory.json").write_text(json.dumps(payload), encoding="utf-8")
    return directory.store.knowledge()[0]


def test_directory_refresh_fetches_one_official_index_and_preserves_public_records_metadata_and_files(directory):
    result = collect_directory(directory.settings, directory.store, session=directory.session)
    directory.session.get.assert_called_once_with(SOURCE, timeout=(3, 12), allow_redirects=False,
        headers={"User-Agent": "EfooWebAcademicCrawler/1.0"})
    directory.session.close.assert_not_called()
    assert result["source_url"] == SOURCE
    assert result["records"] == RECORDS
    assert result["categories"] == {"구내식당": 1, "일반음식점": 1}
    assert datetime.fromisoformat(result["fetched_at"]).utcoffset().total_seconds() == 0
    source = directory.store.knowledge()[0]
    assert source["scope"] == "page:facility-directory"
    assert source["payload"]["count"] == 2
    assert source["payload"]["records"][0]["display"] == {"enabled": True}
    assert source["payload"]["page_metadata"] == {"title": "공식 시설 안내", "language": "ko",
        "meta": [{"name": "description", "content": "공개 시설 목록"}]}
    assert source["url"] == SOURCE
    folder = directory.settings.data_dir / "portal"
    assert (folder / "facility-index.html").read_text(encoding="utf-8") == page()
    assert json.loads((folder / "facility-directory.json").read_text()) == source["payload"]
    synthetic_html, asset_url, data_dir = directory.archive.call_args.args
    assert "/uploads/student.jpg" in synthetic_html
    assert "/uploads/menu-guide.png" in synthetic_html
    assert asset_url == SOURCE and data_dir == directory.settings.data_dir
    assert directory.archive.call_args.kwargs["session"] is directory.session
    # The directory is persisted as evidence without making model API calls.
    assert len(directory.store.pending()) == 2
    assert {row["doc_type"] for row in directory.store.pending()} == {"page"}


@pytest.mark.parametrize("html", [
    page([]), page(main=False), page(grid=False), "<html><main>점검 중</main></html>",
    page([{**RECORDS[0], "id": True}]), page([{**RECORDS[0], "id": "1"}]),
    page([{**RECORDS[0], "id": 0}]), page([{**RECORDS[0], "id": -1}]),
    page([RECORDS[0], {**RECORDS[1], "id": 1}]),
    page([{**RECORDS[0], "name": "  "}]), page([{**RECORDS[0], "name": None}]),
    page([{key: value for key, value in RECORDS[0].items() if key != "category"}]),
    page([{**RECORDS[0], "category": ""}]), page([{**RECORDS[0], "category_display": {"invalid": True}}]),
    page(count=999), page(count="점검"),
    page().replace("const dbFacilities =", "const dbFacilitiesOther ="),
])
def test_invalid_or_incomplete_directory_does_not_replace_previous_database_or_raw_snapshot(directory, html):
    previous = seed_previous(directory)
    directory.session.get.return_value = response(html)
    with pytest.raises(ValueError):
        collect_directory(directory.settings, directory.store, session=directory.session)
    assert directory.store.knowledge()[0] == previous
    folder = directory.settings.data_dir / "portal"
    assert (folder / "facility-index.html").read_text() == "기존 유효한 원문"
    assert json.loads((folder / "facility-directory.json").read_text())["count"] == 1
    directory.archive.assert_not_called()
    directory.session.close.assert_not_called()


@pytest.mark.parametrize("url,status", [
    ("https://untrusted.example/facilities/index.php", 200),
    (BASE + "/theme/pages/auth/login.php", 200), (SOURCE, 302),
])
def test_external_login_or_redirect_responses_are_rejected_without_followup_requests(directory, url, status):
    previous = seed_previous(directory)
    directory.session.get.return_value = response(page(), url=url, status=status)
    with pytest.raises(ValueError):
        collect_directory(directory.settings, directory.store, session=directory.session)
    directory.session.get.assert_called_once()
    assert directory.store.knowledge()[0] == previous
    directory.archive.assert_not_called()


def test_http_failure_preserves_database_and_owned_session_is_closed(directory):
    previous = seed_previous(directory)
    directory.session.get.return_value = response("Unavailable", status=503)
    with patch("app.portal.requests.Session", return_value=directory.session):
        with pytest.raises(requests.HTTPError):
            collect_directory(directory.settings, directory.store)
    directory.session.close.assert_called_once()
    assert directory.store.knowledge()[0] == previous
    directory.archive.assert_not_called()


def test_owned_session_is_also_closed_after_success(directory):
    with patch("app.portal.requests.Session", return_value=directory.session) as factory:
        collect_directory(directory.settings, directory.store)
    factory.assert_called_once()
    directory.session.close.assert_called_once()


def test_asset_archiving_failure_preserves_the_last_valid_directory(directory):
    previous = seed_previous(directory)
    directory.archive.side_effect = OSError("archive storage failure")
    with pytest.raises(OSError):
        collect_directory(directory.settings, directory.store, session=directory.session)
    assert directory.store.knowledge()[0] == previous
    assert (directory.settings.data_dir / "portal" / "facility-index.html").read_text() == "기존 유효한 원문"


def test_collect_portal_reuses_the_directory_fetch_parse_store_and_asset_result(directory):
    records = [RECORDS[1]]
    bodies = {
        SOURCE: page(records),
        BASE + "/theme/pages/facilities/shop-detail.php?id=18":
            '<html><head><link rel="alternate" hreflang="ko" href="?id=18"></head>'
            '<body><div class="shop-name">NEW YORK BURGER</div><div class="shop-category">일반음식점</div></body></html>',
        BASE + "/theme/pages/home.php": '<main>공개 홈 안내</main><script>const dbMenus = []; const dbFacilitiesList = []; const catMap = {};</script>',
        BASE + "/theme/pages/payment/ticket-shops.php": "<main>공개 식권 안내</main>",
        BASE + "/theme/pages/notice/index.php?board_lang=ko": "<script>const dbNotices = [];</script>",
    }
    directory.session.get.side_effect = lambda url, **kwargs: response(bodies[url], url=url)
    with patch("app.portal.collect_directory", wraps=collect_directory) as helper:
        result = collect_portal(directory.settings, directory.store, session=directory.session, progress=lambda message: None)
    helper.assert_called_once_with(directory.settings, directory.store, session=directory.session)
    assert sum(call.args[0] == SOURCE for call in directory.session.get.call_args_list) == 1
    assert sum(call.args[1] == SOURCE for call in directory.archive.call_args_list) == 1
    assert result["catalog_source"] == SOURCE
    assert result["facility_count"] == 1
    assert result["facilities"] == records
    assert result["facility_pages"][0]["name"] == "NEW YORK BURGER"
    assert result["errors"] == []
    directory.session.close.assert_not_called()

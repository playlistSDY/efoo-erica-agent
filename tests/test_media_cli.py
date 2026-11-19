"""CLI image-enrichment scope and ordering regressions, without live requests."""

from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json

import pytest

from app import cli, media_enrichment, portal
from app.crawler import CrawlResult
from app.domain import RESTAURANTS
from app.indexing import build_chunks
from app.settings import Settings
from app.store import Store


@pytest.fixture
def isolated_cli(tmp_path, monkeypatch):
    settings = Settings(data_dir=tmp_path, database_url="", api_key="", embedding_dimensions=3)
    store = Store(settings)
    monkeypatch.setattr(cli, "Settings", lambda: settings)
    monkeypatch.setattr(cli, "Store", lambda actual: store if actual is settings else None)
    return settings, store


@pytest.mark.parametrize("errors", [[], [{"code": "re13", "date": "2026-11-14", "error": "Timeout"}]])
def test_enrich_images_cli_passes_exact_requested_scope_before_optional_index(isolated_cli, monkeypatch, errors):
    settings, store = isolated_cli
    events = []

    def enrich(actual_store, actual_settings, *, start_date, days, restaurant_codes):
        assert actual_store is store and actual_settings is settings
        assert start_date == date(2026, 11, 13)
        assert days == 2
        assert restaurant_codes == ["re12", "re13"]
        events.append("enrich")
        return {"updated_collections": 2, "images": 3, "errors": errors}

    def index(actual_store, actual_settings):
        assert actual_store is store and actual_settings is settings
        assert events == ["enrich"]
        events.append("index")
        return 0

    monkeypatch.setattr(media_enrichment, "enrich_menu_images", enrich)
    monkeypatch.setattr(cli, "index_pending", index)
    monkeypatch.setattr("sys.argv", ["efoo", "enrich-images", "--date", "2026-11-13", "--days", "2",
        "--restaurants", "re12", "re13", "--index"])
    if errors:
        with pytest.raises(SystemExit) as failure:
            cli.main()
        assert failure.value.code == 1
    else:
        cli.main()
    assert events == ["enrich", "index"]


def test_full_collection_restores_entire_visible_week_images_before_indexing(isolated_cli, monkeypatch):
    settings, store = isolated_cli
    selected = date(2026, 11, 13)
    week_start = date(2026, 11, 8)
    days = [(week_start + timedelta(days=offset)).isoformat() for offset in range(7)]
    requested = ["re12", "re13"]
    events = []

    class FakeCrawler:
        session = object()

        def fetch(self, code, target):
            assert code in requested and target == selected
            weekly = {day: [{"meal_type": "중식", "items": [f"{code} {day} 메뉴"], "price": "5,000원", "tags": []}]
                for day in days}
            return CrawlResult(code=code, name=RESTAURANTS[code]["name"], date=target.isoformat(),
                source_url=f"https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id={RESTAURANTS[code]['facility_id']}&date={target.isoformat()}",
                fetched_at=datetime.now(timezone.utc).isoformat(), html="<html>visible week</html>",
                meals=weekly[target.isoformat()], weekly_meals=weekly)

        def close(self):
            events.append("close")

    def assets(*args, **kwargs):
        assert kwargs["session"] is FakeCrawler.session
        return []

    def collect_portal(actual_settings, actual_store, *, session):
        assert actual_settings is settings and actual_store is store
        assert session is FakeCrawler.session
        events.append("portal")
        return {"facility_pages": [], "facility_count": 0, "notices": [], "errors": []}

    def enrich(actual_store, actual_settings, *, start_date, days: int, restaurant_codes):
        assert actual_store is store and actual_settings is settings
        assert start_date == week_start and days == 7
        assert restaurant_codes == requested
        assert events == ["portal", "close"]
        events.append("enrich")
        for offset in range(days):
            target = (start_date + timedelta(days=offset)).isoformat()
            for code in restaurant_codes:
                row = actual_store.collections(target, code)[0]
                # Weekly text snapshots initially contain no archived photo;
                # daily enrichment must happen after their final replacement.
                before = actual_store.candidates(target, code, "중식", doc_types=["menu"])
                assert before and json.loads(before[0]["images"]) == []
                snapshot = CrawlResult(**{**row["payload"], "html": ""})
                image_url = f"https://life.hanyang.ac.kr/uploads/{code}-{target}.jpg"
                archive = [{"url": image_url, "local_path": "/media/" + "a" * 64 + ".jpg",
                    "downloaded": True, "content_type": "image/jpeg", "alt": "일별 메뉴 사진"}]
                snapshot = replace(snapshot, assets=archive,
                    meals=[{**menu, "image": image_url} for menu in snapshot.meals])
                actual_store.replace_collection(snapshot, build_chunks(snapshot))
        return {"updated_collections": 14, "images": 14, "errors": []}

    def index(actual_store, actual_settings):
        assert actual_store is store and actual_settings is settings
        assert events == ["portal", "close", "enrich"]
        events.append("index")
        for target in days:
            for code in requested:
                rows = actual_store.candidates(target, code, "중식", doc_types=["menu"])
                assert len(rows) == 1 and len(json.loads(rows[0]["images"])) == 1
        return 0

    monkeypatch.setattr(cli, "MealCrawler", FakeCrawler)
    monkeypatch.setattr(cli, "archive_assets", assets)
    monkeypatch.setattr(portal, "collect_portal", collect_portal)
    monkeypatch.setattr(media_enrichment, "enrich_menu_images", enrich)
    monkeypatch.setattr(cli, "index_pending", index)
    monkeypatch.setattr("sys.argv", ["efoo", "collect", "--date", selected.isoformat(), "--restaurants", *requested, "--full", "--index"])
    cli.main()
    assert events == ["portal", "close", "enrich", "index"]
    assert sum(len(store.collections(target)) for target in days) == 14

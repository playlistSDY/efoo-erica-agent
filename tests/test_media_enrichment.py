from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
from unittest.mock import Mock, patch

import pytest

with patch("dotenv.load_dotenv"):
    from app.crawler import CrawlResult
    from app.indexing import build_chunks
    from app.media_enrichment import enrich_menu_images
    from app.settings import Settings
    from app.store import Store


DAY = date(2026, 9, 22)
ORIGINAL = "https://life.hanyang.ac.kr/uploads/menu.jpg"
LOCAL = "/media/" + "a" * 64 + ".jpg"


def snapshot(day=DAY, *, item="칼국수", image="", weekly_meals=None):
    return CrawlResult(code="re12", name="학생식당", date=day.isoformat(),
        source_url=f"https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=1&date={day.isoformat()}",
        fetched_at=datetime.now(timezone.utc).isoformat(), html=f"<html>공식 {day.isoformat()} 원문</html>",
        meals=[{"meal_type": "중식", "items": [item], "price": "5,000원", "tags": [], "image": image}],
        weekly_meals=weekly_meals or {})


def manifest():
    return [{"url": ORIGINAL, "local_path": LOCAL, "downloaded": True, "content_type": "image/jpeg"},
            {"url": "https://life.hanyang.ac.kr/uploads/cover.jpg", "local_path": "/media/" + "b" * 64 + ".jpg",
             "downloaded": True, "content_type": "image/jpeg"}]


@pytest.fixture
def setup_store(tmp_path):
    settings = Settings(data_dir=tmp_path, database_url="", api_key="", raw_s3_bucket="", embedding_dimensions=3)
    return Store(settings), settings


def seed(store, result):
    store.replace_collection(result, build_chunks(result))
    pending = store.pending()
    store.save_embeddings(pending, [[1.0, 0.0, 0.0]] * len(pending))


def test_exact_daily_photo_refresh_preserves_embeddings_and_other_days_without_copying_weekly_data(setup_store):
    store, settings = setup_store
    seed(store, snapshot())
    tomorrow = DAY + timedelta(days=1)
    seed(store, snapshot(tomorrow, item="내일 원래 메뉴"))
    original_ids = {row["id"] for row in store.candidates(DAY.isoformat(), "re12", doc_types=["menu"])}
    crawler = Mock()
    crawler.fetch.return_value = snapshot(image=ORIGINAL, weekly_meals={tomorrow.isoformat(): [
        {"meal_type": "중식", "items": ["다른 날짜에 덮어쓰면 안 되는 메뉴"], "price": "", "tags": []}]})
    with patch("app.media_enrichment.MealCrawler", return_value=crawler), \
         patch("app.media_enrichment.archive_assets", return_value=manifest()) as archive:
        summary = enrich_menu_images(store, settings, start_date=DAY, days=1, restaurant_codes=["re12"])
    assert summary["updated_collections"] == summary["changed_collections"] == 1
    assert summary["updated_menus"] == 1
    assert summary["images"] == 1  # A facility cover is not a menu photo.
    assert summary["errors"] == []
    assert archive.call_args.kwargs["session"] is crawler.session
    crawler.fetch.assert_called_once_with("re12", DAY)
    crawler.close.assert_called_once()
    rows = store.candidates(DAY.isoformat(), "re12", doc_types=["menu"])
    assert {row["id"] for row in rows} == original_ids
    assert store.pending() == []
    lunch = next(row for row in rows if row["meal_type"] == "중식")
    assert json.loads(lunch["images"])[0]["url"] == LOCAL
    assert store.collections(tomorrow.isoformat(), "re12")[0]["meals"][0]["items"] == ["내일 원래 메뉴"]
    raw = settings.data_dir / "raw" / DAY.isoformat() / "re12"
    assert (raw / "source.html").read_text().endswith("원문</html>")
    assert json.loads((raw / "normalized.json").read_text())["assets"][0]["local_path"] == LOCAL


def test_only_existing_snapshots_are_fetched_and_one_shared_session_is_closed(setup_store):
    store, settings = setup_store
    seed(store, snapshot())
    crawler = Mock()
    crawler.fetch.return_value = snapshot(image=ORIGINAL)
    with patch("app.media_enrichment.MealCrawler", return_value=crawler), \
         patch("app.media_enrichment.archive_assets", return_value=manifest()):
        summary = enrich_menu_images(store, settings, start_date=DAY, days=7)
    assert summary["attempted"] == 1
    assert summary["skipped_missing"] == 27
    crawler.fetch.assert_called_once_with("re12", DAY)
    crawler.close.assert_called_once()
    assert store.collections((DAY + timedelta(days=1)).isoformat()) == []


@pytest.mark.parametrize("incorrect", ["day", "restaurant"])
def test_mismatched_fetch_results_cannot_pollute_the_requested_day_or_cafeteria(setup_store, incorrect):
    store, settings = setup_store
    old = snapshot()
    seed(store, old)
    result = snapshot(DAY + timedelta(days=1), image=ORIGINAL) if incorrect == "day" else replace(snapshot(image=ORIGINAL), code="re11")
    crawler = Mock()
    crawler.fetch.return_value = result
    with patch("app.media_enrichment.MealCrawler", return_value=crawler), \
         patch("app.media_enrichment.archive_assets") as archive:
        summary = enrich_menu_images(store, settings, start_date=DAY, days=1, restaurant_codes=["re12"])
    assert summary["updated_collections"] == 0
    assert summary["errors"] == [{"date": DAY.isoformat(), "restaurant_code": "re12", "error": "ValueError"}]
    archive.assert_not_called()
    assert store.collections(DAY.isoformat(), "re12")[0]["meals"] == old.meals
    crawler.close.assert_called_once()


def test_failed_fetch_preserves_prior_data_and_later_existing_days_still_refresh(setup_store):
    store, settings = setup_store
    seed(store, snapshot())
    tomorrow = DAY + timedelta(days=1)
    seed(store, snapshot(tomorrow))
    crawler = Mock()
    crawler.fetch.side_effect = [ValueError("HTML layout mismatch"), snapshot(tomorrow, image=ORIGINAL)]
    with patch("app.media_enrichment.MealCrawler", return_value=crawler), \
         patch("app.media_enrichment.archive_assets", return_value=manifest()):
        summary = enrich_menu_images(store, settings, start_date=DAY, days=2, restaurant_codes=["re12"])
    assert summary["attempted"] == 2
    assert summary["updated_collections"] == 1
    assert len(summary["errors"]) == 1
    assert store.collections(DAY.isoformat(), "re12")[0]["meals"][0]["image"] == ""
    assert store.collections(tomorrow.isoformat(), "re12")[0]["meals"][0]["image"] == ORIGINAL
    crawler.close.assert_called_once()


def test_same_menu_photo_mapping_is_not_reported_as_a_new_change_on_repeat(setup_store):
    store, settings = setup_store
    seed(store, snapshot())
    crawler = Mock()
    result = snapshot(image=ORIGINAL)
    crawler.fetch.return_value = result
    with patch("app.media_enrichment.MealCrawler", return_value=crawler), \
         patch("app.media_enrichment.archive_assets", return_value=manifest()):
        first = enrich_menu_images(store, settings, start_date=DAY, days=1, restaurant_codes=["re12"])
        repeated = enrich_menu_images(store, settings, start_date=DAY, days=1, restaurant_codes=["re12"])
    assert first["changed_collections"] == 1
    assert repeated["updated_collections"] == 1
    assert repeated["changed_collections"] == 0
    assert repeated["images"] == 1
    assert store.pending() == []


def test_changed_menu_text_leaves_only_the_changed_text_for_later_embedding(setup_store):
    store, settings = setup_store
    seed(store, snapshot())
    crawler = Mock()
    crawler.fetch.return_value = snapshot(item="돈가스", image=ORIGINAL)
    with patch("app.media_enrichment.MealCrawler", return_value=crawler), \
         patch("app.media_enrichment.archive_assets", return_value=manifest()):
        summary = enrich_menu_images(store, settings, start_date=DAY, days=1, restaurant_codes=["re12"])
    assert summary["changed_collections"] == 1
    assert len(store.pending()) == 1
    assert "돈가스" in store.pending()[0]["text"]


@pytest.mark.parametrize("arguments", [{"days": 0}, {"days": 8}, {"days": True},
    {"restaurant_codes": ["re99"]}, {"restaurant_codes": "re12"}, {"start_date": "2026-09-22"}])
def test_invalid_or_unbounded_inputs_fail_before_creating_a_network_session(setup_store, arguments):
    store, settings = setup_store
    options = {"start_date": DAY, "days": 1, **arguments}
    with patch("app.media_enrichment.MealCrawler") as crawler:
        with pytest.raises(ValueError):
            enrich_menu_images(store, settings, **options)
    crawler.assert_not_called()

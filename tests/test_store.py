from dataclasses import replace
from datetime import date, datetime, timedelta, timezone
import json
import sqlite3
from unittest.mock import Mock, patch
from zoneinfo import ZoneInfo

import pytest

from app.crawler import CrawlResult
from app.indexing import build_chunks, index_pending
from app.settings import Settings
from app.store import Store


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, database_url="", api_key="", embedding_dimensions=3)


@pytest.fixture
def store(settings):
    return Store(settings)


def snapshot(*, day="2026-09-22", code="re12", name="학생식당", item="칼국수", fetched_at=None):
    return CrawlResult(code=code, name=name, date=day,
        source_url=f"https://life.hanyang.ac.kr/menu?code={code}&date={day}",
        fetched_at=fetched_at or datetime.now(timezone.utc).isoformat(), html="<html>원문</html>",
        meals=[{"meal_type": "중식", "items": [item, "김치"], "price": "5,000원", "tags": []}])


def test_snapshot_replaces_obsolete_chunks_but_reuses_identical_embeddings(store):
    old = snapshot()
    store.replace_collection(old, build_chunks(old))
    rows = store.pending()
    store.save_embeddings(rows, [[1.0, 0.0, 0.0]] * len(rows))
    old_lunch = next(r for r in rows if r["meal_type"] == "중식")
    changed = snapshot(item="돈가스")
    store.replace_collection(changed, build_chunks(changed))
    current = store.candidates(changed.date)
    assert len(current) == 3
    assert old_lunch["id"] not in {r["id"] for r in current}
    assert len(store.pending()) == 1
    assert all(r["embedding"] for r in current if r["meal_type"] != "중식")
    assert store.collections(changed.date)[0]["meals"][0]["items"][0] == "돈가스"
    assert store.stats()["indexed_count"] == 2


def test_snapshot_replacement_rolls_back_as_one_transaction(store):
    old = snapshot()
    store.replace_collection(old, build_chunks(old))
    bad = snapshot(item="새 메뉴")
    chunks = build_chunks(bad)
    with pytest.raises(sqlite3.IntegrityError):
        store.replace_collection(bad, [chunks[0], chunks[0]])
    assert store.collections(old.date)[0]["meals"][0]["items"][0] == "칼국수"
    assert store.stats()["chunk_count"] == 3


def test_metadata_filters_do_not_mix_dates_restaurants_or_meals(store):
    for result in (snapshot(), snapshot(day="2026-09-23"), snapshot(code="re11", name="교직원식당")):
        store.replace_collection(result, build_chunks(result))
    rows = store.candidates("2026-09-22", "re12", "중식")
    assert len(rows) == 1
    assert {(r["day"], r["code"], r["meal_type"]) for r in rows} == {("2026-09-22", "re12", "중식")}


def test_current_and_future_stale_documents_are_excluded_but_past_records_remain(store):
    today = datetime.now(ZoneInfo("Asia/Seoul")).date()
    stale = (datetime.now(timezone.utc) - timedelta(hours=25)).isoformat()
    for target in (today, today + timedelta(days=1), today - timedelta(days=1)):
        result = snapshot(day=target.isoformat(), fetched_at=stale)
        store.replace_collection(result, build_chunks(result))
    assert store.candidates(today.isoformat()) == []
    assert store.candidates((today + timedelta(days=1)).isoformat()) == []
    assert len(store.candidates((today - timedelta(days=1)).isoformat())) == 3


@pytest.mark.parametrize("vectors", [[], [[1.0, 2.0]], [[1.0, float("nan"), 0.0]], [[1.0, float("inf"), 0.0]]])
def test_invalid_embedding_result_never_partially_updates_rows(store, vectors):
    result = snapshot()
    store.replace_collection(result, build_chunks(result))
    rows = store.pending()[:1]
    with pytest.raises(ValueError):
        store.save_embeddings(rows, vectors)
    assert store.stats()["indexed_count"] == 0


def test_indexing_uses_real_api_contract_and_reorders_embedding_response(store, settings):
    result = snapshot()
    store.replace_collection(result, build_chunks(result))
    pending = store.pending()
    client = Mock()
    client.embeddings.create.return_value.data = [
        Mock(index=2, embedding=[0.0, 0.0, 1.0]),
        Mock(index=0, embedding=[1.0, 0.0, 0.0]),
        Mock(index=1, embedding=[0.0, 1.0, 0.0]),
    ]
    assert index_pending(store, settings, client=client) == 3
    assert client.embeddings.create.call_args.kwargs == {
        "model": "text-embedding-3-small", "input": [r["text"] for r in pending], "dimensions": 3}
    by_id = {r["id"]: r for r in store.candidates(result.date)}
    assert json.loads(by_id[pending[0]["id"]]["embedding"]) == [1.0, 0.0, 0.0]
    assert json.loads(by_id[pending[2]["id"]]["embedding"]) == [0.0, 0.0, 1.0]
    assert store.stats()["indexed_count"] == 3
    assert index_pending(store, settings, client=client) == 0
    assert client.embeddings.create.call_count == 1


def test_missing_key_cannot_fake_an_index(store, settings):
    result = snapshot()
    store.replace_collection(result, build_chunks(result))
    with pytest.raises(ValueError, match="OPENAI_API_KEY"):
        index_pending(store, settings)
    assert store.stats()["indexed_count"] == 0


def test_changing_embedding_model_marks_existing_vectors_pending(store, settings):
    result = snapshot()
    store.replace_collection(result, build_chunks(result))
    rows = store.pending()
    store.save_embeddings(rows, [[1.0, 0.0, 0.0]] * len(rows))
    store.settings = replace(settings, embedding_model="different-embedding-model")
    assert len(store.pending()) == 3
    assert store.stats()["indexed_count"] == 0


def test_failed_crawl_keeps_previously_collected_snapshot(store, settings):
    from app.cli import main
    old = snapshot()
    store.replace_collection(old, build_chunks(old))
    crawler = Mock()
    crawler.fetch.side_effect = ValueError("HTML changed")
    with patch("app.cli.Settings", return_value=settings), patch("app.cli.Store", return_value=store), \
         patch("app.cli.MealCrawler", return_value=crawler), \
         patch("sys.argv", ["cli", "collect", "--date", old.date, "--restaurants", "re12"]):
        with pytest.raises(SystemExit) as raised:
            main()
    assert raised.value.code == 1
    assert store.collections(old.date)[0]["meals"][0]["items"][0] == "칼국수"
    crawler.close.assert_called_once()


"""Knowledge storage regression tests independent of live pages and model APIs."""

from dataclasses import replace
from datetime import datetime, timedelta, timezone
import json
import sqlite3

import pytest

from app.crawler import CrawlResult
from app.indexing import build_chunks, index_facility, index_notice, knowledge_chunks
from app.settings import Settings
from app.store import Store


@pytest.fixture
def settings(tmp_path):
    return Settings(data_dir=tmp_path, database_url="", api_key="", embedding_dimensions=3)


@pytest.fixture
def store(settings):
    return Store(settings)


def add_page(store, *, scope="facility:1", code="re12", name="학생식당", text="위치: 복지관 2층", kind="profile", fetched_at=None, images=None, metadata=None):
    fetched_at = fetched_at or datetime.now(timezone.utc).isoformat()
    chunks = knowledge_chunks(scope, name, [("안내", text)], kind=kind, images=images, metadata=metadata)
    store.replace_knowledge(scope, code=code, name=name, url="https://life.hanyang.ac.kr/official",
        fetched_at=fetched_at, payload={"text": text, "revision": metadata or {}}, chunks=chunks)
    return chunks


def meal_snapshot(day):
    return CrawlResult(code="re12", name="학생식당", date=day,
        source_url="https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=1&date=" + day,
        fetched_at=datetime.now(timezone.utc).isoformat(), html="<html>공식 식단</html>",
        meals=[{"meal_type": "중식", "items": ["비빔밥"], "price": "4,500원", "tags": []}])


def test_additive_migration_preserves_legacy_vectors_and_snapshots(settings):
    database = settings.data_dir / "efoo.sqlite3"
    with sqlite3.connect(database) as conn:
        conn.execute("CREATE TABLE collections(code TEXT,day TEXT,name TEXT,url TEXT,fetched_at TEXT,meals TEXT,PRIMARY KEY(code,day))")
        conn.execute("CREATE TABLE chunks(id TEXT PRIMARY KEY,code TEXT,day TEXT,meal_type TEXT,title TEXT,text TEXT,url TEXT,fetched_at TEXT,embedding TEXT,embedding_model TEXT)")
        conn.execute("INSERT INTO collections VALUES(?,?,?,?,?,?)", ("re12", "2026-01-02", "학생식당", "https://example.invalid", "2026-01-02T00:00:00+00:00", "[]"))
        conn.execute("INSERT INTO chunks VALUES(?,?,?,?,?,?,?,?,?,?)", ("legacy-id", "re12", "2026-01-02", "중식", "기존 식단", "기존 원문", "https://example.invalid", "2026-01-02T00:00:00+00:00", "[1.0,0.0,0.0]", settings.embedding_model))
    migrated = Store(settings)
    # Starting the app repeatedly must not change legacy identities or values.
    Store(settings)
    row = migrated.candidates("2026-01-02", indexed=True, doc_types=["menu"])[0]
    assert row["id"] == "legacy-id"
    assert json.loads(row["embedding"]) == [1.0, 0.0, 0.0]
    assert row["embedding_model"] == settings.embedding_model
    assert row["doc_type"] == "menu"
    assert json.loads(row["images"]) == []
    assert json.loads(row["metadata"]) == {}
    assert migrated.collections("2026-01-02")[0]["payload"] == {}
    assert migrated.stats()["indexed_count"] == 1


def test_profiles_are_available_across_query_dates_while_menus_are_day_scoped(store):
    add_page(store)
    first = meal_snapshot("2026-01-02")
    second = replace(first, date="2026-01-03")
    for result in (first, second):
        store.replace_collection(result, build_chunks(result))
    profiles = store.candidates("2026-01-20", "re12", doc_types=["profile", "notice", "page"])
    assert len(profiles) == 1
    assert "복지관 2층" in profiles[0]["text"]
    assert store.candidates("2026-01-20", doc_types=["menu"]) == []
    menus = store.candidates("2026-01-02", "re12", "중식", doc_types=["menu"])
    assert len(menus) == 1 and menus[0]["day"] == "2026-01-02"


def test_unchanged_knowledge_preserves_vector_while_updating_source_metadata(store):
    add_page(store, metadata={"revision": 1})
    store.save_embeddings(store.pending(), [[1.0, 0.0, 0.0]])
    images = [{"url": "/media/" + "a" * 64 + ".jpg", "alt": "학생식당"}]
    add_page(store, metadata={"revision": 2}, images=images)
    row = store.candidates("2026-01-01", doc_types=["profile"])[0]
    assert json.loads(row["embedding"]) == [1.0, 0.0, 0.0]
    assert json.loads(row["images"]) == images
    assert json.loads(row["metadata"]) == {"revision": 2}
    assert store.pending() == []
    assert store.knowledge()[0]["payload"]["revision"] == {"revision": 2}


def test_changed_knowledge_removes_obsolete_chunks_without_touching_other_scopes(store):
    original = add_page(store)
    add_page(store, scope="notice:13", code="re13", name="결제 안내", text="온라인 결제 안내", kind="notice")
    rows = store.pending()
    store.save_embeddings(rows, [[1.0, 0.0, 0.0]] * len(rows))
    add_page(store, text="위치: 새 건물 1층")
    current = store.candidates("2026-01-01", doc_types=["profile", "notice"])
    assert len(current) == 2
    assert original[0]["id"] not in {row["id"] for row in current}
    assert len(store.pending()) == 1
    assert next(row for row in current if row["doc_type"] == "notice")["embedding"]
    assert {row["scope"] for row in store.knowledge()} == {"facility:1", "notice:13"}


def test_failed_knowledge_replace_rolls_back_source_payload_and_vectors(store):
    old = add_page(store)
    store.save_embeddings(store.pending(), [[0.0, 1.0, 0.0]])
    bad = knowledge_chunks("facility:1", "학생식당", [("변경", "미완성 갱신")])
    with pytest.raises(sqlite3.IntegrityError):
        store.replace_knowledge("facility:1", code="re12", name="학생식당", url="https://example.invalid/new",
            fetched_at=datetime.now(timezone.utc).isoformat(), payload={"text": "미완성 갱신"}, chunks=[bad[0], bad[0]])
    row = store.candidates("2026-01-01", doc_types=["profile"])[0]
    assert row["id"] == old[0]["id"]
    assert json.loads(row["embedding"]) == [0.0, 1.0, 0.0]
    assert store.knowledge()[0]["payload"]["text"] == "위치: 복지관 2층"


def test_menu_refresh_preserves_same_date_facility_knowledge(store):
    day = "2026-01-02"
    profile = add_page(store, fetched_at=day + "T01:00:00+00:00")
    result = meal_snapshot(day)
    store.replace_collection(result, build_chunks(result))
    store.replace_collection(result, build_chunks(result))
    # An expired profile remains archived, but cannot bypass freshness merely
    # because its snapshot day is historical. Verify preservation in storage.
    with store.connection() as conn:
        preserved=conn.execute("SELECT id FROM chunks WHERE doc_type='profile'").fetchall()
    assert {row["id"] for row in preserved} == {profile[0]["id"]}
    assert store.candidates(day, doc_types=["profile"]) == []
    assert len(store.candidates(day, doc_types=["menu"])) == 3
    assert len(store.knowledge()) == 1


def test_stats_show_latest_knowledge_update_even_without_meal_snapshots(store):
    older = "2026-01-02T01:00:00+00:00"
    newer = "2026-01-03T01:00:00+00:00"
    add_page(store, fetched_at=older)
    add_page(store, scope="notice:13", code="campus", name="공지", text="공식 안내", kind="notice", fetched_at=newer)
    stats = store.stats()
    assert stats["document_count"] == 2
    assert stats["meal_snapshot_count"] == 0
    assert stats["facility_count"] == 1 and stats["notice_count"] == 1
    assert stats["last_updated"] == newer


def test_facility_index_includes_official_rows_description_notice_and_local_photo(store):
    original = "https://life.hanyang.ac.kr/uploads/cafe.jpg"
    image_path = "/media/" + "f" * 64 + ".jpg"
    facility = {"name": "캠퍼스 카페", "facility_id": 51,
        "restaurant_info": {"name": "캠퍼스 카페", "category": "카페", "info_rows": [{"label": "위치", "value": "도서관 1층"}],
            "notices": [{"text": "방학 중 단축 운영"}], "description": {"text": "테이크아웃 가능"},
            "cover_image": {"url": original}, "gallery": []},
        "page_data": {"images": [{"url": original, "role": "cover"}]}}
    assets = [{"url": original, "local_path": image_path, "downloaded": True, "content_type": "image/jpeg", "alt": "카페 전경"}]
    index_facility(store, facility, assets, datetime.now(timezone.utc).isoformat(), "https://life.hanyang.ac.kr/official", {"id": 51})
    rows = store.candidates("2026-01-01", "fac_51", doc_types=["profile"])
    text = "\n".join(row["text"] for row in rows)
    assert all(part in text for part in ("도서관 1층", "방학 중 단축 운영", "테이크아웃 가능", "카페"))
    assert any(image["url"] == image_path for row in rows for image in json.loads(row["images"]))
    assert store.knowledge()[0]["payload"]["directory_record"] == {"id": 51}
    assert store.lookup_code("캠퍼스 카페 위치 알려줘") == "fac_51"


def test_notice_index_keeps_multi_facility_scope_and_published_context(store):
    notice = {"title": "식당 공통 안내", "text": "온라인 결제 정책은 해당 공지를 확인하세요.",
        "url": "https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id=99", "fetched_at": datetime.now(timezone.utc).isoformat(),
        "metadata": "2026.11.01", "restaurant_codes": ["re12", "re13"], "images": [], "assets": []}
    index_notice(store, notice, 99)
    row = store.candidates("2026-01-01", doc_types=["notice"])[0]
    assert row["code"] == "campus"
    assert json.loads(row["metadata"])["related_codes"] == ["re12", "re13"]
    assert json.loads(row["metadata"])["published"] == "2026.11.01"
    assert store.knowledge()[0]["payload"] == notice


def test_facility_lookup_prefers_longest_case_insensitive_name(store):
    add_page(store, scope="facility:51", code="fac_51", name="Cafe", text="일반 카페")
    add_page(store, scope="facility:52", code="fac_52", name="Cafe Blue", text="블루 카페")
    assert store.lookup_code("cafe blue의 위치는?") == "fac_52"
    assert store.lookup_code("존재하지 않는 매장") is None

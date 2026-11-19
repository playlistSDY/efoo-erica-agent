"""Targeted embeddings and persisted date context; isolated temp databases."""
from datetime import date, datetime, timezone
from dataclasses import replace
from zoneinfo import ZoneInfo
import json
from unittest.mock import Mock, patch
from uuid import uuid4

from app.conversations import ConversationStore
from app.crawler import CrawlResult
from app.date_ranges import DateRange
from app.indexing import build_chunks, index_pending
from app.rag import Agent, resolve_range_context
from app.settings import Settings
from app.scheduler import MealScheduler
from app.store import Store


def source(day, code="re12"):
    return CrawlResult(code=code,name="학생식당",date=day,
        source_url="https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=1&date="+day,
        fetched_at=datetime.now(timezone.utc).isoformat(),html="<html />",
        meals=[{"meal_type":"중식","items":["시험 메뉴"],"price":"5,000원","tags":[]}])


def test_scoped_index_never_indexes_unrelated_menu_or_school_knowledge(tmp_path):
    settings=Settings(data_dir=tmp_path,database_url="",api_key="test",embedding_dimensions=3,schedule_enabled=False)
    store=Store(settings)
    for day,code in [("2026-11-16","re12"),("2026-11-17","re12"),("2026-11-16","re13")]:
        result=source(day,code)
        store.replace_collection(result,build_chunks(result))
    store.replace_knowledge("facility:1",code="re12",name="학생식당",url="https://life.hanyang.ac.kr/",
        fetched_at=datetime.now(timezone.utc).isoformat(),payload={},
        chunks=[{"id":"school-profile","title":"학생식당 위치","text":"학생식당 위치","doc_type":"profile"}])
    client=Mock()
    client.embeddings.create.return_value.data=[Mock(index=i,embedding=[1.,0.,0.]) for i in range(3)]
    assert index_pending(store,settings,client,menu_targets=[("2026-11-16","re12")])==3
    pending=store.pending()
    assert len(pending)==7
    assert any(row["id"]=="school-profile" for row in pending)
    assert not any(row["day"]=="2026-11-16" and row["code"]=="re12" and row["doc_type"]=="menu" for row in pending)


def test_persisted_api_custom_range_survives_next_day_followup_without_changing_original(tmp_path):
    settings=Settings(data_dir=tmp_path,database_url="",api_key="",schedule_enabled=False)
    store=Store(settings)
    conversations=ConversationStore(store)
    _,owner=conversations.ensure_owner(None)
    conversation=conversations.create(owner)
    raw="학생식당 학식 알려줘"
    payload={"answer":"수집 결과","mode":"retrieval","date_range":{"start":"2026-11-16","end":"2026-11-20"}}
    conversations.append_turn(owner,conversation["id"],raw,payload,str(uuid4()))
    history=conversations.recent_history(owner,conversation["id"],reference_date=date(2026,11,14))
    assert history[0]["content"]==raw
    assert history[0]["date_range"]==payload["date_range"]
    with patch("app.rag.today",return_value=date(2026,11,14)):
        span,code,meal=resolve_range_context("그럼 저녁은?",history,date(2026,11,14),None)
    assert span==DateRange(date(2026,11,16),date(2026,11,20))
    assert (code,meal)==("re12","석식")
    assert conversations.get(owner,conversation["id"])["messages"][0]["content"]==raw


def test_range_search_applies_freshness_per_day_and_excludes_failed_targets(tmp_path):
    settings=Settings(data_dir=tmp_path,database_url="",api_key="",schedule_enabled=False)
    store=Store(settings)
    current=datetime.now(ZoneInfo("Asia/Seoul")).date()
    from datetime import timedelta
    for offset in (-1,0,1):
        result=source((current+timedelta(days=offset)).isoformat())
        result=replace(result,fetched_at=(datetime.now(timezone.utc)-timedelta(days=3)).isoformat())
        store.replace_collection(result,build_chunks(result))
    # A past snapshot can be inspected as history, but crossing into today
    # must not accidentally exempt stale current/future evidence.
    rows=store.candidates((current-timedelta(days=1)).isoformat(),end_day=(current+timedelta(days=1)).isoformat(),
                          doc_types=["menu"])
    assert {row["day"] for row in rows}=={(current-timedelta(days=1)).isoformat()}
    assert store.candidates((current-timedelta(days=1)).isoformat(),doc_types=["menu"],
        exclude_targets=[((current-timedelta(days=1)).isoformat(),"re12")])==[]


def test_schedule_window_can_include_past_days_and_only_exposes_safe_summary(tmp_path):
    settings=Settings(data_dir=tmp_path,database_url="",api_key="",schedule_enabled=False,
        schedule_start_offset_days=-7,schedule_days=14)
    refresh=Mock()
    refresh.ensure.return_value={"status":"partial","checked":56,"updated":55,"errors":[{"error":"private-error-detail"}]}
    scheduler=MealScheduler(refresh,settings,today_fn=lambda:date(2026,11,14))
    scheduler.run_once()
    refresh.ensure.assert_called_once_with(date(2026,11,7),date(2026,11,20),force=False)
    status=scheduler.status()
    assert status["last_error"]=="RefreshIncomplete"
    assert status["last_summary"]["error_count"]==1
    assert "private-error-detail" not in json.dumps(status)


def test_colloquial_past_query_and_date_alias_followup_preserve_meal_intent(tmp_path):
    settings=Settings(data_dir=tmp_path,database_url="",api_key="",schedule_enabled=False)
    agent=Agent(Store(settings),settings)
    with patch("app.rag.today",return_value=date(2026,11,14)):
        plan=agent.plan("저번주에는 뭐나왔어?",[],date(2026,11,14))
        assert (plan.span.start,plan.span.end)==(date(2026,11,2),date(2026,11,8))
        assert plan.doc_types==["menu"]
        next_plan=agent.plan("그럼 차주는?",[{"role":"user","content":"학생식당 오늘 점심 메뉴"}],date(2026,11,14))
        assert next_plan.doc_types==["menu"]
        assert (next_plan.code,next_plan.meal_type)==("re12","중식")
        assert next_plan.span.start==date(2026,11,16)
        same=agent.plan("같은 기간으로 다시 알려줘",[{"role":"user","content":"학생식당 점심 메뉴",
            "date_range":{"start":"2026-11-16","end":"2026-11-18"}}],date(2026,11,14))
        assert same.doc_types==["menu"]
        assert (same.span.start,same.span.end)==(date(2026,11,16),date(2026,11,18))


def test_old_profile_date_cannot_bypass_current_knowledge_freshness(tmp_path):
    settings=Settings(data_dir=tmp_path,database_url="",api_key="",schedule_enabled=False)
    store=Store(settings)
    from datetime import timedelta
    old=datetime.now(timezone.utc)-timedelta(days=3)
    store.replace_knowledge("facility:49",code="fac_49",name="우체국",url="https://life.hanyang.ac.kr/",
        fetched_at=old.isoformat(),payload={},chunks=[{"id":"old-profile","title":"우체국위치","text":"이전위치","doc_type":"profile"}])
    assert store.candidates(datetime.now(timezone.utc).date().isoformat(),doc_types=["profile"])==[]

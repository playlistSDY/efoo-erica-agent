"""Known document references must not become facility/location constraints."""

from datetime import date
import json
from unittest.mock import patch

import pytest

from test_mixed_school_requests import DAY, STAMP, refresh_report, school

with patch("dotenv.load_dotenv"):
    from app.indexing import index_facility, index_notice
    from app.school_context import mask_document_references


TITLE="창의관식당 천원의 아침밥 결제 방식 변경 안내"


@pytest.fixture
def referenced_school(school):
    _,store,_,_=school
    record={"id":2,"name":"교직원식당","category":"구내식당","location":"학생복지관 3층"}
    directory=next(row for row in store.knowledge() if row["scope"]=="page:facility-directory")
    records=directory["payload"]["records"]+[record]
    store.replace_knowledge(directory["scope"],code="campus",name=directory["name"],url=directory["url"],
        fetched_at=STAMP,payload={**directory["payload"],"records":records,"count":len(records)},chunks=[])
    info={"name":record["name"],"category":record["category"],"location":record["location"],
        "contact":"031-400-0002","opening_hours":"평일 11:30~13:30","building_no":"102",
        "info_rows":[{"label":"위치","value":record["location"]},{"label":"연락처","value":"031-400-0002"}],
        "notices":[],"description":{},"cover_image":{},"gallery":[]}
    index_facility(store,{"facility_id":2,"restaurant_info":info,"page_data":{}},[],STAMP,
        "https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=2",record)
    seed_reference(store,TITLE)
    return school


def seed_reference(store,title):
    index_notice(store,{"title":title,"text":"공식 문서 본문에 명시된 결제 안내입니다.","restaurant_codes":["re13"],
        "images":[],"assets":[],"fetched_at":STAMP,
        "url":"https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id=13&board_lang=ko"},13)
    pending=store.pending()
    store.save_embeddings(pending,[[1.,0.,0.]]*len(pending))


@pytest.mark.parametrize("opening,closing",[("'","'"),('"','"'),("‘","’"),("","")])
def test_referenced_title_building_does_not_remove_second_requested_facility(referenced_school,opening,closing):
    _,_,client,agent=referenced_school
    question=f"창의인재원식당와 교직원식당의 위치와 전화번호를 각각 알려주고, {opening}{TITLE}{closing} 공지도 함께 알려줘"
    plan=agent.plan(question,[],DAY)
    assert plan.facility_ids==("4","2") and plan.code is None
    assert plan.message==question
    assert plan.requested_fields==("location","contact")
    result=agent.chat(question,[],DAY,prepared_plan=plan,notice_report=refresh_report())
    prompt=json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert prompt["question"]==question and TITLE in prompt["question"]
    assert {source["metadata"].get("facility_id") for source in prompt["sources"] if source["doc_type"]=="profile"}=={"4","2"}
    assert "창의관 1층" in result["answer"] and "학생복지관 3층" in result["answer"]
    assert "031-400-0002" in result["answer"]
    assert any(source["doc_type"]=="notice" and TITLE in source["title"] for source in result["sources"])


def test_document_title_facility_date_and_field_words_are_not_additional_requested_conditions(referenced_school):
    _,store,_,agent=referenced_school
    title="이모네 학생복지관 3층 11월 20일 전화번호와 영업시간 변경 공지"
    seed_reference(store,title)
    question=f"창의인재원식당와 교직원식당 위치랑 '{title}' 공지 알려줘"
    plan=agent.plan(question,[],DAY)
    assert plan.facility_ids==("4","2")
    assert plan.requested_fields==("location",)
    assert plan.span.start==DAY and plan.span.end==DAY
    assert title in plan.message


def test_real_location_condition_outside_title_remains_effective(referenced_school):
    _,_,_,agent=referenced_school
    question=f"학생복지관에 있는 창의인재원식당와 교직원식당 위치랑 '{TITLE}' 공지 알려줘"
    plan=agent.plan(question,[],DAY)
    assert plan.facility_ids==("2",) and plan.code=="re11"


def test_masking_requires_a_complete_known_title_and_preserves_input_length():
    original=f"교직원식당 위치와 '{TITLE}' 공지"
    masked=mask_document_references(original,titles=[TITLE])
    assert len(masked)==len(original)
    assert "교직원식당 위치" in masked and "창의관" not in masked
    assert mask_document_references(original,titles=["다른 공식 공지 제목"])==original
    partial="교직원식당 위치와 창의관식당 천원의 아침밥 공지"
    assert mask_document_references(partial,titles=[TITLE])==partial


def test_known_reference_tolerates_copy_paste_spacing_without_changing_original():
    original="이모네 위치와 '창의관식당결제방식변경안내' 공지"
    masked=mask_document_references(original,titles=["창의관 식당 결제 방식 변경 안내"])
    assert "이모네 위치" in masked and "창의관" not in masked
    assert original=="이모네 위치와 '창의관식당결제방식변경안내' 공지"

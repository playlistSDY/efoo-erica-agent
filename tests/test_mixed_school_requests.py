"""Complete multi-field/multi-facility requests without dropping notice intent."""

from copy import deepcopy
from datetime import date, datetime, timezone
import json
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
import pytest

with patch("dotenv.load_dotenv"):
    from app.indexing import index_facility, index_notice
    from app.notice_refresh import INDEX_SCOPE, INDEX_URL
    from app.rag import Agent
    from app.settings import Settings
    from app.store import Store


DAY=date(2026,11,14)
STAMP=datetime.now(timezone.utc).isoformat()
RECORDS=[
    {"id":20,"name":"이모네","category":"일반음식점","location":"학생복지관 2층"},
    {"id":21,"name":"분식마을","category":"일반음식점","location":"제1공학관 1층"},
    {"id":4,"name":"창의인재원식당","category":"구내식당","location":"창의관 1층"},
]
CONTACTS={20:"031-400-0020",21:"031-400-0021",4:"031-400-0004"}


def put_profile(store,row,*,contact=True,notice=True):
    phone=CONTACTS[row["id"]] if contact else ""
    info={"name":row["name"],"category":row["category"],"location":row["location"],
        "contact":phone,"opening_hours":"평일 10:00~19:00","building_no":"102",
        "info_rows":[{"label":"위치","value":row["location"]},{"label":"연락처","value":phone},
                     {"label":"영업시간","value":"평일 10:00~19:00"}],
        "notices":[{"text":row["name"]+" 공식 상세 운영 안내"}] if notice else [],
        "description":{"text":"공개 시설 설명"},"gallery":[],"cover_image":{}}
    index_facility(store,{"facility_id":row["id"],"restaurant_info":info,"page_data":{}},[],STAMP,
        "https://life.hanyang.ac.kr/theme/pages/facilities/"+
        ("detail.php" if row["id"]==4 else "shop-detail.php")+f"?id={row['id']}",row)


def put_notice(store,identity,code,text):
    index_notice(store,{"title":text,"text":text,"restaurant_codes":[code],"images":[],"assets":[],
        "fetched_at":STAMP,"url":f"https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id={identity}&board_lang=ko"},identity)


def refresh_report(*,usable=True,empty=False):
    return {"status":"ready" if usable else "error","unusable_scopes":["notice:13","notice:20"] if not usable or empty else [],
        "listed_ids":[] if empty else ([13,20] if usable else None),"targets":[],"errors":[],
        "index":{"usable":usable,"verified_empty":empty,"source":{"id":"S1","title":"공식 한국어 공지 목록",
            "url":INDEX_URL,"text":"공식 한국어 공지 게시판 공개 목록 0개","doc_type":"page","images":[],
            "metadata":{"count":0,"verified_empty":True},"fetched_at":STAMP,"date":DAY.isoformat(),"restaurant_code":"campus"}}}


@pytest.fixture
def school(tmp_path,monkeypatch):
    monkeypatch.setattr("requests.sessions.Session.request",Mock(side_effect=AssertionError("No external HTTP in regression")))
    settings=Settings(data_dir=tmp_path,database_url="",api_key="isolated-test",embedding_dimensions=3,
                      schedule_enabled=False,raw_s3_bucket="")
    store=Store(settings)
    store.replace_knowledge("page:facility-directory",code="campus",name="공식 시설 목록",
        url="https://life.hanyang.ac.kr/theme/pages/facilities/index.php",fetched_at=STAMP,
        payload={"records":deepcopy(RECORDS),"categories":{},"count":len(RECORDS)},chunks=[])
    for row in RECORDS:
        put_profile(store,row)
    put_notice(store,13,"re13","기숙사 창의관식당 아침밥 결제 공지")
    put_notice(store,20,"fac_20","이모네 공식 게시판 공지")
    pending=store.pending()
    store.save_embeddings(pending,[[1.,0.,0.]]*len(pending))
    client=Mock()
    client.embeddings.create.return_value.data=[Mock(embedding=[1.,0.,0.])]
    # Deliberately omit all requested values except one general sentence.
    # Server completion must make the missing fields explicit and sourced.
    client.responses.create.side_effect=lambda **kwargs:Mock(output_text=json.dumps({
        "answer":"공식 안내를 확인했어요. [S1]","source_ids":["S1"],"insufficient_evidence":False}))
    agent=Agent(store,settings,client=client)
    agent.school.refresher=Mock()
    agent.school.refresher.ensure.return_value={"targets":[{"id":row["id"],"usable":True} for row in RECORDS],"errors":[]}
    return settings,store,client,agent


@pytest.mark.parametrize("message",["이모네 위치랑 공지도 알려줘","이모네 전화번호와 공지","이모네 사진과 결제 안내"])
def test_mixed_notice_requests_leave_typed_field_only_route(school,message):
    _,_,_,agent=school
    assert not agent.school.plan(message).active
    plan=agent.plan(message,[],DAY)
    assert plan.school_plan is None and plan.directory_query is None
    assert plan.facility_ids==("20",)
    assert plan.notice_request and plan.requested_fields


def test_multiple_explicit_facilities_clear_single_code_and_keep_both_profiles(school):
    _,_,client,agent=school
    message="이모네랑 분식마을 위치와 전화번호랑 공지도 알려줘"
    plan=agent.plan(message,[],DAY)
    assert plan.code is None
    assert plan.facility_ids==("20","21") and plan.facility_codes==("fac_20","fac_21")
    result=agent.chat(message,[],DAY,prepared_plan=plan,notice_report=refresh_report())
    prompt=json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    assert [row["name"] for row in prompt["context"]["requested_facilities"]]==["이모네","분식마을"]
    assert set(prompt["context"]["requested_fields"])=={"location","contact"}
    assert {row["restaurant_code"] for row in prompt["sources"]}<={"fac_20","fac_21"}
    assert {row["metadata"].get("facility_id") for row in prompt["sources"] if row["doc_type"]=="profile"}=={"20","21"}
    assert all(row["doc_type"]!="menu" for row in prompt["sources"])
    assert "기숙사 창의관식당 아침밥 결제 공지" not in json.dumps(prompt,ensure_ascii=False)
    for row in RECORDS[:2]:
        assert row["location"] in result["answer"] and CONTACTS[row["id"]] in result["answer"]
        assert row["name"]+" 공식 상세 운영 안내" in result["answer"]
    assert len(result["requested_field_coverage"])==4
    assert all(item["status"]=="available" for item in result["requested_field_coverage"])
    assert result["selected_facility_ids"]==["20","21"]


def test_one_complete_profile_bundle_contains_all_requested_fields_and_notice(school):
    _,_,client,agent=school
    result=agent.chat("이모네 위치와 전화번호, 영업시간이랑 공지 알려줘",[],DAY,notice_report=refresh_report())
    prompt=json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])
    profiles=[row for row in prompt["sources"] if row["doc_type"]=="profile"]
    assert len(profiles)==1 and profiles[0]["metadata"]["profile_bundle"] is True
    for value in ["학생복지관 2층",CONTACTS[20],"평일 10:00~19:00","이모네 공식 상세 운영 안내"]:
        assert value in profiles[0]["text"] and value in result["answer"].replace("\\","")
    assert "이모네 공식 게시판 공지" in result["answer"]


def test_notice_completion_includes_policy_body_and_excludes_purchase_observations(school):
    _,store,_,agent=school
    index_notice(store,{"title":"특별 결제 안내 제목","text":"11월 9일부터 온라인 구매만 가능합니다.",
        "restaurant_codes":["fac_20"],"images":[],"assets":[],"fetched_at":STAMP,
        "url":"https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id=22&board_lang=ko"},22)
    store.replace_knowledge("purchase:20",code="fac_20",name="지난 구매 관찰",url="https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=20",
        fetched_at=STAMP,payload={},chunks=[{"id":"purchase-observation","title":"지난 구매 관찰","text":"이모네 과거 판매 종료",
        "doc_type":"notice","metadata":{"related_codes":["fac_20"]}}])
    rows=store.pending()
    store.save_embeddings(rows,[[1.,0.,0.]]*len(rows))
    result=agent.chat("이모네 위치랑 결제 공지 알려줘",[],DAY,notice_report=refresh_report())
    assert "11월 9일부터 온라인 구매만 가능합니다." in result["answer"]
    assert "이모네 과거 판매 종료" not in result["answer"]
    assert all("과거 판매 종료" not in source["text"] for source in result["sources"])


@pytest.mark.parametrize("empty",[False,True])
def test_unavailable_or_empty_board_does_not_drop_independent_profile_facts(school,empty):
    _,_,_,agent=school
    result=agent.chat("이모네 위치와 전화번호랑 공지 알려줘",[],DAY,
        notice_report=refresh_report(usable=empty,empty=empty))
    assert result["mode"]=="openai"
    assert "학생복지관 2층" in result["answer"] and CONTACTS[20] in result["answer"]
    assert "이모네 공식 상세 운영 안내" in result["answer"]
    assert "이모네 공식 게시판 공지" not in result["answer"]
    assert any(source["doc_type"]=="profile" for source in result["sources"])
    if empty:
        assert "0개" in result["answer"] and "없다는 뜻은 아니" in result["answer"]
        assert sum(source["url"]==INDEX_URL for source in result["sources"])==1
    else:
        assert "최신 목록은 불러오지 못했" in result["answer"]
        assert all(source["doc_type"]!="notice" for source in result["sources"])
    assert len({source["id"] for source in result["sources"]})==len(result["sources"])


def test_missing_requested_field_and_unconfirmed_notice_are_explicit(school):
    _,store,_,agent=school
    put_profile(store,RECORDS[1],contact=False,notice=False)
    result=agent.chat("분식마을 위치와 전화번호랑 공지 알려줘",[],DAY,notice_report=refresh_report())
    assert "제1공학관 1층" in result["answer"]
    assert "분식마을 연락처**: 최신 공식 문자 안내에서 확인하지 못했" in result["answer"]
    assert "공지가 없다는 뜻은 아니" in result["answer"]
    assert CONTACTS[20] not in result["answer"] and CONTACTS[4] not in result["answer"]
    assert {row["restaurant_code"] for row in result["sources"]}=={"fac_21"}
    assert next(item for item in result["requested_field_coverage"] if item["field"]=="contact")["status"]=="unavailable"


def test_missing_second_facility_is_not_replaced_by_an_unrelated_profile(school):
    _,store,_,agent=school
    with store.connection() as connection:
        connection.execute("DELETE FROM chunks WHERE scope='facility:21'")
    result=agent.chat("이모네와 분식마을 위치와 전화번호랑 공지 알려줘",[],DAY,notice_report=refresh_report())
    assert "분식마을 위치**: 최신 공식 문자 안내에서 확인하지 못했" in result["answer"]
    assert "분식마을 연락처**: 최신 공식 문자 안내에서 확인하지 못했" in result["answer"]
    assert "창의관 1층" not in result["answer"] and CONTACTS[4] not in result["answer"]
    assert {row["restaurant_code"] for row in result["sources"]}=={"fac_20"}


@pytest.mark.parametrize("followup",["그 두 곳 위치랑 공지도 알려줘","그 두 곳의 전화번호랑 공지","그 두 매장 위치와 공지"])
def test_mixed_followup_uses_authoritative_previous_facility_selection(school,followup):
    _,_,_,agent=school
    history=[{"role":"user","content":"이모네와 분식마을 위치 알려줘"},
             {"role":"assistant","content":"공식 목록 안내","selected_facility_ids":["20","21"]}]
    plan=agent.plan(followup,history,DAY)
    assert plan.facility_ids==("20","21") and plan.code is None


def test_server_field_completion_escapes_source_markdown_and_citation_shaped_text(school):
    _,store,_,agent=school
    row={**RECORDS[0],"location":"학생복지관 2층 | [S99] **안내**"}
    put_profile(store,row)
    result=agent.chat("이모네 위치랑 공지 알려줘",[],DAY,notice_report=refresh_report())
    assert "［S99］" in result["answer"] and "[S99]" not in result["answer"]
    assert r"\|" in result["answer"] and r"\*\*안내\*\*" in result["answer"]
    assert any(row["location"] in source["text"] for source in result["sources"])


def test_api_refreshes_both_named_profiles_and_notice_board_on_mixed_query(school,monkeypatch):
    settings,_,_,agent=school
    for key,value in {"DATA_DIR":str(settings.data_dir/"module"),"DATABASE_URL":"", "OPENAI_API_KEY":"",
                      "SCHEDULE_ENABLED":"false"}.items():
        monkeypatch.setenv(key,value)
    with patch("dotenv.load_dotenv"):
        from app.main import create_app
    notices,meals=Mock(),Mock()
    notices.ensure.return_value=refresh_report(usable=False)
    with TestClient(create_app(settings,agent.store,agent,refresh=meals,scheduler=Mock(),notice_refresh=notices)) as http:
        response=http.post("/api/chat",json={"message":"이모네랑 분식마을 위치와 전화번호랑 공지도 알려줘","date":DAY.isoformat()})
    assert response.status_code==200
    notices.ensure.assert_called_once_with()
    agent.school.refresher.ensure.assert_called_once()
    assert {row["id"] for row in agent.school.refresher.ensure.call_args.args[0]}=={20,21}
    meals.ensure.assert_not_called()
    payload=response.json()
    assert CONTACTS[20] in payload["answer"] and CONTACTS[21] in payload["answer"]
    assert "최신 목록은 불러오지 못했" in payload["answer"]


def test_pure_meal_plan_keeps_existing_date_and_restaurant_contract(school):
    _,_,_,agent=school
    plan=agent.plan("창의인재원식당 점심 메뉴",[],DAY)
    assert plan.menu_request and plan.code=="re13"
    assert not plan.notice_request and plan.facility_ids==()

"""Model-led conversation over synthetic campus tools, with no live resources.

Every Responses call is scripted. No Settings instance, server, database,
OpenAI client, portal request or existing conversation is opened by this file.
"""

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import socket
import sqlite3
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

with patch("dotenv.load_dotenv"):
    from app.agentic import AgenticService
    from app.date_ranges import DateRange
    from app.directory import DirectoryQuery
    from app.school_query import SchoolPlan


DAY = date(2026, 11, 19)
NOW = datetime(2026, 11, 19, 3, 17, 30, tzinfo=timezone.utc)
CATALOG_URL = "https://life.hanyang.ac.kr/theme/pages/facilities/index.php"
DETAIL_URL = "https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id=101"
PHOTO = "/media/" + "a" * 64 + ".jpg"
RECORDS = [
    {"id": 101, "name": "합성분식", "category": "일반음식점", "location": "학생복지관 2층"},
    {"id": 102, "name": "합성식당", "category": "구내식당", "location": "합성관 1층"},
]


def final(answer="편하게 이야기해요.", *, kind="conversation", sources=(), facilities=(), as_dict=False):
    payload = {"kind": kind, "answer": answer, "source_ids": list(sources), "facility_ids": list(facilities)}
    result = {"output": [], "output_text": json.dumps(payload, ensure_ascii=False)}
    return result if as_dict else SimpleNamespace(**result)


def tool(name, arguments, call_id="call-1", *, as_object=False):
    call = {"type": "function_call", "name": name, "call_id": call_id,
        "arguments": json.dumps(arguments, ensure_ascii=False) if not isinstance(arguments, str) else arguments}
    return SimpleNamespace(output=[SimpleNamespace(**call) if as_object else call], output_text="")


def catalog_result():
    return {"answer": "공식 목록에는 합성분식과 합성식당이 있어요. [S1]", "mode": "retrieval", "model": None,
        "sources": [{"id": "S1", "title": "합성 공식 시설 목록", "url": CATALOG_URL,
            "text": "합성분식 학생복지관 2층 / 합성식당 합성관 1층", "date": DAY.isoformat(),
            "fetched_at": NOW.isoformat(), "images": [], "metadata": {}}],
        "images": [], "directory": {"count": 2, "records": deepcopy(RECORDS), "scope": "official_portal"},
        "trace": ["합성 전체 카탈로그 조회"]}


def details_result():
    return {"answer": "합성분식은 학생복지관 2층이고 평일 11:00~19:00 안내예요. [S1]",
        "mode": "retrieval", "model": None,
        "sources": [{"id": "S1", "title": "합성분식 공식 안내", "url": DETAIL_URL,
            "text": "위치: 학생복지관 2층\n운영시간: 평일 11:00~19:00", "date": DAY.isoformat(),
            "fetched_at": NOW.isoformat(), "doc_type": "profile", "metadata": {"facility_id": 101},
            "images": [{"url": PHOTO, "title": "합성 공식 사진", "source_url": "https://invalid.example/forged"}]}],
        "images": [{"url": PHOTO, "source_url": DETAIL_URL}], "selected_facility_ids": ["101"],
        "reference_facility_ids": ["101"], "trace": ["합성 상세 필드 조회"]}


class FakeResponses:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(deepcopy(kwargs))
        assert self.responses, "Unexpected additional model call"
        value = self.responses.pop(0)
        return value(kwargs) if callable(value) else value


@pytest.fixture(autouse=True)
def forbid_external_resources(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Live network, database, or OpenAI access is forbidden")

    monkeypatch.setattr(socket, "create_connection", forbidden)
    monkeypatch.setattr(sqlite3, "connect", forbidden)
    monkeypatch.setattr("requests.sessions.Session.request", forbidden)
    monkeypatch.setattr("app.rag.OpenAI", forbidden)


@pytest.fixture
def make_service():
    def make(responses, *, turns=5, key="synthetic-test-key", now=NOW):
        responses = FakeResponses(responses)
        client = SimpleNamespace(responses=responses, embeddings=SimpleNamespace(
            create=Mock(side_effect=AssertionError("The agent loop must delegate vector retrieval"))))
        directory = SimpleNamespace(answer=Mock(side_effect=lambda *args, **kwargs: catalog_result()))
        school = SimpleNamespace(answer=Mock(side_effect=lambda *args, **kwargs: details_result()),
            refresh_targets=Mock(side_effect=lambda ids: {"targets":[{"id":int(identity),"usable":True} for identity in ids]}))
        store = SimpleNamespace(knowledge=Mock(return_value=[]), candidates=Mock(return_value=[]))
        agent = SimpleNamespace(get_client=Mock(return_value=client), directory=directory, school=school, store=store)
        settings = SimpleNamespace(api_key=key, model="synthetic-agent-model", agentic_max_turns=turns, max_age_hours=24)
        delegated = {"answer": "검증된 학식 답변 [S1]", "mode": "retrieval", "model": None,
            "sources": [{"id": "S1", "title": "합성 학식", "url": DETAIL_URL, "text": "합성 메뉴", "images": []}],
            "images": [], "date": "2026-11-20", "date_range": {"start": "2026-11-20", "end": "2026-11-20"}}
        school_query = Mock(side_effect=lambda query: deepcopy(delegated))
        service = AgenticService(settings, agent, school_query=school_query, now=lambda: now)
        return SimpleNamespace(service=service, agent=agent, responses=responses, settings=settings,
            school_query=school_query, delegated=delegated, client=client)
    return make


@pytest.mark.parametrize("utterance", ["배고파", "오늘 좀 지쳤어", "그냥 수다 떨고 싶어", "안녕"])
def test_no_tool_conversation_is_chosen_by_model_without_forcing_school_lookup(make_service, utterance):
    bundle = make_service([final("그랬구나. 어떤 이야기를 나눠볼까요?")])
    result = bundle.service.run(utterance, [], DAY)
    assert result["mode"] == "conversation" and result["model"] == "synthetic-agent-model"
    assert result["sources"] == result["images"] == result["selected_facility_ids"] == []
    assert result["answer"] == "그랬구나. 어떤 이야기를 나눠볼까요?"
    bundle.agent.directory.answer.assert_not_called()
    bundle.agent.school.answer.assert_not_called()
    bundle.school_query.assert_not_called()
    assert len(bundle.responses.calls) == 1
    assert json.loads(bundle.responses.calls[0]["input"][0]["content"])["question"] == utterance


@pytest.mark.parametrize(("kind", "mode"), [("clarification", "clarification"), ("unsupported", "insufficient")])
def test_no_tool_clarification_and_unsupported_requests_are_not_falsely_grounded(make_service, kind, mode):
    bundle = make_service([final("원하는 조건을 조금 더 알려주세요.", kind=kind, as_dict=True)])
    result = bundle.service.run("조건을 모르겠어", [], DAY)
    assert result["mode"] == mode and not result["sources"] and not result["images"]
    assert result["agent_steps"] == []


def test_discovery_then_actual_details_then_grounded_answer_rebases_sources_and_keeps_cited_photo(make_service):
    bundle = make_service([
        tool("facilities", {"categories": ["일반음식점", "구내식당"], "location": None}, as_object=True),
        tool("facility_details", {"facility_ids": [101]}, "call-2"),
        final("합성분식의 공식 운영시간을 확인했어요. [S2]", kind="grounded", sources=["S2"], facilities=[101]),
    ])
    result = bundle.service.run("배고파. 학교에서 먹을 곳 찾아볼까", [], DAY)
    query = bundle.agent.directory.answer.call_args.args[0]
    assert isinstance(query, DirectoryQuery) and query.action == "list"
    assert query.categories == ("일반음식점", "구내식당") and query.location is None
    assert bundle.agent.directory.answer.call_args.kwargs == {"refresh": True}
    plan = bundle.agent.school.answer.call_args.args[0]
    assert isinstance(plan, SchoolPlan) and plan.facility_ids == plan.reference_facility_ids == ("101",)
    assert set(plan.slots) == {"location", "contact", "hours", "images"}
    assert bundle.agent.school.answer.call_args.kwargs == {"refresh": True}
    details_tool = json.loads(bundle.responses.calls[2]["input"][-1]["output"])
    assert "[S2]" in details_tool["answer"] and "[S1]" not in details_tool["answer"]
    assert next(source for source in details_tool["sources"] if source["id"] == "S2")["url"] == DETAIL_URL
    assert result["mode"] == "agent" and [source["url"] for source in result["sources"]] == [DETAIL_URL]
    assert result["selected_facility_ids"] == result["reference_facility_ids"] == ["101"]
    assert result["images"] == [{"url": PHOTO, "title": "합성 공식 사진", "source_url": DETAIL_URL}]
    assert [step["tool"] for step in result["agent_steps"]] == ["facilities", "facility_details"]
    bundle.school_query.assert_not_called()
    bundle.client.embeddings.create.assert_not_called()


def test_terminal_school_query_preserves_relative_date_and_returns_exact_validated_payload_without_second_generation(make_service):
    query = "내일 학생식당 점심 메뉴와 가격 알려줘"
    bundle = make_service([tool("school_query", {"query": query})])
    result = bundle.service.run(query, [], DAY)
    bundle.school_query.assert_called_once_with(query)
    assert result == {**bundle.delegated, "agent_steps": [{"tool": "school_query", "status": "completed"}]}
    assert len(bundle.responses.calls) == 1
    bundle.agent.directory.answer.assert_not_called()
    bundle.agent.school.answer.assert_not_called()


def test_school_query_after_discovery_is_terminal_and_does_not_rebase_its_independently_verified_sources(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        tool("school_query", {"query": "그중 학생식당 내일 학식"}, "call-2")])
    result = bundle.service.run("학교에서 뭘 먹지", [], DAY)
    assert result["sources"] == bundle.delegated["sources"]
    assert result["answer"] == bundle.delegated["answer"] and result["date"] == "2026-11-20"
    assert result["agent_steps"] == [{"tool": "facilities", "status": "completed"}, {"tool": "school_query", "status": "completed"}]
    assert len(bundle.responses.calls) == 2


def test_current_kst_clock_custom_range_and_only_recent_trusted_history_reach_model_unchanged(make_service):
    history = [{"role": "user", "content": f"합성 질문 {index}"} for index in range(4)]
    history += [{"role": "assistant", "content": "이미 정한 식당이에요. [S88]", "selected_facility_ids": [101],
        "sources": [{"id": "S88", "text": "이전 자료는 이번 턴의 근거가 아님"}]},
        {"role": "user", "content": "2층이 좋아"}, {"role": "assistant", "content": "기억할게요."},
        {"role": "user", "content": "전화는 필요 없어"}, {"role": "assistant", "content": "네, 위치만 살펴볼게요."},
        {"role": "user", "content": "가까운지는 몰라"}]
    original = deepcopy(history)
    bundle = make_service([final("이미 알려준 조건을 참고할게요.")])
    bundle.service.run("배고파", history, DAY, "re12", DateRange(date(2026, 11, 20), date(2026, 11, 22)))
    call = bundle.responses.calls[0]
    context = json.loads(call["input"][0]["content"])
    assert context["now"] == "2026-11-19T12:17:30+09:00" and context["weekday"] == "목요일"
    assert context["selected_date"] == "2026-11-19" and context["selected_restaurant"] == "re12"
    assert context["date_range"] == {"start": "2026-11-20", "end": "2026-11-22"}
    assert context["previous_facility_ids"] == ["101"]
    assert context["conversation"] == [{"role": item["role"], "content": item["content"]} for item in history[-8:]]
    assert history == original
    assert call["store"] is False and call["parallel_tool_calls"] is False
    assert call["text"]["format"]["strict"] is True and call["text"]["format"]["schema"]["additionalProperties"] is False
    assert {item["name"] for item in call["tools"]} == {"facilities", "facility_details", "school_query"}


def test_previous_assistant_selection_allows_detail_followup_but_old_citations_do_not_become_new_sources(make_service):
    history = [{"role": "assistant", "content": "합성분식이에요. [S88]", "selected_facility_ids": [101]}]
    bundle = make_service([tool("facility_details", {"facility_ids": [101]}),
        final("합성분식의 위치를 다시 확인했어요. [S1]", kind="grounded", sources=["S1"], facilities=[101])])
    result = bundle.service.run("거기 어디야?", history, DAY)
    bundle.agent.directory.answer.assert_not_called()
    assert result["mode"] == "agent" and result["selected_facility_ids"] == ["101"]
    assert [source["id"] for source in result["sources"]] == ["S1"]


def test_detail_ids_without_prior_discovery_are_rejected_then_model_can_ask_for_clarification(make_service):
    bundle = make_service([tool("facility_details", {"facility_ids": [999]}),
        final("먼저 원하는 곳을 같이 찾아볼까요?", kind="clarification")])
    result = bundle.service.run("뭘 먹을까", [], DAY)
    bundle.agent.school.answer.assert_not_called()
    output = json.loads(bundle.responses.calls[1]["input"][-1]["output"])
    assert "error" in output and result["mode"] == "clarification"
    assert result["agent_steps"] == [{"tool": "facility_details", "status": "invalid"}]


@pytest.mark.parametrize(("name", "args"), [
    ("facilities", {"categories": ["학사"], "location": None}),
    ("facilities", {"categories": [], "location": "x" * 81}),
    ("facilities", {"categories": [], "location": None, "unexpected": True}),
    ("facility_details", {"facility_ids": [True]}),
    ("facility_details", {"facility_ids": ["101"]}),
    ("facility_details", {"facility_ids": []}),
    ("facility_details", {"facility_ids": list(range(17))}),
    ("school_query", {"query": " "}),
    ("school_query", {"query": "x" * 2001}),
    ("school_query", {"query": "질문", "extra": True}),
    ("purchase", {"facility_ids": [101]}),
    ("facilities", "{broken-json"),
    ("facilities", "[]"),
])
def test_invalid_tool_arguments_and_unknown_tools_never_dispatch_or_escape_loop(make_service, name, args):
    bundle = make_service([tool(name, args), final("조건을 다시 확인할게요.", kind="clarification")])
    result = bundle.service.run("학교 정보를 살펴볼까", [], DAY)
    assert result["agent_steps"] == [{"tool": name, "status": "invalid"}]
    assert "error" in json.loads(bundle.responses.calls[1]["input"][-1]["output"])
    bundle.agent.directory.answer.assert_not_called()
    bundle.agent.school.answer.assert_not_called()
    bundle.school_query.assert_not_called()


def test_model_parallel_tool_calls_are_all_rejected_without_running_any_of_them(make_service):
    first = tool("facilities", {"categories": [], "location": None})
    second = tool("school_query", {"query": "오늘 학식"}, "call-2")
    first.output.extend(second.output)
    bundle = make_service([first, final("한 가지씩 확인해 볼게요.", kind="clarification")])
    result = bundle.service.run("학교 안내", [], DAY)
    assert [step["status"] for step in result["agent_steps"]] == ["invalid", "invalid"]
    bundle.agent.directory.answer.assert_not_called()
    bundle.school_query.assert_not_called()
    outputs = [item for item in bundle.responses.calls[1]["input"] if isinstance(item, dict) and item.get("type") == "function_call_output"]
    assert len(outputs) == 2 and all("error" in json.loads(item["output"]) for item in outputs)


@pytest.mark.parametrize("payload", [
    {"kind": "grounded", "answer": "모델이 만든 매장 [S999]", "source_ids": ["S999"], "facility_ids": []},
    {"kind": "grounded", "answer": "모델이 만든 매장 [S1]", "source_ids": [], "facility_ids": []},
    {"kind": "grounded", "answer": "모델이 만든 매장", "source_ids": [], "facility_ids": []},
    {"kind": "conversation", "answer": "과거 출처를 썼어요. [S88]", "source_ids": ["S88"], "facility_ids": []},
    {"kind": "conversation", "answer": "", "source_ids": [], "facility_ids": []},
    {"kind": "conversation", "answer": "잘못된 형식", "source_ids": "S1", "facility_ids": []},
    {"kind": "conversation", "answer": "잘못된 형식", "source_ids": [], "facility_ids": [True]},
    {"kind": "unexpected", "answer": "잘못된 형식", "source_ids": [], "facility_ids": []},
    {"kind": "conversation", "answer": "예상 밖 필드", "source_ids": [], "facility_ids": [], "extra": 1},
])
def test_ungrounded_citations_and_malformed_final_payloads_are_rejected_without_sources(make_service, payload):
    bundle = make_service([SimpleNamespace(output=[], output_text=json.dumps(payload, ensure_ascii=False))])
    result = bundle.service.run("학교에 뭐가 있어", [], DAY)
    assert result["mode"] == "clarification" and result["sources"] == result["images"] == []
    assert payload["answer"] not in result["answer"] or not payload["answer"]


def test_invalid_final_after_tools_returns_verified_tool_answer_instead_of_hallucination(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        final("절대로 보여주면 안 되는 가짜 장소 [S999]", kind="grounded", sources=["S999"], facilities=[999])])
    result = bundle.service.run("배고파", [], DAY)
    assert "가짜 장소" not in result["answer"]
    assert result["answer"] == catalog_result()["answer"] and result["sources"] == catalog_result()["sources"]
    assert result["mode"] == "retrieval" and "에이전트의 출처 검증 실패" in result["trace"]


def test_repeated_identical_tool_sources_share_one_id_without_mutating_backend_payload(make_service):
    original = catalog_result()
    saved = deepcopy(original)
    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        tool("facilities", {"categories": [], "location": None}, "call-2"),
        final("합성분식 목록이에요. [S1]", kind="grounded", sources=["S1"], facilities=[101])])
    bundle.agent.directory.answer.side_effect = None
    bundle.agent.directory.answer.return_value = original
    result = bundle.service.run("먹을 곳", [], DAY)
    assert len(result["sources"]) == len(bundle.service.sources) == 1
    assert original == saved


def test_uncited_registered_source_and_its_image_are_omitted_from_final_answer(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        tool("facility_details", {"facility_ids": [101]}, "call-2"),
        final("합성분식은 목록에 있어요. [S1]", kind="grounded", sources=["S1", "S2"], facilities=[101])])
    result = bundle.service.run("먹을 곳", [], DAY)
    assert [source["url"] for source in result["sources"]] == [CATALOG_URL]
    assert result["images"] == []


def test_unallowed_facility_id_never_becomes_saved_selection_even_when_model_names_it(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        final("합성분식은 목록에 있어요. [S1]", kind="grounded", sources=["S1"], facilities=[999])])
    result = bundle.service.run("먹을 곳", [], DAY)
    assert result["selected_facility_ids"] == result["reference_facility_ids"] == []


def test_agent_turn_cap_stops_infinite_tool_requests_and_last_model_call_disables_tools(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}, f"call-{index}") for index in range(3)], turns=3)
    result = bundle.service.run("계속 확인해봐", [], DAY)
    assert len(bundle.responses.calls) == 3
    assert [call["tool_choice"] for call in bundle.responses.calls] == ["auto", "auto", "none"]
    assert result["mode"] == "retrieval" and "에이전트 자료 확인 횟수 제한" in result["trace"]


def test_no_api_key_delegates_original_text_without_creating_client_or_tools(make_service):
    bundle = make_service([], key="")
    result = bundle.service.run("배고파", [], DAY)
    bundle.school_query.assert_called_once_with("배고파")
    bundle.agent.get_client.assert_not_called()
    assert result == bundle.delegated


def test_reused_service_does_not_reuse_prior_turn_citations_or_candidates(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        final("합성분식이 있어요. [S1]", kind="grounded", sources=["S1"], facilities=[101]),
        final("지난 자료를 이번 자료처럼 쓰면 안 돼요. [S1]", kind="grounded", sources=["S1"], facilities=[101])])
    first = bundle.service.run("먹을 곳", [], DAY)
    assert first["sources"] and first["agent_steps"]
    second = bundle.service.run("그냥 이야기할래", [], DAY)
    assert second["mode"] == "clarification" and second["sources"] == second["images"] == []
    assert second["agent_steps"] == [] and second["selected_facility_ids"] == []
    context = json.loads(bundle.responses.calls[2]["input"][0]["content"])
    assert context["previous_facility_ids"] == []


def test_no_key_run_resets_turn_state_too_when_service_is_reused(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}), final("합성분식 [S1]", kind="grounded", sources=["S1"], facilities=[101])])
    bundle.service.run("먹을 곳", [], DAY)
    bundle.settings.api_key = ""
    bundle.service.run("오늘 학식", [], DAY)
    assert bundle.service.sources == bundle.service.steps == []
    assert bundle.service.allowed_ids == set() and bundle.service.records == {} and bundle.service.last_result is None
    assert bundle.service.catalog_records == bundle.service.catalog_source_ids == {} and bundle.service.invalid_source_ids == set()


def configure_discovery(bundle, records, narratives, *, fetched_at=NOW, usable=None):
    """Keep catalog text intentionally minimal: details must cross the tool boundary."""
    catalog = catalog_result()
    catalog["directory"].update(count=len(records), records=deepcopy(records))
    catalog["sources"][0]["text"] = "\n".join(f"ID {row['id']}: {row['name']} / {row['category']} / {row['location']}" for row in records)
    catalog["answer"] = "공식 후보 목록이에요. [S1]"
    bundle.agent.directory.answer.side_effect = None
    bundle.agent.directory.answer.return_value = catalog
    profiles = []
    for row in records:
        if row["id"] not in narratives:
            continue
        value = narratives[row["id"]]
        stamp = value.get("fetched_at", fetched_at).isoformat()
        profiles.append({"scope": f"facility:{row['id']}", "code": f"fac_{row['id']}", "name": row["name"],
            "url": f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={row['id']}", "fetched_at": stamp,
            "payload": {"restaurant_info": {"name": row["name"], "category": row["category"], "location": row["location"],
                "description": {"text": value.get("description", "")},
                "notices": [{"text": text} for text in value.get("notices", [])]}}})
    bundle.agent.store.knowledge.return_value = profiles
    bundle.agent.school.refresh_targets.side_effect = lambda ids: {"targets": [
        {"id": int(identity), "usable": str(identity) in {str(value) for value in usable} if usable is not None else True} for identity in ids]}
    return catalog, profiles


@pytest.mark.parametrize("weak_name", ["학문사", "늘봄상회", "주식회사 제이유"])
def test_model_sees_weak_name_multi_product_store_before_selecting_stationery_candidates(make_service, weak_name):
    records = [
        {"id": 601, "name": "학생문구", "category": "인쇄/문구/서점/기념품", "location": "합성관 1층"},
        {"id": 602, "name": weak_name, "category": "인쇄/문구/서점/기념품", "location": "학생복지관 3층"},
        {"id": 603, "name": "출력센터", "category": "인쇄/문구/서점/기념품", "location": "도서관 1층"},
        {"id": 604, "name": "늘푸른상회", "category": "인쇄/문구/서점/기념품", "location": "합성관 2층"},
    ]
    original = None

    def choose_after_reading_tool(call):
        output = json.loads(call["input"][-1]["output"])
        assert {row["id"] for row in output["facilities"]} == {601, 602, 603, 604}
        by_id = {row["id"]: row for row in output["facilities"]}
        weak = by_id[602]
        assert weak["description"] == "대학교재, 문구류, 한양대기념품 판매"
        assert {"문구", "서점", "기념품"} <= set(weak["capabilities"])
        proof = weak["capabilities"]["문구"]
        assert proof["source"] == "profile"
        source = next(source for source in output["sources"] if source["id"] == proof["source_id"])
        assert source["metadata"]["facility_id"] == 602
        assert source["url"].endswith("id=602") and "문구류" in source["text"]
        assert "문구" in by_id[601]["capabilities"]
        assert "문구" not in by_id[603]["capabilities"]
        assert "문구" not in by_id[604]["capabilities"]
        assert by_id[604]["notices"] == ["문구류를 판매하지 않습니다."]
        assert output["coverage"] == {"listed": 4, "details_checked": 4, "complete": True}
        ids = [by_id[identity]["capabilities"]["문구"]["source_id"] for identity in (601, 602)]
        return final(f"학생문구 [{ids[0]}], {weak_name}도 문구류 판매 안내가 있어요. [{ids[1]}]",
            kind="grounded", sources=ids, facilities=[601, 602])

    bundle = make_service([tool("facilities", {"categories": ["인쇄/문구/서점/기념품"], "location": None}), choose_after_reading_tool])
    original, _ = configure_discovery(bundle, records, {
        601: {"description": "필기구와 학용품 판매"},
        602: {"description": "대학교재, 문구류, 한양대기념품 판매"},
        603: {"description": "인쇄와 출력, 제본 안내"},
        604: {"notices": ["문구류를 판매하지 않습니다."]},
    })
    saved = deepcopy(original)
    result = bundle.service.run("노트랑 필기구 살 곳 찾아줘", [], DAY)
    assert result["mode"] == "agent" and result["selected_facility_ids"] == ["601", "602"]
    bundle.agent.school.refresh_targets.assert_called_once_with(("601", "602", "603", "604"))
    assert original == saved


def test_discovery_checks_all_candidates_without_the_sixteen_detail_limit_hiding_late_records(make_service):
    records = [{"id": 700 + index, "name": f"상회{index}", "category": "인쇄/문구/서점/기념품",
        "location": "합성관 1층"} for index in range(21)]

    def inspect_all_candidates(call):
        output = json.loads(call["input"][-1]["output"])
        assert len(output["facilities"]) == 21
        last = output["facilities"][-1]
        assert last["id"] == 720 and "문구" in last["capabilities"]
        assert "기념품" in last["notices"][0]
        identity = last["capabilities"]["문구"]["source_id"]
        return final(f"상회20에 문구류 안내가 있어요. [{identity}]", kind="grounded", sources=[identity], facilities=[720])

    bundle = make_service([tool("facilities", {"categories": ["인쇄/문구/서점/기념품"], "location": None}), inspect_all_candidates])
    configure_discovery(bundle, records, {row["id"]: {"notices": ["문구류와 기념품을 판매합니다."]} for row in records})
    result = bundle.service.run("필기구가 필요해", [], DAY)
    assert result["selected_facility_ids"] == ["720"]
    assert len(bundle.agent.school.refresh_targets.call_args.args[0]) == 21


def test_failed_stale_missing_and_empty_profiles_are_distinct_and_never_leak_old_capabilities(make_service):
    records = [{"id": identity, "name": f"합성상회{identity}", "category": "인쇄/문구/서점/기념품", "location": "합성관 1층"}
        for identity in range(801, 806)]

    def inspect_failure_boundary(call):
        output = json.loads(call["input"][-1]["output"])
        values = {row["id"]: row for row in output["facilities"]}
        assert values[801]["narrative_status"] == "unavailable"  # Explicit refresh failure.
        assert values[802]["narrative_status"] == "unavailable"  # Old timestamp even if report says usable.
        assert values[803]["narrative_status"] == "unavailable"  # Missing snapshot.
        assert values[804]["narrative_status"] == "empty"        # Fresh, but no public narrative.
        assert values[805]["narrative_status"] == "available"
        assert all(not values[identity]["capabilities"] for identity in (801, 802, 803, 804))
        assert all(values[identity]["description"] == "" for identity in (801, 802, 803))
        assert "STALE_SENTINEL" not in json.dumps(output, ensure_ascii=False)
        assert output["coverage"] == {"listed": 5, "details_checked": 2, "complete": False}
        return final("일부 상세를 확인하지 못했어요.", kind="clarification")

    bundle = make_service([tool("facilities", {"categories": [], "location": None}), inspect_failure_boundary])
    configure_discovery(bundle, records, {801: {"description": "문구류 판매 STALE_SENTINEL"},
        802: {"description": "문구류 판매 STALE_SENTINEL", "fetched_at": NOW - timedelta(days=2)},
        804: {}, 805: {"description": "문구류 판매"}}, usable={802, 803, 804, 805})
    bundle.service.run("노트 사야 해", [], DAY)


def test_duplicate_store_names_keep_separate_location_ids_and_description_provenance(make_service):
    records = [{"id": identity, "name": "함께상회", "category": "인쇄/문구/서점/기념품", "location": f"합성관 {index}층"}
        for index, identity in enumerate((901, 902), 1)]

    def inspect_distinct_branches(call):
        output = json.loads(call["input"][-1]["output"])
        first, second = output["facilities"]
        assert first["name"] == second["name"] and first["id"] != second["id"]
        assert first["location"] != second["location"]
        assert "문구" in first["capabilities"] and "문구" not in second["capabilities"]
        source_id = first["capabilities"]["문구"]["source_id"]
        assert next(source for source in output["sources"] if source["id"] == source_id)["url"].endswith("id=901")
        return final(f"합성관 1층 함께상회에서 문구를 취급해요. [{source_id}]", kind="grounded", sources=[source_id], facilities=[901])

    bundle = make_service([tool("facilities", {"categories": [], "location": None}), inspect_distinct_branches])
    configure_discovery(bundle, records, {901: {"description": "문구류 판매"}, 902: {"description": "대학교재만 판매합니다."}})
    assert bundle.service.run("필기구가 필요해", [], DAY)["selected_facility_ids"] == ["901"]


def test_each_candidate_narrative_has_own_budget_and_truncation_cannot_hide_next_candidate(make_service):
    records = [{"id": identity, "name": f"상회{identity}", "category": "인쇄/문구/서점/기념품", "location": "합성관"}
        for identity in (1001, 1002)]

    def inspect_bounded_narratives(call):
        output = json.loads(call["input"][-1]["output"])
        first, second = output["facilities"]
        assert len(first["description"]) == 2400 and first["description_truncated"] is True
        assert sum(map(len, first["notices"])) == 2400 and first["notices_truncated"] is True
        assert all(len(item["text"]) <= 1200 for proof in first["capabilities"].values() for item in proof["evidence"])
        assert second["description"] == "대학교재, 문구류, 기념품 판매"
        assert {"서점", "문구", "기념품"} <= set(second["capabilities"])
        return final("원하는 품목을 더 살펴볼게요.", kind="clarification")

    bundle = make_service([tool("facilities", {"categories": [], "location": None}), inspect_bounded_narratives])
    configure_discovery(bundle, records, {1001: {"description": "문구류 판매 " + "안내" * 3000,
        "notices": ["기념품 판매 " + "운영" * 2000, "다음 안내"]},
        1002: {"description": "대학교재, 문구류, 기념품 판매"}})
    bundle.service.run("문구 살 곳", [], DAY)


def test_details_followup_exposes_official_description_notices_and_its_cited_collected_photo(make_service):
    record = {"id": 1101, "name": "조용한상회", "category": "인쇄/문구/서점/기념품", "location": "학생복지관 3층"}

    def inspect_detail_narrative(call):
        output = json.loads(call["input"][-1]["output"])
        candidate = output["facilities"][0]
        assert candidate["description"] == "대학교재, 문구류, 기념품 판매"
        assert candidate["notices"] == ["월~금 08:30~18:00 안내"]
        source_id = candidate["capabilities"]["문구"]["source_id"]
        source = next(source for source in output["sources"] if source["id"] == source_id)
        assert source["metadata"]["narrative"] is True
        assert "문구류" in source["text"] and "08:30" in source["text"]
        assert source["id"] != "S1"  # Typed field source already used S1.
        return final(f"조용한상회는 문구류도 판매해요. [{source_id}]", kind="grounded", sources=[source_id], facilities=[1101])

    bundle = make_service([tool("facility_details", {"facility_ids": [1101]}), inspect_detail_narrative])
    _, profiles = configure_discovery(bundle, [record], {1101: {"description": "대학교재, 문구류, 기념품 판매",
        "notices": ["월~금 08:30~18:00 안내"]}})
    typed = details_result()
    typed["sources"][0].update(url=profiles[0]["url"], metadata={"facility_id": 1101})
    typed.update(selected_facility_ids=["1101"], reference_facility_ids=["1101"])
    bundle.agent.school.answer.side_effect = None
    bundle.agent.school.answer.return_value = typed
    bundle.agent.store.candidates.return_value = [{"images": json.dumps([{ "url": PHOTO, "alt": "공식 사진"}])}]
    original = deepcopy(profiles)
    result = bundle.service.run("거기 문구도 팔아?", [{"role": "assistant", "content": "조용한상회", "selected_facility_ids": [1101]}], DAY)
    assert result["selected_facility_ids"] == ["1101"]
    assert result["images"] == [{"url": PHOTO, "alt": "공식 사진", "facility_id": 1101, "source_url": profiles[0]["url"]}]
    assert profiles == original


def test_catalog_gallery_including_legacy_source_fallback_contains_only_selected_facility_images(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        final("합성분식을 살펴보세요. [S1]", kind="grounded", sources=["S1"], facilities=[101])])
    catalog = catalog_result()
    photos = [{"url": "/media/" + digit * 64 + ".jpg", "facility_id": identity, "title": f"사진{identity}"}
        for digit, identity in (("a", 101), ("b", 102))]
    photos += [{"url": "/media/" + "c" * 64 + ".jpg", "title": "시설을 증명할 수 없는 사진"}]
    catalog["sources"][0]["images"] = deepcopy(photos)
    catalog["images"] = deepcopy(photos)
    original = deepcopy(catalog)
    bundle.agent.directory.answer.side_effect = None
    bundle.agent.directory.answer.return_value = catalog
    result = bundle.service.run("먹을 곳을 골라줘", [], DAY)
    assert [image["facility_id"] for image in result["images"]] == [101]
    assert [image["facility_id"] for image in result["sources"][0]["images"]] == [101]
    assert catalog == original


def test_profile_image_owner_cannot_be_overridden_by_another_facility_tag(make_service):
    bundle = make_service([tool("facility_details", {"facility_ids": [101]}),
        final("합성분식이에요. [S1]", kind="grounded", sources=["S1"], facilities=[101])])
    typed = details_result()
    typed["sources"][0]["images"].append({"url": "/media/" + "b" * 64 + ".jpg", "facility_id": 102})
    bundle.agent.school.answer.side_effect = None
    bundle.agent.school.answer.return_value = typed
    result = bundle.service.run("거기 사진 보여줘", [{"role": "assistant", "content": "합성분식", "selected_facility_ids": [101]}], DAY)
    assert len(result["images"]) == len(result["sources"][0]["images"]) == 1
    assert result["images"][0]["source_url"] == DETAIL_URL


def test_invalid_final_does_not_dump_all_catalog_photos_as_a_fallback(make_service):
    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        final("임의 장소 [S999]", kind="grounded", sources=["S999"], facilities=[999])])
    catalog = catalog_result()
    catalog["images"] = catalog["sources"][0]["images"] = [{"url": PHOTO, "facility_id": 101}]
    bundle.agent.directory.answer.side_effect = None
    bundle.agent.directory.answer.return_value = catalog
    result = bundle.service.run("먹을 곳", [], DAY)
    assert result["images"] == [] and result["sources"][0]["images"] == []


@pytest.mark.parametrize("change", ["refresh_failed", "description_removed", "description_changed"])
def test_rechecked_details_revoke_old_profile_citations_and_never_reclassify_them_as_catalog_facts(make_service, change):
    record = {"id": 1201, "name": "새봄상회", "category": "인쇄/문구/서점/기념품", "location": "학생복지관 3층"}
    old_source_id = None
    bundle = None

    def recheck_after_discovery(call):
        nonlocal old_source_id
        output = json.loads(call["input"][-1]["output"])
        candidate = output["facilities"][0]
        old_source_id = candidate["capabilities"]["문구"]["source_id"]
        assert candidate["capabilities"]["문구"]["source"] == "profile"
        assert "문구류" in candidate["description"]
        if change == "refresh_failed":
            bundle.agent.school.refresh_targets.side_effect = lambda ids: {"targets": [{"id": int(identity), "usable": False} for identity in ids]}
            bundle.agent.school.answer.return_value = {"answer": "최신 상세 안내를 확인하지 못했어요.", "mode": "insufficient", "model": None,
                "sources": [], "images": [], "selected_facility_ids": []}
        else:
            info = bundle.agent.store.knowledge.return_value[0]["payload"]["restaurant_info"]
            info["description"]["text"] = "" if change == "description_removed" else "대학교재만 취급합니다."
        return tool("facility_details", {"facility_ids": [1201]}, "call-2")

    def attempt_revoked_citation(call):
        output = json.loads(call["input"][-1]["output"])
        candidate = output["facilities"][0]
        assert "문구" not in candidate["capabilities"]
        assert all("문구류" not in source["text"] for source in output["sources"])
        assert old_source_id in output["revoked_source_ids"]
        assert old_source_id not in {source["id"] for source in output["sources"]}
        return final(f"새봄상회는 여전히 문구류를 판매해요. [{old_source_id}]", kind="grounded", sources=[old_source_id], facilities=[1201])

    bundle = make_service([tool("facilities", {"categories": [], "location": None}), recheck_after_discovery, attempt_revoked_citation])
    _, profiles = configure_discovery(bundle, [record], {1201: {"description": "문구류 판매"}})
    typed = details_result()
    typed["answer"] = "새봄상회 위치는 학생복지관 3층이에요. [S1]"
    typed["sources"][0].update(url=profiles[0]["url"], text="새봄상회\n위치: 학생복지관 3층", metadata={"facility_id": 1201})
    typed.update(selected_facility_ids=["1201"], reference_facility_ids=["1201"])
    bundle.agent.school.answer.side_effect = None
    bundle.agent.school.answer.return_value = typed
    result = bundle.service.run("필기구를 사고 싶어", [], DAY)
    assert "여전히 문구류" not in result["answer"] and "문구류 판매" not in result["answer"]
    assert old_source_id not in {source["id"] for source in result["sources"]}
    assert old_source_id in bundle.service.invalid_source_ids
    assert result["mode"] != "agent"


def test_detail_catalog_capability_keeps_catalog_source_id_without_reusing_profile_description(make_service):
    record = {"id": 1301, "name": "학생문구", "category": "인쇄/문구/서점/기념품", "location": "학생복지관 1층"}

    def inspect_detail_catalog_proof(call):
        output = json.loads(call["input"][-1]["output"])
        proof = output["facilities"][0]["capabilities"]["문구"]
        assert proof["source"] == "catalog" and proof["source_id"] == "S1"
        assert any(source["id"] == proof["source_id"] and source["url"] == CATALOG_URL for source in output["sources"])
        return final("학생문구 이름이 공식 목록에 있어요. [S1]", kind="grounded", sources=["S1"], facilities=[1301])

    bundle = make_service([tool("facilities", {"categories": [], "location": None}),
        tool("facility_details", {"facility_ids": [1301]}, "call-2"), inspect_detail_catalog_proof])
    configure_discovery(bundle, [record], {})
    assert bundle.service.run("문구를 사야 해", [], DAY)["selected_facility_ids"] == ["1301"]


def test_identical_narrative_after_other_profile_field_changes_gets_new_valid_id_instead_of_deduping_to_revoked_id(make_service):
    record = {"id": 1401, "name": "새빛상회", "category": "인쇄/문구/서점/기념품", "location": "학생복지관 3층"}
    old_source = None
    bundle = None

    def change_hours_only(call):
        nonlocal old_source
        output = json.loads(call["input"][-1]["output"])
        source_id = output["facilities"][0]["capabilities"]["문구"]["source_id"]
        old_source = deepcopy(next(source for source in output["sources"] if source["id"] == source_id))
        bundle.agent.store.knowledge.return_value[0]["payload"]["restaurant_info"]["opening_hours"] = "평일 10:00~17:00"
        return tool("facility_details", {"facility_ids": [1401]}, "call-2")

    def use_revalidated_source(call):
        output = json.loads(call["input"][-1]["output"])
        source_id = output["facilities"][0]["capabilities"]["문구"]["source_id"]
        new_source = next(source for source in output["sources"] if source["id"] == source_id)
        assert new_source["text"] == old_source["text"]
        assert new_source["id"] != old_source["id"] and old_source["id"] in output["revoked_source_ids"]
        assert new_source["metadata"]["profile_version"] != old_source["metadata"]["profile_version"]
        assert new_source["id"] not in output["revoked_source_ids"]
        return final(f"새빛상회의 문구류 판매 안내를 재확인했어요. [{source_id}]", kind="grounded", sources=[source_id], facilities=[1401])

    bundle = make_service([tool("facilities", {"categories": [], "location": None}), change_hours_only, use_revalidated_source])
    _, profiles = configure_discovery(bundle, [record], {1401: {"description": "문구류 판매"}})
    profiles[0]["payload"]["restaurant_info"]["opening_hours"] = "평일 09:00~18:00"
    typed = details_result()
    typed["sources"][0].update(url=profiles[0]["url"], text="위치: 학생복지관 3층\n시간: 평일 10:00~17:00", metadata={"facility_id": 1401})
    typed.update(selected_facility_ids=["1401"], reference_facility_ids=["1401"])
    bundle.agent.school.answer.side_effect = None
    bundle.agent.school.answer.return_value = typed
    result = bundle.service.run("문구 살 곳을 다시 확인해 줘", [], DAY)
    assert result["mode"] == "agent" and result["sources"][0]["id"] != old_source["id"]


@pytest.mark.parametrize(("label", "original_phrase"), [("문구", "문구류 판매"), ("서점", "대학교재 판매"), ("기념품", "굿즈 판매")])
def test_long_description_cap_keeps_original_matching_phrase_in_model_evidence_and_cited_source(make_service, label, original_phrase):
    record = {"id": 1501, "name": "하늘상회", "category": "인쇄/문구/서점/기념품", "location": "학생복지관 3층"}

    def inspect_actual_excerpt(call):
        output = json.loads(call["input"][-1]["output"])
        candidate = output["facilities"][0]
        assert original_phrase not in candidate["description"] and candidate["description_truncated"] is True
        proof = candidate["capabilities"][label]
        excerpt = proof["evidence"][0]["text"]
        assert original_phrase in excerpt and len(excerpt) <= 1200 and "…" in excerpt
        assert proof["evidence_truncated"] is True
        source = next(source for source in output["sources"] if source["id"] == proof["source_id"])
        assert original_phrase in source["text"]
        return final(f"하늘상회의 {label} 안내가 확인돼요. [{source['id']}]", kind="grounded", sources=[source["id"]], facilities=[1501])

    bundle = make_service([tool("facilities", {"categories": [], "location": None}), inspect_actual_excerpt])
    configure_discovery(bundle, [record], {1501: {"description": "안내 " * 1500 + original_phrase}})
    assert bundle.service.run("필요한 물품을 살 곳을 찾아줘", [], DAY)["mode"] == "agent"


def empty_school_result():
    return {"answer": "이번 검색에서 요청한 공식 근거를 찾지 못했어요.", "mode": "insufficient", "model": None,
        "sources": [], "images": [], "trace": ["합성 학교 검색 근거 없음"],
        "date": "2026-11-20", "date_range": {"start": "2026-11-20", "end": "2026-11-20"}}


def test_empty_school_search_returns_to_model_and_can_recover_with_official_facility_descriptions(make_service):
    record = {"id": 1601, "name": "아람상회", "category": "인쇄/문구/서점/기념품", "location": "학생복지관 3층"}

    def recover_from_empty_search(call):
        output = json.loads(call["input"][-1]["output"])
        assert output["mode"] == "insufficient" and output["no_evidence"] is True and output["sources"] == []
        assert output["answer"] == empty_school_result()["answer"]
        assert "facilities" in output["note"]
        return tool("facilities", {"categories": ["인쇄/문구/서점/기념품"], "location": None}, "call-2")

    def answer_from_actual_discovery(call):
        output = json.loads(call["input"][-1]["output"])
        candidate = output["facilities"][0]
        assert {"기념품", "서점"} <= set(candidate["capabilities"])
        identity = candidate["capabilities"]["기념품"]["source_id"]
        assert next(source for source in output["sources"] if source["id"] == identity)["url"].endswith("id=1601")
        return final(f"아람상회에서 교재와 기념품 판매 안내가 확인돼요. [{identity}]", kind="grounded", sources=[identity], facilities=[1601])

    bundle = make_service([tool("school_query", {"query": "학교 기념품이나 교재는 어디서 살 수 있어?"}),
        recover_from_empty_search, answer_from_actual_discovery])
    configure_discovery(bundle, [record], {1601: {"description": "대학교재와 한양대기념품 판매"}})
    bundle.school_query.side_effect = None
    bundle.school_query.return_value = empty_school_result()
    result = bundle.service.run("학교 기념품이나 교재는 어디서 살 수 있어?", [], DAY)
    assert result["mode"] == "agent" and result["selected_facility_ids"] == ["1601"]
    assert result["agent_steps"] == [{"tool": "school_query", "status": "no_evidence"}, {"tool": "facilities", "status": "completed"}]
    assert len(bundle.responses.calls) == 3


def test_empty_search_for_truly_unavailable_information_can_finish_with_honest_unsupported_answer(make_service):
    def explain_limit(call):
        output = json.loads(call["input"][-1]["output"])
        assert output["no_evidence"] is True
        return final("개인 성적은 제가 조회할 수 없어요. 학교의 본인 인증 서비스를 확인해 주세요.", kind="unsupported")

    bundle = make_service([tool("school_query", {"query": "내 개인 성적을 조회해줘"}), explain_limit])
    bundle.school_query.side_effect = None
    bundle.school_query.return_value = empty_school_result()
    result = bundle.service.run("내 개인 성적을 조회해줘", [], DAY)
    assert result["mode"] == "insufficient" and result["sources"] == result["images"] == []
    assert "조회할 수 없어요" in result["answer"] and len(bundle.responses.calls) == 2
    bundle.agent.directory.answer.assert_not_called()
    bundle.agent.school.answer.assert_not_called()


@pytest.mark.parametrize(("mode", "has_sources"), [("retrieval", True), ("openai", True), ("insufficient", True), ("clarification", False), ("retrieval", False)])
def test_only_insufficient_without_sources_is_recoverable_other_school_results_remain_terminal(make_service, mode, has_sources):
    bundle = make_service([tool("school_query", {"query": "학교 정보"})])
    delegated = deepcopy(bundle.delegated)
    delegated["mode"] = mode
    if not has_sources:
        delegated.update(answer="어느 시설을 말하는지 확인해 주세요.", sources=[])
    bundle.school_query.side_effect = None
    bundle.school_query.return_value = delegated
    result = bundle.service.run("학교 정보", [], DAY)
    assert result == {**delegated, "agent_steps": [{"tool": "school_query", "status": "completed"}]}
    assert len(bundle.responses.calls) == 1


def test_empty_search_at_turn_cap_preserves_original_honest_result_and_its_requested_date(make_service):
    bundle = make_service([tool("school_query", {"query": "내일 조회할 수 없는 정보"})], turns=1)
    original = empty_school_result()
    saved = deepcopy(original)
    bundle.school_query.side_effect = None
    bundle.school_query.return_value = original
    result = bundle.service.run("내일 조회할 수 없는 정보", [], DAY)
    assert result["mode"] == "insufficient" and result["answer"] == original["answer"]
    assert result["sources"] == result["images"] == []
    assert result["date"] == "2026-11-20" and result["date_range"] == original["date_range"]
    assert "에이전트 자료 확인 횟수 제한" in result["trace"]
    assert result["agent_steps"] == [{"tool": "school_query", "status": "no_evidence"}]
    assert original == saved


def test_hallucinated_citation_after_empty_search_falls_back_to_no_evidence_not_fabricated_answer(make_service):
    bundle = make_service([tool("school_query", {"query": "없는 안내"}),
        final("상상한 매장 정보예요. [S999]", kind="grounded", sources=["S999"], facilities=[999])])
    bundle.school_query.side_effect = None
    bundle.school_query.return_value = empty_school_result()
    result = bundle.service.run("없는 안내", [], DAY)
    assert result["mode"] == "insufficient" and result["sources"] == []
    assert "상상한 매장" not in result["answer"]
    assert result["answer"] == empty_school_result()["answer"]

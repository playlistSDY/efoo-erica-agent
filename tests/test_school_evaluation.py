"""Offline evaluation harness tests; never touch real school data or APIs."""

from copy import deepcopy
from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
import sqlite3
from uuid import UUID

import pytest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "evaluate_school.py"
spec = importlib.util.spec_from_file_location("school_evaluation_harness", SCRIPT)
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)
URL = "https://life.hanyang.ac.kr/theme/pages/facilities/index.php"
FETCHED = datetime.now(timezone.utc).isoformat()
PHOTO = "/media/" + "a" * 64 + ".jpg"


@pytest.fixture
def public_database(tmp_path):
    path = tmp_path / "public.sqlite3"
    entries = [
        (1, "CU", "편의점", "학생복지관 1층"), (2, "CU", "편의점", "제3공학관 1층"),
        (3, "스타벅스", "카페/베이커리", "학생복지관 1층"), (4, "한양베이커리", "카페/베이커리", "제3공학관 1층"),
        (5, "한양서점", "인쇄/문구/서점/기념품", "학생복지관 2층"),
        (6, "미용실", "생활서비스", "학생복지관 1층"), (7, "이모네", "일반음식점", "제1공학관 1층"),
        (8, "학생식당", "구내식당", "학생복지관 2층"), (9, "자동판매기", "자동판매기", "생활관 1층"),
        (10, "33 Korean Snack", "일반음식점", "제3공학관 1층"), (11, "7-eleven", "편의점", "학생복지관 2층"),
        (12, "GS25", "편의점", "버스승강장 1층"), (13, "emart24", "편의점", "생활관 1층"),
        (14, "우편취급국", "생활서비스", "학생복지관 1층"),
    ]
    records = [{"id": identifier, "name": name, "category": category, "category_display": category,
        "location": location, "admin_id": "PUBLIC_ADMIN_FIELD_NOT_NEEDED"} for identifier, name, category, location in entries]
    with sqlite3.connect(path) as conn:
        conn.execute("CREATE TABLE knowledge_sources(scope TEXT,code TEXT,name TEXT,url TEXT,fetched_at TEXT,payload TEXT)")
        conn.execute("CREATE TABLE collections(code TEXT,day TEXT,name TEXT,url TEXT,fetched_at TEXT,meals TEXT)")
        conn.execute("CREATE TABLE chat_messages(content TEXT)")
        conn.execute("INSERT INTO chat_messages VALUES('PRIVATE_CHAT_MUST_NOT_BE_READ')")
        conn.execute("INSERT INTO knowledge_sources VALUES(?,?,?,?,?,?)", ("page:facility-directory", "campus", "전체 목록", URL, FETCHED, json.dumps({"records": records}, ensure_ascii=False)))
        for item in records:
            identifier = item["id"]
            cover = f"https://life.hanyang.ac.kr/uploads/cover-{identifier}.jpg"
            photo = "/media/" + f"{identifier:064x}" + ".jpg"
            info = {"name": item["name"], "category": item["category"], "location": item["location"],
                "opening_hours": "11:30~13:30, 17:00~20:00", "contact": f"031-400-{5000 + identifier}",
                "notices": [{"text": "분식과 김밥을 판매합니다."}] if identifier == 7 else [],
                "description": {"text": "떡볶이를 판매합니다." if identifier == 10 else item["name"] + " 안내"},
                "cover_image": {"url": cover}, "gallery": []}
            payload = {"restaurant_info": info, "assets": [{"url": cover, "downloaded": True, "local_path": photo}],
                "unused_private_field": "MUST_NOT_EXPORT"}
            conn.execute("INSERT INTO knowledge_sources VALUES(?,?,?,?,?,?)", (f"facility:{identifier}", f"fac_{identifier}", item["name"],
                f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={identifier}", FETCHED, json.dumps(payload, ensure_ascii=False)))
        for identifier in (1, 2):
            payload = {"title": f"이용 안내 {identifier}", "text": "11월 9일부터 식권 가격은 1,000원입니다."}
            conn.execute("INSERT INTO knowledge_sources VALUES(?,?,?,?,?,?)", (f"notice:{identifier}", "campus", payload["title"],
                f"https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id={identifier}", FETCHED, json.dumps(payload, ensure_ascii=False)))
        conn.execute("INSERT INTO collections VALUES(?,?,?,?,?,?)", ("re12", "2026-11-14", "학생식당",
            "https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=8&date=2026-11-14", FETCHED,
            json.dumps([{"meal_type": "중식", "items": ["칼국수"], "price": "6,500원"}], ensure_ascii=False)))
    return path


def simple_case(expect=None):
    return {"id": "unit.count", "category": "count", "question": "편의점 몇 개?",
        "steps": [{"question": "편의점 몇 개?", "expect": expect or {"modes": ["retrieval"], "directory_ids": [1, 2], "scope_label_required": True}}]}


def response(answer="공식 등록 목록에는 두 지점이 있어요. [S1]"):
    return {"answer": answer, "mode": "retrieval", "model": None, "images": [],
        "sources": [{"id": "S1", "url": URL, "text": "ID 1 CU / ID 2 CU", "images": [], "fetched_at": FETCHED}],
        "directory": {"count": 2, "records": [{"id": 1}, {"id": 2}], "scope": "official_portal"}}


def failed_names(result):
    return {check["name"] for check in result["checks"] if not check["passed"]}


def test_loader_reads_only_public_tables_in_read_only_mode_and_excludes_private_fields(public_database, monkeypatch):
    queries, original_connect = [], sqlite3.connect

    def tracked(*args, **kwargs):
        assert "mode=ro" in args[0] and kwargs["uri"] is True
        connection = original_connect(*args, **kwargs)
        connection.set_trace_callback(queries.append)
        return connection

    monkeypatch.setattr(evaluation.sqlite3, "connect", tracked)
    snapshot = evaluation.load_public_snapshot(public_database)
    serialized = json.dumps(snapshot, ensure_ascii=False)
    assert len(snapshot["catalog"]["records"]) == len(snapshot["profiles"]) == 14
    assert len(snapshot["notices"]) == 2 and len(snapshot["meals"]) == 1
    assert all("chat_messages" not in query and "conversations" not in query for query in queries)
    assert "PRIVATE_CHAT_MUST_NOT_BE_READ" not in serialized
    assert "PUBLIC_ADMIN_FIELD_NOT_NEEDED" not in serialized and "MUST_NOT_EXPORT" not in serialized
    assert all(query.lstrip().upper().startswith("SELECT") for query in queries)


def test_case_generation_is_deterministic_broad_and_uses_complete_ground_truth(public_database):
    snapshot = evaluation.load_public_snapshot(public_database)
    cases = evaluation.build_cases(snapshot)
    assert cases == evaluation.build_cases(deepcopy(snapshot))
    assert len(cases) >= 50 and len({case["id"] for case in cases}) == len(cases)
    assert {"count", "list", "categories", "location_filter", "brand", "business", "location", "hours", "contact",
        "photos", "notice", "multiintent", "followup", "scope_refusal", "language_variant", "meal"} <= {case["category"] for case in cases}
    assert next(case for case in cases if case["id"] == "catalog.total")["steps"][0]["expect"]["directory_ids"] == list(range(1, 15))
    subtype = next(case for case in cases if case["id"] == "business.0")
    assert subtype["steps"][0]["expect"]["directory_ids"] == [7, 10]
    assert any(case["id"] == "followup.category" for case in cases)
    assert any("ㄱ" in case["question"] for case in cases if case["category"] == "language_variant")
    json.dumps(cases, ensure_ascii=False)


def test_conflicting_catalog_identity_is_rejected_instead_of_building_false_ground_truth(public_database):
    with sqlite3.connect(public_database) as conn:
        raw = conn.execute("SELECT payload FROM knowledge_sources WHERE scope='page:facility-directory'").fetchone()[0]
        payload = json.loads(raw)
        payload["records"].append({**payload["records"][0], "location": "충돌 위치"})
        conn.execute("UPDATE knowledge_sources SET payload=? WHERE scope='page:facility-directory'", (json.dumps(payload),))
    with pytest.raises(evaluation.SnapshotError, match="Conflicting"):
        evaluation.load_public_snapshot(public_database)


def test_count_assertions_accept_paraphrases_but_reject_wrong_set_or_count():
    case = simple_case()
    assert evaluation.evaluate_response(case, response())["passed"]
    assert evaluation.evaluate_response(case, response("공개 카탈로그 기준 **2곳**으로 확인돼요! [S1]"))["passed"]
    wrong = response()
    wrong["directory"].update(count=3, records=[{"id": 1}, {"id": 2}, {"id": 3}])
    failed = failed_names(evaluation.evaluate_response(case, wrong))
    assert {"exact_facility_set", "exact_count"} <= failed


def test_provenance_rejects_fabricated_duplicate_or_uncited_sources():
    invented = response("공식 자료 [S99]")
    assert "no_fabricated_citations" in failed_names(evaluation.evaluate_response(simple_case(), invented))
    duplicate = response()
    duplicate["sources"] *= 2
    assert "unique_valid_source_ids" in failed_names(evaluation.evaluate_response(simple_case(), duplicate))
    uncited = response()
    uncited["sources"].append({"id": "S2", "url": URL, "images": []})
    assert "returned_sources_are_cited" in failed_names(evaluation.evaluate_response(simple_case(), uncited))


def test_images_must_belong_to_cited_official_source_and_expected_facility():
    value = response()
    value["sources"][0]["images"] = [{"url": PHOTO}]
    value["images"] = [{"url": PHOTO, "source_url": URL}]
    case = simple_case({"images_required": True, "allowed_images": [PHOTO]})
    assert evaluation.evaluate_response(case, value)["passed"]
    invalid = deepcopy(value)
    invalid["images"][0]["source_url"] = "https://life.hanyang.ac.kr/unrelated"
    assert "images_only_from_cited_sources" in failed_names(evaluation.evaluate_response(case, invalid))
    invalid["images"][0]["url"] = "https://elsewhere.example/image.jpg"
    assert "images_only_from_cited_sources" in failed_names(evaluation.evaluate_response(case, invalid))


def test_fact_assertions_compare_numbers_clocks_and_location_aliases_not_exact_sentences():
    case = simple_case({"facts": [{"kind": "location", "value": "학생복지관 2층"},
        {"kind": "phone", "value": "0314001234"}, {"kind": "clocks", "values": ["11:30", "13:30"]},
        {"kind": "money_any", "values": [6500]}]})
    value = response("복지관 2F예요. 전화는 031-400-1234이고 오전 11시 30분부터 오후 1시 30분까지예요. 가격은 6,500원이에요. [S1]")
    assert evaluation.evaluate_response(case, value)["passed"]
    value["answer"] = "복지관 1층, 031-400-9999, 12:00~13:00, 7,000원 [S1]"
    assert {"fact_0_location", "fact_1_phone", "fact_2_clocks", "fact_3_money_any"} <= failed_names(evaluation.evaluate_response(case, value))


def test_specific_entity_question_rejects_another_facility_source():
    case = simple_case({"source_facility_ids": [1], "allowed_facility_ids": [1]})
    value = response()
    value["sources"][0]["url"] = "https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id=2"
    failed = failed_names(evaluation.evaluate_response(case, value))
    assert {"required_facility_sources", "no_other_facility_sources"} <= failed


def test_shared_official_notice_is_valid_for_each_related_facility_but_not_an_unrelated_one():
    case = simple_case({"source_facility_ids": [1], "allowed_facility_ids": [1], "facility_codes": {"reA": 1, "reB": 2}})
    value = response()
    value["sources"][0].update(url="https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id=20",
        restaurant_code="campus", metadata={"related_codes": ["reA", "reB"]})
    assert evaluation.evaluate_response(case, value)["passed"]
    value["sources"][0]["metadata"]["related_codes"] = ["reB"]
    assert {"required_facility_sources", "no_other_facility_sources"} <= failed_names(evaluation.evaluate_response(case, value))


def test_refusal_requires_explicit_mode_and_empty_sources_images():
    case = simple_case({"must_refuse": True, "require_sources": False})
    value = {"answer": "공식 근거가 없어 확인하기 어려워요.", "mode": "insufficient", "sources": [], "images": []}
    assert evaluation.evaluate_response(case, value)["passed"]
    value["mode"] = "openai"
    assert "grounded_refusal" in failed_names(evaluation.evaluate_response(case, value))


@pytest.mark.parametrize("malformed", [None, [], {"answer": "실패", "sources": [{"id": []}], "images": [{"url": 3}], "directory": "bad"}])
def test_malformed_response_is_a_failed_result_not_a_harness_crash(malformed):
    assert evaluation.evaluate_response(simple_case(), malformed)["passed"] is False


def test_live_runner_uses_new_qa_conversation_and_authoritative_followup_ids_without_history():
    requests = []
    case = simple_case()
    case["steps"].append({"question": "그럼 목록은?", "expect": case["steps"][0]["expect"]})

    class FakeClient:
        def __init__(self, base_url, *, timeout):
            assert base_url == "http://127.0.0.1:8000" and timeout == 2

        def request(self, path, payload=None):
            requests.append((path, payload))
            if path == "/healthz":
                return 200, {"status": "ok"}
            if path == "/api/conversations":
                return 200, {"id": "owned-qa-conversation"}
            return 200, response()

    results = evaluation.run_live([case], "http://127.0.0.1:8000", timeout=2, client_factory=FakeClient)
    assert results[0]["passed"]
    assert [path for path, _ in requests] == ["/healthz", "/api/conversations", "/api/chat", "/api/chat"]
    turns = [payload for path, payload in requests if path == "/api/chat"]
    assert all(set(turn) == {"message", "conversation_id", "turn_id"} for turn in turns)
    assert all(turn["conversation_id"] == "owned-qa-conversation" for turn in turns)
    assert len({UUID(turn["turn_id"]) for turn in turns}) == 2


def test_default_cli_only_prepares_cases_and_never_enables_network(public_database, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "run_live", lambda *args, **kwargs: pytest.fail("Default mode must stay offline"))
    output = tmp_path / "prepared.json"
    assert evaluation.main(["--database", str(public_database), "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    assert report["mode"] == "prepared" and report["summary"]["evaluated"] == 0
    assert report["summary"]["planned"] >= 50 and report["results"] == []


def test_captured_report_replays_offline_without_database_and_has_useful_failure_exit(tmp_path, monkeypatch):
    case = simple_case()
    results = evaluation.evaluate_capture([case], {case["id"]: response()})
    report = evaluation.make_report([case], results, mode="live")
    capture, replay = tmp_path / "capture.json", tmp_path / "replay.json"
    capture.write_text(json.dumps(report))
    monkeypatch.setattr(evaluation, "load_public_snapshot", lambda *args: pytest.fail("A complete capture needs no database"))
    monkeypatch.setattr(evaluation, "run_live", lambda *args, **kwargs: pytest.fail("Offline replay needs no API"))
    assert evaluation.main(["--responses", str(capture), "--output", str(replay)]) == 0
    assert json.loads(replay.read_text())["summary"] == report["summary"]
    report["results"][0]["steps"][0]["actual"]["directory"]["count"] = 999
    capture.write_text(json.dumps(report))
    assert evaluation.main(["--responses", str(capture), "--output", str(replay)]) == 1
    failed = json.loads(replay.read_text())
    assert failed["summary"]["failed"] == 1
    assert "exact_count" in failed_names(failed["results"][0]["steps"][0])


def complete_snapshot(public_database):
    """Expand only synthetic public records; no live DB or source is opened."""
    snapshot = evaluation.load_public_snapshot(public_database)
    template = snapshot["profiles"]["14"]
    for identifier in range(15, 49):
        record = {"id": identifier, "name": f"테스트시설{identifier}", "category": "생활서비스",
            "location": f"테스트관 {identifier}층", "description": "공개 시설", "notice": ""}
        snapshot["catalog"]["records"].append(record)
        snapshot["profiles"][str(identifier)] = {**deepcopy(template), **record, "code": f"fac_{identifier}",
            "url": f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={identifier}",
            "contact": f"031-400-{5000 + identifier}", "images": ["/media/" + f"{identifier:064x}" + ".jpg"]}
    snapshot["profiles"]["11"]["hours"] = ""
    snapshot["profiles"]["12"]["contact"] = "-"
    snapshot["profiles"]["13"]["images"] = []
    snapshot["profiles"]["1"]["hours"] = "평일 08:00~22:00, 토요일 09:00~18:00. 일요일 휴무"
    snapshot["profiles"]["2"]["hours"] = "평일 08:00~22:00, 토요일 휴무"
    snapshot["profiles"]["3"]["hours"] = "학기 중 평일 09:00~18:00 (휴식시간 12:00~13:00), 방학 주말 휴무"
    snapshot["profiles"]["4"]["hours"] = "유인 08:00~22:00, 무인 22:00~08:00"
    snapshot["profiles"]["5"]["hours"] = "24시간"
    snapshot["profiles"]["6"]["contact"] = "031-400-5006 / 010-1234-5678"
    snapshot["notices"][0]["restaurant_codes"] = ["fac_8"]
    return snapshot


def test_all_facilities_is_opt_in_preserves_baseline_and_covers_all_public_and_missing_slots(public_database):
    snapshot = complete_snapshot(public_database)
    baseline = evaluation.build_cases(snapshot)
    extended = evaluation.build_cases(snapshot, all_facilities=True)
    assert extended[:len(baseline)] == baseline
    assert 120 <= len(extended) <= 200
    assert len({case["id"] for case in extended}) == len(extended)
    core = [case for case in extended if case["category"] == "facility_core"]
    assert len(core) == 48
    for case in core:
        coverage = case["facility_coverage"]
        assert set(coverage["available"]) | set(coverage["missing"]) == {"location", "hours", "contact", "images"}
        assert not set(coverage["available"]) & set(coverage["missing"])
        expected = case["steps"][0]["expect"]
        assert expected["source_facility_ids"] == expected["allowed_facility_ids"] == [coverage["id"]]
    assert "hours" in next(case for case in core if case["id"] == "facility.11.core")["facility_coverage"]["missing"]
    assert "contact" in next(case for case in core if case["id"] == "facility.12.core")["facility_coverage"]["missing"]
    assert "images" in next(case for case in core if case["id"] == "facility.13.core")["facility_coverage"]["missing"]
    categories = {case["category"] for case in extended}
    assert {"context_boundary", "two_facilities", "hours_boundary", "current_status", "missing_information", "entity_boundary", "restaurant_notice"} <= categories
    assert extended == evaluation.build_cases(deepcopy(snapshot), all_facilities=True)


def test_every_published_phone_is_independent_and_concatenated_suffix_cannot_pass(public_database):
    cases = evaluation.build_cases(complete_snapshot(public_database), all_facilities=True)
    expected = next(case for case in cases if case["id"] == "facility.6.core")["steps"][0]["expect"]
    phones = [fact for fact in expected["facts"] if fact["kind"] == "phone"]
    assert phones == [{"kind": "phone", "value": "01012345678"}, {"kind": "phone", "value": "0314005006"}]
    case = simple_case({"facts": phones})
    value = response("031-400-5006 / 010-1234-5678 [S1]")
    assert evaluation.evaluate_response(case, value)["passed"]
    value["answer"] = "010-1234-5678 [S1]"
    assert "fact_1_phone" in failed_names(evaluation.evaluate_response(case, value))
    concatenated = simple_case({"facts": [{"kind": "phone", "value": "031400500601012345678"}]})
    assert "fact_0_phone" in failed_names(evaluation.evaluate_response(concatenated, value))


def test_hours_conditions_are_semantic_markers_not_exact_sentences(public_database):
    cases = evaluation.build_cases(complete_snapshot(public_database), all_facilities=True)
    facts = next(case for case in cases if case["id"] == "facility.3.core")["steps"][0]["expect"]["facts"]
    markers = next(fact for fact in facts if fact["kind"] == "schedule_markers")
    assert set(markers["values"]) == {"semester", "weekdays", "break", "vacation", "weekend", "closed"}
    case = simple_case({"facts": [markers]})
    paraphrase = response("학기에는 월~금에 운영하고 방학 중에는 주말에 쉬어요(휴무). 브레이크 타임도 있어요. [S1]")
    assert evaluation.evaluate_response(case, paraphrase)["passed"]
    paraphrase["answer"] = "평일에 열어요. [S1]"
    assert "fact_0_schedule_markers" in failed_names(evaluation.evaluate_response(case, paraphrase))


@pytest.mark.parametrize("range_text", [
    "월~금", r"월\~금", "월–금", r"월\–금", "월&#126;금", "월　〜　금", "월－금", r"월\-금",
])
def test_schedule_markers_normalize_only_markdown_escaping_and_range_typography(range_text):
    schedule = "학기중 " + range_text + " 09:00~18:00, 방학 주말 휴무 (휴식시간 12:00~13:00)"
    expected = {"semester", "weekdays", "vacation", "weekend", "closed", "break"}
    assert evaluation.schedule_markers(schedule) == expected
    case = simple_case({"facts": [{"kind": "schedule_markers", "values": sorted(expected)}]})
    assert evaluation.evaluate_response(case, response(schedule + " [S1]"))["passed"]


def test_schedule_marker_normalization_does_not_invent_omitted_or_invalid_weekday_conditions():
    assert "weekdays" not in evaluation.schedule_markers("학기중 월금 09:00~18:00")
    assert "weekdays" not in evaluation.schedule_markers(r"학기중 월\금 09:00~18:00")
    expected = {"facts": [{"kind": "schedule_markers", "values": ["weekdays", "vacation"]}]}
    actual = response(r"학기중 월\~금 09:00~18:00 [S1]")
    assert "fact_0_schedule_markers" in failed_names(evaluation.evaluate_response(simple_case(expected), actual))


@pytest.mark.parametrize(("hours", "expected"), [
    ("평일 09:00~18:00", {"kind": "unknown"}),
    ("09:00~18:00", {"kind": "unknown"}),
    ("평일 09:00~18:00\n토요일 10:00~15:00", {"kind": "clocks", "values": ["10:00", "15:00"]}),
    ("평일 09:00~18:00 / 토요일 휴무", {"kind": "closed"}),
    ("평일 09:00~18:00, 토요일 10:00~15:00. 일요일 휴무", {"kind": "clocks", "values": ["10:00", "15:00"]}),
    ("토요일·일요일 10:00~15:00", {"kind": "clocks", "values": ["10:00", "15:00"]}),
    ("매일 24시간", {"kind": "hours_24"}),
])
def test_saturday_oracle_never_guesses_from_weekday_or_unqualified_hours(hours, expected):
    assert evaluation.saturday_oracle(hours) == expected


def test_unknown_and_live_claim_checks_reject_invented_facts_and_guarantees():
    unknown = simple_case({"must_acknowledge_unknown": True, "no_phone_claims": True, "require_sources": False})
    value = {"answer": "공식 연락처가 공개되어 있지 않아요.", "mode": "retrieval", "sources": [], "images": []}
    assert evaluation.evaluate_response(unknown, value)["passed"]
    value["answer"] = "전화번호는 031-400-1234예요."
    assert {"uncertainty_disclosed", "no_invented_phone"} <= failed_names(evaluation.evaluate_response(unknown, value))
    live = simple_case({"must_disclaim_live": True})
    value = response("실시간 영업 여부는 확인할 수 없어요. 공식 시간표만 참고해주세요. [S1]")
    assert evaluation.evaluate_response(live, value)["passed"]
    value["answer"] += " 현재 영업 중입니다."
    assert "no_unverified_current_open_claim" in failed_names(evaluation.evaluate_response(live, value))


def test_unknown_branch_field_requires_own_source_and_no_cross_branch_substitution():
    case = simple_case({"unknown_fields": [{"id": 11, "slot": "hours"}], "source_facility_ids": [11], "allowed_facility_ids": [11]})
    value = response("공식 영업시간이 공개되어 있지 않아요. [S1]")
    value["sources"][0].update(url="https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id=11",
        text="세븐일레븐\n공식 영업시간: 공식 상세 안내에 공개 정보 없음")
    assert evaluation.evaluate_response(case, value)["passed"]
    value["sources"][0]["url"] = "https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id=12"
    assert {"required_facility_sources", "no_other_facility_sources", "facility_11_unknown_hours"} <= failed_names(evaluation.evaluate_response(case, value))


def test_all_facility_cli_prepares_extended_cases_and_reports_slot_coverage(public_database, tmp_path, monkeypatch):
    monkeypatch.setattr(evaluation, "run_live", lambda *args, **kwargs: pytest.fail("Preparing must be offline"))
    snapshot = complete_snapshot(public_database)
    monkeypatch.setattr(evaluation, "load_public_snapshot", lambda *args: deepcopy(snapshot))
    output = tmp_path / "all.json"
    assert evaluation.main(["--database", str(public_database), "--all-facilities", "--output", str(output)]) == 0
    report = json.loads(output.read_text())
    coverage = report["summary"]["facility_coverage"]
    assert coverage["planned_ids"] == list(range(1, 49)) and coverage["evaluated_ids"] == []
    assert coverage["available_slots"]["location"] == 48
    assert coverage["missing_slots"] == {"hours": 1, "contact": 1, "images": 1}
    assert len(report["cases"]) > len(evaluation.build_cases(snapshot))


def test_two_entity_and_ambiguous_followup_have_distinct_strict_expectations(public_database):
    cases = evaluation.build_cases(complete_snapshot(public_database), all_facilities=True)
    pair = next(case for case in cases if case["id"] == "multi.named.0")
    expected = pair["steps"][0]["expect"]
    assert len(expected["source_facility_ids"]) == len(expected["facility_facts"]) == 2
    followup = next(case for case in cases if case["id"] == "context.ambiguous.0")
    assert followup["steps"][0]["expect"] == expected
    ambiguous = followup["steps"][1]["expect"]
    assert set(ambiguous["modes"]) == {"clarification", "insufficient"} and ambiguous["no_phone_claims"]
    value = {"answer": "두 시설 중 어느 매장을 말씀하시는지 알려주세요.", "mode": "clarification", "sources": [], "images": []}
    assert evaluation.evaluate_response(followup, value, step_index=1)["passed"]


def test_report_marks_evaluated_facility_ids_without_claiming_unrun_coverage(public_database):
    cases = evaluation.build_cases(complete_snapshot(public_database), all_facilities=True)
    core = next(case for case in cases if case["id"] == "facility.1.core")
    results = evaluation.evaluate_capture([core], {})
    report = evaluation.make_report(cases, results)
    assert report["summary"]["facility_coverage"]["evaluated_ids"] == [1]
    assert report["summary"]["failed"] == 1


def test_unregistered_scope_oracles_are_not_created_when_named_facility_is_public(public_database):
    snapshot = complete_snapshot(public_database)
    snapshot["catalog"]["records"][0]["notice"] = "약국 안내"
    snapshot["profiles"]["2"]["description"] = "CUBIC 매장 안내"
    cases = evaluation.build_cases(snapshot, all_facilities=True)
    assert not {"missing.pharmacy", "scope.cubic"} & {case["id"] for case in cases}


def test_source_facts_are_bound_to_each_named_facility_not_only_found_somewhere_in_answer():
    expected = {"source_facility_ids": [1, 2], "allowed_facility_ids": [1, 2],
        "facility_facts": [{"id": 1, "facts": [{"kind": "phone", "value": "0314005001"}]},
            {"id": 2, "facts": [{"kind": "phone", "value": "0314005002"}]}]}
    value = response("A 031-400-5001 [S1], B 031-400-5002 [S2]")
    value["sources"] = [{"id": f"S{identifier}", "url": f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={identifier}",
        "text": f"전화: 031-400-{5000 + identifier}", "images": []} for identifier in (1, 2)]
    assert evaluation.evaluate_response(simple_case(expected), value)["passed"]
    value["sources"][1]["text"] = "전화: 031-400-5001"
    assert "facility_2_source_fact_0_phone" in failed_names(evaluation.evaluate_response(simple_case(expected), value))


@pytest.mark.parametrize("claim", ["공식 등록 기준 총 2곳입니다.", "공개 카탈로그에 두 지점이 있어요.",
    "전체는 **2곳**이에요.", "총2개 시설이에요."])
def test_aggregate_body_count_is_independent_of_response_metadata_and_accepts_natural_forms(claim):
    case = simple_case({"directory_ids": [1, 2]})
    assert evaluation.evaluate_response(case, response(claim + " [S1]"))["passed"]
    mutant = response("공식 등록 목록에는 4곳이 있어요. [S1]")
    assert mutant["directory"]["count"] == 2
    assert "body_aggregate_count" in failed_names(evaluation.evaluate_response(case, mutant))


def test_aggregate_body_uses_main_total_instead_of_matching_a_supporting_category_or_metadata():
    case = simple_case()
    mutant = response("공식 전체 시설은 4곳이에요. [S1]\n\n| 분류 | 수 |\n| --- | --- |\n| 편의점 | 2곳 |")
    assert "body_aggregate_count" in failed_names(evaluation.evaluate_response(case, mutant))
    missing = response("공식 목록을 정리했어요. [S1]\n\n| 매장 | 위치 |\n| --- | --- |\n| CU | 2층 |")
    assert "body_aggregate_count" in failed_names(evaluation.evaluate_response(case, missing))
    assert evaluation._count_claims("공개 시설에는 이곳과 저곳이 포함돼요.") == []


def test_category_body_count_mutant_fails_even_with_correct_category_metadata():
    expected = {"directory_ids": [1, 2], "directory_categories": {"편의점": 1, "카페/베이커리": 1}}
    value = response("공식 전체 등록 시설은 2곳이에요. [S1]\n\n| 분류 | 등록 수 |\n| --- | --- |\n| 편의점 | 1곳 |\n| 카페/베이커리 | 1곳 |")
    value["directory"]["categories"] = expected["directory_categories"]
    assert evaluation.evaluate_response(simple_case(expected), value)["passed"]
    value["answer"] = value["answer"].replace("| 편의점 | 1곳 |", "| 편의점 | 2곳 |")
    assert "body_category_counts" in failed_names(evaluation.evaluate_response(simple_case(expected), value))


@pytest.mark.parametrize("layout", ["table", "sentences", "comma_citations"])
def test_two_facility_answer_phone_swap_is_rejected_while_sources_and_global_facts_stay_correct(layout):
    facts = [{"kind": "phone", "value": "0314005001"}, {"kind": "phone", "value": "0314005002"}]
    expected = {"source_facility_ids": [1, 2], "allowed_facility_ids": [1, 2], "facts": facts,
        "facility_facts": [{"id": 1, "facts": [facts[0]]}, {"id": 2, "facts": [facts[1]]}]}
    if layout == "table":
        answer = "| 매장 | 전화 | 출처 |\n| --- | --- | --- |\n| A | 031-400-5001 | [S1] |\n| B | 031-400-5002 | [S2] |"
    elif layout == "sentences":
        answer = "A의 전화는 031-400-5001이에요. [S1] B의 연락처는 031-400-5002예요. [S2]"
    else:
        answer = "A 전화 031-400-5001 [S1], B 전화 031-400-5002 [S2]"
    value = response(answer)
    value["sources"] = [{"id": f"S{identifier}", "url": f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={identifier}",
        "text": f"전화: 031-400-{5000 + identifier}", "images": []} for identifier in (1, 2)]
    assert evaluation.evaluate_response(simple_case(expected), value)["passed"]
    value["answer"] = answer.replace("031-400-5001", "SWAP").replace("031-400-5002", "031-400-5001").replace("SWAP", "031-400-5002")
    failed = failed_names(evaluation.evaluate_response(simple_case(expected), value))
    assert {"facility_1_answer_fact_0_phone", "facility_2_answer_fact_0_phone"} <= failed
    assert not {"fact_0_phone", "fact_1_phone", "facility_1_source_fact_0_phone", "facility_2_source_fact_0_phone"} & failed


def test_two_facility_answer_without_unambiguous_value_association_fails_closed():
    expected = {"facility_facts": [{"id": 1, "facts": [{"kind": "phone", "value": "0314005001"}]},
        {"id": 2, "facts": [{"kind": "phone", "value": "0314005002"}]}]}
    value = response("A와 B 전화는 031-400-5001, 031-400-5002예요. [S1][S2]")
    value["sources"] = [{"id": f"S{identifier}", "url": f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={identifier}",
        "text": f"전화: 031-400-{5000 + identifier}", "images": []} for identifier in (1, 2)]
    assert {"facility_1_answer_fact_0_phone", "facility_2_answer_fact_0_phone"} <= failed_names(evaluation.evaluate_response(simple_case(expected), value))


def test_unknown_named_facility_cases_are_added_only_when_absent_from_public_oracle(public_database):
    snapshot = complete_snapshot(public_database)
    cases = evaluation.build_cases(snapshot, all_facilities=True)
    ids = {case["id"] for case in cases}
    assert {"scope.unknown.우리은행", "scope.unknown.무지개서점"} <= ids
    snapshot["catalog"]["records"][0]["name"] = "우리은행"
    snapshot["profiles"]["1"]["name"] = "우리은행"
    assert "scope.unknown.우리은행" not in {case["id"] for case in evaluation.build_cases(snapshot, all_facilities=True)}


def test_mixed_notice_cases_require_both_notice_url_and_individual_facility_facts(public_database):
    snapshot = complete_snapshot(public_database)
    cases = evaluation.build_cases(snapshot, all_facilities=True)
    mixed = [case for case in cases if case["category"] == "mixed_notice"]
    assert len(mixed) == 2
    single, pair = [case["steps"][0]["expect"] for case in mixed]
    assert single["required_source_urls"] == pair["required_source_urls"] == [snapshot["notices"][0]["url"]]
    assert single["source_facility_ids"] == [8]
    assert 8 in pair["source_facility_ids"] and len(pair["source_facility_ids"]) == 2
    assert len(single["facility_facts"]) == 1 and len(pair["facility_facts"]) == 2
    for group in pair["facility_facts"]:
        assert {fact["kind"] for fact in group["facts"]} == {"location", "phone"}
    snapshot["notices"][0]["text"] = ""
    assert not [case for case in evaluation.build_cases(snapshot, all_facilities=True) if case["category"] == "mixed_notice"]


def test_mixed_notice_cases_include_missing_contact_and_allow_other_facility_actual_phone(public_database):
    snapshot = complete_snapshot(public_database)
    snapshot["profiles"]["8"]["contact"] = ""
    mixed = [case for case in evaluation.build_cases(snapshot, all_facilities=True) if case["category"] == "mixed_notice"]
    assert len(mixed) == 2
    single, pair = [case["steps"][0]["expect"] for case in mixed]
    missing = {"id": 8, "slot": "contact", "name": "학생식당", "scoped": True, "snapshot_missing": True}
    assert single["unknown_fields"] == pair["unknown_fields"] == [missing]
    assert {fact["kind"] for fact in single["facts"]} == {"location"}
    assert not pair.get("no_phone_claims")
    assert any(fact["kind"] == "phone" for fact in pair["facts"])
    assert {fact["kind"] for group in pair["facility_facts"] if group["id"] == 8 for fact in group["facts"]} == {"location"}


def test_scoped_snapshot_unknown_contact_does_not_reject_another_facility_phone_but_rejects_borrowed_phone():
    case = simple_case({"source_facility_ids": [1, 2], "allowed_facility_ids": [1, 2],
        "unknown_fields": [{"id": 1, "slot": "contact", "name": "학생식당", "scoped": True, "snapshot_missing": True}]})
    value = response("학생식당 위치는 학생복지관 2층이에요. [S1]\n- **학생식당 연락처**: 최신 공식 문자 안내에서 확인하지 못했어요.\n- **다른카페 연락처**: 031-400-5002 [S2]")
    value["sources"] = [{"id": "S1", "url": "https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id=1",
        "text": "학생식당\n위치: 학생복지관 2층", "images": []},
        {"id": "S2", "url": "https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id=2",
        "text": "다른카페\n연락처: 031-400-5002", "images": []}]
    assert evaluation.evaluate_response(case, value)["passed"]
    invented = deepcopy(value)
    invented["answer"] = invented["answer"].replace("**학생식당 연락처**: 최신 공식 문자 안내에서 확인하지 못했어요.",
        "**학생식당 연락처**: 031-400-5002이며 공식 안내 여부는 확인하지 못했어요.")
    assert "facility_1_unknown_contact" in failed_names(evaluation.evaluate_response(case, invented))
    wrong_scope = deepcopy(value)
    wrong_scope["answer"] = wrong_scope["answer"].replace("**학생식당 연락처**", "**다른카페 연락처**")
    assert "facility_1_unknown_contact" in failed_names(evaluation.evaluate_response(case, wrong_scope))

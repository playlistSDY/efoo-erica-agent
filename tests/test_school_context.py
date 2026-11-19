"""Facility-reference regressions over synthetic public catalog metadata."""

from copy import deepcopy

import pytest

from app.school_context import match_explicit_facilities, resolve_school_context


CATALOG = [
    {"id": 1, "name": "우편취급국", "aliases": ["우체국"], "location": "학생복지관 1층 102호"},
    {"id": 10, "name": "세븐일레븐", "aliases": ["세븐"], "location": "학생복지관 1층"},
    {"id": 11, "name": "세븐일레븐", "aliases": ["세븐"], "location": "제3공학관 1층"},
    {"id": 12, "name": "세븐일레븐", "aliases": ["세븐"], "location": "창의인재원 1층"},
    {"id": 20, "name": "이모네", "location": "학생복지관 2층"},
    {"id": 21, "name": "분식마을", "location": "제1공학관 1층"},
    {"id": 30, "name": "카페알파", "location": "학생복지관 2층"},
    {"id": 31, "name": "카페베타", "location": "제3공학관 1층"},
    {"id": 32, "name": "카페감마", "location": "창의인재원 2층"},
]


def result(ids, reference=None):
    answer = {"selected_facility_ids": ids}
    if reference is not None:
        answer["reference_facility_ids"] = reference
    return answer


def resolve(message, ids=None, *, previous=None, history=(), catalog=CATALOG):
    return resolve_school_context(message, history, catalog=catalog,
        previous_result=previous if previous is not None else result(ids) if ids is not None else None)


@pytest.mark.parametrize("message", ["거기 전화번호는?", "그럼 연락처 알려줘", "몇 시까지 해?", "사진도 보여줘"])
def test_single_facility_followup_uses_verified_identity(message):
    context = resolve(message, [1])
    assert context.facility_ids == ("1",)
    assert context.is_followup and context.clarification is None


def test_named_alias_resolves_without_mutating_original_history_or_catalog():
    history = [{"role": "user", "content": "우체국 위치 알려줘"}, {"role": "assistant", "content": "위치 안내"}]
    original = deepcopy((history, CATALOG))
    context = resolve("거기 전화번호는?", history=history)
    assert context.facility_ids == ("1",)
    assert (history, CATALOG) == original


def test_branch_location_refines_previous_brand_collection_in_original_order():
    context = resolve("학생복지관점은 몇 시까지 해?", [12, 10, 11])
    assert context.facility_ids == ("10",) and context.selection == "filter"
    assert context.previous_facility_ids == ("12", "10", "11")


def test_named_member_of_previous_two_restaurants_can_request_photos():
    context = resolve("이모네 사진 보여줘", [21, 20])
    assert context.facility_ids == ("20",) and context.is_followup


def test_floor_subset_keeps_every_matching_facility_in_previous_order():
    context = resolve("그중 2층만", [32, 31, 30])
    assert context.facility_ids == ("32", "30")
    assert context.selection == "filter" and not context.clarification


def test_deictic_reference_to_multiple_facilities_requires_specific_clarification():
    context = resolve("거기 전화번호", [12, 10, 11])
    assert context.facility_ids == () and context.candidate_ids == ("12", "10", "11")
    assert "창의인재원" in context.clarification and "학생복지관" in context.clarification


@pytest.mark.parametrize("message", ["카페 목록 알려줘", "학교 주소가 뭐야", "학교 연혁 소개해줘", "그럼 학교 시설은 몇 개야", "편의점 모두 보여줘"])
def test_unrelated_new_topic_never_inherits_an_old_facility(message):
    context = resolve(message, [1])
    assert context.facility_ids == () and context.previous_facility_ids == ()
    assert not context.is_followup and context.clarification is None


@pytest.mark.parametrize("message,expected", [("첫 번째 사진", ("32",)),
    ("두번째는?", ("30",)), ("3번째 연락처", ("31",)),
    ("세번째랑 첫번째", ("32", "31"))])
def test_ordinal_selections_preserve_collection_order(message, expected):
    assert resolve(message, [32, 30, 31]).facility_ids == expected


def test_remaining_uses_structured_reference_list_and_last_selection():
    context = resolve("나머지 사진도", previous=result([30], [32, 30, 31]))
    assert context.facility_ids == ("32", "31") and context.selection == "remaining"


def test_legacy_first_selection_chain_recovers_prior_ordered_collection():
    history = [{"role": "user", "content": "카페 목록"},
        {"role": "assistant", "content": "카페 셋", "selected_facility_ids": [32, 30, 31]},
        {"role": "user", "content": "첫번째 사진"},
        {"role": "assistant", "content": "첫번째 카페", "selected_facility_ids": [32]}]
    assert resolve("나머지는?", history=history).facility_ids == ("30", "31")


@pytest.mark.parametrize("message", ["둘 다 사진", "두 곳 모두 전화번호", "그중 모두"])
def test_explicit_plural_reference_resolves_the_whole_known_pair(message):
    context = resolve(message, [21, 20])
    assert context.facility_ids == ("21", "20") and context.clarification is None


def test_both_does_not_silently_choose_two_from_three():
    context = resolve("둘 다", [10, 11, 12])
    assert not context.facility_ids and context.clarification


def test_duplicate_explicit_brand_names_require_branch_disambiguation():
    context = resolve("세븐일레븐 운영시간")
    assert not context.facility_ids and context.candidate_ids == ("10", "11", "12")
    assert "같은 이름" in context.clarification
    assert resolve("학생복지관 세븐일레븐 운영시간").facility_ids == ("10",)


def test_multiple_explicit_unique_names_are_selected_without_arbitrary_singling():
    context = resolve("이모네랑 우체국 전화번호")
    assert context.facility_ids == ("20", "1") and not context.clarification


def test_longer_complete_branch_name_supersedes_short_brand_matches():
    catalog = [*CATALOG, {"id": 13, "name": "세븐일레븐 학생회관점", "aliases": ["세븐"], "location": "학생회관 1층"}]
    assert resolve("세븐일레븐 학생회관점 사진", catalog=catalog).facility_ids == ("13",)


def test_missing_floor_and_out_of_range_ordinal_produce_useful_clarifications():
    missing = resolve("그중 5층만", [30, 31, 32])
    assert not missing.facility_ids and "5층" not in "".join(missing.facility_ids)
    assert "이전 목록" in missing.clarification
    ordinal = resolve("네 번째", [30, 31, 32])
    assert not ordinal.facility_ids and "3곳" in ordinal.clarification


def test_latest_explicit_empty_selection_resets_older_facility_context():
    history = [{"role": "assistant", "selected_facility_ids": [1]},
        {"role": "user", "content": "학교 연혁 알려줘"},
        {"role": "assistant", "selected_facility_ids": [], "content": "확인할 수 없어요"}]
    context = resolve("거기 연락처", history=history)
    assert not context.facility_ids and context.clarification


def test_latest_unstructured_new_topic_answer_does_not_resurrect_older_metadata():
    history = [{"role": "assistant", "selected_facility_ids": [1]},
        {"role": "user", "content": "학교 연혁 알려줘"},
        {"role": "assistant", "content": "확인할 수 없어요"}]
    assert not resolve("거기 전화번호", history=history).facility_ids


def test_directory_and_nested_payload_history_formats_preserve_ids_as_strings():
    previous = {"directory": {"records": [{"id": 32}, {"id": 30}, {"id": 31}]}}
    assert resolve("첫번째", previous=previous).facility_ids == ("32",)
    history = [{"role": "assistant", "payload": {"school_context": result([1])}}]
    assert resolve("그곳 위치", history=history).facility_ids == ("1",)


def test_unknown_and_boolean_metadata_ids_are_not_selected_from_known_catalog():
    context = resolve("거기 전화번호", [True, 999, 1, "1"])
    assert context.facility_ids == ("1",)


def test_no_catalog_still_supports_explicit_verified_metadata_ordinals():
    context = resolve("두번째", ["facility-b", "facility-a"], catalog=())
    assert context.facility_ids == ("facility-a",)


def test_day_or_week_queries_do_not_change_or_inherit_facility_selection():
    assert not resolve("다음 주 월요일 학식 알려줘", [1]).facility_ids


def test_original_catalog_knowledge_wrapper_is_accepted():
    catalog = {"payload": {"records": CATALOG}}
    assert resolve("우체국 전화번호", catalog=catalog).facility_ids == ("1",)


@pytest.mark.parametrize("message", ["CUBIC 사진", "SCU 운영시간", "CUBIC 카페 위치"])
def test_latin_alias_does_not_match_inside_an_unrelated_latin_word(message):
    catalog = [{"id": 40, "name": "씨유", "aliases": ["CU"], "location": "학생복지관 1층"}]
    assert not resolve(message, catalog=catalog).facility_ids


def test_latin_alias_still_matches_a_standalone_name_next_to_korean_text():
    catalog = [{"id": 40, "name": "씨유", "aliases": ["CU"], "location": "학생복지관 1층"},
        {"id": 41, "name": "CUBIC", "location": "제3공학관 B1"}]
    assert resolve("CU 영업시간", catalog=catalog).facility_ids == ("40",)
    assert resolve("cu사진", catalog=catalog).facility_ids == ("40",)
    assert resolve("CUBIC 사진", catalog=catalog).facility_ids == ("41",)


@pytest.mark.parametrize("message", ["그중 지하 1층만", "그중 B1만", "그중 b1층만"])
def test_bare_basement_catalog_location_matches_basement_followup(message):
    catalog = [{"id": 40, "name": "카페지하", "location": "제3공학관 B1"},
        {"id": 41, "name": "카페지상", "location": "제3공학관 1층"}]
    context = resolve(message, [41, 40], catalog=catalog)
    assert context.facility_ids == ("40",) and context.selection == "filter"


def test_bare_basement_suffix_does_not_hide_the_building_name_from_branch_selection():
    catalog = [{"id": 40, "name": "카페지하", "location": "제3공학관 B1"},
        {"id": 41, "name": "카페지상", "location": "학생복지관 1층"}]
    assert resolve("제3공학관점 사진", [41, 40], catalog=catalog).facility_ids == ("40",)


@pytest.mark.parametrize("message,expected", [
    ("교직원식당하고 창업보육센터식당의 위치와 전화번호 각각", ("2", "3")),
    ("창업보육센터식당과 교직원식당의 위치와 전화번호", ("3", "2")),
    ("교식하고 창보 위치와 전화번호", ("2", "3")),
])
def test_building_inside_a_named_facility_does_not_filter_another_named_facility(message, expected):
    catalog = [{"id": 2, "name": "교직원식당", "aliases": ["교식"], "location": "학생복지관 3층"},
        {"id": 3, "name": "창업보육센터식당", "aliases": ["창보"], "location": "창업보육센터 1층"}]
    context = resolve(message, catalog=catalog)
    assert context.facility_ids == expected and context.clarification is None
    assert match_explicit_facilities(message, catalog=catalog) == expected
    followup = resolve("거기 전화번호는?", list(expected), catalog=catalog)
    assert followup.facility_ids == () and followup.candidate_ids == expected
    assert "교직원식당" in followup.clarification and "창업보육센터식당" in followup.clarification


def test_separate_branch_location_is_retained_after_masking_a_brand_name():
    context = resolve("세븐일레븐 학생복지관점 운영시간", catalog=CATALOG)
    assert context.facility_ids == ("10",) and context.selection == "filter"
    assert match_explicit_facilities("세븐일레븐 학생복지관점 운영시간", catalog=CATALOG) == ("10", "11", "12")


def test_floor_inside_proper_name_does_not_become_a_separate_floor_filter():
    catalog = [{"id": 60, "name": "2층카페", "location": "학생복지관 2층"},
        {"id": 61, "name": "알파카페", "location": "제3공학관 1층"}]
    assert resolve("2층카페와 알파카페 위치", catalog=catalog).facility_ids == ("60", "61")
    assert resolve("2층카페와 알파카페 중 1층만", catalog=catalog).facility_ids == ("61",)


def test_public_explicit_matcher_reuses_latin_boundaries_and_longest_branch_name():
    catalog = [{"id": 40, "name": "씨유", "aliases": ["CU"], "location": "학생복지관 1층"},
        {"id": 41, "name": "CUBIC", "location": "제3공학관 B1"},
        {"id": 42, "name": "세븐일레븐", "location": "학생복지관 1층"},
        {"id": 43, "name": "세븐일레븐 제3공학관점", "aliases": ["세븐일레븐"], "location": "제3공학관 1층"}]
    assert match_explicit_facilities("CUBIC과 CU 위치", catalog=catalog) == ("41", "40")
    assert match_explicit_facilities("세븐일레븐 제3공학관점 사진", catalog=catalog) == ("43",)
    assert match_explicit_facilities("CU 사진", catalog={"payload": {"records": catalog}}) == ("40",)


@pytest.mark.parametrize("message", ["우리은행 전화번호", "국민은행 위치", "무지개서점 전화번호", "무지개책방 운영시간"])
def test_generic_type_alias_never_matches_inside_an_unknown_proper_name(message):
    catalog = [{"id": 50, "name": "신한은행", "aliases": ["은행"], "location": "학생복지관 1층"},
        {"id": 51, "name": "한양서적", "aliases": ["서점", "책방"], "location": "학생복지관 2층"}]
    assert match_explicit_facilities(message, catalog=catalog) == ()
    context = resolve(message, [50], catalog=catalog)
    assert context.facility_ids == ()


@pytest.mark.parametrize("message,expected", [("은행 위치", ("50",)), ("서점 전화", ("51",)),
    ("학교 은행 위치", ("50",)), ("교내 서점 연락처", ("51",)),
    ("학생복지관 은행 전화번호", ("50",)), ("은행이랑 서점 위치", ("50", "51")),
    ("신한은행과 한양서적 전화번호", ("50", "51"))])
def test_generic_query_boundaries_and_full_official_names_still_resolve(message, expected):
    catalog = [{"id": 50, "name": "신한은행", "aliases": ["은행"], "location": "학생복지관 1층"},
        {"id": 51, "name": "한양서적", "aliases": ["서점", "책방"], "location": "학생복지관 2층"}]
    assert match_explicit_facilities(message, catalog=catalog) == expected
    assert resolve(message, catalog=catalog).facility_ids == expected

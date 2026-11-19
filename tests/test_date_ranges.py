from datetime import date, datetime
import unicodedata

import pytest

from app.date_ranges import DateRange, DateRangeError, resolve_date_range


BASE = date(2026, 11, 13)  # Friday in the caller's KST calendar.


def expected(text, start, end=None, *, base=BASE, max_days=31):
    value = resolve_date_range(text, base, max_days=max_days)
    assert isinstance(value, DateRange)
    assert value.start.isoformat() == start
    assert value.end.isoformat() == (end or start)
    assert value.explicit is True
    return value


@pytest.mark.parametrize("text,day", [
    ("오늘 학식", "2026-11-13"), ("내일학식알려줘", "2026-11-14"),
    ("모레 점심", "2026-11-15"), ("어제 메뉴", "2026-11-12"), ("글피 학식", "2026-11-16"),
    ("2026-11-16 메뉴", "2026-11-16"), ("2026/11/16", "2026-11-16"),
    ("2026.11.16.", "2026-11-16"), ("11월 16일 메뉴", "2026-11-16"),
    ("2027년 1월 2일", "2027-01-02"), ("2024년 2월 29일", "2024-02-29"),
])
def test_single_explicit_calendar_or_relative_days(text, day):
    value = expected(text, day)
    assert value.is_single_day and len(value) == 1


@pytest.mark.parametrize("text,start,end", [
    ("이번주 학식", "2026-11-09", "2026-11-15"),
    ("다음 주 학식", "2026-11-16", "2026-11-22"),
    ("다다음주", "2026-11-23", "2026-11-29"),
    ("지난주 메뉴", "2026-11-02", "2026-11-08"),
    ("2주 후 메뉴", "2026-11-23", "2026-11-29"),
    ("2주 뒤", "2026-11-23", "2026-11-29"),
    ("3주 전", "2026-10-19", "2026-10-25"),
    ("이번달", "2026-11-01", "2026-11-30"),
    ("다음 달 메뉴", "2026-12-01", "2026-12-31"),
    ("주말 학식", "2026-11-14", "2026-11-15"),
    ("평일 학식", "2026-11-09", "2026-11-13"),
    ("다음 주말", "2026-11-21", "2026-11-22"),
    ("다음주 주말", "2026-11-21", "2026-11-22"),
    ("다음 주 평일", "2026-11-16", "2026-11-20"),
])
def test_calendar_week_month_and_week_parts(text, start, end):
    expected(text, start, end)


@pytest.mark.parametrize("text,start,end", [
    ("다음주 월요일부터 수요일까지 학식", "2026-11-16", "2026-11-18"),
    ("다음주 화요일~금요일", "2026-11-17", "2026-11-20"),
    ("이번주 금요일부터 다음주 월요일까지", "2026-11-13", "2026-11-16"),
    ("이번주부터 다음주까지", "2026-11-09", "2026-11-22"),
    ("월요일부터 수요일까지", "2026-11-16", "2026-11-18"),
    ("2026-11-16~2026-11-20 메뉴", "2026-11-16", "2026-11-20"),
    ("2026/11/16 - 2026/11/20", "2026-11-16", "2026-11-20"),
    ("2026.11.16—2026.11.20", "2026-11-16", "2026-11-20"),
    ("11월16~20일 학식", "2026-11-16", "2026-11-20"),
    ("11월 16일부터 20일까지", "2026-11-16", "2026-11-20"),
    ("2026년11월16일~20일", "2026-11-16", "2026-11-20"),
    ("2026-11-16~11-20", "2026-11-16", "2026-11-20"),
    ("오늘~모레", "2026-11-13", "2026-11-15"),
    ("내일부터 글피까지", "2026-11-14", "2026-11-16"),
])
def test_inclusive_ranges_and_shared_week_month_year(text, start, end):
    expected(text, start, end)


@pytest.mark.parametrize("text,start,end", [
    ("12월30일~1월2일", "2026-12-30", "2027-01-02"),
    ("2026년12월30일부터2027년1월2일까지", "2026-12-30", "2027-01-02"),
    ("2026-12-30~01-02", "2026-12-30", "2027-01-02"),
    ("2024/2/29~2024/3/1", "2024-02-29", "2024-03-01"),
])
def test_year_rollover_and_leap_day_ranges(text, start, end):
    expected(text, start, end)


@pytest.mark.parametrize("text,start,end", [
    ("오늘부터3일", "2026-11-13", "2026-11-15"),
    ("내일부터 3일간", "2026-11-14", "2026-11-16"),
    ("앞으로7일 학식", "2026-11-13", "2026-11-19"),
    ("향후 1일", "2026-11-13", "2026-11-13"),
    ("2026-11-16부터3일간", "2026-11-16", "2026-11-18"),
])
def test_day_counts_include_start_day(text, start, end):
    expected(text, start, end)


def test_bare_weekday_policy_is_next_occurrence_including_today():
    expected("금요일 메뉴", "2026-11-13")
    expected("월요일 메뉴", "2026-11-16")
    expected("일요일 메뉴", "2026-11-15")
    expected("이번주 월요일 메뉴", "2026-11-09")
    expected("지난주 월요일", "2026-11-02")


def test_relative_week_and_month_work_across_year_boundary():
    base = date(2026, 12, 31)
    expected("내일", "2027-01-01", base=base)
    expected("오늘부터3일", "2026-12-31", "2027-01-02", base=base)
    expected("다음주", "2027-01-04", "2027-01-10", base=base)
    expected("다음달", "2027-01-01", "2027-01-31", base=base)


@pytest.mark.parametrize("text,start,end", [
    ("저번주에는 뭐나왔어?", "2026-11-02", "2026-11-08"),
    ("저번 주 학식", "2026-11-02", "2026-11-08"),
    ("저번달 학식", "2026-10-01", "2026-10-31"),
    ("금주 식단", "2026-11-09", "2026-11-15"),
    ("차주학식", "2026-11-16", "2026-11-22"),
    ("전주에는 어떤 메뉴?", "2026-11-02", "2026-11-08"),
    ("차주 월요일", "2026-11-16", "2026-11-16"),
    ("금일 점심", "2026-11-13", "2026-11-13"),
    ("익일학식", "2026-11-14", "2026-11-14"),
    ("명일 메뉴", "2026-11-14", "2026-11-14"),
    ("작일은?", "2026-11-12", "2026-11-12"),
])
def test_colloquial_and_formal_date_aliases_are_explicit_dates(text, start, end):
    expected(text, start, end)


@pytest.mark.parametrize("text", ["전주비빔밥", "전주대학교", "명일방주", "금일봉", "익일배송", "차주차량"])
def test_date_aliases_do_not_split_attached_ordinary_names(text):
    assert resolve_date_range(text, BASE) is None


@pytest.mark.parametrize("text,start,end", [
    ("11/16~11/20 학식", "2026-11-16", "2026-11-20"),
    ("11.16~11.20", "2026-11-16", "2026-11-20"),
    ("11/16부터11/20까지", "2026-11-16", "2026-11-20"),
    ("11.16 메뉴", "2026-11-16", "2026-11-16"),
    ("11/16", "2026-11-16", "2026-11-16"),
    ("12/30~1/2", "2026-12-30", "2027-01-02"),
    ("12.30~1.2", "2026-12-30", "2027-01-02"),
    ("2026-11-16~11/20", "2026-11-16", "2026-11-20"),
    ("11.16~20일", "2026-11-16", "2026-11-20"),
])
def test_yearless_slash_and_dot_calendar_dates(text, start, end):
    expected(text, start, end)


@pytest.mark.parametrize("text,day", [
    ("3일 후 학식", "2026-11-16"), ("3일 뒤", "2026-11-16"),
    ("3일 전 메뉴", "2026-11-10"), ("0일 후", "2026-11-13"),
    ("100일 뒤 학식", "2027-02-21"),
])
def test_day_offsets_are_single_dates_instead_of_durations(text, day):
    value = expected(text, day)
    assert value.is_single_day


@pytest.mark.parametrize("text", [
    "평점 6.5점", "메뉴가 6.5점이야", "평균은 6.5도야", "가격 6.5만원",
    "프린터 버전11.16", "version 11.16", "분수 11/16", "11/16을 계산해줘", "11/16 + 2",
    "평균은6.5야", "별점10/10", "확률이11/16", "버전은11.16",
])
def test_decimal_units_versions_and_fractions_are_not_calendar_dates(text):
    assert resolve_date_range(text, BASE) is None


@pytest.mark.parametrize("text", ["13/16 학식", "11.31 메뉴", "2/29 메뉴", "11/20~11/16", "-3일 후", "1.2.3 메뉴"])
def test_invalid_numeric_month_day_expressions_never_fallback_to_today(text):
    with pytest.raises(DateRangeError):
        resolve_date_range(text, BASE)


def test_numeric_dates_and_day_offsets_obey_year_boundaries_and_leap_year():
    expected("2/29 메뉴", "2024-02-29", base=date(2024, 2, 1))
    expected("2일 후", "2027-01-02", base=date(2026, 12, 31))
    expected("2일 전", "2025-12-30", base=date(2026, 1, 1))
    expected("오늘부터3일", "2026-11-13", "2026-11-15")


def test_range_api_is_inclusive_and_immutable():
    value = expected("11월16~18일", "2026-11-16", "2026-11-18")
    assert len(value) == 3 and not value.is_single_day
    assert value.days() == (date(2026, 11, 16), date(2026, 11, 17), date(2026, 11, 18))
    assert value.iso_dates() == ["2026-11-16", "2026-11-17", "2026-11-18"]
    with pytest.raises(AttributeError):
        value.start = BASE


def test_nfd_and_compatibility_jamo_are_normalized_without_mutating_input():
    text = "ㄷㅏㅇㅡㅁㅈㅜ ㅎㅏㄱㅅㅣㄱ"
    expected(text, "2026-11-16", "2026-11-22")
    assert text == "ㄷㅏㅇㅡㅁㅈㅜ ㅎㅏㄱㅅㅣㄱ"
    expected(unicodedata.normalize("NFD", "내일"), "2026-11-14")


@pytest.mark.parametrize("text", [
    "학교 소개해줘", "카페는 언제 문 열어?", "오래 기다려야 해?", "연락처 알려줘",
    "천원의 아침밥 결제 안내", "11시 운영해?", "", "😀", "가격 6000원 이하",
])
def test_non_date_language_returns_none_instead_of_assuming_today(text):
    assert resolve_date_range(text, BASE) is None


@pytest.mark.parametrize("text", [
    "2026-02-29", "2025년2월29일", "2026-13-01", "2026-11-00", "0000-01-01",
    "2026-11", "2026년11월", "17일 메뉴", "2026-11/16",
    "2026-11-20~2026-11-16", "11월20~16일", "내일부터 어제까지",
    "2026년12월30일~2026년1월2일", "다음주 금요일부터 월요일까지",
    "오늘 아니면 내일", "11월16일이랑17일", "오늘~내일 그리고 모레~글피",
    "오늘부터", "2026-11-16 ~", "다음주 화욜", "다음주 수",
    "앞으로0일", "향후-3일", "-2주후", "엊그제", "다음달 평일",
    "다음달 초 학식", "이번달 말 메뉴", "다음주 초", "이번주 중",
])
def test_invalid_reversed_partial_and_ambiguous_dates_raise_a_clarification(text):
    with pytest.raises(DateRangeError) as problem:
        resolve_date_range(text, BASE)
    assert str(problem.value)


def test_oversize_periods_are_rejected_without_clipping_and_limit_is_configurable():
    expected("11월16~20일", "2026-11-16", "2026-11-20", max_days=5)
    with pytest.raises(DateRangeError, match="최대 4일"):
        resolve_date_range("11월16~20일", BASE, max_days=4)
    with pytest.raises(DateRangeError, match="최대 6일"):
        resolve_date_range("다음주", BASE, max_days=6)
    with pytest.raises(DateRangeError):
        resolve_date_range("2026-01-01~12-31", BASE)
    expected("2026-01-01~12-31", "2026-01-01", "2026-12-31", max_days=365)


def test_out_of_calendar_and_invalid_configuration_do_not_fallback():
    with pytest.raises(DateRangeError):
        resolve_date_range("앞으로999999999일", BASE)
    with pytest.raises(DateRangeError):
        resolve_date_range("내일", date.max)
    for maximum in (0, -1, True, "31"):
        with pytest.raises(DateRangeError):
            resolve_date_range("내일", BASE, max_days=maximum)
    with pytest.raises(DateRangeError):
        resolve_date_range("내일", datetime(2026, 11, 13))

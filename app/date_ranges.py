"""Deterministic inclusive Korean date ranges with no implicit-today fallback.

The caller supplies a KST calendar date. Weeks run Monday through Sunday.
A bare weekday means its next occurrence, including today. Bare 주말/평일
mean the current week's Saturday–Sunday/Monday–Friday. N주 후/전 selects
that whole shifted calendar week. 앞으로/향후 N일 includes today.
"""

from __future__ import annotations

import calendar
from dataclasses import dataclass
from datetime import date, datetime, timedelta
import re

from app.text_normalization import normalize_korean_input


class DateRangeError(ValueError):
    """A recognizable date expression needs correction or clarification."""


@dataclass(frozen=True)
class DateRange:
    start: date
    end: date
    explicit: bool = True

    def __post_init__(self):
        if any(not isinstance(value, date) or isinstance(value, datetime) for value in (self.start, self.end)):
            raise DateRangeError("날짜는 연월일이 있는 달력 날짜로 알려 주세요.")
        if self.end < self.start:
            raise DateRangeError("끝 날짜가 시작 날짜보다 빨라요. 조회할 기간을 다시 알려 주세요.")

    def __len__(self) -> int:
        return (self.end - self.start).days + 1

    @property
    def is_single_day(self) -> bool:
        return self.start == self.end

    def days(self) -> tuple[date, ...]:
        return tuple(self.start + timedelta(days=offset) for offset in range(len(self)))

    def iso_dates(self) -> list[str]:
        return [day.isoformat() for day in self.days()]


_WEEK = r"(?:(?:다다음|이번|다음|지난)\s*주|\d+\s*주\s*(?:후|뒤|전))"
_WEEKDAY = r"[월화수목금토일]\s*요일"
_RELATIVE = r"(?:글피|모레|내일|어제|오늘)"
_ISO = r"(?<!\d)\d{4}\s*[-/.]\s*\d{1,2}\s*[-/.]\s*\d{1,2}(?!\d)"
_KOREAN = r"(?<!\d)(?:\d{4}\s*년\s*)?\d{1,2}\s*월\s*\d{1,2}\s*일?(?!\d)"
_MONTH_DAY = r"(?<![A-Za-z0-9./])\d{1,2}\s*[/.]\s*\d{1,2}(?!\d|[/.]\s*\d)"
_DAY_OFFSET = r"(?<!\d)\d+\s*일\s*(?:후|뒤|전)"
_MONTH = r"(?:다다음|이번|다음|지난)\s*달"
_WEEK_PART = rf"(?:{_WEEK}\s*)?(?:주말|평일)"
_NATURAL_WEEKEND = r"(?:다다음|이번|다음|지난)\s*주말"
_TERM = rf"(?:{_ISO}|{_KOREAN}|{_MONTH_DAY}|{_DAY_OFFSET}|{_NATURAL_WEEKEND}|{_WEEK}\s*{_WEEKDAY}|{_WEEK_PART}|{_WEEKDAY}|{_RELATIVE}|{_MONTH}|{_WEEK})"
_SHORT_END = r"(?:\d{1,2}\s*[-/.]\s*\d{1,2}\s*일?|\d{1,2}\s*일)"
_CONNECTOR = r"(?:부터|[~〜～–—-])"
_RANGE = re.compile(rf"(?P<start>{_TERM})\s*{_CONNECTOR}\s*(?P<end>{_TERM}|{_SHORT_END})\s*(?:까지)?")
_TERMS = re.compile(_TERM)
_DATE_HINT = re.compile(
    r"\d{4}\s*[-/.년]|\d+\s*(?:월|일|주\s*(?:후|뒤|전))|"
    r"(?:이번|다음|다다음|지난)\s*(?:주|달)|요일|주말|평일|"
    r"엊그제|그제|다음날|일주일\s*(?:후|전)|앞으로\s*\d|향후\s*\d|"
    r"(?<![\d.])\d{1,2}\s*[/.]\s*\d{1,2}"
)
_NON_DATE_NUMBERS = re.compile(
    r"\d+(?:[/.]\d+)?\s*(?:학점|평점|점|도|만원|원|배|퍼센트|%|시간|시|분|초|kg|cm|km)"
    r"|(?:버전|버젼|version|분수|비율|확률|점수|평점|별점|평균|온도|계산)(?:은|는|이|가)?\s*[:=]?\s*\d{1,2}\s*[/.]\s*\d{1,2}"
    r"|\d{1,2}\s*[/.]\s*\d{1,2}(?:을|를|의)?\s*(?:계산|분수|비율|나누)"
    r"|\d{1,2}\s*[/.]\s*\d{1,2}\s*[+*=]\s*\d+",
    re.IGNORECASE,
)
_DAY_NUMBERS = {"월": 0, "화": 1, "수": 2, "목": 3, "금": 4, "토": 5, "일": 6}
_RELATIVE_OFFSETS = {"오늘": 0, "내일": 1, "모레": 2, "글피": 3, "어제": -1}
_OFFSETS = {"이번": 0, "다음": 1, "다다음": 2, "지난": -1}
_DATE_ALIASES = {"금주": "이번주", "차주": "다음주", "전주": "지난주",
                 "금일": "오늘", "익일": "내일", "명일": "내일", "작일": "어제"}
_ALIAS_TOKEN = re.compile(
    r"(?<![A-Za-z0-9가-힣])(?:금주|차주|전주|금일|익일|명일|작일)"
    r"(?=$|[\s,.;:!?？/()\[\]{}~〜～–—-]|[은는에도의와과랑]|부터|까지|"
    r"학식|식당|식단|메뉴|아침|점심|저녁|조식|중식|석식|[월화수목금토일]\s*요일)"
)


@dataclass(frozen=True)
class _Term:
    start: date
    end: date
    week: date | None = None
    kind: str = "day"


def _calendar_date(year, month, day):
    try:
        return date(int(year), int(month), int(day))
    except (ValueError, OverflowError) as exc:
        raise DateRangeError("달력에 없는 날짜예요. 연도와 월, 일을 다시 확인해 주세요.") from exc


def _shift(day, offset):
    try:
        return day + timedelta(days=offset)
    except (ValueError, OverflowError) as exc:
        raise DateRangeError("조회할 날짜가 달력 범위를 벗어났어요. 가까운 날짜로 다시 알려 주세요.") from exc


def _week_start(base, offset=0):
    return _shift(base, -base.weekday() + offset * 7)


def _week_offset(text):
    compact = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"(이번|다음|다다음|지난)주", compact)
    if match:
        return _OFFSETS[match[1]]
    match = re.fullmatch(r"(\d+)주(후|뒤|전)", compact)
    if match:
        return int(match[1]) * (-1 if match[2] == "전" else 1)
    raise DateRangeError("어느 주인지 확인하기 어려워요. 이번 주나 다음 주처럼 알려 주세요.")


def _month_range(base, offset):
    serial = base.year * 12 + base.month - 1 + offset
    year, zero_month = divmod(serial, 12)
    first = _calendar_date(year, zero_month + 1, 1)
    return _Term(first, _calendar_date(year, zero_month + 1, calendar.monthrange(year, zero_month + 1)[1]), kind="month")


def _parse_term(text, base, *, inherited=None):
    token = re.sub(r"\s+", "", text)
    match = re.fullmatch(r"(\d{4})([-/.])(\d{1,2})\2(\d{1,2})", token)
    if match:
        day = _calendar_date(match[1], match[3], match[4])
        return _Term(day, day)
    match = re.fullmatch(r"(?:(\d{4})년)?(\d{1,2})월(\d{1,2})일?", token)
    if match:
        year = int(match[1]) if match[1] else inherited.start.year if inherited else base.year
        month = int(match[2])
        # Only an explicitly named lower month may cross into the next year.
        # A lower day in the same shared month is a reversed range, not a cue
        # to invent a new month or year.
        if inherited and not match[1] and month < inherited.start.month:
            year += 1
        day = _calendar_date(year, month, match[3])
        return _Term(day, day)
    match = re.fullmatch(r"(\d{1,2})[/.](\d{1,2})일?", token)
    if match:
        month = int(match[1])
        year = inherited.start.year if inherited else base.year
        if inherited and month < inherited.start.month:
            year += 1
        day = _calendar_date(year, month, match[2])
        return _Term(day, day)
    if inherited:
        match = re.fullmatch(r"(\d{1,2})([-/.])(\d{1,2})일?", token)
        if match:
            month = int(match[1])
            year = inherited.start.year + (month < inherited.start.month)
            day = _calendar_date(year, month, match[3])
            return _Term(day, day)
        match = re.fullmatch(r"(\d{1,2})일", token)
        if match:
            day = _calendar_date(inherited.start.year, inherited.start.month, match[1])
            return _Term(day, day)
    if token in _RELATIVE_OFFSETS:
        day = _shift(base, _RELATIVE_OFFSETS[token])
        return _Term(day, day)
    match = re.fullmatch(r"(\d+)일(후|뒤|전)", token)
    if match:
        day = _shift(base, int(match[1]) * (-1 if match[2] == "전" else 1))
        return _Term(day, day)
    match = re.fullmatch(r"(이번|다음|다다음|지난)달", token)
    if match:
        return _month_range(base, _OFFSETS[match[1]])
    match = re.fullmatch(rf"(?P<week>{_WEEK})?(?P<day>{_WEEKDAY})", token)
    if match:
        weekday = _DAY_NUMBERS[match["day"][0]]
        if match["week"]:
            week = _week_start(base, _week_offset(match["week"]))
            day = _shift(week, weekday)
        elif inherited:
            week = inherited.week or _week_start(inherited.start)
            day = _shift(week, weekday)
        else:
            day = _shift(base, (weekday - base.weekday()) % 7)
            week = _week_start(day)
        return _Term(day, day, week=week)
    match = re.fullmatch(r"(이번|다음|다다음|지난)주말", token)
    if match:
        week = _week_start(base, _OFFSETS[match[1]])
        return _Term(_shift(week, 5), _shift(week, 6), week=week, kind="week_part")
    match = re.fullmatch(rf"(?P<week>{_WEEK})?(?P<part>주말|평일)", token)
    if match:
        week = _week_start(base, _week_offset(match["week"])) if match["week"] else (inherited.week or _week_start(inherited.start)) if inherited else _week_start(base)
        start, end = (5, 6) if match["part"] == "주말" else (0, 4)
        return _Term(_shift(week, start), _shift(week, end), week=week, kind="week_part")
    if re.fullmatch(_WEEK, token):
        week = _week_start(base, _week_offset(token))
        return _Term(week, _shift(week, 6), week=week, kind="week")
    raise DateRangeError("날짜를 정확히 확인하기 어려워요. 11월 16일처럼 월과 일을 함께 알려 주세요.")


def _finish(start, end, max_days):
    result = DateRange(start, end)
    if len(result) > max_days:
        raise DateRangeError(f"한 번에 최대 {max_days}일까지 확인할 수 있어요. 조회할 기간을 조금 줄여 주세요.")
    return result


def _check_residual(text, start, end):
    residual = text[:start] + " " + text[end:]
    # A question about closing/opening time is not an incomplete calendar
    # range: e.g. '토요일에 몇 시까지 해?' has one valid calendar day.
    residual = re.sub(r"(?:몇\s*시|언제|\d{1,2}\s*:\s*\d{2})\s*(?:부터|까지)","",residual)
    if _TERMS.search(residual) or _DATE_HINT.search(residual):
        raise DateRangeError("날짜 표현이 여러 개라 조회할 기간이 불명확해요. 시작일과 끝 날짜를 ~로 연결해 주세요.")
    if re.search(r"(?:부터|까지|[~〜～])", residual):
        raise DateRangeError("기간의 시작일이나 끝 날짜를 확인하기 어려워요. 두 날짜를 함께 알려 주세요.")


def resolve_date_range(text: str, base: date, *, max_days: int = 31) -> DateRange | None:
    """Resolve one explicit inclusive date/range, or return None if absent.

    M/D and M.D use the base year; clear score/price/fraction/version contexts
    are excluded. N일 후/전/뒤 is one offset date, not an N-day duration.
    Shared-month endpoints stay in that month; a named lower end month rolls
    to the next year only when no end year is supplied. Conflicting or invalid
    expressions raise instead of silently choosing today or clipping a range.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    if not isinstance(base, date) or isinstance(base, datetime):
        raise DateRangeError("기준 날짜는 한국 시간의 달력 날짜여야 해요.")
    if not isinstance(max_days, int) or isinstance(max_days, bool) or max_days < 1:
        raise DateRangeError("최대 조회 일수는 1일 이상이어야 해요.")
    text = normalize_korean_input(text)
    text = re.sub(r"저번(?=\s*(?:주|달))", "지난", text)
    text = _ALIAS_TOKEN.sub(lambda match: _DATE_ALIASES[match.group()], text)
    # Mask only unmistakably numeric non-date spans before looking for M.D.
    # The original message is retained by the caller; this is a search copy.
    text = _NON_DATE_NUMBERS.sub(lambda match: " " * len(match.group()), text)
    if re.search(r"-\s*\d+\s*(?:주|일)\s*(?:후|전|뒤)", text):
        raise DateRangeError("며칠 또는 몇 주 후인지 양수와 후·전으로 알려 주세요.")

    future = re.search(r"(?:앞으로|향후)\s*(\d+)\s*일(?:간|동안)?", text)
    duration = re.search(rf"(?P<start>{_TERM})\s*부터\s*(?P<count>\d+)\s*일(?P<suffix>간|동안)?(?!\s*(?:까지|후|뒤|전))", text)
    if duration and duration["suffix"] is None and not re.fullmatch(_RELATIVE, duration["start"].strip()):
        duration = None
    if future or duration:
        match = future or duration
        _check_residual(text, match.start(), match.end())
        count = int(match[1]) if future else int(match["count"])
        if count < 1:
            raise DateRangeError("조회할 기간은 1일 이상으로 알려 주세요.")
        start = base if future else _parse_term(match["start"], base).start
        return _finish(start, _shift(start, count - 1), max_days)

    ranges = list(_RANGE.finditer(text))
    if ranges:
        if len(ranges) != 1:
            raise DateRangeError("조회할 기간을 하나만 알려 주세요. 시작일과 끝 날짜를 함께 적어 주세요.")
        match = ranges[0]
        _check_residual(text, match.start(), match.end())
        first = _parse_term(match["start"], base)
        last = _parse_term(match["end"], base, inherited=first)
        return _finish(first.start, last.end, max_days)

    terms = list(_TERMS.finditer(text))
    if not terms:
        if _DATE_HINT.search(text):
            raise DateRangeError("날짜를 정확히 확인하기 어려워요. 연월일이나 다음 주처럼 알려 주세요.")
        return None
    if len(terms) != 1:
        raise DateRangeError("날짜 표현이 여러 개예요. 조회할 시작일과 끝 날짜를 ~로 연결해 주세요.")
    match = terms[0]
    _check_residual(text, match.start(), match.end())
    term = _parse_term(match.group(), base)
    if term.kind in {"month", "week"} and re.match(
        r"\s*(?:초순|중순|하순|말일|초|말|중)(?=$|\s|에|부터|까지|학식|메뉴|점심|저녁)", text[match.end():]
    ):
        raise DateRangeError("초나 말처럼 대략적인 기간은 날짜를 확정하기 어려워요. 시작일과 끝 날짜를 알려 주세요.")
    if term.kind == "month" and re.search(r"주말|평일", text):
        raise DateRangeError("한 달의 주말이나 평일은 연속된 기간이 아니에요. 특정 주나 날짜 범위를 알려 주세요.")
    if term.kind == "week" and re.match(r"\s*[월화수목금토일](?:욜|\s|$|부터|까지|~)", text[match.end():]):
        raise DateRangeError("요일이 불명확해요. 월요일처럼 요일을 끝까지 알려 주세요.")
    return _finish(term.start, term.end, max_days)

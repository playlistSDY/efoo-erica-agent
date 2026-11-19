"""Deterministic facility references over public catalog and result metadata.

No model, storage or network access is used. Callers should persist the selected
IDs and ``previous_facility_ids or facility_ids`` as reference_facility_ids in
assistant metadata, so ordinal/remaining selections survive reconnects.
"""

from collections.abc import Mapping
from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class SchoolContext:
    facility_ids: tuple[str, ...] = ()
    candidate_ids: tuple[str, ...] = ()
    previous_facility_ids: tuple[str, ...] = ()
    is_followup: bool = False
    selection: str = "none"
    clarification: str | None = None


def _plain(value):
    return re.sub(r"[\W_]+", "", unicodedata.normalize("NFKC", str(value)).casefold())


def _identity(value):
    if isinstance(value, bool) or not isinstance(value, (str, int)):
        return None
    return str(value).strip() or None


def _ids(values, known):
    if not isinstance(values, (list, tuple)):
        return ()
    result = []
    for value in values:
        identity = _identity(value)
        if identity and (not known or identity in known) and identity not in result:
            result.append(identity)
    return tuple(result)


def _records(catalog):
    if isinstance(catalog, Mapping):
        container = catalog.get("payload", catalog)
        catalog = container.get("records", ()) if isinstance(container, Mapping) else ()
    if not isinstance(catalog, (list, tuple)):
        return {}
    result = {}
    for row in catalog:
        if not isinstance(row, Mapping):
            continue
        identity = _identity(row.get("id", row.get("facility_id")))
        if identity and row.get("name") and identity not in result:
            result[identity] = row
    return result


def _snapshot(value, known):
    """Return None for absent metadata; an explicit empty list resets context."""
    if isinstance(value, SchoolContext):
        return value.facility_ids, value.previous_facility_ids or value.facility_ids
    if not isinstance(value, Mapping):
        return None
    for key in ("selected_facility_ids", "facility_ids"):
        if key in value:
            selected = _ids(value[key], known)
            reference = _ids(value.get("reference_facility_ids", value.get("previous_facility_ids")), known)
            return selected, reference or selected
    directory = value.get("directory")
    if isinstance(directory, Mapping) and isinstance(directory.get("records"), list):
        selected = _ids([row.get("id") for row in directory["records"] if isinstance(row, Mapping)], known)
        return selected, selected
    for key in ("school_context", "answer_metadata", "payload"):
        nested = _snapshot(value.get(key), known)
        if nested is not None:
            return nested
    return None


_REFERENCE = re.compile(r"거기|그곳|그중|그가게|그매장|그식당|그시설|그카페|그지점|해당시설|이중|나머지")
_ORDINAL = re.compile(r"(첫|두|세|네|다섯|여섯|일곱|여덟|아홉|열|\d+)(?:번째|째|번(?!호))")
_ORDINAL_VALUES = {word: number for number, word in enumerate(
    ("첫", "두", "세", "네", "다섯", "여섯", "일곱", "여덟", "아홉", "열"), 1)}
_SLOT = re.compile(
    r"(?:전화번호|전화|연락처|연락|주소|위치|어디|어딨어|어딨|영업시간|운영시간|시간|사진|이미지|몇시(?:까지|부터)?|언제)"
    r"(?:은|는|이|가|을|를|도|좀|만|어디|뭐|알려|보여|줘|주세요|해|나요|야|요|까|지|열어|닫아|있어|가능해)*$"
)


def _slot_only(text):
    text=re.sub(r"^(?:그럼|그러면|그리고)", "", text)
    operating=re.fullmatch(r"(?:오늘|내일|모레|지금|현재)(?:은|는)?(?:안)?(?:해|하나|하나요|해요|쉬어|쉬나요|열어|열려|영업해|영업하나|문열어|문열려)(?:요)?",text)
    return bool(_SLOT.fullmatch(text) or operating)


def _collective(text):
    if re.search(r"둘다|(?:그|이)?두(?:곳|매장|시설|가게|식당|카페|지점)(?:다|모두|의|에서|은|는|도|$|위치|전화|연락|사진|공지|결제|운영|영업)|두개(?:다|모두)", text):
        return "both"
    if text in {"모두", "전부", "전체"} or re.search(r"(?:그중|그곳들)(?:모두|전부|전체)", text):
        return "all"
    return None


def _dependent(text):
    return bool(_REFERENCE.search(text) or _ORDINAL.search(text) or _collective(text) or _slot_only(text))


_GENERIC_ALIASES = {"은행", "서점", "책방"}
_GENERIC_PREFIXES = ("학교", "학교내", "학교에", "학교에서", "교내", "교내에", "캠퍼스", "한양대", "한양대학교",
    "에리카", "erica", "지금", "현재", "오늘", "내일", "그럼", "그러면", "그리고", "랑", "하고", "및", "와", "과")
_GENERIC_SUFFIX = re.compile(r"^(?:위치|전화|연락|어디|영업|운영|몇|언제|사진|이미지|건물|주소|문의|이름|정보|안내|소개|"
    r"은|는|이|가|에|의|을|를|도|만|좀|부터|까지|랑|하고|과|와|주말|평일|공휴일|[월화수목금토일]요일|24시간|지금|오늘)")


def _generic_alias_boundary(text, start, end, location_prefixes):
    # Compact Korean text has no spaces left. A generic type is a reference
    # only at a query boundary, after known query grammar, or after a catalog
    # location. Unknown proper-name prefixes (우리/국민/무지개...) are not aliases.
    prefix = text[:start]
    if prefix and not prefix.endswith((*_GENERIC_PREFIXES, *location_prefixes)):
        return False
    suffix = text[end:]
    return not suffix or bool(_GENERIC_SUFFIX.match(suffix))


def _name_spans(text, records):
    found = []
    location_prefixes = tuple({term for row in records.values() for term in _location_terms(row)})
    for identity, row in records.items():
        aliases = row.get("aliases", ())
        if not isinstance(aliases, (list, tuple)):
            aliases = ()
        for name in (row["name"], *aliases):
            term = _plain(name)
            if not term:
                continue
            if re.search(r"[a-z]", term):
                pattern = r"(?<![a-z])" + re.escape(term) + r"(?![a-z])"
            else:
                pattern = re.escape(term)
            for match in re.finditer(pattern, text):
                if (term in _GENERIC_ALIASES and term != _plain(row["name"])
                    and not _generic_alias_boundary(text, match.start(), match.end(), location_prefixes)):
                    continue
                found.append((match.start(), match.end(), identity))
    # A complete branch name supersedes a brand/name nested within that span.
    found = [item for item in found if not any(
        other[0] <= item[0] and other[1] >= item[1] and other[1] - other[0] > item[1] - item[0]
        for other in found)]
    return sorted(found, key=lambda item: item[0])


def _name_matches(text, records):
    result = []
    for _, _, identity in _name_spans(text, records):
        if identity not in result:
            result.append(identity)
    return tuple(result)


def match_explicit_facilities(message, *, catalog=()):
    """Return catalog-matched IDs in utterance order, before contextual filters.

    Longer complete names supersede nested aliases, and Latin aliases retain
    ASCII word boundaries. No history or location condition is applied here.
    """
    return _name_matches(_plain(message), _records(catalog))


def mask_document_references(message, *, titles=()):
    """Hide known document-title spans only in an interpretation copy.

    A referenced document may name a building, facility, date, or information
    field. Those words are part of the document identifier rather than extra
    constraints on the facilities requested outside that title. Callers keep
    the original utterance for retrieval, generation, storage and display.
    """
    text=str(message)
    spans=[]
    for title in sorted({str(value).strip() for value in titles if isinstance(value,str) and value.strip()},
                        key=len,reverse=True):
        # Official titles may be copied with different spacing or line wraps.
        pattern=r"\s*".join(re.escape(piece) for piece in title.split())
        for match in re.finditer(pattern,text,re.I):
            spans.append(match.span())
    if not spans:
        return text
    characters=list(text)
    for start,end in spans:
        characters[start:end]=" "*(end-start)
    return "".join(characters)


def _floor(value):
    match = re.search(r"(지하|B)?\s*(\d+)\s*(?:층|F)", str(value), re.I)
    if match:
        return int(match[2]) * (-1 if match[1] else 1)
    # Public catalog locations also use bare basement notation, e.g. B1.
    basement = re.search(r"(?<![A-Za-z0-9])B\s*(\d+)(?!\d|[A-Za-z])", str(value), re.I)
    return -int(basement[1]) if basement else None


def _location_terms(row):
    location = str(row.get("location") or "")
    prefix = re.split(r"(?:지하\s*|B)?\d+\s*(?:층|F)|(?<![A-Za-z0-9])B\s*\d+(?!\d|[A-Za-z])",
        location, maxsplit=1, flags=re.I)[0]
    terms = {_plain(piece) for piece in re.split(r"[,/·()]", prefix)}
    return {term for term in terms if len(term) >= 2}


def _filter_location(text, pool, records):
    # A building/floor written inside a facility's own proper name is not an
    # extra query condition. Mask the matched spans, preserving any separate
    # branch/floor phrase such as "세븐일레븐 학생복지관점".
    remaining = list(text)
    for start, end, _ in _name_spans(text, records):
        remaining[start:end] = " " * (end - start)
    text = "".join(remaining)
    mentioned = set()
    for row in records.values():
        mentioned.update(term for term in _location_terms(row) if term in text)
    # If both a full building name and its substring appear, keep the full one.
    mentioned = {term for term in mentioned if not any(term != other and term in other for other in mentioned)}
    floor = _floor(text)
    if not mentioned and floor is None:
        return tuple(pool), False
    selected = []
    for identity in pool:
        row = records.get(identity, {})
        location = _plain(row.get("location", ""))
        if mentioned and not any(term in location for term in mentioned):
            continue
        if floor is not None and _floor(row.get("location", "")) != floor:
            continue
        selected.append(identity)
    return tuple(selected), True


def _previous(history, previous_result, records):
    known = set(records)
    if previous_result is not None:
        return _snapshot(previous_result, known) or ((), ())
    items = list(history or ())
    for index in range(len(items) - 1, -1, -1):
        item = items[index]
        if not isinstance(item, Mapping):
            continue
        snapshot = _snapshot(item, known)
        if snapshot is not None:
            selected, reference = snapshot
            # Legacy histories may have only selected IDs. Recover a preceding
            # collection only across an explicitly dependent question chain.
            if selected and selected == reference and len(selected) == 1:
                user = next((entry for entry in reversed(items[:index])
                    if isinstance(entry, Mapping) and entry.get("role") == "user"), None)
                text = _plain(user.get("content", "")) if user else ""
                _, location_filter = _filter_location(text, tuple(records), records)
                if _dependent(text) or location_filter:
                    for earlier in reversed(items[:index]):
                        if not isinstance(earlier, Mapping):
                            continue
                        older = _snapshot(earlier, known)
                        if older is not None:
                            if len(older[1]) > 1 and set(selected).issubset(older[1]):
                                return selected, older[1]
                            if not older[0]:
                                break
                        elif earlier.get("role") == "user" and not _dependent(_plain(earlier.get("content", ""))):
                            break
            return snapshot
        if item.get("role") == "assistant":
            # Do not leap over an unrelated answer to resurrect an old result.
            prior = next((entry for entry in reversed(items[:index])
                if isinstance(entry, Mapping) and entry.get("role") == "user"), None)
            selected = _name_matches(_plain(prior.get("content", "")), records) if prior else ()
            return selected, selected
    return (), ()


def _clarify(ids, reference, records, reason=None):
    options = []
    for number, identity in enumerate(ids, 1):
        row = records.get(identity, {})
        name = str(row.get("name") or f"시설 ID {identity}")
        place = str(row.get("location") or "")
        options.append(f"{number}. {name}" + (f" ({place})" if place else ""))
    prompt = reason or "어느 시설을 말씀하시는지 확인해 주세요."
    if options:
        prompt += " " + " / ".join(options)
    prompt += " 시설명이나 지점을 알려 주세요."
    return SchoolContext((), tuple(ids), tuple(reference), True, "ambiguous", prompt)


def _resolved(ids, reference, followup, selection):
    return SchoolContext(tuple(ids), tuple(ids), tuple(reference), followup, selection)


def resolve_school_context(message, history, *, catalog=(), previous_result=None):
    """Resolve explicit facilities, a subset, an ordinal, or a safe follow-up.

    IDs are strings and collection order is retained. Ambiguous references
    yield no selected facility plus a specific clarification and candidate IDs.
    Category/topic routing belongs to the caller; broad category words are not
    expanded here and only catalog-provided names/aliases identify facilities.
    """
    text = _plain(message)
    records = _records(catalog)
    selected_before, reference = _previous(history, previous_result, records)
    explicit = _name_matches(text, records)
    collective = _collective(text)
    ordinals = list(_ORDINAL.finditer(text))

    if explicit:
        pool = explicit
        if reference and (_dependent(text) or re.search(r"사진|영업시간|운영시간|전화번호|연락처|몇시", text)):
            intersect = tuple(identity for identity in reference if identity in explicit)
            if intersect:
                pool = intersect
        filtered, has_filter = _filter_location(text, pool, records)
        multiple_named = (len({_plain(records[identity]["name"]) for identity in pool}) > 1
            and bool(re.search(r"랑|하고|및|와|과|,", str(message))))
        if not filtered:
            return _clarify(pool, reference, records, "말씀하신 위치·층에 맞는 시설을 확인할 수 없어요.")
        if len(filtered) > 1 and not multiple_named and not collective:
            return _clarify(filtered, reference, records, "같은 이름의 시설이 여러 곳이에요.")
        followup = bool(reference and set(filtered).issubset(reference))
        return _resolved(filtered, reference if followup else (), followup, "filter" if has_filter else "explicit")

    filtered, has_filter = _filter_location(text, reference, records)
    if has_filter and reference:
        if not filtered:
            return _clarify(reference, reference, records, "이전 목록에 말씀하신 위치·층의 시설이 없어요.")
        return _resolved(filtered, reference, True, "filter")

    if ordinals:
        if not reference:
            return _clarify((), (), records, "번호로 선택할 이전 시설 목록이 없어요.")
        numbers = {_ORDINAL_VALUES.get(match[1], int(match[1]) if match[1].isdigit() else 0) for match in ordinals}
        if any(number < 1 or number > len(reference) for number in numbers):
            return _clarify(reference, reference, records, f"이전 목록은 {len(reference)}곳이라 해당 번호를 선택할 수 없어요.")
        chosen = tuple(identity for index, identity in enumerate(reference, 1) if index in numbers)
        if re.search(r"말고|제외", text):
            chosen = tuple(identity for identity in reference if identity not in chosen)
        return _resolved(chosen, reference, True, "ordinal")

    if "나머지" in text:
        remaining = tuple(identity for identity in reference if identity not in selected_before)
        if not remaining:
            return _clarify(reference, reference, records, "나머지를 고를 기준 시설이 명확하지 않아요.")
        return _resolved(remaining, reference, True, "remaining")

    if collective:
        if not reference or (collective == "both" and len(reference) != 2):
            return _clarify(reference, reference, records, "함께 선택할 두 시설을 확인해 주세요." if collective == "both" else "이전 시설 목록이 없어요.")
        return _resolved(reference, reference, True, collective)

    if _dependent(text):
        if len(selected_before) == 1:
            return _resolved(selected_before, reference, True, "previous")
        return _clarify(selected_before or reference, reference, records)
    return SchoolContext()

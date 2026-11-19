"""Generate and evaluate grounded school questions without model-text matching.

Default operation only reads public SQLite tables and prepares JSON cases:
  python scripts/evaluate_school.py --database data/efoo.sqlite3 --output cases.json
Opt in to every catalog facility's public and missing detail fields:
  python scripts/evaluate_school.py --database data/efoo.sqlite3 --all-facilities --output complete-cases.json
Explicit live execution creates fresh, isolated QA conversations:
  python scripts/evaluate_school.py --database data/efoo.sqlite3 --base-url http://127.0.0.1:8000 --output report.json
Replay a complete captured report offline, without a database or API:
  python scripts/evaluate_school.py --responses report.json --output replay.json

No settings module, environment variables, API keys, existing chat messages or
browser cookies are read. Only the explicit --base-url option enables HTTP.
"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import html
from html.parser import HTMLParser
from http.cookiejar import CookieJar
import json
from pathlib import Path
import re
import sqlite3
import sys
import unicodedata
from urllib.error import HTTPError, URLError
from urllib.parse import parse_qs, quote, urlparse
from urllib.request import HTTPCookieProcessor, Request, build_opener
from uuid import uuid4


SCHEMA_VERSION = 1
OFFICIAL_HOST = "life.hanyang.ac.kr"
LOCAL_IMAGE = re.compile(r"^/media/[a-f0-9]{64}\.(?:jpg|jpeg|png|gif|webp|avif)$")
SOURCE_ID = re.compile(r"^S[1-9]\d*$")
BRANDS = {
    "세븐일레븐": (r"7[\s-]*eleven|세븐\s*일레븐", "7-eleven"),
    "CU": (r"(?<![a-z])cu(?![a-z])|씨유", "씨유"),
    "GS25": (r"gs\s*25|지에스\s*25", "GS25"),
    "이마트24": (r"e[\s-]*mart\s*24|이마트\s*24", "emart24"),
    "스타벅스": (r"starbucks|스타벅스", "STARBUCKS"),
}
BUSINESS_TERMS = {
    "분식": ("분식", "떡볶이", "김밥"),
    "서점": ("서점", "서적", "bookstore"),
    "미용실": ("미용실", "헤어", "hair"),
    "꽃집": ("꽃집", "플라워", "flower"),
}


class SnapshotError(ValueError):
    pass


class _PublicText(HTMLParser):
    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts, self.ignored = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in {"script", "style"}:
            self.ignored += 1

    def handle_endtag(self, tag):
        if tag in {"script", "style"} and self.ignored:
            self.ignored -= 1

    def handle_data(self, data):
        if not self.ignored:
            self.parts.append(data)


def plain(value):
    if not isinstance(value, str):
        return ""
    if "<" in value:
        parser = _PublicText()
        parser.feed(value)
        return " ".join(parser.parts).strip()
    return value.strip()


def official_url(value):
    if not isinstance(value, str):
        return False
    try:
        url = urlparse(value)
        return url.scheme in {"http", "https"} and url.hostname == OFFICIAL_HOST and not url.username and not url.password
    except (ValueError, TypeError):
        return False


def normalized(value):
    return re.sub(r"[^a-z0-9가-힣]", "", unicodedata.normalize("NFKC", html.unescape(str(value))).casefold())


def location_form(value):
    value = unicodedata.normalize("NFKC", str(value))
    value = re.sub(r"B\s*(\d+)", r"지하\1층", value, flags=re.I)
    value = re.sub(r"(\d+)\s*F", r"\1층", value, flags=re.I)
    value = re.sub(r"(?<!학생)복지관", "학생복지관", value)
    value = value.replace("기숙사", "창의인재원").replace("제삼공학관", "제3공학관")
    value = re.sub(r"(?<!제)([1-5])공학관", r"제\1공학관", value)
    return normalized(value)


def clock_values(text):
    result = set()
    pattern = r"(?P<period>오전|오후)?\s*(?<!\d)(?P<hour>[0-1]?\d|2[0-4])\s*(?::\s*(?P<minute>\d{2})|시(?!간)(?:\s*(?P<kminute>\d{1,2})\s*분)?)"
    for match in re.finditer(pattern, unicodedata.normalize("NFKC", str(text))):
        hour, minute = int(match["hour"]), int(match["minute"] or match["kminute"] or 0)
        if minute > 59 or (hour == 24 and minute):
            continue
        if match["period"] == "오후" and hour < 12:
            hour += 12
        if match["period"] == "오전" and hour == 12:
            hour = 0
        result.add(f"{hour:02d}:{minute:02d}")
    return result


def phone_values(text):
    values = set()
    text = re.sub(r"[‐‑–—]", "-", unicodedata.normalize("NFKC", str(text)))
    pattern = r"(?:\+82[-\s]?)?0?\d{1,2}[-\s.)]+\d{3,4}[-\s]+\d{4}"
    for match in re.finditer(pattern, text):
        digits = re.sub(r"\D", "", match.group())
        values.add("0" + digits[2:] if digits.startswith("82") else digits)
    values.update(match.group() for match in re.finditer(r"(?<!\d)0\d{8,10}(?!\d)", text))
    return values


def money_values(text):
    values = set()
    for match in re.finditer(r"(?<![\d,])(\d{1,3}(?:,\d{3})+|\d+)\s*원", str(text)):
        values.add(int(match[1].replace(",", "")))
    for match in re.finditer(r"(?:(\d+)\s*만)?\s*(?:(\d+)\s*천)?\s*원", str(text)):
        if match[1] or match[2]:
            values.add(int(match[1] or 0) * 10000 + int(match[2] or 0) * 1000)
    return values


def schedule_markers(text):
    """Meaningful schedule conditions; punctuation and wording can differ."""
    # The UI escapes Markdown punctuation (e.g. 월\~금) to preserve the
    # official schedule. Normalize only that presentation, not its conditions.
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~～〜–—−‐‑‒])", r"\1", text)
    text = text.translate(str.maketrans({separator: "~" for separator in "～〜–—−‐‑‒"}))
    patterns = {
        "weekdays": r"평일|월\s*[~～–—-]\s*금|월요일.{0,5}금요일",
        "weekend": r"주말|토\s*[·/,~～–—-]\s*일",
        "saturday": r"토요일|(?<![가-힣])토(?=[\s:·/,~～–—(\d-])",
        "sunday": r"일요일|(?<![가-힣])일(?=[\s:·/,~～–—(\d-])",
        "holiday": r"공휴일|공휴",
        "semester": r"학기|학기중",
        "vacation": r"방학",
        "break": r"브레이크|break|휴식|휴게|준비\s*시간",
        "closed": r"휴무|휴점|미운영|운영\s*하지\s*않",
        "staffed": r"(?<!무)유인",
        "unstaffed": r"무인",
    }
    return {name for name, pattern in patterns.items() if re.search(pattern, text, re.I)}


def _fact_result(fact, text):
    kind = fact["kind"]
    if kind == "location":
        return location_form(fact["value"]) in location_form(text), text
    if kind == "phone":
        actual = sorted(phone_values(text))
        # Each complete published phone is an independent assertion. A string
        # made by concatenating two phones must never pass via suffix matching.
        value = re.sub(r"\D", "", fact["value"])
        return value in actual, actual
    if kind == "clocks":
        actual = sorted(clock_values(text))
        return set(fact["values"]) <= set(actual), actual
    if kind == "hours_24":
        return bool(re.search(r"24\s*(?:시간|h)", text, re.I)), text
    if kind == "schedule_markers":
        actual = sorted(schedule_markers(text))
        return set(fact["values"]) <= set(actual), actual
    if kind == "money_any":
        actual = sorted(money_values(text))
        return bool(set(fact["values"]) & set(actual)), actual
    return False, "Unknown fact assertion"


def _acknowledges_unknown(text):
    return bool(re.search(r"확인.{0,15}(?:못|어려|없|않)|(?:공개|등록|수집|근거|정보).{0,18}(?:없|않|못|미공개|미등록)|"
        r"(?:알|단정).{0,10}(?:없|어려)|명시.{0,8}(?:않|없)|모르|불명확|어느|어떤.{0,12}(?:시설|매장|곳)|"
        r"(?:시설|매장|대상).{0,15}(?:선택|특정|알려.{0,5}주)|미공개|미등록", str(text)))


def _count_number(value):
    if value.isdigit():
        return int(value)
    ones = {"영": 0, "공": 0, "한": 1, "하나": 1, "두": 2, "둘": 2, "세": 3, "셋": 3,
        "네": 4, "넷": 4, "다섯": 5, "여섯": 6, "일곱": 7, "여덟": 8, "아홉": 9,
        "일": 1, "이": 2, "삼": 3, "사": 4, "오": 5, "육": 6, "칠": 7, "팔": 8, "구": 9}
    if value in ones:
        return ones[value]
    for prefix, tens in (("아흔", 90), ("여든", 80), ("일흔", 70), ("예순", 60), ("쉰", 50),
            ("마흔", 40), ("서른", 30), ("스물", 20), ("스무", 20), ("열", 10)):
        if value.startswith(prefix) and value[len(prefix):] in {"", *ones}:
            return tens + ones.get(value[len(prefix):], 0)
    match = re.fullmatch(r"([이삼사오육칠팔구]?)십([일이삼사오육칠팔구]?)", value)
    return (ones.get(match[1], 1) * 10 + ones.get(match[2], 0)) if match else None


def _count_claims(text):
    """Read explicit facility counters, excluding floors, dates and phone digits."""
    text = unicodedata.normalize("NFKC", html.unescape(str(text)))
    text = re.sub(r"\\([\\`*_{}\[\]()#+\-.!|>~])", r"\1", text)
    text = re.sub(r"[*_`]", "", text)
    result = []
    for match in re.finditer(r"((?<!\d)[0-9]+|(?<![가-힣])[가-힣]+)\s*(곳|군데|개소|개|지점)"
            r"(?=$|[^가-힣]|(?:이에요|입니다|예요|이|은|는|을|를|으로|로|만|도)(?:$|[^가-힣]))", text):
        # 공개 is a noun/adjective, not 공(0)+개; 이곳 is a demonstrative,
        # not 이(2)+곳. Single Sino-Korean numerals are not natural counters.
        if match[1] in {"공", "일", "이", "삼", "사", "오", "육", "칠", "팔", "구"}:
            continue
        number = _count_number(match[1])
        if number is not None:
            result.append(number)
    return result


def _primary_count(answer):
    """The first facility count in narrative, before any supporting table.

    Category tables and building floor numbers cannot stand in for the main
    answer. If no explicit narrative count is supplied, validation fails rather
    than treating the response's own directory metadata as proof of its prose.
    """
    for line in str(answer).splitlines():
        if line.lstrip().startswith("|"):
            break
        counts = _count_claims(line)
        if counts:
            return counts[0]
    return None


def _category_body_counts(answer, categories):
    result = {}
    for line in str(answer).splitlines():
        cells = re.split(r"(?<!\\)\|", line.strip().strip("|")) if line.lstrip().startswith("|") else [line]
        for category in categories:
            if normalized(category) not in normalized(cells[0]):
                continue
            claims = _count_claims(" ".join(cells[1:]) if len(cells) > 1 else line)
            if len(claims) == 1:
                result[category] = claims[0]
    return result


def _facility_answer_scopes(answer, sources, code_map, facility_id):
    """Collect only table rows or citation-bearing sentences owned by one ID.

    A sentence citing both compared facilities is deliberately not proof of
    which value belongs to which facility. That requires an unambiguous row or
    sentence, and an uncheckable association is a failed assertion.
    """
    owners = {source.get("id"): _facility_ids([source], code_map) for source in sources}
    scopes = []
    for line in str(answer).splitlines():
        units = [line] if line.lstrip().startswith("|") else re.split(
            r"(?<=[.!?])\s+(?!\[S\d+\])|(?<=\])(?:\s*[,;]\s*|\s+(?=[^\[\s]))", line)
        for unit in units:
            cited = set(re.findall(r"\[(S\d+)\]", unit))
            identities = set().union(*(owners.get(citation, set()) for citation in cited)) if cited else set()
            if identities == {facility_id}:
                scopes.append(unit)
    return "\n".join(scopes)


def _row_value(info, *labels):
    return next((plain(row.get("value")) for row in info.get("info_rows", [])
        if isinstance(row, dict) and any(label in str(row.get("label", "")) for label in labels)), "")


def _profile_images(payload, info):
    cover = info.get("cover_image") if isinstance(info.get("cover_image"), dict) else {}
    wanted = {cover.get("url", "")}
    wanted.update(item.get("url", "") for item in info.get("gallery", []) if isinstance(item, dict))
    wanted.update(item.get("url", "") for item in payload.get("page_data", {}).get("images", [])
        if isinstance(item, dict) and item.get("role") != "menu")
    images = set()
    for asset in payload.get("assets", []):
        if not isinstance(asset, dict) or not asset.get("downloaded"):
            continue
        local = asset.get("local_path", "")
        if LOCAL_IMAGE.fullmatch(local) and (asset.get("url") in wanted or asset.get("role") in {"cover", "gallery"}):
            images.add(local)
    return sorted(images)


def load_public_snapshot(database):
    """Read only explicit public source tables through a read-only connection."""
    path = Path(database).resolve()
    if not path.is_file():
        raise SnapshotError("Public database file does not exist")
    connection = sqlite3.connect("file:" + quote(str(path), safe="/") + "?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    try:
        rows = [dict(row) for row in connection.execute(
            "SELECT scope,code,name,url,fetched_at,payload FROM knowledge_sources "
            "WHERE scope='page:facility-directory' OR scope LIKE 'facility:%' OR scope LIKE 'notice:%'")]
        try:
            meals = [dict(row) for row in connection.execute("SELECT code,day,name,url,fetched_at,meals FROM collections")]
        except sqlite3.OperationalError:
            meals = []
    except sqlite3.OperationalError as exc:
        raise SnapshotError("Public source tables are unavailable") from exc
    finally:
        connection.close()
    for row in rows:
        try:
            row["payload"] = json.loads(row["payload"])
            if not isinstance(row["payload"], dict):
                raise SnapshotError("A public source payload must be an object")
        except (ValueError, TypeError) as exc:
            raise SnapshotError("A public source payload is invalid JSON") from exc
    catalog = next((row for row in rows if row["scope"] == "page:facility-directory"), None)
    raw = (catalog or {}).get("payload", {}).get("records")
    if not catalog or not official_url(catalog["url"]) or not isinstance(raw, list) or not raw:
        raise SnapshotError("A complete official facility catalog is required")
    unique = {}
    for item in raw:
        if not isinstance(item, dict) or isinstance(item.get("id"), bool) or not isinstance(item.get("id"), int):
            raise SnapshotError("Facility IDs must be integers")
        value = {"id": item["id"], "name": plain(item.get("name")),
            "category": plain(item.get("category_display") or item.get("category")),
            "location": plain(item.get("location")), "description": plain(item.get("description")), "notice": plain(item.get("notice"))}
        if not value["name"] or not value["category"]:
            raise SnapshotError("Facility name/category is missing")
        if value["id"] in unique and unique[value["id"]] != value:
            raise SnapshotError("Conflicting duplicate facility IDs")
        unique[value["id"]] = value
    profiles, notices = {}, []
    for row in sorted(rows, key=lambda row: row["scope"]):
        if not official_url(row["url"]):
            continue
        payload = row["payload"]
        if row["scope"].startswith("facility:"):
            try:
                identifier = int(row["scope"].split(":", 1)[1])
            except ValueError:
                continue
            if identifier not in unique:
                continue
            info = payload.get("restaurant_info", {})
            if not isinstance(info, dict):
                continue
            description = info.get("description") if isinstance(info.get("description"), dict) else {}
            profiles[str(identifier)] = {"id": identifier, "name": unique[identifier]["name"],
                "code": row["code"], "category": unique[identifier]["category"], "url": row["url"], "fetched_at": row["fetched_at"],
                "location": plain(info.get("location")) or _row_value(info, "위치", "주소") or unique[identifier]["location"],
                "hours": plain(info.get("opening_hours")) or _row_value(info, "영업시간", "운영시간"),
                "contact": plain(info.get("contact")) or _row_value(info, "연락", "전화"),
                "description": plain(description.get("text")),
                "notices": [plain(item.get("text")) for item in info.get("notices", []) if isinstance(item, dict)],
                "images": _profile_images(payload, info)}
        elif row["scope"].startswith("notice:"):
            notices.append({"id": row["scope"].split(":", 1)[1], "url": row["url"], "fetched_at": row["fetched_at"],
                "title": plain(payload.get("title") or row["name"]), "text": plain(payload.get("text")),
                "restaurant_codes": [code for code in payload.get("restaurant_codes", []) if isinstance(code, str)]})
    clean_meals = []
    for row in meals:
        if not official_url(row["url"]):
            continue
        try:
            menus = json.loads(row["meals"])
        except (ValueError, TypeError):
            continue
        clean_meals.append({**row, "meals": menus})
    return {"catalog": {"url": catalog["url"], "fetched_at": catalog["fetched_at"],
        "records": sorted(unique.values(), key=lambda item: item["id"])},
        "profiles": profiles, "notices": notices, "meals": clean_meals}


def _positive_business(text, words):
    for word in words:
        for match in re.finditer(re.escape(word), text, re.I):
            following = text[match.end():match.end() + 35]
            if not re.match(r"\s*(?:집|점)?(?:이|은|는|가|을|를)?\s*(?:아니|아닌|아닙|아님|없|(?:판매|취급)\s*(?:하지|안)|메뉴\s*없)", following):
                return True
    return False


def _business_ids(snapshot, words):
    found = []
    for item in snapshot["catalog"]["records"]:
        texts = [item["name"], item.get("description", ""), item.get("notice", "")]
        profile = snapshot["profiles"].get(str(item["id"]))
        if profile:
            texts += [profile["name"], profile["description"], *profile["notices"]]
        if _positive_business("\n".join(texts), words):
            found.append(item["id"])
    return found


def compatibility_jamo(text):
    initials, vowels = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ", "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
    finals = "\0ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
    output = []
    for char in text:
        offset = ord(char) - 0xAC00
        if 0 <= offset < 11172:
            output.extend((initials[offset // 588], vowels[offset // 28 % 21]))
            if offset % 28:
                output.append(finals[offset % 28])
        else:
            output.append(char)
    return "".join(output)


def build_cases(snapshot, *, all_facilities=False):
    """Build deterministic assertion manifests from public records and facts."""
    records, profiles = snapshot["catalog"]["records"], snapshot["profiles"]
    by_id = {item["id"]: item for item in records}
    categories = sorted({item["category"] for item in records})
    cases = []

    def add(identifier, category, question=None, expect=None, *, steps=None):
        cases.append({"id": identifier, "category": category, "question": question or steps[-1]["question"],
            "steps": steps or [{"question": question, "expect": expect or {}}]})

    def aggregate(ids, *, names=False, counts=None):
        value = {"modes": ["retrieval"], "directory_ids": sorted(ids), "scope_label_required": True}
        if names:
            value["mention_names"] = sorted({by_id[identifier]["name"] for identifier in ids})
        if counts is not None:
            value["directory_categories"] = counts
        return value

    def detail(profile, facts=(), **extra):
        return {"source_facility_ids": [profile["id"]], "allowed_facility_ids": [profile["id"]],
            "facility_codes": {item["code"]: item["id"] for item in profiles.values()}, "facts": list(facts), **extra}

    for index, category in enumerate(categories):
        ids = [item["id"] for item in records if item["category"] == category]
        add(f"category.{index}.count", "count", f"학교에 {category} 몇 곳 있어?", aggregate(ids))
        add(f"category.{index}.list", "list", f"{category} 목록을 전부 알려줘", aggregate(ids, names=True))
    all_ids = sorted(by_id)
    add("catalog.total", "count", "학교 시설은 총 몇 곳이야?", aggregate(all_ids))
    add("catalog.types", "categories", "학교 시설 종류별로 몇 개 있어?", aggregate(all_ids, counts=dict(Counter(item["category"] for item in records))))
    add("catalog.all", "list", "공식 시설 목록 전체를 알려줘", aggregate(all_ids, names=True))
    convenience = [item["id"] for item in records if item["category"] == "편의점"]
    if convenience:
        for index, question in enumerate(["학교에 편의점 몇개 있어?", "편의점이 몇 군데야?", "편의점 개수 알려줘", "몇 개의 편의점이 있어?", "캠퍼스 편의점 어디어디 있어?"]):
            add(f"convenience.variant.{index}", "language_variant", question, aggregate(convenience, names=index == 4))
        for index, question in enumerate(["학교에편의점몇개있어?", compatibility_jamo("학교에 편의점 몇개 있어?")]):
            add(f"convenience.input.{index}", "language_variant", question, aggregate(convenience))
    for index, (brand, (pattern, alias)) in enumerate(BRANDS.items()):
        ids = [item["id"] for item in records if re.search(pattern, item["name"], re.I)]
        if ids:
            add(f"brand.{index}.count", "brand", f"{brand} 몇 개 있어?", aggregate(ids))
            add(f"brand.{index}.alias", "brand", f"{alias} 지점 위치를 전부 알려줘", aggregate(ids, names=True))
    location_groups = []
    for place in sorted({re.split(r"\s+", item["location"])[0] for item in records if item["location"]}):
        for category in ("편의점", "카페/베이커리"):
            ids = [item["id"] for item in records if item["category"] == category and location_form(place) in location_form(item["location"])]
            if ids:
                location_groups.append((place, category, ids))
    for index, (place, category, ids) in enumerate(location_groups[:6]):
        add(f"location.filter.{index}", "location_filter", f"{place}에 {category} 몇 곳 있어?", aggregate(ids))
    if len(profiles) == len(records):
        for index, (label, words) in enumerate(BUSINESS_TERMS.items()):
            ids = _business_ids(snapshot, words)
            if ids:
                add(f"business.{index}", "business", f"학교에 {label}집 찾아줘" if label == "분식" else f"학교에 {label} 찾아줘", aggregate(ids, names=True))
    ordered_profiles = sorted(profiles.values(), key=lambda value: (value["category"], value["id"]))
    distinct = []
    for profile in ordered_profiles:
        if profile["category"] not in {item["category"] for item in distinct}:
            distinct.append(profile)
    distinct.extend(profile for profile in ordered_profiles if profile not in distinct)
    selected = distinct[:6]
    name_counts = Counter(item["name"] for item in records)

    def entity(profile):
        return profile["name"] if name_counts[profile["name"]] == 1 else profile["location"] + "의 " + profile["name"]

    for profile in selected:
        if profile["location"]:
            add(f"detail.{profile['id']}.location", "location", f"{entity(profile)} 위치 알려줘", detail(profile, [{"kind": "location", "value": profile["location"]}]))
    with_hours = [profile for profile in distinct if clock_values(profile["hours"])][:6]
    for profile in with_hours:
        add(f"detail.{profile['id']}.hours", "hours", f"{entity(profile)} 운영시간 알려줘", detail(profile, [{"kind": "clocks", "values": sorted(clock_values(profile["hours"]))}]))
    with_contact = [profile for profile in distinct if phone_values(profile["contact"])][:6]
    for profile in with_contact:
        add(f"detail.{profile['id']}.contact", "contact", f"{entity(profile)} 연락처 알려줘", detail(profile, [{"kind": "phone", "value": sorted(phone_values(profile["contact"]))[0]}]))
    with_photos = [profile for profile in distinct if profile["images"]][:4]
    for profile in with_photos:
        add(f"detail.{profile['id']}.photos", "photos", f"{entity(profile)} 사진 보여줘", detail(profile, images_required=True, allowed_images=profile["images"]))
    for notice in snapshot["notices"][:3]:
        facts = []
        values = sorted(money_values(notice["text"]))
        if values:
            facts.append({"kind": "money_any", "values": values})
        question = f"'{notice['title']}' 공지 내용 알려줘" + (". 금액이 있다면 그 금액도 알려줘" if facts else "")
        add(f"notice.{notice['id']}", "notice", question, {"required_source_urls": [notice["url"]], "facts": facts})
    for profile in with_contact[:3]:
        facts = [{"kind": "location", "value": profile["location"]}, {"kind": "phone", "value": sorted(phone_values(profile["contact"]))[0]}]
        add(f"multi.{profile['id']}", "multiintent", f"{entity(profile)} 위치랑 전화번호 같이 알려줘", detail(profile, facts))
    for place in dict.fromkeys(group[0] for group in location_groups):
        first_ids = [item["id"] for item in records if item["category"] == "편의점" and location_form(place) in location_form(item["location"])]
        second_ids = [item["id"] for item in records if item["category"] == "카페/베이커리" and location_form(place) in location_form(item["location"])]
        if first_ids and second_ids:
            add("followup.category", "followup", steps=[
                {"question": f"{place} 편의점 몇 개 있어?", "expect": aggregate(first_ids)},
                {"question": "그럼 카페는?", "expect": aggregate(second_ids)}])
            break
    if with_contact:
        profile = with_contact[0]
        add("followup.entity", "followup", steps=[
            {"question": f"{entity(profile)} 위치 알려줘", "expect": detail(profile, [{"kind": "location", "value": profile["location"]}])},
            {"question": "그럼 전화번호는?", "expect": detail(profile, [{"kind": "phone", "value": sorted(phone_values(profile["contact"]))[0]}])}])
    for index, question in enumerate(["파리 내일 날씨 알려줘", "내 계좌 비밀번호 알려줘", "개인 학생 성적 조회해줘",
        "서울캠퍼스 편의점 몇 개 있어?", "공식 근거 없이 졸업 요건을 정확히 알려줘", "지금 카페에 남은 좌석 수 알려줘"]):
        add(f"scope.{index}", "scope_refusal", question, {"must_refuse": True, "require_sources": False})
    priced = []
    for collection in sorted(snapshot["meals"], key=lambda row: (row["day"], row["code"]), reverse=True):
        for menu in collection["meals"]:
            prices = money_values(menu.get("price", ""))
            if prices:
                priced.append((collection, menu, sorted(prices)[0]))
                break
        if len(priced) == 3:
            break
    for index, (collection, menu, price) in enumerate(priced):
        ids = parse_qs(urlparse(collection["url"]).query).get("id", [])
        expected = {"facts": [{"kind": "money_any", "values": [price]}], "required_dates": [collection["day"]]}
        if ids and ids[0].isdigit():
            expected.update(source_facility_ids=[int(ids[0])], allowed_facility_ids=[int(ids[0])])
        add(f"meal.price.{index}", "meal", f"{collection['day']} {collection['name']} {menu.get('meal_type', '')} 가격 알려줘", expected)
    if all_facilities:
        cases.extend(_all_facility_cases(snapshot))
    return cases


def known_field(value):
    return isinstance(value, str) and bool(normalized(value)) and normalized(value) not in {"정보없음", "없음", "미등록", "미공개", "정보미등록"}


def saturday_oracle(hours):
    """Use only explicitly Saturday/daily conditions, never infer from weekdays.

    An unqualified timetable does not prove Saturday opening. The oracle keeps
    that uncertainty instead of guessing the current day or season.
    """
    text = re.sub(r"토요일\s*[·/,와과]+\s*일요일", "주말", str(hours))
    match = re.search(r"(?:토요일|토(?=\s*[:·/ ,(]|\s*\d)|주말|매일|연중무휴)", text)
    if not match:
        return {"kind": "unknown"}
    segment = text[match.start():]
    segment = re.split(r"\n|;|평일|월요일|화요일|수요일|목요일|금요일|일요일|공휴일|방학|학기", segment, maxsplit=1)[0]
    if re.search(r"휴무|휴점|미운영|운영\s*(?:안|하지\s*않)|닫", segment):
        return {"kind": "closed"}
    clocks = sorted(clock_values(segment))
    if clocks:
        return {"kind": "clocks", "values": clocks}
    if re.search(r"24\s*(?:시간|h)", segment, re.I):
        return {"kind": "hours_24"}
    return {"kind": "unknown"}


def _all_facility_cases(snapshot):
    records, profiles = snapshot["catalog"]["records"], snapshot["profiles"]
    counts = Counter(item["name"] for item in records)
    code_map = {item["code"]: item["id"] for item in profiles.values()}
    cases = []

    def entity(item):
        return item["name"] if counts[item["name"]] == 1 else item.get("location", "") + "의 " + item["name"]

    def expect_for(items, **extra):
        ids = [item["id"] for item in items]
        return {"source_facility_ids": ids, "allowed_facility_ids": ids,
            "facility_codes": code_map, **extra}

    def add(identifier, category, question=None, expect=None, *, steps=None, coverage=None):
        case = {"id": identifier, "category": category, "question": question or steps[-1]["question"],
            "steps": steps or [{"question": question, "expect": expect or {}}]}
        if coverage:
            case["facility_coverage"] = coverage
        cases.append(case)

    def facts_for(item, slots=("location", "hours", "contact")):
        facts = []
        if "location" in slots and known_field(item.get("location", "")):
            facts.append({"kind": "location", "value": item["location"]})
        if "hours" in slots:
            clocks = sorted(clock_values(item.get("hours", "")))
            if clocks:
                facts.append({"kind": "clocks", "values": clocks})
            if re.search(r"24\s*(?:시간|h)", item.get("hours", ""), re.I):
                facts.append({"kind": "hours_24"})
            markers = sorted(schedule_markers(item.get("hours", "")))
            if markers:
                facts.append({"kind": "schedule_markers", "values": markers})
        if "contact" in slots:
            for phone in sorted(phone_values(item.get("contact", ""))):
                facts.append({"kind": "phone", "value": phone})
        return facts

    # One request per facility covers every slot without multiplying 48 by 4.
    # Missing fields are intentional tests, not silently skipped coverage.
    for record in records:
        profile = profiles.get(str(record["id"]))
        public = profile or record
        available = [slot for slot in ("location", "hours", "contact") if known_field(public.get(slot, ""))]
        if public.get("images"):
            available.append("images")
        missing = [slot for slot in ("location", "hours", "contact", "images") if slot not in available]
        expected = expect_for([public], facts=facts_for(public), allowed_images=public.get("images", []))
        if profile:
            expected["facility_facts"] = [{"id": public["id"], "facts": facts_for(public)}]
            expected["unknown_fields"] = [{"id": public["id"], "slot": slot} for slot in missing if slot != "images"]
            expected["images_required"] = bool(public.get("images"))
            if "contact" in missing:
                expected["no_phone_claims"] = True
            if "hours" in missing:
                expected["no_clock_claims"] = True
        else:
            expected.update(modes=["retrieval", "openai", "insufficient"], require_sources=False)
            expected.pop("source_facility_ids")
            expected["must_acknowledge_unknown"] = True
        add(f"facility.{record['id']}.core", "facility_core",
            f"{entity(public)} 위치, 영업시간, 전화번호와 사진을 함께 알려줘. 공개되지 않은 항목은 모른다고 알려줘.",
            expected, coverage={"id": record["id"], "requested": ["location", "hours", "contact", "images"],
                "available": available, "missing": missing})

    ordered = sorted(profiles.values(), key=lambda item: item["id"])
    contact = [item for item in ordered if phone_values(item.get("contact", ""))]
    hours = [item for item in ordered if known_field(item.get("hours", ""))]
    for index, item in enumerate(contact[:3]):
        add(f"context.{item['id']}.short_contact", "context_boundary", steps=[
            {"question": f"{entity(item)} 위치 알려줘", "expect": expect_for([item], facts=facts_for(item, ("location",)))},
            {"question": ("전화도" if index % 2 == 0 else "연락처는?"), "expect": expect_for([item], facts=facts_for(item, ("contact",)))}])
    for index, item in enumerate(hours[:3]):
        add(f"context.{item['id']}.short_hours", "context_boundary", steps=[
            {"question": f"{entity(item)} 위치 알려줘", "expect": expect_for([item], facts=facts_for(item, ("location",)))},
            {"question": "그럼 운영시간은?", "expect": expect_for([item], facts=facts_for(item, ("hours",)))}])
    for item in contact[:2]:
        add(f"context.{item['id']}.short_location", "context_boundary", steps=[
            {"question": f"{entity(item)} 전화번호 알려줘", "expect": expect_for([item], facts=facts_for(item, ("contact",)))},
            {"question": "어디야?", "expect": expect_for([item], facts=facts_for(item, ("location",)))}])
    for index in range(min(4, len(contact) // 2)):
        pair = contact[index * 2:index * 2 + 2]
        expected = expect_for(pair, facts=[fact for item in pair for fact in facts_for(item, ("location", "contact"))],
            facility_facts=[{"id": item["id"], "facts": facts_for(item, ("location", "contact"))} for item in pair])
        question = f"{entity(pair[0])}하고 {entity(pair[1])}의 위치와 전화번호를 각각 알려줘"
        add(f"multi.named.{index}", "two_facilities", question, expected)
        if index < 2:
            add(f"context.ambiguous.{index}", "context_boundary", steps=[
                {"question": question, "expect": expected},
                {"question": "거기 전화번호는?", "expect": {"modes": ["clarification", "insufficient"],
                    "require_sources": False, "allowed_facility_ids": [item["id"] for item in pair],
                    "must_acknowledge_unknown": True, "no_phone_claims": True}}])
    add("context.no_antecedent", "context_boundary", "거기 전화번호랑 위치 알려줘",
        {"must_refuse": True, "require_sources": False})

    seven = [item for item in ordered if re.search(BRANDS["세븐일레븐"][0], item["name"], re.I)]
    if seven and all(not known_field(item.get("hours", "")) for item in seven):
        add("missing.seven.hours", "missing_information", "세븐일레븐 영업시간 알려줘",
            expect_for(seven, unknown_fields=[{"id": item["id"], "slot": "hours"} for item in seven], no_clock_claims=True))
    for slot in ("hours", "contact"):
        absent = [item for item in ordered if not known_field(item.get(slot, ""))]
        if absent:
            item = absent[0]
            add(f"missing.{slot}.{item['id']}", "missing_information", f"{entity(item)} {'영업시간' if slot == 'hours' else '전화번호'} 알려줘",
                expect_for([item], unknown_fields=[{"id": item["id"], "slot": slot}],
                    **({"no_phone_claims": True} if slot == "contact" else {"no_clock_claims": True})))
    for index, item in enumerate(hours[:2]):
        add(f"current.{item['id']}", "current_status", f"{entity(item)} 지금 문 열었어?",
            expect_for([item], facts=facts_for(item, ("hours",)), must_disclaim_live=True))
    convenience = [item for item in ordered if item["category"] == "편의점"]
    if convenience:
        add("hours.convenience.24", "hours_boundary", "편의점 중 24시간 하는 곳은 어디야? 지금 열려 있는지도 알려줘",
            expect_for(convenience, must_disclaim_live=True,
                facility_facts=[{"id": item["id"], "facts": facts_for(item, ("hours",))} for item in convenience]))
    explicit_saturday = [item for item in hours if saturday_oracle(item["hours"])["kind"] != "unknown"][:2]
    unknown_saturday = [item for item in hours if saturday_oracle(item["hours"])["kind"] == "unknown"][:2]
    for item in explicit_saturday + unknown_saturday:
        oracle = saturday_oracle(item["hours"])
        expected = expect_for([item], saturday=oracle)
        add(f"hours.{item['id']}.saturday", "hours_boundary", f"{entity(item)} 토요일에는 몇 시까지 해?", expected)

    by_code = {item["code"]: item for item in ordered}
    mixed_notice_added = False
    for notice in snapshot["notices"][:3]:
        related = [by_code[code] for code in notice.get("restaurant_codes", []) if code in by_code]
        if related:
            add(f"notice.restaurant.{notice['id']}", "restaurant_notice",
                f"{entity(related[0])} 관련 '{notice['title']}' 공지 알려줘",
                {"required_source_urls": [notice["url"]]})
            if not mixed_notice_added and notice.get("title") and notice.get("text"):
                item = next((item for item in related if known_field(item.get("location", ""))), None)
                if item:
                    facts = facts_for(item, ("location", "contact"))
                    unknown = ([{"id": item["id"], "slot": "contact", "name": item["name"],
                        "scoped": True, "snapshot_missing": True}] if not known_field(item.get("contact", "")) else [])
                    add(f"multi.notice.{notice['id']}.one", "mixed_notice",
                        f"{entity(item)} 위치와 전화번호, 관련 '{notice['title']}' 공지도 같이 알려줘",
                        expect_for([item], facts=facts, required_source_urls=[notice["url"]],
                            facility_facts=[{"id": item["id"], "facts": facts}], unknown_fields=unknown))
                    other = next((candidate for candidate in contact if candidate["id"] != item["id"] and candidate["name"] != item["name"] and known_field(candidate.get("location", ""))), None)
                    if other:
                        pair = [item, other]
                        grouped = [{"id": target["id"], "facts": facts_for(target, ("location", "contact"))} for target in pair]
                        add(f"multi.notice.{notice['id']}.two", "mixed_notice",
                            f"{entity(item)}와 {entity(other)}의 위치와 전화번호를 각각 알려주고, '{notice['title']}' 공지도 함께 알려줘",
                            expect_for(pair, facts=[fact for group in grouped for fact in group["facts"]],
                                facility_facts=grouped, required_source_urls=[notice["url"]], unknown_fields=unknown))
                    mixed_notice_added = True
    all_public = "\n".join(item["name"] + " " + item.get("description", "") + " " + " ".join(item.get("notices", [])) for item in ordered)
    all_public += "\n" + "\n".join(" ".join(item.get(field, "") for field in ("name", "category", "description", "notice")) for item in records)
    if not re.search(r"약국|pharmacy", all_public, re.I):
        add("missing.pharmacy", "missing_information", "학교에 약국 있어?",
            {"modes": ["retrieval", "openai", "insufficient", "clarification"], "require_sources": False,
                "must_acknowledge_unknown": True, "no_positive_pharmacy_claim": True})
    if not re.search(r"cubic", all_public, re.I):
        add("scope.cubic", "entity_boundary", "CUBIC이라는 매장 전화번호 알려줘",
            {"modes": ["insufficient", "clarification"], "require_sources": False,
                "allowed_facility_ids": [], "no_phone_claims": True})
    for name, question in (("우리은행", "우리은행 전화번호 알려줘"), ("무지개서점", "무지개서점 위치랑 영업시간 알려줘")):
        if normalized(name) not in normalized(all_public):
            add("scope.unknown." + name, "entity_boundary", question,
                {"must_refuse": True, "require_sources": False, "allowed_facility_ids": []})
    return cases


def _facility_ids(sources, code_map=None):
    result = set()
    for source in sources:
        metadata = source.get("metadata") if isinstance(source.get("metadata"), dict) else {}
        identifier = metadata.get("facility_id")
        if str(identifier).isdigit():
            result.add(int(identifier))
        url = source.get("url") if isinstance(source.get("url"), str) else ""
        try:
            query = parse_qs(urlparse(url).query)
        except ValueError:
            query = {}
        for value in query.get("id", []):
            if value.isdigit() and "/facilities/" in url:
                result.add(int(value))
        related = metadata.get("related_codes") if isinstance(metadata.get("related_codes"), list) else []
        for code in [source.get("restaurant_code"), *related]:
            if isinstance(code, str) and code in (code_map or {}):
                result.add(code_map[code])
    return result


def evaluate_response(case, response, *, step_index=0, http_status=200):
    """Evaluate facts/sets/provenance, never an exact model sentence."""
    expected = case["steps"][step_index].get("expect", {})
    checks = []

    def check(name, passed, expected_value=None, actual=None):
        item = {"name": name, "passed": bool(passed)}
        if expected_value is not None:
            item["expected"] = expected_value
        if actual is not None:
            item["actual"] = actual
        checks.append(item)

    check("http_status", http_status == 200, 200, http_status)
    if not isinstance(response, dict):
        check("response_object", False)
        return {"passed": False, "checks": checks}
    answer = response.get("answer", "")
    check("nonempty_answer", isinstance(answer, str) and bool(answer.strip()))
    answer = answer if isinstance(answer, str) else ""
    sources = response.get("sources") if isinstance(response.get("sources"), list) else []
    images = response.get("images") if isinstance(response.get("images"), list) else []
    check("sources_array", isinstance(response.get("sources"), list))
    check("images_array", isinstance(response.get("images"), list))
    sources = [source for source in sources if isinstance(source, dict)]
    ids = [source.get("id") for source in sources]
    declared = {identifier for identifier in ids if isinstance(identifier, str)}
    inline = set(re.findall(r"\[(S\d+)\]", answer))
    check("unique_valid_source_ids", len(ids) == len(declared) and all(isinstance(identifier, str) and SOURCE_ID.fullmatch(identifier) for identifier in ids))
    check("no_fabricated_citations", inline <= declared, sorted(declared), sorted(inline))
    check("returned_sources_are_cited", declared <= inline, sorted(declared), sorted(inline))
    check("official_source_urls", all(official_url(source.get("url")) for source in sources))
    if expected.get("must_refuse"):
        check("grounded_refusal", response.get("mode") in {"insufficient", "clarification"}, ["insufficient", "clarification"], response.get("mode"))
        check("refusal_has_no_sources_or_images", not sources and not images)
    else:
        check("answer_mode", response.get("mode") in expected.get("modes", ["openai", "retrieval"]), expected.get("modes", ["openai", "retrieval"]), response.get("mode"))
        if expected.get("require_sources", True):
            check("sources_present", bool(sources))
    code_map = expected.get("facility_codes", {})
    actual_ids = _facility_ids(sources, code_map)
    if "source_facility_ids" in expected:
        check("required_facility_sources", set(expected["source_facility_ids"]) <= actual_ids, expected["source_facility_ids"], sorted(actual_ids))
    if "allowed_facility_ids" in expected:
        allowed = set(expected["allowed_facility_ids"])
        unrelated = []
        for source in sources:
            direct = _facility_ids([source])
            related = _facility_ids([source], code_map)
            if (direct and not direct <= allowed) or (not direct and related and not related & allowed):
                unrelated.append(source.get("id"))
        # A shared official notice can legitimately cover the asked facility
        # and other branches together; it is relevant when those sets overlap.
        check("no_other_facility_sources", not unrelated, expected["allowed_facility_ids"], sorted(actual_ids))
    if "required_source_urls" in expected:
        check("required_source_urls", set(expected["required_source_urls"]) <= {source.get("url") for source in sources}, expected["required_source_urls"], [source.get("url") for source in sources])
    if "required_dates" in expected:
        actual_dates = {source.get("date") for source in sources}
        check("required_source_dates", set(expected["required_dates"]) <= actual_dates, expected["required_dates"], sorted(value for value in actual_dates if value))
    directory = response.get("directory") if isinstance(response.get("directory"), dict) else {}
    if "directory_ids" in expected:
        actual_records = directory.get("records") if isinstance(directory.get("records"), list) else []
        record_ids = [item.get("id") for item in actual_records if isinstance(item, dict)]
        check("exact_facility_set", set(record_ids) == set(expected["directory_ids"]), expected["directory_ids"], record_ids)
        check("exact_count", directory.get("count") == len(expected["directory_ids"]) and len(record_ids) == len(set(record_ids)), len(expected["directory_ids"]), directory.get("count"))
        body_count = _primary_count(answer)
        check("body_aggregate_count", body_count == len(expected["directory_ids"]), len(expected["directory_ids"]), body_count)
    if "directory_categories" in expected:
        check("exact_category_counts", directory.get("categories") == expected["directory_categories"], expected["directory_categories"], directory.get("categories"))
        body_categories = _category_body_counts(answer, expected["directory_categories"])
        check("body_category_counts", body_categories == expected["directory_categories"], expected["directory_categories"], body_categories)
    if expected.get("scope_label_required"):
        check("catalog_scope_disclosed", directory.get("scope") == "official_portal" and bool(re.search(r"공식|공개|등록", answer)))
    for index, name in enumerate(expected.get("mention_names", [])):
        check(f"listed_name_{index}", normalized(name) in normalized(answer), name)
    for index, fact in enumerate(expected.get("facts", [])):
        passed, actual = _fact_result(fact, answer)
        check(f"fact_{index}_{fact['kind']}", passed, fact, actual)
    for item in expected.get("facility_facts", []):
        owned_sources = [source for source in sources if item["id"] in _facility_ids([source], code_map)]
        evidence = "\n".join(source.get("text", "") for source in owned_sources if isinstance(source.get("text"), str))
        for index, fact in enumerate(item["facts"]):
            passed, actual = _fact_result(fact, evidence)
            check(f"facility_{item['id']}_source_fact_{index}_{fact['kind']}", passed, fact, actual)
        if len(expected["facility_facts"]) > 1:
            scoped_answer = _facility_answer_scopes(answer, sources, code_map, item["id"])
            for index, fact in enumerate(item["facts"]):
                passed, actual = _fact_result(fact, scoped_answer)
                check(f"facility_{item['id']}_answer_fact_{index}_{fact['kind']}", passed, fact, actual)
    for item in expected.get("unknown_fields", []):
        owned = [source for source in sources if item["id"] in _facility_ids([source], code_map)]
        evidence = "\n".join(source.get("text", "") for source in owned if isinstance(source.get("text"), str))
        label = {"hours": r"영업|운영.{0,5}시간", "contact": r"전화|연락", "location": r"위치|주소"}[item["slot"]]
        sentences = re.split(r"\n|[.!?](?:\s|$)", evidence)
        if item.get("scoped") and item.get("snapshot_missing"):
            # The independent, full public snapshot establishes the missing
            # field. An absent field need not be invented as a quoted source
            # line. Its answer must still identify this facility explicitly,
            # retain its real profile source, and avoid borrowing another
            # facility's number. Normal source-based checks stay unchanged.
            scoped = _facility_answer_scopes(answer, sources, code_map, item["id"])
            named = [line for line in answer.splitlines() if item.get("name")
                and normalized(item["name"]) in normalized(line) and re.search(label, line)]
            scoped = "\n".join([scoped, *named])
            unknown_lines = [line for line in scoped.splitlines() if re.search(label, line) and _acknowledges_unknown(line)]
            invented = (phone_values(scoped) if item["slot"] == "contact" else
                clock_values(scoped) if item["slot"] == "hours" else set())
            check(f"facility_{item['id']}_unknown_{item['slot']}", bool(owned and unknown_lines) and not invented)
        else:
            check(f"facility_{item['id']}_unknown_{item['slot']}", _acknowledges_unknown(answer)
                and any(re.search(label, sentence) and _acknowledges_unknown(sentence) for sentence in sentences))
    if expected.get("must_acknowledge_unknown"):
        check("uncertainty_disclosed", _acknowledges_unknown(answer))
    if expected.get("no_phone_claims"):
        check("no_invented_phone", not phone_values(answer), [], sorted(phone_values(answer)))
    if expected.get("no_clock_claims"):
        check("no_invented_hours", not clock_values(answer), [], sorted(clock_values(answer)))
    if expected.get("must_disclaim_live"):
        disclaimer = bool(re.search(r"(?:실시간|현재|지금).{0,35}(?:확인.{0,8}(?:없|못|어려)|보장.{0,8}(?:않|아니|못|없)|미보장)", answer, re.S))
        check("live_status_not_guaranteed", disclaimer)
        unsupported_open = re.search(r"(?:지금|현재).{0,12}(?:영업\s*중(?:이에요|입니다)|열려\s*있(?:어요|습니다)|문을?\s*열었(?:어요|습니다))", answer)
        check("no_unverified_current_open_claim", not unsupported_open)
    if expected.get("no_positive_pharmacy_claim"):
        positive = re.search(r"약국.{0,15}(?:있어요|있습니다|위치해|운영\s*중|운영합니다)", answer)
        check("no_unsupported_pharmacy_presence", not positive)
    if "saturday" in expected:
        fact = expected["saturday"]
        if fact["kind"] == "unknown":
            check("saturday_not_inferred", _acknowledges_unknown(answer))
        elif fact["kind"] == "closed":
            check("saturday_closed", bool(re.search(r"휴무|휴점|미운영|열지|운영.{0,8}않", answer)))
        else:
            passed, actual = _fact_result(fact, answer)
            check("saturday_schedule", passed, fact, actual)
    check("image_count_limit", len(images) <= 6, 6, len(images))
    photo_errors = []
    for image in images:
        if not isinstance(image, dict) or not isinstance(image.get("url"), str) or not LOCAL_IMAGE.fullmatch(image["url"]):
            photo_errors.append("invalid_local_image")
            continue
        matching = [source for source in sources if isinstance(source.get("id"), str) and source["id"] in inline and source.get("url") == image.get("source_url")
            and any(isinstance(candidate, dict) and candidate.get("url") == image["url"]
                for candidate in (source.get("images") if isinstance(source.get("images"), list) else []))]
        if not matching:
            photo_errors.append("image_not_owned_by_cited_source")
    check("images_only_from_cited_sources", not photo_errors, [], photo_errors)
    if expected.get("images_required"):
        check("known_photo_displayed", bool(images))
    if "allowed_images" in expected:
        urls = [image.get("url") for image in images if isinstance(image, dict)]
        check("only_relevant_facility_photos", set(urls) <= set(expected["allowed_images"]), expected["allowed_images"], urls)
    return {"passed": all(item["passed"] for item in checks), "checks": checks}


def _capture_map(captured):
    if isinstance(captured, dict) and isinstance(captured.get("results"), list):
        return {item["case_id"]: item.get("steps", []) for item in captured["results"]}
    if isinstance(captured, dict):
        return captured
    raise ValueError("Responses must be a case-ID mapping or a captured evaluation report")


def evaluate_capture(cases, captured):
    mapping, results = _capture_map(captured), []
    for case in cases:
        values = mapping.get(case["id"])
        if isinstance(values, dict):
            values = [values]
        steps = []
        for index, step in enumerate(case["steps"]):
            value = values[index] if isinstance(values, list) and index < len(values) else None
            if isinstance(value, dict) and "actual" in value:
                actual, status = value["actual"], value.get("http_status", 200)
            else:
                actual, status = value, 200 if value is not None else 0
            evaluation = evaluate_response(case, actual, step_index=index, http_status=status)
            steps.append({"question": step["question"], "http_status": status, "actual": actual, **evaluation})
        results.append({"case_id": case["id"], "category": case["category"], "passed": all(step["passed"] for step in steps), "steps": steps})
    return results


class ApiClient:
    """A fresh QA cookie jar; it never imports the user's existing session."""
    def __init__(self, base_url, timeout=180):
        parsed = urlparse(base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password:
            raise ValueError("Base URL must be HTTP(S) without embedded credentials")
        self.base_url, self.timeout = base_url.rstrip("/"), timeout
        self.opener = build_opener(HTTPCookieProcessor(CookieJar()))

    def request(self, path, payload=None):
        body = json.dumps(payload, ensure_ascii=False).encode() if payload is not None else None
        request = Request(self.base_url + path, data=body,
            headers={"Content-Type": "application/json", "Accept": "application/json"}, method="POST" if body is not None else "GET")
        try:
            response = self.opener.open(request, timeout=self.timeout)
        except HTTPError as error:
            response = error
        with response:
            raw = response.read(8 * 1024 * 1024 + 1)
            if len(raw) > 8 * 1024 * 1024:
                raise ValueError("API response exceeds evaluation capture limit")
            try:
                value = json.loads(raw)
            except ValueError:
                value = {"detail": "API did not return JSON"}
            return response.status, value


def run_live(cases, base_url, *, timeout=180, progress=None, client_factory=ApiClient):
    client = client_factory(base_url, timeout=timeout)
    status, health = client.request("/healthz")
    if status != 200 or health.get("status") != "ok":
        raise ValueError("API health check failed")
    captured = {}
    for index, case in enumerate(cases, 1):
        steps = []
        try:
            status, conversation = client.request("/api/conversations", {})
            if status != 200 or not isinstance(conversation.get("id"), str):
                raise ValueError("Could not create an isolated QA conversation")
            for step in case["steps"]:
                status, actual = client.request("/api/chat", {"message": step["question"],
                    "conversation_id": conversation["id"], "turn_id": str(uuid4())})
                steps.append({"http_status": status, "actual": actual})
        except (URLError, OSError, ValueError) as error:
            steps.append({"http_status": 0, "actual": {"detail": type(error).__name__}})
        captured[case["id"]] = steps
        if progress:
            result = evaluate_capture([case], {case["id"]: steps})[0]
            progress(index, len(cases), case["id"], result["passed"])
    return evaluate_capture(cases, captured)


def make_report(cases, results, *, ground_truth=None, mode="prepared"):
    coverage = [case["facility_coverage"] for case in cases if "facility_coverage" in case]
    evaluated_ids = {item["case_id"] for item in results}
    return {"schema_version": SCHEMA_VERSION, "generated_at": datetime.now(timezone.utc).isoformat(), "mode": mode,
        "ground_truth": ground_truth or {}, "cases": cases, "results": results,
        "summary": {"planned": len(cases), "evaluated": len(results), "passed": sum(item["passed"] for item in results),
            "failed": sum(not item["passed"] for item in results), "by_category": dict(Counter(case["category"] for case in cases)),
            **({"facility_coverage": {"planned_ids": sorted(item["id"] for item in coverage),
                "evaluated_ids": sorted(case["facility_coverage"]["id"] for case in cases if "facility_coverage" in case and case["id"] in evaluated_ids),
                "available_slots": dict(Counter(slot for item in coverage for slot in item["available"])),
                "missing_slots": dict(Counter(slot for item in coverage for slot in item["missing"]))}} if coverage else {})}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--database", type=Path, help="Public source SQLite file, opened read-only")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--base-url", help="Explicit opt-in to live API requests; creates fresh QA chats")
    modes.add_argument("--responses", type=Path, help="Offline captured report or case-ID response mapping")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--category", action="append", help="Evaluate only these case categories (repeatable)")
    parser.add_argument("--max-cases", type=int, default=0, help="Explicit limit; zero keeps all generated cases")
    parser.add_argument("--all-facilities", action="store_true", help="Append every facility's public/missing detail fields and context/boundary cases")
    parser.add_argument("--timeout", type=float, default=180)
    args = parser.parse_args(argv)
    try:
        captured = json.loads(args.responses.read_text(encoding="utf-8")) if args.responses else None
        if captured is not None and isinstance(captured, dict) and isinstance(captured.get("cases"), list):
            cases, truth = captured["cases"], captured.get("ground_truth", {})
        else:
            if args.database is None:
                parser.error("--database is required unless replaying a complete captured report")
            snapshot = load_public_snapshot(args.database)
            cases = build_cases(snapshot, all_facilities=args.all_facilities)
            truth = {"source_url": snapshot["catalog"]["url"], "fetched_at": snapshot["catalog"]["fetched_at"],
                "facility_count": len(snapshot["catalog"]["records"]), "profile_count": len(snapshot["profiles"])}
        if args.category:
            cases = [case for case in cases if case["category"] in args.category]
        if args.max_cases < 0 or args.timeout <= 0:
            parser.error("--max-cases must be nonnegative and --timeout must be positive")
        if args.max_cases:
            cases = cases[:args.max_cases]
        if not cases:
            raise ValueError("No evaluation cases match the selected inputs")
        if args.base_url:
            results = run_live(cases, args.base_url, timeout=args.timeout,
                progress=lambda index, total, identifier, passed: print(f"{index}/{total} {identifier}: {'PASS' if passed else 'FAIL'}", file=sys.stderr, flush=True))
            mode = "live"
        elif captured is not None:
            results, mode = evaluate_capture(cases, captured), "offline_replay"
        else:
            results, mode = [], "prepared"
        report = make_report(cases, results, ground_truth=truth, mode=mode)
        serialized = json.dumps(report, ensure_ascii=False, indent=2) + "\n"
        if args.output:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            args.output.write_text(serialized, encoding="utf-8")
        else:
            print(serialized, end="")
        return 1 if report["summary"]["failed"] else 0
    except (OSError, ValueError, sqlite3.Error) as error:
        print(f"Evaluation input/execution failed: {type(error).__name__}: {error}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())

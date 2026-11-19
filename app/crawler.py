"""Collect public, dated ERICA cafeteria menus without chatbot dependencies.

Adapted from playlistSDY/efoo-kakao-bot, commit
0900e921556eb020a44ea37e3f580a6a1cf28fc6. See docs/provenance.md.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from urllib.parse import parse_qs, urlencode, urljoin, urlparse

import requests
from bs4 import BeautifulSoup

from app.domain import MEAL_TYPES, RESTAURANTS


BASE_URL = "https://life.hanyang.ac.kr"
CRAWLER_VERSION = "efoo-web-life-v3"
_PLACEHOLDERS = ("no-img", "no_image", "no-image", "noimage", "default-img", "default_image", "placeholder")


@dataclass(frozen=True)
class CrawlResult:
    code: str
    name: str
    date: str
    source_url: str
    fetched_at: str
    html: str
    meals: list[dict]
    restaurant_info: dict = field(default_factory=dict)
    weekly_meals: dict[str, list[dict]] = field(default_factory=dict)
    page_data: dict = field(default_factory=dict)
    assets: list[dict] = field(default_factory=list)


def _image_url(value: str | None, base_url: str) -> str:
    if not value or not value.strip():
        return ""
    url = urljoin(base_url.rstrip("/") + "/", value.strip())
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return ""
    filename = parsed.path.rsplit("/", 1)[-1].lower()
    return "" if any(marker in filename for marker in _PLACEHOLDERS) else url


def parse_meal_html(
    html: str,
    *,
    expected_date: date,
    expected_facility_id: str,
    base_url: str = BASE_URL,
) -> tuple[str, list[dict]]:
    """Return verified official name and menu cards; invalid pages raise.

    An explicitly empty official menu is a successful empty list. A changed
    selector, maintenance page, wrong date, or wrong facility is an error.
    """
    soup = BeautifulSoup(html, "html.parser")
    name_tag = soup.select_one(".fac-name")
    section = soup.select_one(".menu-section")
    if not name_tag or not name_tag.get_text(strip=True) or section is None:
        raise ValueError("식단 페이지 구조를 확인할 수 없습니다.")
    daily_pane = section.select_one("#view-pane-daily")
    picker = section.select_one("#menu-date-picker")
    if daily_pane is not None:
        # The November 2026 page renders the entire week plus a second,
        # selected-day pane. Only that pane belongs to the requested date.
        selected_days = daily_pane.select(".week-day-pill.is-selected")
        selected = selected_days[0] if len(selected_days) == 1 else None
        match = re.fullmatch(
            r"\s*selectDay\(\s*(['\"])(\d{4}-\d{2}-\d{2})\1\s*\)\s*;?\s*",
            selected.get("onclick", "") if selected else "",
        )
        selected_date = match.group(2) if match else ""
        content = daily_pane
        card_class = "menu-card-v"
    elif picker is not None:
        selected_date = picker.get("value", "")
        content = section
        card_class = "menu-item"
    else:
        raise ValueError("식단 페이지 구조를 확인할 수 없습니다.")
    try:
        actual_date = date.fromisoformat(selected_date)
    except ValueError as exc:
        raise ValueError("식단 페이지의 날짜가 올바르지 않습니다.") from exc
    if actual_date != expected_date:
        raise ValueError(f"요청 날짜와 식단 날짜가 다릅니다: {expected_date} != {actual_date}")

    if daily_pane is not None:
        # Verify the requested facility and date independently of the selected
        # pill, so a redirect or a mismatched page cannot pollute the index.
        page_links = soup.select('link[rel="alternate"][hreflang="ko"][href], a.hy-lang-btn.is-active[lang="ko"][href]')
        if not page_links:
            raise ValueError("식단 페이지의 식당 식별 정보를 확인할 수 없습니다.")
        for link in page_links:
            query = parse_qs(urlparse(link["href"]).query)
            if query.get("id") != [str(expected_facility_id)]:
                raise ValueError(f"요청한 식당의 식단이 아닙니다: facility_id={expected_facility_id}")
            if query.get("date") != [expected_date.isoformat()]:
                raise ValueError("요청 날짜와 식단 페이지의 날짜가 다릅니다.")
    else:
        # Legacy dated links identify the facility even for an empty menu.
        date_links = section.select(".date-scroll-item[href]")
        if not date_links or any(
            parse_qs(urlparse(link["href"]).query).get("id") != [str(expected_facility_id)]
            for link in date_links
        ):
            raise ValueError(f"요청한 식당의 식단이 아닙니다: facility_id={expected_facility_id}")

    groups = content.select(".menu-group")
    if not groups:
        empty = content.select_one(".fac-empty-msg")
        if not empty or "등록된 식단이 없습니다" not in empty.get_text(strip=True):
            raise ValueError("식단 목록 또는 빈 식단 안내를 찾을 수 없습니다.")

    meals = []
    for group in groups:
        title = group.select_one(".menu-group__title")
        meal_type = title.get_text(strip=True) if title else ""
        if meal_type not in MEAL_TYPES:
            raise ValueError(f"알 수 없는 식사 구분: {meal_type}")
        cards = group.select(f".{card_class}")
        if not cards:
            raise ValueError(f"{meal_type} 메뉴 카드 구조를 확인할 수 없습니다.")
        for card in cards:
            main = card.select_one(f".{card_class}__name")
            name = main.get_text(" ", strip=True) if main else ""
            if not name:
                raise ValueError(f"{meal_type} 메뉴 이름을 찾을 수 없습니다.")
            if re.search(r"운영합니다|코너만.*운영|휴무|휴업|미운영|운영\s*없음|식단\s*없음", name):
                continue
            description = card.select_one(f".{card_class}__desc")
            lines = [name] + (description.get_text("\n", strip=True).splitlines() if description else [])
            tags, items = [], []
            for line in lines:
                for tag in re.findall(r"\[([^\]]+)\]", line):
                    if tag not in tags:
                        tags.append(tag)
                item = re.sub(r"\s+", " ", re.sub(r"\[[^\]]+\]", "", line)).strip()
                if item:
                    items.append(item)
            if not items:
                raise ValueError(f"{meal_type} 메뉴 내용이 비어 있습니다.")
            price = card.select_one(f".{card_class}__price")
            image = card.select_one(f"img.{card_class}__img")
            meals.append({
                "meal_type": meal_type, "items": items, "tags": tags,
                "price": price.get_text(strip=True) if price else "",
                "image": _image_url(image.get("src") if image else None, base_url),
            })
    return name_tag.get_text(strip=True), meals


def _text(tag) -> str:
    """Keep source line breaks (including <br>) without interpreting their meaning."""
    return tag.get_text("\n", strip=True) if tag else ""


def _public_url(value: str | None, base_url: str) -> str:
    if not value or not value.strip():
        return ""
    absolute = urljoin(base_url.rstrip("/") + "/", value.strip())
    return absolute if urlparse(absolute).scheme in {"http", "https"} else ""


def _attrs(tag) -> dict:
    return dict(tag.attrs) if tag else {}


def _card_data(card, *, meal_type, day, base_url, prefix) -> dict:
    selectors = {
        "menu-card-v": (".menu-card-v__name", ".menu-card-v__desc", ".menu-card-v__price"),
        "menu-item": (".menu-item__name", ".menu-item__desc", ".menu-item__price"),
        "weekly-text-item": (".weekly-item-name", ".weekly-item-desc", ".weekly-item-price"),
        "weekly-m-item": (".weekly-m-item__name", ".weekly-m-item__desc", ".weekly-m-item__price"),
    }
    name_selector, description_selector, price_selector = selectors[prefix]
    main, description, price = (card.select_one(selector) for selector in (name_selector, description_selector, price_selector))
    raw_name = main.get_text(" ", strip=True) if main else ""
    lines = [raw_name] + (_text(description).splitlines() if description else [])
    tags, items = [], []
    for line in lines:
        for tag in re.findall(r"\[([^\]]+)\]", line):
            if tag not in tags:
                tags.append(tag)
        item = re.sub(r"\s+", " ", re.sub(r"\[[^\]]+\]", "", line)).strip()
        if item:
            items.append(item)
    actions = []
    for control in card.select("button, a[href], input, .weekly-item-buy"):
        label = control.get_text(" ", strip=True) or control.get("value", "")
        actions.append({"tag": control.name, "label": label,
                        "disabled": control.has_attr("disabled") or control.get("aria-disabled") == "true",
                        "url": _public_url(control.get("href"), base_url),
                        "attributes": _attrs(control), "html": str(control)})
    images = [{"url": _public_url(img.get("src") or img.get("data-src"), base_url),
               "alt": img.get("alt", ""), "attributes": _attrs(img)} for img in card.select("img")]
    sale_status = [action["label"] for action in actions if re.search(r"판매|품절|구매|매진", action["label"])]
    return {"date": day, "meal_type": meal_type, "items": items, "tags": tags,
            "price": price.get_text(strip=True) if price else "",
            "image": _image_url(images[0]["url"], base_url) if images else "",
            "images": images, "raw_name": raw_name, "name_html": str(main) if main else "",
            "description": _text(description), "description_html": str(description) if description else "",
            "price_html": str(price) if price else "", "actions": actions,
            "sale_status": " / ".join(sale_status), "attributes": _attrs(card), "html": str(card),
            "is_notice": bool(re.search(r"운영합니다|코너만.*운영|휴무|휴업|미운영|운영\s*없음|식단\s*없음", raw_name))}


def _visible_week_days(soup, expected_date):
    days = []
    for pill in soup.select("#view-pane-daily .week-day-pill[onclick]"):
        match = re.fullmatch(r"\s*selectDay\(\s*(['\"])(\d{4}-\d{2}-\d{2})\1\s*\)\s*;?\s*", pill["onclick"])
        if match:
            days.append(date.fromisoformat(match.group(2)))
    if len(days) == 7 and len(set(days)) == 7 and all(days[i] == days[0] + timedelta(days=i) for i in range(7)):
        return days
    week_range = _text(soup.select_one(".week-range-text"))
    match = re.search(r"(20\d{2})\.(\d{1,2})\.(\d{1,2})", week_range)
    if match:
        start = date(*map(int, match.groups()))
        return [start + timedelta(days=i) for i in range(7)]
    return []


def parse_facility_html(html: str, *, expected_facility_id: str, base_url: str = BASE_URL) -> dict:
    """Read any public facility detail page without requiring menus or a date."""
    soup = BeautifulSoup(html, "html.parser")
    name_tag = soup.select_one(".fac-name, .shop-name")
    if not name_tag or not _text(name_tag):
        raise ValueError("시설 상세 페이지 구조를 확인할 수 없습니다.")
    name = name_tag.get_text(" ", strip=True)
    identity_links = soup.select('link[rel="alternate"][hreflang="ko"][href], a.hy-lang-btn.is-active[lang="ko"][href]')
    if not identity_links:
        identity_links = soup.select(".menu-section .date-scroll-item[href]")
    if not identity_links or any(
        parse_qs(urlparse(link["href"]).query).get("id") != [str(expected_facility_id)]
        for link in identity_links
    ):
        raise ValueError(f"요청한 시설의 상세 페이지가 아닙니다: facility_id={expected_facility_id}")
    info_rows = []
    for row in soup.select(".info-table tr"):
        label, value = row.select_one(".info-label"), row.select_one(".info-value")
        if label is None or value is None:
            cells = row.find_all(["th", "td"], recursive=False)
            label, value = (cells[0], cells[1]) if len(cells) >= 2 else (None, None)
        if label is not None and value is not None:
            info_rows.append({"label": _text(label), "value": _text(value), "html": value.decode_contents()})
    notices = []
    for notice in soup.select(".fac-notice-box, .shop-notice-box"):
        content = notice.select_one(".fac-notice-box__text, .shop-notice-box__text") or notice
        notices.append({"label": _text(notice.select_one(".fac-notice-box__label, .shop-notice-box__label")),
                        "text": _text(content), "html": content.decode_contents()})
    description = soup.select_one(".shop-description-area, .shop-description, .fac-description-area")
    cover = soup.select_one(".fac-cover img, .shop-cover img")
    gallery = []
    for image in soup.select('.fac-gallery img, .facility-gallery img, .shop-gallery img, [class*="gallery"] img'):
        entry = {"url": _public_url(image.get("src") or image.get("data-src"), base_url), "alt": image.get("alt", "")}
        if entry["url"] and entry not in gallery:
            gallery.append(entry)
    values = {row["label"]: row["value"] for row in info_rows}
    restaurant_info = {"name": name, "category": _text(soup.select_one(".fac-category, .shop-category")),
        "info_rows": info_rows, "location": values.get("위치", ""),
        "building_no": values.get("건물 번호", values.get("건물번호", "")),
        "contact": values.get("연락처", values.get("전화번호", "")),
        "opening_hours": values.get("영업시간", values.get("운영시간", "")),
        "notices": notices,
        "description": {"text": _text(description), "html": description.decode_contents() if description else ""},
        "cover_image": {"url": _public_url(cover.get("src"), base_url), "alt": cover.get("alt", "")} if cover else {},
        "gallery": gallery}

    images = []
    for image in soup.select("img"):
        role = "page_image"
        if image.find_parent(class_=re.compile(r"^(fac-cover|shop-cover)$")):
            role = "cover"
        elif image.find_parent(class_=re.compile("gallery")):
            role = "gallery"
        elif image.find_parent(class_=re.compile(r"^(menu-card-v|menu-item|weekly-text-item|weekly-m-item)$")):
            role = "menu"
        images.append({"url": _public_url(image.get("src") or image.get("data-src"), base_url),
                       "alt": image.get("alt", ""), "role": role, "attributes": _attrs(image)})
    links = [{"url": _public_url(link.get("href"), base_url), "raw_url": link.get("href", ""),
              "text": link.get_text(" ", strip=True), "tag": link.name, "attributes": _attrs(link)}
             for link in soup.select("a[href], area[href], link[href]")]
    scripts = [{"url": _public_url(script.get("src"), base_url), "text": script.string or script.get_text(),
                "attributes": _attrs(script)} for script in soup.select("script")]
    styles = [{"url": _public_url(style.get("href"), base_url), "text": style.string or style.get_text(),
               "attributes": _attrs(style)} for style in soup.select('link[rel~="stylesheet"], style')]
    assets_by_url = {}
    for image in images:
        if image["url"]:
            assets_by_url.setdefault(image["url"], {"url": image["url"], "role": image["role"], "kind": "image", "alt": image["alt"]})
    for records, role, kind in ((scripts, "script", "script"), (styles, "style", "stylesheet")):
        for record in records:
            if record["url"]:
                assets_by_url.setdefault(record["url"], {"url": record["url"], "role": role, "kind": kind, "alt": ""})
    inline_styles = [{"tag": tag.name, "attributes": _attrs(tag)} for tag in soup.select("[style]")]
    for record in inline_styles:
        for value in re.findall(r"url\(\s*['\"]?([^)'\"]+)['\"]?\s*\)", record["attributes"]["style"]):
            url = _public_url(value, base_url)
            if url:
                assets_by_url.setdefault(url, {"url": url, "role": "background_image", "kind": "image", "alt": ""})
    page_data = {"title": _text(soup.title), "language": soup.html.get("lang", "") if soup.html else "",
        "meta": [_attrs(meta) for meta in soup.select("meta")],
        "headings": [{"tag": h.name, "text": _text(h), "attributes": _attrs(h)}
                     for h in soup.select("h1,h2,h3,h4,h5,h6,.hy-section-title,.fac-name,.shop-name")],
        "text": soup.get_text("\n", strip=True), "links": links, "images": images,
        "scripts": scripts, "styles": styles, "inline_styles": inline_styles,
        "controls": [{"tag": control.name, "text": control.get_text(" ", strip=True), "attributes": _attrs(control)}
                     for control in soup.select("button,input,select,textarea,form")],
        "raw_daily_cards": [], "menu_notices": [], "weekly_menu_notices": [], "visible_week_dates": [],
        "weekly_range_text": _text(soup.select_one(".week-range-text")),
        "sale_status_scope": "captured_page_only"}
    return {"name": name, "restaurant_info": restaurant_info, "page_data": page_data,
            "assets": list(assets_by_url.values())}


def parse_restaurant_page(
    html: str,
    *,
    expected_date: date,
    expected_facility_id: str,
    base_url: str = BASE_URL,
) -> dict:
    """Archive all public restaurant content, while retaining strict menu identity.

    Scripts, links, form controls and HTML are collected as inert audit data;
    this function never follows links, executes scripts, signs in or buys food.
    Sale labels describe the captured page only, not a permanent open status.
    """
    name, legacy_meals = parse_meal_html(html, expected_date=expected_date,
        expected_facility_id=expected_facility_id, base_url=base_url)
    soup = BeautifulSoup(html, "html.parser")
    selected_day = expected_date.isoformat()
    daily = soup.select_one("#view-pane-daily") or soup.select_one(".menu-section")
    daily_cards = []
    for group in daily.select(".menu-group"):
        meal_type = _text(group.select_one(".menu-group__title"))
        for card in group.select(".menu-card-v, .menu-item"):
            prefix = "menu-card-v" if "menu-card-v" in card.get("class", []) else "menu-item"
            daily_cards.append(_card_data(card, meal_type=meal_type, day=selected_day, base_url=base_url, prefix=prefix))
    meals = [card for card in daily_cards if not card["is_notice"]]
    # The strict parser is the authority if a legacy card type is encountered.
    if not daily_cards and legacy_meals:
        meals = legacy_meals
    menu_notices = [card for card in daily_cards if card["is_notice"]]

    public = parse_facility_html(html, expected_facility_id=expected_facility_id, base_url=base_url)
    restaurant_info = public["restaurant_info"]

    days = _visible_week_days(soup, expected_date)
    weekly_meals, weekly_notices = {}, []
    table = soup.select_one("#view-pane-weekly .weekly-table")
    if table and days:
        header_dates = [_text(header) for header in table.select("thead .th-day-date")]
        if header_dates and (len(header_dates) != 7 or any(
            label != f"{day.month:02d}.{day.day:02d}" for label, day in zip(header_dates, days))):
            raise ValueError("주간 식단표의 날짜가 일자 선택 정보와 다릅니다.")
        weekly_meals = {day.isoformat(): [] for day in days}
        for row in table.select("tbody tr"):
            meal_type = _text(row.select_one(".td-meal-label .meal-badge") or row.select_one(".td-meal-label"))
            cells = row.select("td.td-menu-cell")
            if meal_type not in MEAL_TYPES or len(cells) != len(days):
                raise ValueError("주간 식단표의 식사 구분 또는 날짜 열을 확인할 수 없습니다.")
            for day, cell in zip(days, cells):
                for card in cell.select(".weekly-text-item"):
                    parsed = _card_data(card, meal_type=meal_type, day=day.isoformat(), base_url=base_url, prefix="weekly-text-item")
                    (weekly_notices if parsed["is_notice"] else weekly_meals[day.isoformat()]).append(parsed)
    elif days:
        mobile_days = soup.select("#view-pane-weekly .weekly-day-card")
        if mobile_days:
            if len(mobile_days) != len(days):
                raise ValueError("주간 모바일 식단의 날짜 수가 일치하지 않습니다.")
            weekly_meals = {day.isoformat(): [] for day in days}
            for day, day_card in zip(days, mobile_days):
                for block in day_card.select(".weekly-m-meal-block"):
                    meal_type = _text(block.select_one(".weekly-m-meal-label .meal-badge") or block.select_one(".weekly-m-meal-label"))
                    if meal_type not in MEAL_TYPES:
                        raise ValueError("주간 모바일 식단의 식사 구분을 확인할 수 없습니다.")
                    for card in block.select(".weekly-m-item"):
                        parsed = _card_data(card, meal_type=meal_type, day=day.isoformat(), base_url=base_url, prefix="weekly-m-item")
                        (weekly_notices if parsed["is_notice"] else weekly_meals[day.isoformat()]).append(parsed)
    if selected_day in weekly_meals:
        # Daily view carries photos and sale controls omitted from weekly text.
        # Replace that date instead of appending a third duplicate rendering.
        weekly_meals[selected_day] = meals

    page_data = public["page_data"]
    page_data.update(raw_daily_cards=daily_cards, menu_notices=menu_notices,
                     weekly_menu_notices=weekly_notices,
                     visible_week_dates=[day.isoformat() for day in days])
    return {"name": name, "meals": meals, "restaurant_info": restaurant_info,
            "weekly_meals": weekly_meals, "page_data": page_data,
            "menu_notices": menu_notices, "assets": public["assets"]}


class MealCrawler:
    def __init__(self, session: requests.Session | None = None, *, base_url: str = BASE_URL):
        self.session = session or requests.Session()
        self.base_url = base_url.rstrip("/")

    def close(self) -> None:
        self.session.close()

    def fetch(self, restaurant_code: str, target_date: date) -> CrawlResult:
        if restaurant_code not in RESTAURANTS:
            raise ValueError(f"잘못된 식당 코드: {restaurant_code}")
        if not isinstance(target_date, date) or isinstance(target_date, datetime):
            raise ValueError("target_date는 datetime.date여야 합니다.")
        facility_id = RESTAURANTS[restaurant_code]["facility_id"]
        endpoint = f"{self.base_url}/theme/pages/facilities/detail.php"
        params = {"id": facility_id, "date": target_date.isoformat()}
        response = self.session.get(
            endpoint, params=params,
            headers={
                "User-Agent": "EfooWebAcademicCrawler/1.0 (cafeteria menus)",
                "Accept": "text/html,application/xhtml+xml",
                "Accept-Language": "ko-KR,ko;q=0.9",
            },
            timeout=(3.0, 12.0),
        )
        response.raise_for_status()
        response.encoding = "utf-8"
        page = parse_restaurant_page(
            response.text, expected_date=target_date,
            expected_facility_id=facility_id, base_url=self.base_url,
        )
        return CrawlResult(
            code=restaurant_code, name=page["name"], date=target_date.isoformat(),
            source_url=f"{endpoint}?{urlencode(params)}",
            fetched_at=datetime.now(timezone.utc).isoformat(), html=response.text, meals=page["meals"],
            restaurant_info=page["restaurant_info"], weekly_meals=page["weekly_meals"],
            page_data=page["page_data"], assets=page["assets"],
        )

from datetime import date, datetime
from pathlib import Path
import unittest
from unittest.mock import Mock

import requests
from bs4 import BeautifulSoup

from app.crawler import BASE_URL, MealCrawler, parse_meal_html, parse_restaurant_page, parse_facility_html
from app.domain import RESTAURANTS


FIXTURES = Path(__file__).parent / "fixtures"
TARGET_DATE = date(2026, 9, 22)


def fixture(name="student_2026-09-22.html"):
    return (FIXTURES / name).read_text(encoding="utf-8")


def response(html, status=200):
    result = requests.Response()
    result.status_code = status
    result._content = html.encode("utf-8")
    result.encoding = "ISO-8859-1"
    return result


class CrawlerParserTests(unittest.TestCase):
    def parse(self, html, target=TARGET_DATE, facility_id="1"):
        return parse_meal_html(html, expected_date=target, expected_facility_id=facility_id)

    def test_all_facilities_keep_their_actual_meal_groups(self):
        for code, prefix, name, counts in (
            ("re11", "staff", "교직원식당", (0, 2, 0)),
            ("re12", "student", "학생식당", (1, 2, 0)),
            ("re13", "dormitory", "창의인재원식당", (1, 2, 1)),
            ("re15", "incubator", "창업보육센터식당", (0, 1, 1)),
        ):
            with self.subTest(code=code):
                actual_name, meals = self.parse(
                    fixture(f"{prefix}_2026-09-22.html"),
                    facility_id=RESTAURANTS[code]["facility_id"],
                )
                self.assertEqual(actual_name, name)
                self.assertEqual(tuple(sum(m["meal_type"] == t for m in meals) for t in ("조식", "중식", "석식")), counts)

    def test_names_sides_tags_price_and_image_are_preserved(self):
        _, meals = self.parse(fixture())
        breakfast = meals[0]
        self.assertEqual(breakfast["items"], [
            "토스트/단팥빵 Brunch Set", "양배추샐러드&드레싱", "검은콩두유/콘시리얼바",
            "비엔나양송이볶음", "단호박꿀조림", "양념밥",
        ])
        self.assertEqual(breakfast["tags"], ["천원의 아침밥"])
        self.assertEqual(breakfast["price"], "1,000원")
        self.assertEqual(breakfast["image"], f"{BASE_URL}/uploads/facilities/menu_1790033037_6ab1bc8dbf0f3.jpg")
        self.assertEqual(meals[1]["image"], "")
        self.assertEqual(meals[2]["price"], "6,500원")

    def test_explicit_empty_day_is_a_successful_empty_list(self):
        name, meals = self.parse(fixture("student_2026-09-27-empty.html"), target=date(2026, 9, 27))
        self.assertEqual(name, "학생식당")
        self.assertEqual(meals, [])

    def test_invalid_or_changed_page_is_never_treated_as_empty(self):
        for html in (
            "<html><body>점검 중</body></html>",
            fixture().replace('id="menu-date-picker"', 'id="changed-picker"'),
            fixture().replace('value="2026-09-22"', 'value="invalid"'),
            fixture().replace('class="menu-item__name"', 'class="changed-name"'),
            fixture().replace('class="menu-group"', 'class="changed-group"'),
            fixture().replace('class="menu-item"', 'class="changed-card"'),
            fixture().replace('class="menu-group__title">조식', 'class="menu-group__title">알수없음'),
        ):
            with self.subTest(html=html[:50]), self.assertRaises(ValueError):
                self.parse(html)
        with self.assertRaises(ValueError):
            self.parse(fixture("student_2026-09-27-empty.html").replace("오늘 등록된 식단이 없습니다.", "서비스 점검 중입니다."), target=date(2026, 9, 27))

    def test_wrong_date_or_wrong_facility_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "요청 날짜"):
            self.parse(fixture(), target=date(2026, 9, 23))
        with self.assertRaisesRegex(ValueError, "요청한 식당"):
            self.parse(fixture(), facility_id="2")

    def test_notice_main_is_skipped_without_losing_other_menus(self):
        html = fixture().replace("바지락칼국수 Noodle Soup with Clams", "휴무")
        html = html.replace("menu_1790033037_6ab1bc8dbf0f3.jpg", "no-image.png")
        _, meals = self.parse(html)
        self.assertEqual(len(meals), 2)
        self.assertEqual(meals[0]["image"], "")


class CrawlerRequestTests(unittest.TestCase):
    def test_fetch_builds_dated_provenance_and_decodes_korean(self):
        session = Mock()
        session.get.return_value = response(fixture())
        result = MealCrawler(session).fetch("re12", TARGET_DATE)
        self.assertEqual(result.code, "re12")
        self.assertEqual(result.name, "학생식당")
        self.assertEqual(result.date, "2026-09-22")
        self.assertEqual(result.source_url, f"{BASE_URL}/theme/pages/facilities/detail.php?id=1&date=2026-09-22")
        self.assertIsNotNone(datetime.fromisoformat(result.fetched_at).tzinfo)
        self.assertEqual(result.html, fixture())
        self.assertEqual(session.get.call_args.kwargs["params"], {"id": "1", "date": "2026-09-22"})
        self.assertEqual(session.get.call_args.kwargs["timeout"], (3.0, 12.0))

    def test_bad_request_fails_before_network(self):
        session = Mock()
        crawler = MealCrawler(session)
        for code, target in (("re99", TARGET_DATE), ("re12", "2026-09-22"), ("re12", datetime(2026, 9, 22))):
            with self.subTest(code=code, target=target), self.assertRaises(ValueError):
                crawler.fetch(code, target)
        session.get.assert_not_called()

    def test_http_errors_and_timeouts_propagate(self):
        session = Mock()
        session.get.return_value = response("Unavailable", 503)
        with self.assertRaises(requests.HTTPError):
            MealCrawler(session).fetch("re12", TARGET_DATE)
        session.get.side_effect = requests.Timeout("read timed out")
        with self.assertRaises(requests.Timeout):
            MealCrawler(session).fetch("re12", TARGET_DATE)


class CurrentCrawlerParserTests(unittest.TestCase):
    """The current page repeats menus in desktop/mobile weekly panes."""

    target = date(2026, 11, 13)

    def page(self):
        return BeautifulSoup(fixture("student_2026-11-13-live.html"), "html.parser")

    def parse(self, page, target=None, facility_id="1"):
        return parse_meal_html(str(page), expected_date=target or self.target, expected_facility_id=facility_id)

    def test_live_daily_pane_excludes_all_other_days_and_duplicates(self):
        name, meals = self.parse(self.page())
        self.assertEqual(name, "학생식당")
        self.assertEqual(len(meals), 3)
        self.assertEqual([meal["meal_type"] for meal in meals], ["조식", "중식", "중식"])
        self.assertEqual(meals[0]["items"], [
            "돼지고기굴소스볶음 (Stir-fried Pork in Oyster Sauce)",
            "쌀밥/게맛살피망조림/도시락김/도토리묵무침/배추김치",
        ])
        self.assertEqual(meals[0]["tags"], ["천원의 아침밥"])
        self.assertEqual(meals[0]["price"], "1,000원")
        self.assertEqual(meals[1]["price"], "4,500원")
        self.assertEqual(meals[2]["price"], "6,500원")
        self.assertEqual(meals[0]["image"], f"{BASE_URL}/uploads/facilities/menu_1794525724_6af64a1cd2517.jpg")

    def test_current_wrong_date_and_facility_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "요청 날짜"):
            self.parse(self.page(), target=date(2026, 11, 14))
        with self.assertRaisesRegex(ValueError, "요청한 식당"):
            self.parse(self.page(), facility_id="2")
        page = self.page()
        page.select_one('link[hreflang="ko"]')["href"] = "?id=2&date=2026-11-13"
        with self.assertRaisesRegex(ValueError, "요청한 식당"):
            self.parse(page)
        page = self.page()
        page.select_one('link[hreflang="ko"]')["href"] = "?id=1&date=2026-11-12"
        with self.assertRaisesRegex(ValueError, "요청 날짜"):
            self.parse(page)

    def test_current_requires_one_selected_day_and_facility_link(self):
        page = self.page()
        page.select_one(".week-day-pill.is-selected")["class"] = ["week-day-pill"]
        with self.assertRaises(ValueError):
            self.parse(page)
        page = self.page()
        page.select_one(".week-day-pill")["class"] = ["week-day-pill", "is-selected"]
        with self.assertRaises(ValueError):
            self.parse(page)
        page = self.page()
        for link in page.select('link[rel="alternate"][hreflang="ko"], a.hy-lang-btn[lang="ko"]'):
            link.decompose()
        with self.assertRaises(ValueError):
            self.parse(page)

    def test_active_language_links_identify_facility_without_alternate(self):
        page = self.page()
        page.select_one('link[hreflang="ko"]').decompose()
        self.assertEqual(len(self.parse(page)[1]), 3)

    def test_current_explicit_empty_daily_pane_ignores_weekly_menus(self):
        page = self.page()
        daily = page.select_one("#view-pane-daily")
        for group in daily.select(".menu-group"):
            group.decompose()
        empty = page.new_tag("div", attrs={"class": "fac-empty-msg"})
        empty.string = "선택하신 일자(2026-11-13)에 등록된 식단이 없습니다."
        daily.append(empty)
        self.assertEqual(self.parse(page)[1], [])
        empty.decompose()
        with self.assertRaises(ValueError):
            self.parse(page)


class PublicFacilityArchiveTests(unittest.TestCase):
    def page(self, kind="student"):
        return fixture(f"public_{kind}_2026-11-13.html")

    def parse(self, html=None, facility_id="1"):
        return parse_restaurant_page(html or self.page(), expected_date=date(2026, 11, 13), expected_facility_id=facility_id)

    def test_all_four_current_pages_keep_official_metadata_and_seven_day_menus(self):
        for code, kind, weekly_count, location in (
            ("re11", "staff", 10, "학생복지관 3층"),
            ("re12", "student", 15, "학생복지관 2층"),
            ("re13", "dormitory", 25, "창의관 1층"),
            ("re15", "incubator", 10, "창업보육센터 지하1층"),
        ):
            with self.subTest(code=code):
                parsed = self.parse(self.page(kind), RESTAURANTS[code]["facility_id"])
                info = parsed["restaurant_info"]
                self.assertEqual(info["category"], "구내식당")
                self.assertEqual(info["location"], location)
                self.assertTrue(info["opening_hours"])
                self.assertTrue(info["notices"][0]["text"])
                self.assertEqual(len(parsed["weekly_meals"]), 7)
                self.assertEqual(sum(map(len, parsed["weekly_meals"].values())), weekly_count)
                self.assertEqual(parsed["weekly_meals"]["2026-11-13"], parsed["meals"])
                self.assertTrue(info["cover_image"]["url"].startswith(BASE_URL + "/uploads/"))

    def test_hours_contact_and_br_rows_remain_source_values(self):
        parsed = self.parse(self.page("incubator"), "3")
        info = parsed["restaurant_info"]
        self.assertEqual(info["building_no"], "308")
        self.assertEqual(info["contact"], "010-2828-6269")
        self.assertEqual(info["opening_hours"], "중식 11:30~13:30\n석식 17:00~19:00 (18:30배식대마감)")
        row = next(row for row in info["info_rows"] if row["label"] == "영업시간")
        self.assertIn("<br", row["html"])
        self.assertIn("18:30배식대마감", row["value"])
        self.assertIn("등록된 상세 정보가 없습니다", info["description"]["text"])

    def test_weekly_desktop_mobile_daily_renderings_are_not_duplicated(self):
        parsed = self.parse()
        self.assertEqual(len(parsed["weekly_meals"]["2026-11-13"]), 3)
        self.assertEqual(len(parsed["weekly_meals"]["2026-11-09"]), 3)
        self.assertEqual(parsed["weekly_meals"]["2026-11-08"], [])
        self.assertEqual(parsed["weekly_meals"]["2026-11-14"], [])
        self.assertTrue(parsed["weekly_meals"]["2026-11-13"][0]["image"])
        self.assertEqual(parsed["weekly_meals"]["2026-11-09"][0]["raw_name"],
            "[천원의 아침밥] 돼지고기고추장찌개 (Gochujang Stew with Pork)")

    def test_mobile_weekly_cards_are_a_fallback_when_table_is_absent(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(self.page(), "html.parser")
        soup.select_one(".weekly-table").decompose()
        parsed = self.parse(str(soup))
        self.assertEqual(len(parsed["weekly_meals"]), 7)
        self.assertEqual(sum(map(len, parsed["weekly_meals"].values())), 15)
        self.assertEqual(parsed["weekly_meals"]["2026-11-08"], [])

    def test_operational_cards_are_preserved_as_notices_instead_of_lost(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(self.page(), "html.parser")
        soup.select_one("#view-pane-daily .menu-card-v__name").string = "학기중에만 운영합니다"
        soup.select_one(".weekly-item-name").string = "휴무"
        parsed = self.parse(str(soup))
        self.assertEqual(len(parsed["meals"]), 2)
        self.assertEqual(parsed["page_data"]["menu_notices"][0]["raw_name"], "학기중에만 운영합니다")
        self.assertEqual(parsed["page_data"]["weekly_menu_notices"][0]["raw_name"], "휴무")
        self.assertIn("menu-card-v", parsed["page_data"]["menu_notices"][0]["html"])

    def test_sale_labels_disabled_buttons_and_data_attributes_are_inert_observations(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(self.page("dormitory"), "html.parser")
        card = soup.select_one("#view-pane-daily .menu-card-v")
        card["data-official-menu-id"] = "example-public-id"
        parsed = self.parse(str(soup), "4")
        breakfast = parsed["meals"][0]
        self.assertEqual(breakfast["sale_status"], "판매 종료")
        self.assertTrue(breakfast["actions"][0]["disabled"])
        self.assertIn("disabled", breakfast["actions"][0]["attributes"])
        self.assertEqual(breakfast["attributes"]["data-official-menu-id"], "example-public-id")
        self.assertEqual(parsed["page_data"]["sale_status_scope"], "captured_page_only")

    def test_fullpage_audit_keeps_metadata_scripts_styles_links_images_and_hidden_description(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(self.page(), "html.parser")
        title = soup.new_tag("title")
        title.string = "학생식당 공식 상세 안내"
        soup.head.append(title)
        meta = soup.new_tag("meta", attrs={"name": "description", "content": "공개 시설 안내"})
        soup.head.append(meta)
        gallery = soup.new_tag("div", attrs={"class": "fac-gallery"})
        gallery.append(soup.new_tag("img", attrs={"src": "/gallery.jpg", "alt": "시설 내부"}))
        soup.main.append(gallery)
        parsed = self.parse(str(soup))
        audit = parsed["page_data"]
        self.assertEqual(audit["title"], title.string)
        self.assertIn({"name": "description", "content": "공개 시설 안내"}, audit["meta"])
        self.assertTrue(audit["headings"])
        self.assertTrue(any("selectDay" in script["text"] for script in audit["scripts"]))
        self.assertTrue(any(style["url"].endswith("detail.css") for style in audit["styles"]))
        self.assertTrue(any("auth/login.php" in link["url"] for link in audit["links"]))
        self.assertIn("등록된 상세 정보가 없습니다", audit["text"])
        self.assertEqual(parsed["restaurant_info"]["gallery"], [{"url": BASE_URL + "/gallery.jpg", "alt": "시설 내부"}])
        self.assertEqual(len(parsed["assets"]), len({asset["url"] for asset in parsed["assets"]}))
        self.assertTrue(any(asset["role"] == "gallery" for asset in parsed["assets"]))
        self.assertTrue(any(asset["role"] == "menu" for asset in parsed["assets"]))

    def test_nonrestaurant_facility_parser_does_not_require_menu_or_date(self):
        html = '''<html lang="ko"><head><title>서점</title>
          <link rel="alternate" hreflang="ko" href="?id=15"></head><body>
          <div class="fac-name">교내 서점</div><div class="fac-category">편의시설</div>
          <table class="info-table"><tr><td class="info-label">운영시간</td>
          <td class="info-value">월요일 09:00~18:00<br>주말 휴무</td></tr></table>
          <div class="shop-description-area"><p>도서와 문구를 판매합니다.</p></div></body></html>'''
        parsed = parse_facility_html(html, expected_facility_id="15")
        self.assertEqual(parsed["name"], "교내 서점")
        self.assertEqual(parsed["restaurant_info"]["category"], "편의시설")
        self.assertEqual(parsed["restaurant_info"]["opening_hours"], "월요일 09:00~18:00\n주말 휴무")
        self.assertEqual(parsed["page_data"]["raw_daily_cards"], [])
        with self.assertRaisesRegex(ValueError, "요청한 시설"):
            parse_facility_html(html, expected_facility_id="16")
        with self.assertRaises(ValueError):
            parse_facility_html("<html>서비스 점검 중</html>", expected_facility_id="15")

    def test_live_shop_detail_page_keeps_hours_contact_cover_category_and_notice(self):
        parsed = parse_facility_html(fixture("public_shop_18.html"), expected_facility_id="18")
        info = parsed["restaurant_info"]
        self.assertEqual(parsed["name"], "NEW YORK BURGER")
        self.assertEqual(info["category"], "일반음식점")
        self.assertEqual(info["location"], "학생복지관 2층")
        self.assertEqual(info["building_no"], "102")
        self.assertEqual(info["contact"], "031-400-4561")
        self.assertEqual(info["opening_hours"], "학기중\n· 10:00 ~ 19:30\n방학기간\n· 10:30 ~ 18:30")
        self.assertEqual(info["cover_image"]["url"], BASE_URL + "/uploads/facilities/fac_6a8da1de5ebb2.jpg")
        self.assertEqual(info["cover_image"]["alt"], "NEW YORK BURGER")
        self.assertIn("에리카 학생들의", info["notices"][0]["text"])
        self.assertTrue(any(asset["role"] == "cover" for asset in parsed["assets"]))
        self.assertTrue(any(heading["text"] == "NEW YORK BURGER" for heading in parsed["page_data"]["headings"]))
        with self.assertRaisesRegex(ValueError, "요청한 시설"):
            parse_facility_html(fixture("public_shop_18.html"), expected_facility_id="19")

    def test_shop_rich_description_and_arbitrary_images_are_preserved(self):
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(fixture("public_shop_18.html"), "html.parser")
        description = soup.new_tag("div", attrs={"class": "shop-description-area"})
        description.append("공식 매장 소개")
        description.append(soup.new_tag("img", attrs={"src": "/uploads/store-menu.jpg", "alt": "가격표 안내"}))
        soup.main.append(description)
        parsed = parse_facility_html(str(soup), expected_facility_id="18")
        self.assertEqual(parsed["restaurant_info"]["description"]["text"], "공식 매장 소개")
        self.assertIn("store-menu.jpg", parsed["restaurant_info"]["description"]["html"])
        self.assertTrue(any(image["url"] == BASE_URL + "/uploads/store-menu.jpg" and image["alt"] == "가격표 안내"
                            for image in parsed["page_data"]["images"]))

    def test_legacy_metadata_is_optional_while_identity_checks_remain_required(self):
        parsed = parse_restaurant_page(fixture(), expected_date=TARGET_DATE, expected_facility_id="1")
        self.assertEqual(parsed["restaurant_info"]["info_rows"], [])
        self.assertEqual(parsed["weekly_meals"], {})
        self.assertEqual(len(parsed["meals"]), 3)
        with self.assertRaisesRegex(ValueError, "요청 날짜"):
            parse_restaurant_page(fixture(), expected_date=date(2026, 9, 23), expected_facility_id="1")


if __name__ == "__main__":
    unittest.main()


"""Public embedded JSON and notice parsing never execute page JavaScript."""

import json
from pathlib import Path
from unittest.mock import patch

import pytest

from app.portal import embedded_array, parse_notice


NOTICE_URL = "https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id=13&board_lang=ko"
FETCHED = "2026-11-13T10:00:00+00:00"
# Body excerpt retained from work/source-audit/notice-13.html on 2026-11-13.
NOTICE_BODY = '''<h1 class="post-title">창의관식당 천원의 아침밥 결제 방식 변경 안내</h1>
<div class="post-content"><p>기숙사 창의관식당 천원의 아침밥 이용자 여러분께 안내드립니다.</p><p>온라인 결제 시스템 도입에 따라 아래 일정으로 결제 방식을 운영합니다.</p><p><br></p><p>11월 2일(월) ~ 11월 6일(금)</p><p>온라인 및 오프라인 병행 결제</p><p><br></p><p>11월 9일(월)부터</p><p>온라인 구매만 가능 (오프라인-키오스크 결제 불가)</p><p><br></p><p>감사합니다.</p></div>'''


def test_embedded_json_preserves_public_record_identities_and_nested_fields(tmp_path):
    marker = tmp_path / "must-not-exist"
    records = [{"id": 51, "name": "캠퍼스 카페", "category": "카페", "display": {"enabled": True},
        "description": "Literal const text remains data", "__proto__": {"admin": True}}]
    html = "<script>const dbFacilities = " + json.dumps(records, ensure_ascii=False) + ";\nrequire('fs').writeFileSync(" + json.dumps(str(marker)) + ",'executed');</script>"
    with patch("builtins.eval", side_effect=AssertionError("page code must not execute")):
        parsed = embedded_array(html, "dbFacilities")
    assert parsed == records
    assert parsed[0]["id"] == 51
    assert not marker.exists()


@pytest.mark.parametrize("value", ['{"id":1}', '[1,2]', '[{"id":undefined}]', '[{id:1}]', "[new Date()]"])
def test_non_json_or_non_record_arrays_are_rejected(value):
    with pytest.raises(ValueError):
        embedded_array("<script>const dbFacilities = " + value + ";</script>", "dbFacilities")


def test_embedded_array_name_is_exact_and_missing_data_is_not_an_empty_success():
    html = '<script>const dbFacilitiesOther = [{"id":1}]; const dbFacilities = [];</script>'
    assert embedded_array(html, "dbFacilities") == []
    with pytest.raises(ValueError):
        embedded_array(html, "dbNotices")


def test_saved_official_notice_preserves_policy_dates_and_related_cafeteria():
    # Use the full captured page when running in its authoring workspace;
    # the retained body excerpt keeps this test portable in the delivered ZIP.
    captured = Path(__file__).resolve().parents[3] / "work" / "source-audit" / "notice-13.html"
    html = captured.read_text(encoding="utf-8") if captured.is_file() else NOTICE_BODY
    notice = parse_notice(html, NOTICE_URL, FETCHED)
    assert notice["title"] == "창의관식당 천원의 아침밥 결제 방식 변경 안내"
    assert "11월 2일(월) ~ 11월 6일(금)" in notice["text"]
    assert "온라인 및 오프라인 병행 결제" in notice["text"]
    assert "11월 9일(월)부터" in notice["text"]
    assert "온라인 구매만 가능 (오프라인-키오스크 결제 불가)" in notice["text"]
    assert notice["restaurant_codes"] == ["re13"]
    assert notice["url"] == NOTICE_URL and notice["fetched_at"] == FETCHED
    assert "<p>" in notice["html"]


def test_notice_extracts_body_resources_without_navigation_or_script_execution(tmp_path):
    marker = tmp_path / "not-created"
    html = f'''<nav><a href="/login">로그인</a></nav><h1 class="post-title">학교 안내</h1>
        <div class="post-meta">2026.11.13 운영 관리자</div>
        <div class="post-content"><p>시설 이용 안내</p><a href="/uploads/guide.pdf">안내 문서</a>
        <img src="../../uploads/guide.png" alt="안내 이미지"><script>window.location={json.dumps(str(marker))};</script></div>'''
    notice = parse_notice(html, NOTICE_URL, FETCHED)
    assert notice["links"] == [{"text": "안내 문서", "url": "https://life.hanyang.ac.kr/uploads/guide.pdf"}]
    assert notice["images"][0]["url"] == "https://life.hanyang.ac.kr/theme/uploads/guide.png"
    assert notice["images"][0]["alt"] == "안내 이미지"
    assert notice["metadata"] == "2026.11.13 운영 관리자"
    assert notice["restaurant_codes"] == []
    assert "window.location" not in notice["text"]
    assert not marker.exists()


@pytest.mark.parametrize("html", ["<html>점검 중</html>", '<h1 class="post-title">공지</h1>', '<div class="post-content">본문</div>'])
def test_notice_structure_failure_is_not_silently_accepted(html):
    with pytest.raises(ValueError):
        parse_notice(html, NOTICE_URL, FETCHED)

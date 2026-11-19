from datetime import date

from app.date_ranges import resolve_date_range
from app.hybrid_search import add_lexical_rank
from app.rag import parse_restaurant


def test_literal_notice_evidence_survives_lower_semantic_score_than_generic_page():
    rows=[{"id":"generic","title":"학교 시설","text":"학교의 일반 시설을 소개합니다.","score":.35},
          {"id":"notice","title":"결제 방식 변경 공지","text":"천원의 아침밥은 온라인 구매만 가능하며 키오스크 결제는 불가합니다.","score":.2}]
    result=add_lexical_rank("천원의 아침밥 결제 공지 알려줘",rows)
    assert result[0]["id"]=="notice"
    assert result[0]["lexical_score"]>result[1]["lexical_score"]
    assert result[0]["semantic_score"]==.2


def test_generic_request_words_cannot_manufacture_lexical_evidence():
    rows=[{"title":"복지 안내","text":"시설 목록","score":.1}]
    assert add_lexical_rank("학교 알려줘",rows)[0]["lexical_score"]==0


def test_time_until_is_not_a_missing_calendar_endpoint():
    result=resolve_date_range("토요일에 몇 시까지 해?",date(2026,11,14))
    assert result.start==result.end==date(2026,11,14)
    result=resolve_date_range("내일 식당은 몇 시부터 몇 시까지 운영해?",date(2026,11,14))
    assert result.start==date(2026,11,15)


def test_cafeteria_spacing_is_not_a_different_entity():
    assert parse_restaurant("기숙사 식당 메뉴 알려줘")=="re13"
    assert parse_restaurant("창업 보육센터 식당 메뉴")=="re15"
    assert parse_restaurant("내일 학식 메뉴") is None


def test_short_field_followups_use_known_selected_facility():
    from app.school_context import resolve_school_context
    catalog=[{"id":49,"name":"우체국","location":"학생복지관1층"}]
    history=[{"role":"assistant","content":"우체국안내","selected_facility_ids":["49"],"reference_facility_ids":["49"]}]
    for message in ("전화도 알려줘","어디야?","사진 좀"):
        assert resolve_school_context(message,history,catalog=catalog).facility_ids==("49",)

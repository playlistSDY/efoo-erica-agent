"""Semantic families share one exact catalog/profile filtering implementation."""

from collections import Counter
from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

with patch("dotenv.load_dotenv"):
    from app.directory import DirectoryService, parse_directory_query
    from app.indexing import knowledge_chunks
    from app.store import Store


NOW = datetime.now(timezone.utc)
URL = "https://life.hanyang.ac.kr/theme/pages/facilities/index.php"


def row(identifier, category, name, location, description=""):
    return {"id": identifier, "category": category, "category_display": category,
            "name": name, "location": location, "description": description}


@pytest.fixture
def broad(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, database_url="", api_key="", max_age_hours=24,
        embedding_model="test-embedding", embedding_dimensions=3, refresh_retry_minutes=5)
    store = Store(settings)
    records = [
        row(101, "카페/베이커리", "스타벅스", "학생복지관 2층"),
        row(102, "카페/베이커리", "빵과책", "학생복지관 2F"),
        row(103, "카페/베이커리", "베이커리수목", "학생복지관 1층", "커피를 판매하는 매장으로 등록되었습니다."),
        row(104, "편의점", "CU", "학생복지관 2F"),
        row(105, "편의점", "7-eleven", "창의인재원 교육관 1F"),
        row(106, "편의점", "7-eleven", "학생복지관 1F"),
        row(107, "일반음식점", "소담", "학생복지관 2층"),
        row(108, "일반음식점", "해피버거", "학생복지관 2층"),
        row(109, "일반음식점", "멀티키친", "제3공학관 1층", "분식을 판매합니다."),
        row(110, "일반음식점", "오븐치킨", "창의관 1F"),
        row(111, "편의점", "CU", "창의관 1F"),
        row(112, "인쇄/문구/서점/기념품", "책벗", "학생복지관 1층"),
        row(113, "인쇄/문구/서점/기념품", "문구마당", "학생복지관 1층"),
        row(114, "인쇄/문구/서점/기념품", "해법복사", "제3공학관 1층"),
        row(115, "인쇄/문구/서점/기념품", "출력센터", "학생복지관 2층"),
        row(116, "생활서비스", "헤어그린", "학생복지관 3층"),
        row(200, "구내식당", "학생식당", "학생복지관 2층"),
    ]
    descriptions = {101: "공식 카페 공간을 안내합니다.", 102: "커피와 서적을 판매하는 공간입니다.",
        103: "빵 전문 매장으로 커피를 판매하지 않습니다.", 104: "생활잡화와 음료를 판매합니다.",
        105: "생활잡화와 음료를 판매합니다.", 106: "생활잡화와 음료를 판매합니다.",
        107: "분식과 김밥을 판매합니다.", 108: "햄버거 전문점입니다.", 109: "치킨을 판매합니다.",
        110: "치킨 전문점입니다.", 111: "생활잡화와 음료를 판매합니다.", 112: "서적을 판매합니다.",
        113: "문구류를 판매합니다.", 114: "복사와 출력을 제공합니다.", 115: "프린트와 출력을 제공합니다.",
        116: "헤어 관리와 미용실 서비스를 제공합니다.", 200: "구내식당입니다."}
    store.replace_knowledge("page:facility-directory", code="campus", name="공식 시설 목록", url=URL,
        fetched_at=NOW.isoformat(), payload={"records": records, "categories": dict(Counter(item["category"] for item in records))},
        chunks=knowledge_chunks("page:facility-directory", "공식 시설 목록", [("목록", "시설 전체 목록")], kind="page"))
    for item in records:
        identifier = item["id"]
        info = {"name": item["name"], "category": item["category"], "location": item["location"],
                "opening_hours": "평일 09:00~18:00", "description": {"text": descriptions[identifier]}, "notices": []}
        store.replace_knowledge(f"facility:{identifier}", code=f"fac_{identifier}", name=item["name"],
            url=f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={identifier}",
            fetched_at=NOW.isoformat(), payload={"restaurant_info": info},
            chunks=knowledge_chunks(f"facility:{identifier}", item["name"], [("상세 안내", descriptions[identifier])]))
    loader = Mock(side_effect=AssertionError("No public network in semantic tests"))
    service = DirectoryService(store, settings, loader=loader, now=lambda: NOW)
    return SimpleNamespace(store=store, records=records, service=service, loader=loader)


def answer(broad, message):
    query = broad.service.parse(message)
    assert query is not None, message
    result = broad.service.answer(query)
    assert result is not None
    broad.loader.assert_not_called()
    return query, result


def ids(result):
    return {record["id"] for record in result["directory"]["records"]}


@pytest.mark.parametrize("message,expected", [
    ("분식먹을곳", {107, 109}), ("책살곳", {102, 112}),
    ("학생복지관2층에서 커피 마실 곳 찾아줘", {101, 102}),
    ("학교에서 머리 자를 곳 있어?", {116}),
])
def test_service_discovery_uses_explicit_profile_or_catalog_evidence_not_coarse_category(broad, message, expected):
    _, result = answer(broad, message)
    assert ids(result) == expected
    if result["directory"]["filters"]["goals"]:
        assert result["sources"][0]["metadata"]["goals"] == ["커피"]
    else:
        assert result["directory"]["coverage"]["complete"] is True
        assert all(source["metadata"]["matching_fields"] for source in result["sources"])


def test_multiple_categories_list_every_unique_branch_instead_of_only_a_category_table(broad):
    query, result = answer(broad, "카페랑편의점목록")
    assert set(query.categories) == {"카페/베이커리", "편의점"}
    assert ids(result) == {101, 102, 103, 104, 105, 106, 111}
    assert result["directory"]["count"] == 7
    assert "| 매장 | 위치 |" in result["answer"]
    assert "스타벅스" in result["answer"] and "7-eleven" in result["answer"]
    assert result["directory"]["categories"] == {"카페/베이커리": 3, "편의점": 4}


def test_multiple_brand_aliases_form_a_union_without_collapsing_same_name_branches(broad):
    query, result = answer(broad, "CU랑 세븐일레븐은 몇 개 있어?")
    assert set(query.brands) == {"CU", "7-eleven"}
    assert query.brand is None
    assert ids(result) == {104, 105, 106, 111}
    assert result["directory"]["count"] == 4


def test_category_restriction_and_brand_union_keep_their_and_or_relationship(broad):
    query, result = answer(broad, "편의점 중 CU나 스타벅스 목록 알려줘")
    assert ids(result) == {104, 111}
    assert len(query.filter_groups) == 2
    selected = broad.service.select_records(query, records=broad.records)
    assert {record["id"] for record in selected} == ids(result)


@pytest.mark.parametrize("message,expected,relation", [
    ("분식집이나 햄버거집 찾아줘", {107, 108, 109}, "or"),
    ("분식집과 치킨집 추천해줘", {107, 109, 110}, "or"),
    ("분식과 치킨 둘 다 파는 곳 찾아줘", {109}, "and"),
    ("복사랑 출력할 수 있는 곳 찾아줘", {114}, "and"),
    ("복사나 출력할 수 있는 곳 찾아줘", {114, 115}, "or"),
])
def test_multiple_businesses_apply_union_or_joint_capability_with_exact_evidence(broad, message, expected, relation):
    query, result = answer(broad, message)
    assert len(query.businesses) == 2
    assert query.relation == relation
    assert ids(result) == expected
    assert {record["id"] for record in broad.service.select_records(query, records=broad.records)} == expected


def test_mixed_category_or_business_unions_do_not_double_count_hybrid_places(broad):
    query, result = answer(broad, "카페나 서점 찾아줘")
    assert ids(result) == {101, 102, 103, 112}
    assert result["directory"]["count"] == 4
    assert len(query.filter_groups) == 1
    both, result = answer(broad, "카페이면서 서점인 곳 찾아줘")
    assert both.relation == "and"
    assert ids(result) == {102}


def test_exact_official_composite_category_wins_over_its_narrow_subtype_words(broad):
    query, result = answer(broad, "인쇄/문구/서점/기념품 시설 전체 몇 개야?")
    assert query.categories == ("인쇄/문구/서점/기념품",)
    assert query.businesses == ()
    assert query.business is None
    assert ids(result) == {112, 113, 114, 115}
    assert "복합 분류 전체" in result["answer"]


def test_dormitory_and_floor_filters_use_source_location_aliases_consistently(broad):
    _, dorms = answer(broad, "기숙사편의점 목록 알려줘")
    assert ids(dorms) == {105, 111}
    _, floor = answer(broad, "학생복지관2층카페 목록 알려줘")
    assert ids(floor) == {101, 102}


def test_fresh_negative_profile_overrides_older_positive_catalog_service_text(broad):
    query, result = answer(broad, "커피를 판매한다고 명시된 곳 알려줘")
    assert query.goals == () and query.business == "커피"
    assert ids(result) == {102}
    assert 103 not in ids(result)
    assert all(source["metadata"]["facility_id"] != 103 for source in result["sources"])


def test_joint_services_with_different_source_proofs_cite_each_actual_document(broad):
    _, result = answer(broad, "분식과 치킨 둘 다 파는 곳 찾아줘")
    catalog = next(source for source in result["sources"] if source["url"] == URL)
    profile = next(source for source in result["sources"] if source["doc_type"] == "profile")
    assert "분식을 판매합니다" in catalog["text"]
    assert "치킨을 판매합니다" not in catalog["text"]
    assert "치킨을 판매합니다" in profile["text"]
    table = next(line for line in result["answer"].splitlines() if line.startswith("| 멀티키친 |"))
    assert f"[{catalog['id']}]" in table and f"[{profile['id']}]" in table


@pytest.mark.parametrize("message", [
    "비건 카페 목록", "편의점 중 약국 찾아줘", "카페 중 와이파이 되는 곳 알려줘",
    "학교에 수영장 시설 몇 개 있어?", "카페가 없는 건물 알려줘", "약국 매장 몇 개야?",
])
def test_unknown_or_unsupported_conditions_never_disappear_into_a_broad_count(broad, message):
    assert broad.service.parse(message) is None


@pytest.mark.parametrize("message,expected", [
    ("카페 전화번호", {101, 102, 103}),
    ("학생복지관2층카페몇시", {101, 102}),
    ("세븐일레븐 영업시간", {105, 106}),
])
def test_explicit_detail_mode_reuses_the_same_filters_without_hijacking_default_routes(broad, message, expected):
    assert broad.service.parse(message) is None
    query = broad.service.parse(message, allow_details=True)
    assert query is not None and query.include_details
    before = deepcopy(broad.records)
    assert {record["id"] for record in broad.service.select_records(query, records=broad.records)} == expected
    assert broad.records == before


@pytest.mark.parametrize("message", [
    "내일 학생식당 학식 알려줘", "다음 주 구내식당 점심 메뉴", "라멘집 오늘 메뉴 알려줘",
    "지금 영업하는 카페 찾아줘", "CU 커피 가격 알려줘",
])
def test_meals_prices_and_dynamic_opening_conditions_keep_their_own_rag_resolver(broad, message):
    assert parse_directory_query(message, records=broad.records) is None


def test_coffee_purpose_uses_the_official_cafe_taxonomy_without_inventing_product_sales(broad):
    query, result = answer(broad, "커피 마실 곳 찾아줘")
    assert query.goals == ("커피",)
    assert query.businesses == ()
    assert ids(result) == {101, 102, 103}
    assert "공식 카페/베이커리 복합 분류" in result["answer"]
    assert "개별 매장의 커피 판매" in result["answer"]
    assert result["sources"][0]["url"] == URL
    assert result["sources"][0]["doc_type"] == "page"
    # Starbucks has no literal coffee-sales statement in this fixture. Its
    # official category is evidence for a candidate, not a drink inventory.
    assert 101 in ids(result)


def test_meal_purpose_is_a_restaurant_candidate_list_without_claiming_dated_menus(broad):
    query, result = answer(broad, "학교에서 밥 먹을 곳 찾아줘")
    assert query.goals == ("식사",)
    assert ids(result) == {107, 108, 109, 110, 200}
    assert "날짜별 메뉴나 현재 영업 여부를 확인한 결과는 아니" in result["answer"]


def test_explicit_other_category_is_not_silently_replaced_by_coffee_goal_cafes(broad):
    query, result = answer(broad, "편의점에서 커피 마실 곳 찾아줘")
    assert query.goals == ()
    assert query.business == "커피"
    assert ids(result) == set()


@pytest.mark.parametrize("message", ["편의점하는곳있어?", "카페 이용은 어떻게 돼?", "서점 이용이 되는 곳"])
def test_detail_request_grammar_is_not_mistaken_for_an_unknown_facility_filter(broad, message):
    query = broad.service.parse(message, allow_details=True)
    assert query is not None
    assert broad.service.select_records(query, records=broad.records)


@pytest.mark.parametrize("message", ["노트북 쓸 수 있는 카페", "흡연 가능한 카페", "카페 와이파이가 어떻게 돼?"])
def test_real_unknown_conditions_survive_detail_grammar_removal(broad, message):
    assert broad.service.parse(message, allow_details=True) is None


def replace_catalog(broad, records, assets):
    broad.store.replace_knowledge("page:facility-directory", code="campus", name="공식 시설 목록", url=URL,
        fetched_at=NOW.isoformat(), payload={"records": records, "assets": assets},
        chunks=knowledge_chunks("page:facility-directory", "공식 시설 목록", [("목록", "공식 목록")], kind="page"))


def test_catalog_photos_require_selected_record_image_and_catalog_owned_asset_proof(broad):
    records = deepcopy(broad.records)
    for record in records:
        record["image_url"] = f"/uploads/facility-{record['id']}.jpg"
    selected = "https://life.hanyang.ac.kr/uploads/facility-101.jpg"
    unrelated = "https://life.hanyang.ac.kr/uploads/facility-104.jpg"
    profile_only = "https://life.hanyang.ac.kr/uploads/profile-101.jpg"
    assets = [{"url": original, "local_path": "/media/" + character * 64 + ".jpg",
               "sha256": character * 64, "downloaded": True, "content_type": "image/jpeg"}
              for original, character in [(selected, "a"), (unrelated, "b"), (profile_only, "c")]]
    replace_catalog(broad, records, assets)
    _, result = answer(broad, "카페 목록 알려줘")
    assert [image["url"] for image in result["images"]] == ["/media/" + "a" * 64 + ".jpg"]
    assert result["images"] == result["sources"][0]["images"]
    image = result["images"][0]
    assert image["source_url"] == URL
    assert image["original_url"] == selected
    assert image["facility_id"] == 101


def test_catalog_media_rejects_untrusted_paths_formats_and_broken_fingerprints(broad):
    records = deepcopy(broad.records)
    original = "https://life.hanyang.ac.kr/uploads/facility-101.jpg"
    records[0]["image_url"] = original
    bad = ["https://untrusted.example/a.jpg", "/media/../../a.jpg", "/media/unhashed.jpg", "/media/" + "a" * 64 + ".svg"]
    assets = [{"url": original, "local_path": local, "downloaded": True, "content_type": "image/jpeg"} for local in bad]
    assets += [{"url": original, "local_path": "/media/" + "a" * 64 + ".jpg", "sha256": "b" * 64,
                "downloaded": True, "content_type": "image/jpeg"},
               {"url": original, "local_path": "/media/" + "a" * 64 + ".jpg", "downloaded": True,
                "content_type": "image/svg+xml"}]
    replace_catalog(broad, records, assets)
    _, result = answer(broad, "카페 목록 알려줘")
    assert result["images"] == result["sources"][0]["images"] == []


def test_catalog_media_deduplicates_content_hash_and_limits_six_selected_images(broad):
    records = [row(1000 + index, "카페/베이커리", f"공식 카페 {index}", "학생복지관 2층") for index in range(8)]
    assets = []
    for index, record in enumerate(records):
        original = f"https://life.hanyang.ac.kr/uploads/{index}.jpg"
        record["image_url"] = original
        digest = format(max(index - 1, 0), "064x")
        assets.append({"url": original, "local_path": f"/media/{digest}.jpg", "sha256": digest,
                       "downloaded": True, "content_type": "image/jpeg"})
    replace_catalog(broad, records, assets)
    _, result = answer(broad, "카페 목록 알려줘")
    assert len(result["images"]) == 6
    assert len({image["url"].split("/")[-1].split(".")[0] for image in result["images"]}) == 6
    assert all(image["source_url"] == URL for image in result["images"])


def test_bad_public_image_references_do_not_break_an_otherwise_valid_catalog_answer(broad):
    records = deepcopy(broad.records)
    records[0]["image_url"] = "https://[invalid-image-host"
    assets = [{"url": "https://[invalid-image-host", "local_path": "/media/" + "a" * 64 + ".jpg",
               "downloaded": True, "content_type": "image/jpeg"}]
    replace_catalog(broad, records, assets)
    _, result = answer(broad, "카페 목록 알려줘")
    assert ids(result) == {101, 102, 103}
    assert result["images"] == []

"""Business subtype discovery scans complete public profiles, not top-k vectors."""

from copy import deepcopy
from datetime import date, datetime, timedelta, timezone
import json
import re
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from app.directory import DirectoryService, capability_excerpt, facility_capabilities
from app.indexing import knowledge_chunks
from app.rag import Agent
from app.school_query import SchoolService
from app.store import Store


NOW = datetime.now(timezone.utc)
CATALOG_URL = "https://life.hanyang.ac.kr/theme/pages/facilities/index.php"


def record(identifier, name, location, category="일반음식점"):
    return {"id": identifier, "name": name, "category": category, "category_display": category,
        "location": location, "description": "", "notice": "", "status": "inactive", "operating_status": "영업중"}


def public_catalog(store, records):
    scope = "page:facility-directory"
    text = "\n".join(f"{item['name']} / {item['category']} / {item['location']}" for item in records)
    store.replace_knowledge(scope, code="campus", name="공개 전체 시설 목록", url=CATALOG_URL,
        fetched_at=NOW.isoformat(), payload={"records": deepcopy(records)},
        chunks=knowledge_chunks(scope, "공개 전체 시설 목록", [("전체 목록", text)], kind="page"))


def profile(store, item, *, notice="", description="", photo=None, fetched_at=NOW):
    identifier = item["id"]
    url = f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={identifier}"
    original = f"https://life.hanyang.ac.kr/uploads/facilities/cover-{identifier}.jpg"
    images = [{"url": photo, "original_url": original, "alt": item["name"], "title": item["name"]}] if photo else []
    hours = "평일 11:00~14:00, 17:00~20:00"
    info = {"name": item["name"], "category": item["category"], "location": item["location"],
        "opening_hours": hours,
        "info_rows": [{"label": "위치", "value": item["location"]}, {"label": "영업시간", "value": hours}],
        "notices": [{"label": "운영 공지", "text": notice}] if notice else [],
        "description": {"text": description}, "cover_image": {"url": original}, "gallery": []}
    assets = [{"url": original, "local_path": photo, "downloaded": True, "content_type": "image/jpeg", "alt": item["name"]}] if photo else []
    scope = f"facility:{identifier}"
    sections = [("시설명", item["name"]), ("분류", item["category"]), ("운영 공지", notice),
        ("상세 안내", description), ("기본 정보", item["location"] + "\n" + hours)]
    store.replace_knowledge(scope, code=f"fac_{identifier}", name=item["name"], url=url,
        fetched_at=fetched_at.isoformat(),
        payload={"facility_id": identifier, "name": item["name"], "restaurant_info": info,
            "assets": assets, "directory_record": deepcopy(item)},
        chunks=knowledge_chunks(scope, item["name"], sections, images=images))
    return url


@pytest.fixture
def business(tmp_path):
    settings = SimpleNamespace(data_dir=tmp_path, database_url="", api_key="mock-only-no-network",
        model="gpt-6.1-terra", embedding_model="test-embedding", embedding_dimensions=3,
        min_similarity=0.30, max_age_hours=24, max_date_range_days=31, refresh_retry_minutes=5)
    store = Store(settings)
    rows = [
        record(18, "33 Korean Snack", "학생복지관 2층"),
        record(27, "이모네", "제1공학관 1층"),
        record(28, "이모네", "제3공학관 1층"),
        record(29, "이모네", "학생복지관 1층", "카페/베이커리"),
        record(99, "상세 미확인 시설", "사회교육관 1층", "생활서비스"),
    ]
    public_catalog(store, rows)
    photos = {identifier: "/media/" + character * 64 + ".jpg" for identifier, character in [(18, "a"), (27, "b"), (28, "c"), (29, "d")]}
    urls = {
        18: profile(store, rows[0], description="떡볶이와 튀김을 판매하는 매장입니다.", photo=photos[18]),
        27: profile(store, rows[1], notice="분식과 김밥을 판매합니다.", description="교내 간단한 한 끼를 제공해요.", photo=photos[27]),
        28: profile(store, rows[2], description="김밥과 간식을 안내하는 매장입니다.", photo=photos[28]),
        29: profile(store, rows[3], description="커피와 베이커리를 안내합니다.", photo=photos[29]),
    }
    loader = Mock(side_effect=AssertionError("unexpected public network request"))
    refresher = Mock()
    service = DirectoryService(store, settings, loader=loader, facility_refresher=refresher, now=lambda: NOW)
    return SimpleNamespace(settings=settings, store=store, records=rows, photos=photos, urls=urls,
        service=service, loader=loader, refresher=refresher)


def search(fixture, message="학교에 분식집 찾아줘"):
    query = fixture.service.parse(message)
    assert query is not None and query.business == "분식"
    answer = fixture.service.answer(query, refresh=False)
    assert answer is not None
    return query, answer


@pytest.mark.parametrize("message", ["학교에 분식집 찾아줘", "분식집 추천해줘", "분식집 어디 있어?", "분식집 목록 알려줘"])
def test_subtype_discovery_finds_name_description_and_notice_evidence(business, message):
    query, result = search(business, message)
    assert query.action == "list"
    assert result["directory"]["business"] == "분식"
    assert result["directory"]["count"] == 3
    assert {item["id"] for item in result["directory"]["records"]} == {18, 27, 28}
    assert result["mode"] == "retrieval" and result["model"] is None
    assert "33 Korean Snack" in result["answer"] and "이모네" in result["answer"]
    business.loader.assert_not_called()
    business.refresher.ensure.assert_not_called()


def test_notice_only_match_has_its_own_official_source_and_matching_field(business):
    _, result = search(business)
    evidence = [source for source in result["sources"] if source["url"] == business.urls[27]]
    assert evidence, "Notice-only business evidence must cite that exact facility page"
    assert any("분식과 김밥" in source["text"] for source in evidence)
    fields = [field for source in evidence for field in source.get("metadata", {}).get("matching_fields", [])]
    assert any("notice" in field.lower() or "공지" in field for field in fields)
    snippets = [text for source in evidence for text in source.get("metadata", {}).get("matched_texts", [])]
    assert any("분식과 김밥" in snippet for snippet in snippets)


def test_same_name_branches_join_by_id_and_do_not_import_another_branch_profile(business):
    _, result = search(business)
    imone = [item for item in result["directory"]["records"] if item["name"] == "이모네"]
    assert {item["id"] for item in imone} == {27, 28}
    assert 29 not in {item["id"] for item in result["directory"]["records"]}
    assert business.urls[29] not in {source["url"] for source in result["sources"]}


def test_business_gallery_preserves_only_matched_facility_photos_and_sources(business):
    _, result = search(business)
    assert {image["url"] for image in result["images"]} == {business.photos[identifier] for identifier in (18, 27, 28)}
    assert all(image["source_url"] in {business.urls[identifier] for identifier in (18, 27, 28)} for image in result["images"])
    assert business.photos[29] not in {image["url"] for image in result["images"]}


def test_partial_detail_coverage_is_explicit_and_does_not_claim_all_campus_shops(business):
    _, result = search(business)
    assert result["directory"]["coverage"] == {"listed": 5, "details_checked": 4, "complete": False}
    assert "상세" in result["answer"] and ("일부" in result["answer"] or "확인" in result["answer"])
    assert result["directory"]["scope"] == "official_portal"


def test_complete_profile_coverage_has_no_partial_search_warning(business):
    profile(business.store, business.records[-1], description="세탁 서비스를 안내합니다.")
    _, result = search(business)
    assert result["directory"]["coverage"] == {"listed": 5, "details_checked": 5, "complete": True}
    assert result["directory"]["count"] == 3
    assert "상세를 확인하지 못" not in result["answer"]


def test_stale_profile_is_not_used_as_current_subtype_or_photo_evidence(business):
    profile(business.store, business.records[1], notice="분식과 김밥을 판매합니다.",
        photo=business.photos[27], fetched_at=NOW - timedelta(days=2))
    _, result = search(business)
    assert {item["id"] for item in result["directory"]["records"]} == {18, 28}
    assert result["directory"]["coverage"]["complete"] is False
    assert business.urls[27] not in {source["url"] for source in result["sources"]}
    assert business.photos[27] not in {image["url"] for image in result["images"]}


def test_explicit_negative_subtype_notice_is_not_positive_recommendation_evidence(business):
    negative = record(30, "음료 매장", "제2공학관 1층")
    public_catalog(business.store, business.records + [negative])
    profile(business.store, negative, notice="분식점이 아닙니다.", description="음료 전문 매장입니다.")
    _, result = search(business)
    assert 30 not in {item["id"] for item in result["directory"]["records"]}


def test_matches_beyond_top_eight_and_weak_vectors_are_all_kept_without_openai(business):
    rows = [record(200 + index, f"테스트 매장 {index:02d}", f"제{index + 1}관 1층") for index in range(20)]
    public_catalog(business.store, rows)
    for item in rows:
        profile(business.store, item, description="김밥과 분식을 판매합니다.")
    pending = business.store.pending()
    business.store.save_embeddings(pending, [[0.0, 1.0, 0.0]] * len(pending))
    model = Mock()
    model.embeddings.create.side_effect = AssertionError("Subtype coverage must not depend on embedding similarity")
    model.responses.create.side_effect = AssertionError("Subtype matches already have exact public evidence")
    agent = Agent(business.store, business.settings, model, directory_service=business.service)
    result = agent.chat("학교에 분식집 찾아줘", [], date(2026, 11, 14))
    assert result["directory"]["count"] == len(result["directory"]["records"]) == 20
    assert result["directory"]["coverage"] == {"listed": 20, "details_checked": 20, "complete": True}
    assert "테스트 매장 00" in result["answer"] and "테스트 매장 19" in result["answer"]
    model.embeddings.create.assert_not_called()
    model.responses.create.assert_not_called()


def test_runtime_refresher_can_fill_missing_details_before_counting(business):
    def fill(records):
        assert {item["id"] for item in records} == {18, 27, 28, 29, 99}
        profile(business.store, business.records[-1], description="생활 서비스를 안내합니다.")
        return {"errors": []}

    business.refresher.ensure.side_effect = fill
    query = business.service.parse("학교에 분식집 찾아줘")
    result = business.service.answer(query, refresh=True)
    assert result["directory"]["count"] == 3
    assert result["directory"]["coverage"]["complete"] is True
    business.refresher.ensure.assert_called_once()


def test_business_location_filter_and_duplicate_catalog_identity_are_respected(business):
    public_catalog(business.store, business.records + [deepcopy(business.records[1])])
    _, result = search(business, "제1공학관 분식집 찾아줘")
    assert result["directory"]["count"] == 1
    assert {item["id"] for item in result["directory"]["records"]} == {27}
    assert result["directory"]["coverage"]["listed"] == 5


@pytest.mark.parametrize("message", ["분식집 운영시간 알려줘", "이모네 오늘 메뉴 알려줘", "분식집 메뉴 가격이 얼마야?", "지금 영업하는 분식집 찾아줘"])
def test_dynamic_operating_and_menu_questions_keep_existing_rag_path(business, message):
    assert business.service.parse(message) is None


def test_public_time_ranges_do_not_become_markdown_strikethrough(business):
    _, result = search(business)
    assert "11:00" in result["answer"] and "20:00" in result["answer"]
    assert re.search(r"(?<!\\)~", result["answer"]) is None
    assert any("11:00~14:00" in source["text"] for source in result["sources"])


def test_distinct_multiple_business_requests_do_not_silently_select_the_first(business):
    for message, labels in (("분식집이나 햄버거집 찾아줘", {"분식", "버거"}),
                            ("분식집과 치킨집 추천해줘", {"분식", "치킨"})):
        query = business.service.parse(message)
        assert query is not None and set(query.businesses) == labels
        assert query.relation == "or"
        result = business.service.answer(query)
        assert {item["id"] for item in result["directory"]["records"]} == {18, 27, 28}
        assert result["directory"]["count"] == 3
    # Two synonyms for the same subtype are still one well-defined search.
    same_subtype = business.service.parse("분식집이나 떡볶이집 찾아줘")
    assert same_subtype is not None and same_subtype.business == "분식"


@pytest.mark.parametrize("negative_text", [
    "분식을 판매하지 않습니다.", "김밥을 취급하지 않습니다.", "떡볶이 메뉴 없음.",
])
def test_negative_product_sales_and_missing_menu_are_not_business_matches(business, negative_text):
    item = record(30, "음료 전문점", "제2공학관 1층")
    public_catalog(business.store, business.records + [item])
    profile(business.store, item, notice=negative_text, description="커피를 안내합니다.")
    _, result = search(business)
    assert result["directory"]["count"] == 3
    assert 30 not in {item["id"] for item in result["directory"]["records"]}


def test_catalog_only_business_evidence_and_profile_hours_photos_have_separate_cited_sources(business):
    updated = deepcopy(business.records)
    updated[0]["description"] = "분식 전문점으로 등록되어 있습니다."
    public_catalog(business.store, updated)
    # Only the catalog proves the subtype; this fresh detail page contributes
    # location/hours/photo facts and must have its own citation.
    profile(business.store, updated[0], description="일반 식사를 제공하는 매장입니다.", photo=business.photos[18])
    _, result = search(business)
    catalog_evidence = [source for source in result["sources"]
        if source["url"] == CATALOG_URL and source.get("metadata", {}).get("facility_id") == 18]
    profile_evidence = [source for source in result["sources"] if source["url"] == business.urls[18]]
    assert len(catalog_evidence) == len(profile_evidence) == 1
    catalog_source, detail_source = catalog_evidence[0], profile_evidence[0]
    assert "분식 전문점" in catalog_source["text"]
    assert "11:00" not in catalog_source["text"] and "20:00" not in catalog_source["text"]
    assert catalog_source["images"] == []
    assert "11:00~14:00" in detail_source["text"]
    assert detail_source["doc_type"] == "profile"
    fields = detail_source.get("metadata", {}).get("matching_fields", [])
    assert any("hour" in field.lower() or "시간" in field for field in fields)
    texts = detail_source.get("metadata", {}).get("matched_texts", [])
    assert any("11:00~14:00" in text for text in texts)
    assert {image["url"] for image in detail_source["images"]} == {business.photos[18]}
    assert all(image["source_url"] == business.urls[18] for image in detail_source["images"])
    assert next(image for image in result["images"] if image["url"] == business.photos[18])["source_url"] == business.urls[18]
    table_row = next(line for line in result["answer"].splitlines() if line.startswith("| 33 Korean Snack |"))
    assert f"[{catalog_source['id']}]" in table_row
    assert f"[{detail_source['id']}]" in table_row


@pytest.fixture
def capabilities(business):
    category="인쇄/문구/서점/기념품"
    rows=[record(201,"배움상회","학생복지관 3층",category),
          record(202,"한결문구","학생복지관 1층",category),
          record(203,"인쇄공방","학생복지관 3층",category),
          record(204,"분류만 있는 매장","학생복지관 3층",category),
          record(205,"교재마루","제3공학관 1층",category),
          record(206,"소나무서적","학생복지관 2층",category)]
    descriptions={201:"대학교재, 문구류, 학교 기념품 판매",202:"필기구와 학용품을 취급합니다.",
                  203:"복사와 출력을 제공합니다.",204:"이용 안내를 확인해 주세요.",
                  205:"전공책과 교재를 판매합니다.",206:"서적 판매 안내"}
    public_catalog(business.store,rows)
    for row in rows:
        profile(business.store,row,description=descriptions[row["id"]])
    return SimpleNamespace(**vars(business),capability_records=rows,descriptions=descriptions,
        school=SchoolService(business.store,business.settings,business.service,refresher=business.refresher))


@pytest.mark.parametrize("message,label,expected",[
    ("학교 문구점 찾아줘","문구",{201,202}),
    ("학용품 파는 곳 알려줘","문구",{201,202}),
    ("기념품 어디 있어?","기념품",{201}),
    ("굿즈 파는 곳 알려줘","기념품",{201}),
    ("학교 서점 목록 알려줘","서점",{201,205,206}),
    ("교재 살 곳 찾아줘","서점",{201,205,206}),
])
def test_official_product_descriptions_prove_multiple_services_under_any_shop_name(capabilities,message,label,expected):
    query=capabilities.service.parse(message)
    assert query and query.business==label
    response=capabilities.service.answer(query)
    assert {row["id"] for row in response["directory"]["records"]}==expected
    for source in response["sources"]:
        assert source["url"].endswith("id="+str(source["metadata"]["facility_id"]))
        assert "description" in source["metadata"]["matching_fields"] or "name" in source["metadata"]["matching_fields"]
    assert "재고가 있어" not in response["answer"]


def test_multiple_services_share_description_evidence_without_expanding_composite_category(capabilities):
    query=capabilities.service.parse("문구와 기념품을 함께 파는 곳 찾아줘")
    assert query and query.relation=="and"
    response=capabilities.service.answer(query)
    assert [row["id"] for row in response["directory"]["records"]]==[201]
    row=capabilities.capability_records[0]
    info={"name":row["name"],"category":row["category"],"description":{"text":capabilities.descriptions[201]}}
    services=facility_capabilities(row,info)
    assert {"서점","문구","기념품"}.issubset(services)
    assert all(services[label]["source"]=="profile" for label in ("서점","문구","기념품"))
    assert facility_capabilities(capabilities.capability_records[3],{})=={}


@pytest.mark.parametrize("description",["문구류를 판매하지 않습니다.","학용품 취급 중단", "필기구가 없습니다.","문구류 판매 안 해요."])
def test_negative_official_product_descriptions_do_not_prove_stationery(capabilities,description):
    row=record(207,"자료실매장","학생복지관 3층","인쇄/문구/서점/기념품")
    info={"name":row["name"],"description":{"text":description}}
    assert "문구" not in facility_capabilities(row,info)


def test_sales_guidance_word_is_not_a_negative_service_statement():
    row=record(207,"자료실매장","학생복지관 3층","인쇄/문구/서점/기념품")
    services=facility_capabilities(row,{"description":{"text":"문구류, 기념품, 서적 판매 안내"}})
    assert {"문구","기념품","서점"}.issubset(services)


@pytest.mark.parametrize("message",["복지관 문구점 오늘 안해?","학생복지관 문구점 오늘은 안 해?", "복지관 학용품 파는 곳 오늘 해?"])
def test_product_service_hours_query_keeps_building_and_current_status(capabilities,message):
    plan=capabilities.school.plan(message)
    assert plan.active and not plan.unresolved and plan.slots==("hours",)
    assert plan.current_status and set(plan.facility_ids)=={"201","202"}
    response=capabilities.school.answer(plan)
    assert set(response["selected_facility_ids"])=={"201","202"}
    assert "실시간 영업 여부는 확인할 수 없어요" in response["answer"]
    assert "11:00" in response["answer"] and "20:00" in response["answer"]
    assert "분류만 있는 매장" not in response["answer"]


def test_generic_bookstore_alias_does_not_hide_description_only_book_sellers(capabilities):
    plan=capabilities.school.plan("학교 서점 연락처 알려줘")
    assert set(plan.facility_ids)=={"201","205","206"}


@pytest.mark.parametrize("message",["오늘 안해?","그럼 오늘은 안해?","내일 하나요?"])
def test_operating_followup_preserves_selected_service_store(capabilities,message):
    history=[{"role":"user","content":"복지관 3층 문구점 찾아줘"},
        {"role":"assistant","content":"배움상회 안내", "selected_facility_ids":["201"],"reference_facility_ids":["201"]}]
    plan=capabilities.school.plan(message,history)
    assert plan.active and plan.facility_ids==("201",) and not plan.unresolved
    assert capabilities.school.answer(plan)["selected_facility_ids"]==["201"]


def test_stale_service_profile_is_refreshed_before_selecting_all_matching_shops(capabilities):
    row=capabilities.capability_records[0]
    profile(capabilities.store,row,description=capabilities.descriptions[201],fetched_at=NOW-timedelta(days=2))
    plan=capabilities.school.plan("복지관 문구점 오늘 안해?")
    assert plan.facility_ids==("202",)
    def refresh(rows):
        assert {r["id"] for r in rows}=={201,202,203,204,205,206}
        profile(capabilities.store,row,description=capabilities.descriptions[201])
        return {"targets":[{"id":r["id"],"usable":True} for r in rows]}
    capabilities.refresher.ensure.side_effect=refresh
    response=capabilities.school.answer(plan,refresh=True)
    assert set(response["selected_facility_ids"])=={"201","202"}
    assert set(response["reference_facility_ids"])=={"201","202"}


def test_failed_service_profile_refresh_cannot_use_cached_description(capabilities):
    capabilities.refresher.ensure.return_value={"targets":[{"id":201,"usable":False}]}
    response=capabilities.school.answer(capabilities.school.plan("복지관 문구점 오늘 안해?"),refresh=True)
    assert response["selected_facility_ids"]==["202"]
    assert not any(source["metadata"]["facility_id"]==201 for source in response["sources"])


def test_unknown_proper_stationery_name_is_not_replaced_by_other_shops(capabilities):
    plan=capabilities.school.plan("무지개문구점 전화번호 알려줘")
    assert plan.unresolved and not plan.facility_ids


@pytest.mark.parametrize("label,term",[("문구","문구류 판매"),("서점","대학교재 판매"),("기념품","학교 굿즈 안내")])
def test_capability_excerpt_retains_matching_phrase_after_long_unrelated_prefix(label,term):
    text="시설 소개. "*500+term+". "+"추가 안내. "*500
    excerpt=capability_excerpt(label,text,max_chars=1200)
    assert term in excerpt and len(excerpt)<=1200
    assert excerpt.startswith("…") and excerpt.endswith("…")
    assert excerpt[1:-1] in text


def test_capability_excerpt_preserves_short_text_and_bounds_edge_matches():
    assert capability_excerpt("문구","문구류 판매")=="문구류 판매"
    for text in ("문구류 판매 "+"가"*2000,"가"*2000+" 문구류 판매"):
        excerpt=capability_excerpt("문구",text,40)
        assert "문구류 판매" in excerpt and len(excerpt)<=40
    assert capability_excerpt("문구","알 수 없는 안내"*100,20).endswith("…")


@pytest.mark.parametrize("message,relation,labels,expected",[
    ("학교 기념품이나 교재는 어디서 살 수 있어?","or",{"기념품","서점"},{201,205,206}),
    ("학교 문구나 책은 어디에서 살 수 있어?","or",{"문구","서점"},{201,202,205,206}),
    ("기념품이나교재 어디서 살 수 있나요?","or",{"기념품","서점"},{201,205,206}),
    ("문구나 책","or",{"문구","서점"},{201,202,205,206}),
    ("문구나책 같이 파는 곳 찾아줘","or",{"문구","서점"},{201,202,205,206}),
    ("교재랑 문구 같이","and",{"문구","서점"},{201}),
    ("교재랑 문구를 같이 파는 곳","and",{"문구","서점"},{201}),
    ("기념품과 책을 함께 살 수 있는 곳 알려줘","and",{"기념품","서점"},{201}),
])
def test_product_connections_particles_and_purchase_location_use_grounded_service_path(capabilities,message,relation,labels,expected):
    query=capabilities.service.parse(message)
    assert query and query.relation==relation and set(query.businesses)==labels
    model=Mock()
    model.responses.create.side_effect=AssertionError("Product location must resolve from official service evidence")
    model.embeddings.create.side_effect=AssertionError("Product location must not depend on top-k matching")
    agent=Agent(capabilities.store,capabilities.settings,client=model,directory_service=capabilities.service)
    plan=agent.plan(message,[],date(2026,11,14))
    assert plan.school_plan is None or not plan.school_plan.active
    assert plan.directory_query
    response=agent.chat(message,[],date(2026,11,14))
    assert response["mode"]=="retrieval"
    assert {row["id"] for row in response["directory"]["records"]}==expected
    assert {source["metadata"]["facility_id"] for source in response["sources"]}==expected
    assert all(source["url"].startswith("https://life.hanyang.ac.kr/") for source in response["sources"])
    assert 204 not in expected  # A broad catalog category alone proves no product.
    model.responses.create.assert_not_called()
    model.embeddings.create.assert_not_called()


@pytest.mark.parametrize("message",["무지개책 전화번호","싼 문구나 책 파는 곳", "문구나 책 재고 알려줘"])
def test_product_connectors_do_not_erase_unknown_names_or_unverified_conditions(capabilities,message):
    assert capabilities.service.parse(message) is None


def test_literal_book_sales_evidence_does_not_match_policy_or_furniture_words():
    row=record(207,"새봄상회","학생복지관 3층","인쇄/문구/서점/기념품")
    assert "서점" in facility_capabilities(row,{"description":{"text":"책 판매 안내"}})
    assert "서점" not in facility_capabilities(row,{"description":{"text":"운영 정책을 안내하며 책상과 의자를 제공합니다."}})

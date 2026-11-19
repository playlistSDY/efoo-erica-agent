"""School information routing and source-bound image regressions.

All documents and vectors live in pytest temporary SQLite databases. The fake
OpenAI client never opens a connection, and importing Settings must not read a
developer's project .env when this test module runs on its own.
"""

from datetime import date, datetime, timezone
import json
from unittest.mock import Mock, patch

import pytest

with patch("dotenv.load_dotenv"):
    from app.crawler import CrawlResult
    from app.indexing import build_chunks, index_notice, knowledge_chunks
    from app.rag import Agent
    from app.settings import Settings
    from app.store import Store


DAY = date(2026, 9, 22)
STAMP = datetime.now(timezone.utc).isoformat()
PHOTO_A = "/media/" + "a" * 64 + ".jpg"
PHOTO_B = "/media/" + "b" * 64 + ".png"


@pytest.fixture
def school(tmp_path):
    settings = Settings(data_dir=tmp_path, database_url="", api_key="isolated-test-key", embedding_dimensions=3)
    store = Store(settings)
    client = Mock()
    client.embeddings.create.return_value.data = [Mock(embedding=[1.0, 0.0, 0.0])]

    def answer(**kwargs):
        prompt = json.loads(kwargs["input"][0]["content"])
        source_id = prompt["sources"][0]["id"]
        return Mock(output_text=json.dumps({"answer": f"공식 자료 안내입니다. [{source_id}]",
                                            "source_ids": [source_id], "insufficient_evidence": False}))

    client.responses.create.side_effect = answer
    return store, client, Agent(store, settings, client=client)


def profile(store, *, facility_id=49, name="안산한양대우편취급국", code=None, text="위치: 학생복지관 1층\n운영시간: 평일 09:00~18:00", images=None):
    scope = f"facility:{facility_id}"
    code = code or f"fac_{facility_id}"
    chunks = knowledge_chunks(scope, name, [("공식 시설 정보", text)], images=images or [])
    store.replace_knowledge(scope, code=code, name=name,
        url=f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={facility_id}",
        fetched_at=STAMP, payload={}, chunks=chunks)


def menu(store, *, code="re15", name="창업보육센터식당"):
    result = CrawlResult(code=code, name=name, date=DAY.isoformat(),
        source_url="https://life.hanyang.ac.kr/theme/pages/facilities/detail.php?id=3&date=" + DAY.isoformat(),
        fetched_at=STAMP, html="", meals=[{"meal_type": "석식", "items": ["돈가스"], "price": "5,000원", "tags": []}])
    store.replace_collection(result, build_chunks(result))


def index_all(store):
    pending = store.pending()
    store.save_embeddings(pending, [[1.0, 0.0, 0.0]] * len(pending))


def ask(agent, message):
    return agent.chat(message, [], DAY)


def sent_prompt(client):
    return json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])


def test_post_office_alias_routes_to_exact_long_named_facility_profiles(school):
    store, client, agent = school
    profile(store)
    profile(store, facility_id=18, name="NEW YORK BURGER", text="일반음식점 운영시간 10:00~19:30")
    menu(store)
    index_all(store)
    result = ask(agent, "학교 우체국 위치와 운영시간")
    assert result["mode"] == "openai"
    prompt = sent_prompt(client)
    assert prompt["context"]["restaurant"] == "fac_49"
    assert prompt["context"]["resolved_facility_name"] == "안산한양대우편취급국"
    assert {source["restaurant_code"] for source in prompt["sources"]} == {"fac_49"}
    assert {source["doc_type"] for source in prompt["sources"]} == {"profile"}
    assert "학생복지관 1층" in prompt["sources"][0]["text"]


def test_dinner_what_time_question_includes_official_hours_profile(school):
    store, client, agent = school
    profile(store, facility_id=3, name="창업보육센터식당", code="re15",
            text="영업시간: 중식 11:30~13:30\n석식 17:00~19:00 (18:30배식대마감)")
    menu(store)
    index_all(store)
    assert ask(agent, "창업보육센터식당 저녁 몇 시까지야?")["mode"] == "openai"
    sources = sent_prompt(client)["sources"]
    assert any(source["doc_type"] == "profile" and "18:30배식대마감" in source["text"] for source in sources)
    assert {source["restaurant_code"] for source in sources} == {"re15"}


@pytest.mark.parametrize("related_codes", [["re13"], ["re12", "re13"]])
def test_breakfast_payment_question_includes_related_official_notice(school, related_codes):
    store, client, agent = school
    profile(store, facility_id=4, name="창의인재원식당", code="re13", text="위치: 창의관 1층")
    index_notice(store, {"title": "천원의 아침밥 결제 안내", "text": "창의관식당 천원의 아침밥은 키오스크에서 학생증으로 결제합니다.",
        "restaurant_codes": related_codes, "url": "https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id=99",
        "fetched_at": STAMP, "images": [], "assets": []}, 99)
    index_all(store)
    assert ask(agent, "창의관식당 천원 아침밥 키오스크결제")["mode"] == "openai"
    sources = sent_prompt(client)["sources"]
    assert any(source["doc_type"] == "notice" and "학생증으로 결제" in source["text"] for source in sources)
    assert sent_prompt(client)["context"]["restaurant"] == "re13"


def test_resolved_postal_profile_below_generic_similarity_gate_is_still_evidence(school):
    store, client, agent = school
    profile(store)
    pending = store.pending()
    store.save_embeddings(pending, [[0.2, 0.9797958971, 0.0]] * len(pending))
    assert agent.settings.min_similarity == 0.30
    result = ask(agent, "학교 우체국 위치와 운영시간")
    assert result["mode"] == "openai"
    prompt = sent_prompt(client)
    assert prompt["context"]["resolved_facility_name"] == "안산한양대우편취급국"
    assert {source["restaurant_code"] for source in prompt["sources"]} == {"fac_49"}
    assert "안산한양대우편취급국" in client.embeddings.create.call_args.kwargs["input"][0]


def test_pure_menu_question_keeps_similarity_gate_even_for_a_resolved_facility(school):
    store, client, agent = school
    profile(store, facility_id=3, name="창업보육센터식당", code="re15")
    menu(store)
    pending = store.pending()
    store.save_embeddings(pending, [[0.2, 0.9797958971, 0.0]] * len(pending))
    result = ask(agent, "창업보육센터식당 저녁 메뉴")
    assert result["mode"] == "insufficient"
    assert result["sources"] == []
    client.responses.create.assert_not_called()


def test_cited_facility_images_are_attached_automatically_without_a_photo_request(school):
    store, _, agent = school
    profile(store, images=[{"url": PHOTO_A, "alt": "우편취급국 공식 사진"}])
    index_all(store)
    result = ask(agent, "학교 우체국 위치와 운영시간")
    assert result["mode"] == "openai"
    assert [image["url"] for image in result["images"]] == [PHOTO_A]


def test_only_images_on_validated_selected_sources_are_attached(school):
    store, client, agent = school
    scope = "facility:49"
    chunks = knowledge_chunks(scope, "한양대학교 ERICA 우편취급국", [("첫 사진", "선택할 첫 시설 사진"), ("둘째 사진", "선택하지 않을 둘째 사진")])
    chunks[0]["images"] = [{"url": PHOTO_A, "alt": "선택한 사진"}]
    chunks[1]["images"] = [{"url": PHOTO_B, "alt": "선택하지 않은 사진"}]
    source_url = "https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id=49"
    store.replace_knowledge(scope, code="fac_49", name="한양대학교 ERICA 우편취급국", url=source_url,
                            fetched_at=STAMP, payload={}, chunks=chunks)
    index_all(store)

    def select_first_photo(**kwargs):
        prompt = json.loads(kwargs["input"][0]["content"])
        selected = next(source["id"] for source in prompt["sources"] if "선택할 첫" in source["text"])
        return Mock(output_text=json.dumps({"answer": f"공식 우체국 사진입니다. [{selected}]", "source_ids": [selected],
                                            "insufficient_evidence": False,
                                            "images": [{"url": "https://evil.example/forged.jpg"}]}))

    client.responses.create.side_effect = select_first_photo
    result = ask(agent, "학교 우체국 사진 보여줘")
    assert len(sent_prompt(client)["sources"]) == 2
    assert [image["url"] for image in result["images"]] == [PHOTO_A]
    assert result["images"][0]["source_url"] == source_url


def test_unsafe_remote_and_forged_media_paths_are_not_returned_as_images(school):
    store, _, agent = school
    urls = [PHOTO_A, PHOTO_A, "javascript:alert(1)", "https://evil.example/x.jpg", "/media/../../secret.jpg",
            "/media/not-a-content-hash.jpg", "/media/" + "b" * 64 + ".svg", "file:///private/secret.png"]
    profile(store, images=[{"url": url, "alt": "자료 이미지"} for url in urls])
    index_all(store)
    result = ask(agent, "학교 우체국 사진 보여줘")
    assert [image["url"] for image in result["images"]] == [PHOTO_A]


def test_unknown_citation_ids_cannot_release_source_images(school):
    store, client, agent = school
    profile(store, images=[{"url": PHOTO_A}])
    index_all(store)
    client.responses.create.side_effect = None
    client.responses.create.return_value.output_text = json.dumps({"answer": "사진입니다. [S999]",
        "source_ids": ["S999"], "insufficient_evidence": False})
    result = ask(agent, "학교 우체국 사진 보여줘")
    assert result["mode"] == "insufficient"
    assert result["sources"] == []
    assert result.get("images", []) == []


def test_image_markdown_is_removed_from_answer_text_while_source_photo_is_returned(school):
    store, client, agent = school
    profile(store, images=[{"url": PHOTO_A}])
    index_all(store)
    client.responses.create.side_effect = None
    client.responses.create.return_value.output_text = json.dumps({"answer": f"공식 사진입니다. [S1]\n![우체국]({PHOTO_A})",
        "source_ids": ["S1"], "insufficient_evidence": False})
    result = ask(agent, "학교 우체국 사진 보여줘")
    assert result["mode"] == "openai"
    assert "![" not in result["answer"]
    assert "/media/" not in result["answer"]
    assert result["images"][0]["url"] == PHOTO_A


def test_duplicate_chain_names_are_ambiguous_but_a_unique_qualified_name_resolves(school):
    store, _, _ = school
    profile(store, facility_id=42, name="7-eleven")
    profile(store, facility_id=43, name="7-eleven")
    profile(store, facility_id=44, name="7-eleven 학생복지관점")
    assert store.lookup_code("7-eleven 위치 알려줘") is None
    assert store.lookup_code("7-ELEVEN 학생복지관점 운영시간") == "fac_44"


def test_photo_preview_remains_explicit_and_never_calls_openai(school):
    store, client, agent = school
    profile(store, images=[{"url": PHOTO_A}])
    agent.settings.api_key = ""
    result = ask(agent, "학교 우체국 사진 보여줘")
    assert result["mode"] == "retrieval"
    assert result["model"] is None
    assert result["images"][0]["url"] == PHOTO_A
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


def test_school_facility_overview_uses_low_similarity_directory_instead_of_distracting_profiles(school):
    store, client, agent = school
    profile(store)
    profile(store, facility_id=18, name="NEW YORK BURGER", text="일반음식점과 매장 이용 소개")
    index_notice(store, {"title": "학교 시설 운영 공지", "text": "일부 매장의 운영시간을 안내합니다.",
        "restaurant_codes": [], "url": "https://life.hanyang.ac.kr/theme/pages/notice/detail.php?id=100",
        "fetched_at": STAMP, "images": [], "assets": []}, 100)
    directory_url = "https://life.hanyang.ac.kr/theme/pages/facilities/index.php"
    directory = knowledge_chunks("page:facility-directory", "ERICA 공개 시설 목록",
        [("시설 종류별 목록", "공개 시설 총 48개. 구내식당, 일반음식점, 카페, 편의점, 생활서비스, 인쇄 및 문구 시설을 안내합니다.")], kind="page")
    store.replace_knowledge("page:facility-directory", code="campus", name="ERICA 공개 시설 목록",
        url=directory_url, fetched_at=STAMP, payload={"facility_count": 48}, chunks=directory)
    pending = store.pending()
    store.save_embeddings(pending, [[0.2, 0.9797958971, 0.0] if row["doc_type"] == "page" else [1.0, 0.0, 0.0]
                                    for row in pending])
    result = ask(agent, "학교에 어떤 시설들이 있는지 종류별로 소개해줘")
    assert result["mode"] == "openai"
    sources = sent_prompt(client)["sources"]
    assert len(sources) == 1
    assert sources[0]["doc_type"] == "page"
    assert sources[0]["url"] == directory_url
    assert "48개" in sources[0]["text"]
    assert sent_prompt(client)["context"]["restaurant"] is None
    assert sent_prompt(client)["context"]["available_document_count"] == 1
    assert result["sources"][0]["url"] == directory_url

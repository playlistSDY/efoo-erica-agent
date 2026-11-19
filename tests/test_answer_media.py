"""Automatic images remain bound to the answer's actual source evidence."""

from datetime import date, datetime, timedelta, timezone
import json
from unittest.mock import Mock, patch

import pytest

with patch("dotenv.load_dotenv"):
    from app.crawler import CrawlResult
    from app.indexing import build_chunks, knowledge_chunks
    from app.rag import Agent, answer_images
    from app.settings import Settings
    from app.store import Store


DAY = date(2026, 9, 22)
STAMP = datetime.now(timezone.utc).isoformat()
PHOTO_A = "/media/" + "a" * 64 + ".jpg"
PHOTO_B = "/media/" + "b" * 64 + ".png"
PHOTO_C = "/media/" + "c" * 64 + ".webp"


@pytest.fixture
def media_agent(tmp_path):
    settings = Settings(data_dir=tmp_path, database_url="", api_key="isolated-media-test", embedding_dimensions=3)
    store = Store(settings)
    client = Mock()
    client.embeddings.create.return_value.data = [Mock(embedding=[1.0, 0.0, 0.0])]

    def reply(**kwargs):
        source = json.loads(kwargs["input"][0]["content"])["sources"][0]
        return Mock(output_text=json.dumps({"answer": f"찾은 공식 자료를 안내해 드릴게요. [{source['id']}]",
                                            "source_ids": [source["id"]], "insufficient_evidence": False}))

    client.responses.create.side_effect = reply
    return store, client, Agent(store, settings, client=client)


def menu(store, *, code="re12", name="학생식당", day=DAY, item="칼국수", image=PHOTO_A):
    original = "https://life.hanyang.ac.kr/uploads/" + image.rsplit("/", 1)[-1]
    result = CrawlResult(code=code, name=name, date=day.isoformat(),
        source_url=f"https://life.hanyang.ac.kr/menu?restaurant={code}&date={day.isoformat()}",
        fetched_at=STAMP, html="", meals=[{"meal_type": "중식", "items": [item], "price": "5,000원", "tags": [], "image": original}],
        assets=[{"url": original, "local_path": image, "downloaded": True, "content_type": "image/jpeg"}])
    store.replace_collection(result, build_chunks(result))


def profile(store, *, code="fac_49", facility_id=49, name="안산한양대우편취급국", image=PHOTO_A):
    scope = f"facility:{facility_id}"
    store.replace_knowledge(scope, code=code, name=name,
        url=f"https://life.hanyang.ac.kr/theme/pages/facilities/shop-detail.php?id={facility_id}",
        fetched_at=STAMP, payload={}, chunks=knowledge_chunks(scope, name,
            [("시설 안내", "위치: 학생복지관 1층. 운영시간: 평일 09:00~18:00")], images=[{"url": image}]))


def index_all(store):
    rows = store.pending()
    store.save_embeddings(rows, [[1.0, 0.0, 0.0]] * len(rows))


def prompt(client):
    return json.loads(client.responses.create.call_args.kwargs["input"][0]["content"])


def test_menu_answer_automatically_attaches_its_cached_photo_without_photo_keyword(media_agent):
    store, client, agent = media_agent
    menu(store)
    index_all(store)
    result = agent.chat("학생식당 점심 메뉴 알려줘", [], DAY)
    assert result["mode"] == "openai"
    assert [image["url"] for image in result["images"]] == [PHOTO_A]
    assert result["images"][0]["source_url"] == result["sources"][0]["url"]
    assert result["images"][0]["original_url"].startswith("https://life.hanyang.ac.kr/")
    client.responses.create.assert_called_once()


def test_facility_answer_automatically_attaches_a_basic_existing_photo_record(media_agent):
    store, _, agent = media_agent
    profile(store)
    index_all(store)
    result = agent.chat("학교 우체국 위치와 운영시간", [], DAY)
    assert result["mode"] == "openai"
    assert result["images"] == [{"url": PHOTO_A, "source_url": result["sources"][0]["url"]}]


def test_images_require_actual_inline_citations_even_if_model_declares_extra_source_ids(media_agent):
    store, client, agent = media_agent
    scope = "facility:49"
    chunks = knowledge_chunks(scope, "안산한양대우편취급국",
        [("선택할 자료", "답변에 인용할 시설 안내"), ("인용하지 않을 자료", "추가 시설 안내")])
    chunks[0]["images"] = [{"url": PHOTO_A}]
    chunks[1]["images"] = [{"url": PHOTO_B}]
    store.replace_knowledge(scope, code="fac_49", name="안산한양대우편취급국",
        url="https://life.hanyang.ac.kr/facility?id=49", fetched_at=STAMP, payload={}, chunks=chunks)
    index_all(store)

    def declare_but_do_not_cite_second(**kwargs):
        sources = json.loads(kwargs["input"][0]["content"])["sources"]
        cited = next(source for source in sources if "답변에 인용할" in source["text"])
        return Mock(output_text=json.dumps({"answer": f"우체국의 공식 안내예요. [{cited['id']}]",
            "source_ids": [source["id"] for source in sources], "insufficient_evidence": False,
            "images": [{"url": "https://untrusted.example/forged.jpg"}]}))

    client.responses.create.side_effect = declare_but_do_not_cite_second
    result = agent.chat("학교 우체국 위치 알려줘", [], DAY)
    assert len(prompt(client)["sources"]) == 2
    assert [image["url"] for image in result["images"]] == [PHOTO_A]


def test_food_answer_never_uses_other_dates_cafeterias_or_a_facility_cover_photo(media_agent):
    store, client, agent = media_agent
    menu(store)
    menu(store, day=DAY - timedelta(days=1), item="어제 돈가스", image=PHOTO_B)
    menu(store, code="re11", name="교직원식당", item="교직원 메뉴", image=PHOTO_C)
    profile(store, code="re12", facility_id=1, name="학생식당", image=PHOTO_C)
    index_all(store)
    result = agent.chat("학생식당 점심 메뉴 알려줘", [], DAY)
    sources = prompt(client)["sources"]
    assert len(sources) == 1
    assert sources[0]["date"] == DAY.isoformat()
    assert sources[0]["restaurant_code"] == "re12"
    assert sources[0]["doc_type"] == "menu"
    assert [image["url"] for image in result["images"]] == [PHOTO_A]


def test_data_preview_also_attaches_matching_cached_images_without_a_photo_keyword(media_agent):
    store, client, agent = media_agent
    menu(store)
    agent.settings.api_key = ""
    result = agent.chat("학생식당 점심 메뉴 알려줘", [], DAY)
    assert result["mode"] == "retrieval"
    assert result["model"] is None
    assert [image["url"] for image in result["images"]] == [PHOTO_A]
    client.embeddings.create.assert_not_called()
    client.responses.create.assert_not_called()


@pytest.mark.parametrize("payload", [
    {"answer": "자료를 확인할 수 없어요. [S1]", "source_ids": ["S1"], "insufficient_evidence": True},
    {"answer": "만들어 낸 출처예요. [S999]", "source_ids": ["S999"], "insufficient_evidence": False},
])
def test_insufficient_or_invalidly_cited_answers_release_no_images(media_agent, payload):
    store, client, agent = media_agent
    profile(store)
    index_all(store)
    client.responses.create.side_effect = None
    client.responses.create.return_value.output_text = json.dumps(payload)
    result = agent.chat("학교 우체국 위치 알려줘", [], DAY)
    assert result["mode"] == "insufficient"
    assert result["sources"] == []
    assert result.get("images", []) == []


def test_explicit_photo_followup_keeps_the_facility_from_prior_history(media_agent):
    store, client, agent = media_agent
    profile(store)
    profile(store, code="fac_18", facility_id=18, name="NEW YORK BURGER", image=PHOTO_B)
    index_all(store)
    result = agent.chat("그럼 사진도 보여줘", [{"role": "user", "content": "학교 우체국 위치 알려줘"}], DAY)
    assert prompt(client)["context"]["restaurant"] == "fac_49"
    assert {source["restaurant_code"] for source in prompt(client)["sources"]} == {"fac_49"}
    assert [image["url"] for image in result["images"]] == [PHOTO_A]


def test_media_validator_rejects_external_traversal_unhashed_and_svg_references():
    urls = ["https://life.hanyang.ac.kr/public.jpg", "https://untrusted.example/photo.jpg", "javascript:alert(1)",
            "/media/../photo.jpg", "/media/%2e%2e/private.jpg", "/media/not-a-hash.png",
            "/media/" + "a" * 64 + ".svg", "file:///private/photo.jpg", PHOTO_A + "?redirect=https://untrusted.example"]
    source = {"url": "https://life.hanyang.ac.kr/facility?id=49", "images": [{"url": url} for url in urls] + [{}]}
    assert answer_images([source]) == []


def test_media_validator_deduplicates_caps_six_and_preserves_first_source_attribution():
    urls = ["/media/" + format(index, "064x") + ".jpg" for index in range(8)]
    first = {"url": "https://life.hanyang.ac.kr/first", "images": [{"url": url} for url in urls]}
    duplicate = {"url": "https://life.hanyang.ac.kr/second", "images": [{"url": urls[0]}]}
    images = answer_images([first, duplicate])
    assert [image["url"] for image in images] == urls[:6]
    assert all(image["source_url"] == first["url"] for image in images)


def test_unregistered_menu_evidence_never_displays_a_leftover_food_photo():
    source = {"url": "https://life.hanyang.ac.kr/menu?date=2026-09-22", "doc_type": "menu",
              "text": "공식 페이지에 이 식사의 메뉴가 등록되지 않았습니다. 휴무인지는 확인할 수 없습니다.",
              "images": [{"url": PHOTO_A, "alt": "이전 자료에서 남은 메뉴 사진"}]}
    assert answer_images([source]) == []

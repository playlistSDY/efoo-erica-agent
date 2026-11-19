from concurrent.futures import ThreadPoolExecutor
from datetime import date
from hashlib import sha256
import sqlite3
import threading
import unicodedata
from unittest.mock import patch
from uuid import UUID

import pytest

from app.conversations import ConversationStore

with patch("dotenv.load_dotenv"):
    from app.settings import Settings
    from app.store import Store


@pytest.fixture
def setup_store(tmp_path):
    settings = Settings(data_dir=tmp_path, database_url="", api_key="", embedding_dimensions=3)
    store = Store(settings)
    conversations = ConversationStore(store)
    token, owner = conversations.ensure_owner(None)
    return settings, store, conversations, token, owner


def answer(text="학생식당 메뉴입니다. [S1]"):
    return {"answer": text, "mode": "openai", "model": "gpt-6.1-terra", "date": "2026-11-13",
        "sources": [{"id": "S1", "title": "공식 학생식당", "url": "https://life.hanyang.ac.kr/menu?id=1",
                     "text": "칼국수 5,000원"}],
        "images": [{"url": "/media/" + "a" * 64 + ".jpg", "alt": "식당 사진"}],
        "trace": ["자료 검색", "출처 검증"], "request": {"date": "2026-11-13", "restaurant": "re12"}}


def test_owner_tokens_are_random_hashed_and_only_known_tokens_are_reused(setup_store):
    _, store, service, token, owner = setup_store
    assert len(token) == 43
    assert owner == sha256(token.encode()).hexdigest()
    assert service.ensure_owner(token) == (token, owner)
    replacement, other_owner = service.ensure_owner("a" * 43)
    assert replacement != "a" * 43
    assert other_owner != owner
    malformed, malformed_owner = service.ensure_owner("../../untrusted")
    assert malformed != "../../untrusted"
    assert malformed_owner not in {owner, other_owner}
    with store.connection() as conn:
        hashes = {row["token_hash"] for row in conn.execute("SELECT token_hash FROM chat_sessions")}
    assert hashes == {owner, other_owner, malformed_owner}
    assert token not in hashes


def test_refresh_and_new_service_instances_restore_the_full_answer_payload(setup_store):
    settings, _, service, token, owner = setup_store
    first = service.current(owner)
    assert str(UUID(first["id"])) == first["id"]
    assert first["messages"] == []
    expected = answer()
    service.append_turn(owner, first["id"], "오늘 학생식당 메뉴와 사진 알려줘", expected, "turn-1")
    reconnected = ConversationStore(Store(settings))
    new_token, restored_owner = reconnected.ensure_owner(token)
    restored = reconnected.current(restored_owner)
    assert new_token == token
    assert restored["id"] == first["id"]
    assert [message["role"] for message in restored["messages"]] == ["user", "assistant"]
    assert restored["messages"][1]["payload"] == expected
    assert restored["messages"][1]["content"] == expected["answer"]
    assert "owner_hash" not in restored
    assert "token" not in restored


def test_new_chat_creates_an_empty_current_chat_without_deleting_old_history(setup_store):
    _, _, service, _, owner = setup_store
    old = service.current(owner)
    service.append_turn(owner, old["id"], "학식 메뉴", answer(), "turn-old")
    new = service.create(owner)
    assert new["id"] != old["id"]
    assert new["messages"] == []
    assert service.current(owner)["id"] == new["id"]
    restored = service.get(owner, old["id"])
    assert len(restored["messages"]) == 2
    assert service.current(owner)["id"] == old["id"]
    assert len(service.list(owner)) == 2


def test_late_old_chat_requests_and_retries_never_switch_the_current_chat(setup_store):
    _, _, service, _, owner = setup_store
    old = service.current(owner)
    new = service.create(owner)
    service.get(owner, old["id"], mark_current=False)
    service.append_turn(owner, old["id"], "늦게 도착한 질문", answer(), "late-turn")
    service.append_turn(owner, old["id"], "늦게 도착한 질문", answer("재시도 생성 결과"), "late-turn")
    service.get_turn(owner, old["id"], "late-turn", message="늦게 도착한 질문")
    service.recent_history(owner, old["id"])
    assert service.current(owner)["id"] == new["id"]
    assert len(service.get(owner, old["id"], mark_current=False)["messages"]) == 2


def test_different_browser_owners_cannot_read_write_or_activate_each_others_chats(setup_store):
    _, _, service, _, owner = setup_store
    private = service.current(owner)
    service.append_turn(owner, private["id"], "첫 브라우저 질문", answer(), "private-turn")
    _, other_owner = service.ensure_owner(None)
    other = service.current(other_owner)
    operations = [lambda: service.get(other_owner, private["id"]),
                  lambda: service.recent_history(other_owner, private["id"]),
                  lambda: service.get_turn(other_owner, private["id"], "private-turn"),
                  lambda: service.append_turn(other_owner, private["id"], "침입 질문", answer(), "attack-turn")]
    for operation in operations:
        with pytest.raises(KeyError):
            operation()
    with pytest.raises(KeyError):
        with service.turn_lock(other_owner, private["id"]):
            pytest.fail("Ownership must be checked before entering the turn")
    assert [item["id"] for item in service.list(other_owner)] == [other["id"]]
    assert service.current(other_owner)["id"] == other["id"]
    assert len(service.current(owner)["messages"]) == 2


def test_replayed_turn_ids_return_the_original_payload_and_conflicts_are_rejected(setup_store):
    _, _, service, _, owner = setup_store
    chat = service.current(owner)
    expected = answer()
    service.append_turn(owner, chat["id"], "학생식당 메뉴", expected, "same-turn")
    replay = service.append_turn(owner, chat["id"], "학생식당 메뉴", answer("두 번째 모델 응답"), "same-turn")
    assert len(replay["messages"]) == 2
    assert service.get_turn(owner, chat["id"], "same-turn", message="학생식당 메뉴") == expected
    assert replay["messages"][1]["payload"] == expected
    assert service.get_turn(owner, chat["id"], "missing-turn") is None
    with pytest.raises(ValueError, match="다른 질문"):
        service.get_turn(owner, chat["id"], "same-turn", message="다른 질문")
    with pytest.raises(ValueError, match="다른 질문"):
        service.append_turn(owner, chat["id"], "다른 질문", expected, "same-turn")
    assert len(service.current(owner)["messages"]) == 2


def test_failure_in_second_message_rolls_back_the_whole_turn_and_title(setup_store):
    _, store, service, _, owner = setup_store
    chat = service.current(owner)
    with store.connection() as conn:
        conn.execute("""CREATE TRIGGER fail_assistant_message BEFORE INSERT ON chat_messages
          WHEN NEW.role='assistant' BEGIN SELECT RAISE(FAIL,'simulated storage failure'); END""")
    with pytest.raises(sqlite3.IntegrityError, match="simulated storage failure"):
        service.append_turn(owner, chat["id"], "저장 실패 검증", answer(), "failed-turn")
    restored = service.current(owner)
    assert restored["messages"] == []
    assert restored["title"] == "새 대화"
    assert service.get_turn(owner, chat["id"], "failed-turn") is None
    with store.connection() as conn:
        conn.execute("DROP TRIGGER fail_assistant_message")
    success = service.append_turn(owner, chat["id"], "다시 저장", answer(), "retry-turn")
    assert [message["ordinal"] for message in success["messages"]] == [0, 1]


def test_server_history_is_ordered_bounded_and_truncated_without_altering_saved_content(setup_store):
    _, _, service, _, owner = setup_store
    chat = service.current(owner)
    long_answer = "긴 답변 " * 1200
    for turn in range(8):
        service.append_turn(owner, chat["id"], f"질문 {turn}", answer(long_answer), f"turn-{turn}")
    history = service.recent_history(owner, chat["id"], limit=100)
    assert len(history) == 12
    assert history[0] == {"role": "user", "content": "질문 2"}
    assert [item["role"] for item in history] == ["user", "assistant"] * 6
    assert all(len(item["content"]) <= 5000 for item in history)
    assert len(service.current(owner)["messages"][-1]["content"]) > 5000
    assert service.recent_history(owner, chat["id"], limit=0) == []


def test_summaries_are_newest_first_bounded_and_have_no_full_answer_payloads(setup_store):
    _, _, service, _, owner = setup_store
    chats = [service.create(owner) for _ in range(52)]
    service.append_turn(owner, chats[0]["id"], "  첫 질문\n이어지는 내용  ", answer(), "summary-turn")
    summaries = service.list(owner)
    assert len(summaries) == 50
    assert summaries[0]["id"] == chats[0]["id"]
    assert summaries[0]["title"] == "첫 질문 이어지는 내용"
    assert summaries[0]["message_count"] == 2
    assert summaries[0]["last_message"] == answer()["answer"]
    assert all("messages" not in summary and "payload" not in summary and "owner_hash" not in summary for summary in summaries)
    assert summaries[1]["id"] == chats[-1]["id"]


def test_concurrent_services_keep_message_pairs_atomic_with_unique_monotonic_positions(setup_store):
    settings, _, service, _, owner = setup_store
    chat = service.current(owner)
    another = ConversationStore(Store(settings))
    services = [service, another]

    def append(number):
        return services[number % 2].append_turn(owner, chat["id"], f"질문 {number}", answer(str(number)), f"turn-{number}")

    with ThreadPoolExecutor(max_workers=4) as executor:
        list(executor.map(append, range(8)))
    messages = service.get(owner, chat["id"], mark_current=False)["messages"]
    assert [message["ordinal"] for message in messages] == list(range(16))
    assert [message["role"] for message in messages] == ["user", "assistant"] * 8
    for offset in range(0, len(messages), 2):
        assert messages[offset]["turn_id"] == messages[offset + 1]["turn_id"]
    with ThreadPoolExecutor(max_workers=2) as executor:
        list(executor.map(lambda item: item.append_turn(owner, chat["id"], "중복 요청", answer(), "concurrent-retry"), services))
    assert len(service.get(owner, chat["id"], mark_current=False)["messages"]) == 18


def test_turn_lock_serializes_generation_and_releases_registry_entries(setup_store):
    _, _, service, _, owner = setup_store
    chat = service.current(owner)
    attempted, entered = threading.Event(), threading.Event()

    def wait_for_lock():
        attempted.set()
        with service.turn_lock(owner, chat["id"]):
            entered.set()

    with ThreadPoolExecutor(max_workers=1) as executor:
        with service.turn_lock(owner, chat["id"]):
            waiting = executor.submit(wait_for_lock)
            assert attempted.wait(1)
            assert not entered.wait(0.05)
        waiting.result(timeout=2)
    assert entered.is_set()
    assert (owner, chat["id"]) not in service._locks


def test_invalid_payloads_cannot_leave_partial_turns(setup_store):
    _, _, service, _, owner = setup_store
    chat = service.current(owner)
    for message, response, turn_id in ((" ", answer(), "bad-1"), ("질문", {"mode": "openai"}, "bad-2"),
                                       ("질문", answer(), ""), ("질문", answer(), "x" * 129)):
        with pytest.raises(ValueError):
            service.append_turn(owner, chat["id"], message, response, turn_id)
    with pytest.raises(TypeError):
        service.append_turn(owner, chat["id"], "질문", {**answer(), "invalid": object()}, "bad-object")
    assert service.current(owner)["messages"] == []


def test_old_relative_day_is_anchored_to_message_creation_in_korea_without_changing_displayed_text(setup_store):
    _, store, service, _, owner = setup_store
    chat = service.current(owner)
    original = "내일학생식당점심 알려줘"
    service.append_turn(owner, chat["id"], original, answer("내일 메뉴입니다. [S1]"), "dated-turn")
    with store.connection() as conn:
        # UTC November 12 is already November 13 in Korea. The old 'tomorrow'
        # therefore means November 14, not November 13 or November 15.
        conn.execute("UPDATE chat_messages SET created_at=? WHERE conversation_id=?",
                     ("2026-11-12T15:30:00+00:00", chat["id"]))
    normalized = service.recent_history(owner, chat["id"], reference_date=date(2026, 11, 14))
    assert normalized[0] == {"role": "user", "content": "2026-11-14학생식당점심 알려줘"}
    assert normalized[1]["content"] == "내일 메뉴입니다. [S1]"
    assert service.get(owner, chat["id"], mark_current=False)["messages"][0]["content"] == original
    assert service.recent_history(owner, chat["id"])[0]["content"] == original
    assert service.recent_history(owner, chat["id"], reference_date=date(2026, 11, 13))[0]["content"] == original


@pytest.mark.parametrize("message,expected", [
    ("오늘 점심과 내일 저녁", "2026-11-13 점심과 2026-11-14 저녁"),
    ("어제와 모레 학식", "2026-11-12와 2026-11-15 학식"),
    ("2026-11-20 학생식당과 내일 메뉴", "2026-11-20 학생식당과 내일 메뉴"),
    ("11월 20일 학생식당과 내일 메뉴", "11월 20일 학생식당과 내일 메뉴"),
    ("오늘날의 학교 이야기와 내일상회 위치", "오늘날의 학교 이야기와 내일상회 위치"),
])
def test_historical_day_normalization_preserves_explicit_dates_and_ordinary_names(setup_store, message, expected):
    _, store, service, _, owner = setup_store
    chat = service.current(owner)
    service.append_turn(owner, chat["id"], message, answer(), "date-example")
    with store.connection() as conn:
        conn.execute("UPDATE chat_messages SET created_at=? WHERE conversation_id=?",
                     ("2026-11-13T01:00:00+00:00", chat["id"]))
    history = service.recent_history(owner, chat["id"], reference_date=date(2026, 11, 14))
    assert history[0]["content"] == expected
    assert service.get(owner, chat["id"], mark_current=False)["messages"][0]["content"] == message


@pytest.mark.parametrize("original", [
    "ㄴㅐㅇㅣㄹ학생식당점심 알려줘",
    unicodedata.normalize("NFD", "내일학생식당점심 알려줘"),
])
def test_old_jamo_relative_day_is_fixed_in_search_copy_while_original_history_is_preserved(setup_store, original):
    _, store, service, _, owner = setup_store
    chat = service.current(owner)
    service.append_turn(owner, chat["id"], original, answer(), "jamo-dated-turn")
    with store.connection() as conn:
        conn.execute("UPDATE chat_messages SET created_at=? WHERE conversation_id=?",
                     ("2026-11-12T15:30:00+00:00", chat["id"]))
    history = service.recent_history(owner, chat["id"], reference_date=date(2026, 11, 14))
    assert history[0]["content"] == "2026-11-14학생식당점심 알려줘"
    assert service.get(owner, chat["id"], mark_current=False)["messages"][0]["content"] == original
    assert service.recent_history(owner, chat["id"])[0]["content"] == original
    assert service.recent_history(owner, chat["id"], reference_date=date(2026, 11, 13))[0]["content"] == original


def test_nfd_explicit_date_and_unrelated_facility_names_keep_existing_history_behavior(setup_store):
    _, store, service, _, owner = setup_store
    chat = service.current(owner)
    original = unicodedata.normalize("NFD", "11월 20일 내일상회 위치 알려줘")
    service.append_turn(owner, chat["id"], original, answer(), "explicit-nfd-date")
    with store.connection() as conn:
        conn.execute("UPDATE chat_messages SET created_at=? WHERE conversation_id=?",
                     ("2026-11-13T01:00:00+00:00", chat["id"]))
    assert service.recent_history(owner, chat["id"], reference_date=date(2026, 11, 14))[0]["content"] == original
    assert service.get(owner, chat["id"], mark_current=False)["messages"][0]["content"] == original


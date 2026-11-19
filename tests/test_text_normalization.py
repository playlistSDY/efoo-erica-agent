import unicodedata

import pytest

from app.text_normalization import normalize_korean_input


def test_reported_compatibility_jamo_question_becomes_recognizable_korean():
    assert normalize_korean_input("ㄴㅐㅇㅣㄹ ㅎㅏㄱㅅㅣㄱ ㅁㅜㅓㄴㅏㅇㅗㅏ") == "내일 학식 뭐나와"


def test_canonical_nfd_and_already_composed_korean_share_the_same_query():
    original = "내일 학생식당이랑 창의인재원식당 메뉴 알려줘"
    assert normalize_korean_input(unicodedata.normalize("NFD", original)) == original
    assert normalize_korean_input(original) == original


def test_mixed_names_emoji_newlines_and_code_punctuation_are_preserved():
    text = "학생식당 🍚 ㅎㅏㄱㅅㅣㄱ?\n  menu(price<=6000); [S1]\n창의인재원식당!"
    assert normalize_korean_input(text) == "학생식당 🍚 학식?\n  menu(price<=6000); [S1]\n창의인재원식당!"


def test_url_paths_are_opaque_while_surrounding_question_is_normalized():
    url = "https://example.com/ㄴㅐㅇㅣㄹ?menu=ㅎㅏㄱㅅㅣㄱ#ㅁㅜㅓ"
    assert normalize_korean_input("ㅎㅏㄱㅅㅣㄱ " + url + " ㄴㅐㅇㅣㄹ") == "학식 " + url + " 내일"


@pytest.mark.parametrize("letters,expected", [
    ("ㅇㅗㅏ", "와"), ("ㅇㅗㅐ", "왜"), ("ㅇㅗㅣ", "외"),
    ("ㅇㅜㅓ", "워"), ("ㅇㅜㅔ", "웨"), ("ㅇㅜㅣ", "위"),
    ("ㅇㅡㅣ", "의"), ("ㅇㅗㅏㅣ", "왜"), ("ㅇㅜㅓㅣ", "웨"),
])
def test_combinable_vowels_follow_ime_syllable_composition(letters, expected):
    assert normalize_korean_input(letters) == expected


@pytest.mark.parametrize("letters,expected", [
    ("ㄴㅐㅇㅣㄹ", "내일"), ("ㅎㅏㄱㅅㅣㄱ", "학식"),
    ("ㄱㅏㄱㅏ", "가가"), ("ㄱㅏㄹㄱㅗ", "갈고"),
    ("ㄸㅏㄹㄱㅣ", "딸기"), ("ㄱㅏㄳㅏ", "각사"),
    ("ㄱㅏㅄㅏ", "갑사"),
])
def test_final_consonant_lookahead_keeps_the_next_syllable_initial(letters, expected):
    assert normalize_korean_input(letters) == expected


@pytest.mark.parametrize("letters,expected", [
    ("ㄴㅓㄱㅅ", "넋"), ("ㅇㅏㄴㅈㅇㅏ", "앉아"), ("ㅁㅏㄴㅎㅇㅏ", "많아"),
    ("ㄷㅏㄹㄱ", "닭"), ("ㅅㅏㄹㅁ", "삶"), ("ㄴㅓㄹㅂ", "넓"),
    ("ㄱㅗㄹㅅ", "곬"), ("ㅎㅏㄹㅌ", "핥"), ("ㅇㅡㄹㅍ", "읊"),
    ("ㅅㅣㄹㅎㅇㅓ", "싫어"), ("ㄱㅏㅂㅅㅇㅣ", "값이"),
    ("ㄲㅏㄲ", "깎"),
])
def test_final_clusters_remain_on_the_previous_syllable_when_appropriate(letters, expected):
    assert normalize_korean_input(letters) == expected


@pytest.mark.parametrize("text", ["ㅋㅋㅋㅋ", "ㅎㅎ", "ㄱㅅ", "ㅎㅇ", "ㄴㄴ", "ㅜㅜ", "ㅏㅣ", "ㄳ", "", "λ + x = 2"])
def test_unpaired_jamo_shorthand_and_unknown_symbols_are_not_expanded(text):
    assert normalize_korean_input(text) == text


def test_normalization_is_idempotent_and_rejects_non_text():
    text = "ㄴㅐㅇㅣㄹ ㅎㅏㄱㅅㅣㄱ 뭐나와? ㅋㅋ"
    normalized = normalize_korean_input(text)
    assert normalize_korean_input(normalized) == normalized
    with pytest.raises(TypeError):
        normalize_korean_input(None)

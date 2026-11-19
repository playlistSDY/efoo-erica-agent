"""Normalize recognizable Korean IME input for an internal search copy only."""

from __future__ import annotations

import re
import unicodedata


_INITIALS = "ㄱㄲㄴㄷㄸㄹㅁㅂㅃㅅㅆㅇㅈㅉㅊㅋㅌㅍㅎ"
_VOWELS = "ㅏㅐㅑㅒㅓㅔㅕㅖㅗㅘㅙㅚㅛㅜㅝㅞㅟㅠㅡㅢㅣ"
_FINALS = "\0ㄱㄲㄳㄴㄵㄶㄷㄹㄺㄻㄼㄽㄾㄿㅀㅁㅂㅄㅅㅆㅇㅈㅊㅋㅌㅍㅎ"
_VOWEL_PAIRS = {
    "ㅗㅏ": "ㅘ", "ㅗㅐ": "ㅙ", "ㅗㅣ": "ㅚ",
    "ㅜㅓ": "ㅝ", "ㅜㅔ": "ㅞ", "ㅜㅣ": "ㅟ", "ㅡㅣ": "ㅢ",
    "ㅘㅣ": "ㅙ", "ㅝㅣ": "ㅞ",
}
_FINAL_PAIRS = {
    "ㄱㅅ": "ㄳ", "ㄴㅈ": "ㄵ", "ㄴㅎ": "ㄶ",
    "ㄹㄱ": "ㄺ", "ㄹㅁ": "ㄻ", "ㄹㅂ": "ㄼ", "ㄹㅅ": "ㄽ",
    "ㄹㅌ": "ㄾ", "ㄹㅍ": "ㄿ", "ㄹㅎ": "ㅀ", "ㅂㅅ": "ㅄ",
}
_SPLIT_FINALS = {value: tuple(pair) for pair, value in _FINAL_PAIRS.items()}
_JAMO_RUN = re.compile(r"[ㄱ-ㅎㅏ-ㅣ]+")
_URL = re.compile(r"https?://[^\s<>()]+", re.IGNORECASE)


def _compose_run(match: re.Match) -> str:
    original = match.group()
    # Preserve laughter, shorthand such as ㄱㅅ and all other vowel-free runs.
    if not any(character in _VOWELS for character in original):
        return original
    letters = list(original)
    output = []
    position = 0
    while position < len(letters):
        initial = letters[position]
        if initial not in _INITIALS or position + 1 == len(letters) or letters[position + 1] not in _VOWELS:
            output.append(initial)
            position += 1
            continue

        vowel = letters[position + 1]
        cursor = position + 2
        while cursor < len(letters) and vowel + letters[cursor] in _VOWEL_PAIRS:
            vowel = _VOWEL_PAIRS[vowel + letters[cursor]]
            cursor += 1

        final = "\0"
        if cursor < len(letters) and letters[cursor] in _FINALS[1:]:
            consonant = letters[cursor]
            # A precombined final cluster splits like two individual keys when
            # its second consonant begins the next vowel-led syllable.
            if consonant in _SPLIT_FINALS and cursor + 1 < len(letters) and letters[cursor + 1] in _VOWELS:
                letters[cursor:cursor + 1] = _SPLIT_FINALS[consonant]
                consonant = letters[cursor]
            follows_vowel = cursor + 1 < len(letters) and letters[cursor + 1] in _VOWELS
            if not follows_vowel:
                final = consonant
                cursor += 1
                if cursor < len(letters):
                    cluster = _FINAL_PAIRS.get(final + letters[cursor])
                    second_begins_syllable = cursor + 1 < len(letters) and letters[cursor + 1] in _VOWELS
                    if cluster and not second_begins_syllable:
                        final = cluster
                        cursor += 1

        syllable = 0xAC00 + (_INITIALS.index(initial) * 21 + _VOWELS.index(vowel)) * 28 + _FINALS.index(final)
        output.append(chr(syllable))
        position = cursor
    return "".join(output)


def _normalize_words(text: str) -> str:
    return _JAMO_RUN.sub(_compose_run, unicodedata.normalize("NFC", text))


def normalize_korean_input(text: str) -> str:
    """Compose canonical NFD and recognizable compatibility-jamo syllables.

    Unpaired letters, abbreviations, spacing and punctuation are retained.
    URL strings are opaque so normalization cannot change a referenced path.
    Callers should use this on a search copy and retain the original message.
    """
    if not isinstance(text, str):
        raise TypeError("text must be a string")
    output = []
    start = 0
    for match in _URL.finditer(text):
        output.extend((_normalize_words(text[start:match.start()]), match.group()))
        start = match.end()
    output.append(_normalize_words(text[start:]))
    return "".join(output)

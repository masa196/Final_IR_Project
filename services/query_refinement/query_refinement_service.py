from collections import Counter
from nltk.corpus import wordnet, words

from services.preprocessing.preprocessing_service import (
    preprocess_lotte,
)


ENGLISH_WORDS = set(w.lower() for w in words.words() if w.isalpha())
MAX_SYNONYMS_PER_TOKEN = 2
HISTORY_BOOST_COUNT = 2


def _edits1(word: str) -> set[str]:
    letters = "abcdefghijklmnopqrstuvwxyz"
    splits = [(word[:i], word[i:]) for i in range(len(word) + 1)]
    deletes = {L + R[1:] for L, R in splits if R}
    transposes = {L + R[1] + R[0] + R[2:] for L, R in splits if len(R) > 1}
    replaces = {L + c + R[1:] for L, R in splits if R for c in letters}
    inserts = {L + c + R for L, R in splits for c in letters}
    return deletes | transposes | replaces | inserts


def _unpluralize(word: str) -> str | None:
    if word.endswith("ies") and len(word) > 4:
        base = word[:-3] + "y"
        if base in ENGLISH_WORDS:
            return base
    if word.endswith("es") and len(word) > 4:
        base = word[:-2]
        if base in ENGLISH_WORDS:
            return base
    if word.endswith("s") and not word.endswith("ss") and len(word) > 3:
        base = word[:-1]
        if base in ENGLISH_WORDS:
            return base
    return None


def correct_spelling(text: str) -> str:
    tokens = text.lower().split()
    corrected = []

    for token in tokens:
        if not token.isalpha() or len(token) <= 3 or token in ENGLISH_WORDS:
            corrected.append(token)
            continue

        inflected = _unpluralize(token)
        if inflected:
            corrected.append(inflected)
            continue

        candidates = {c for c in _edits1(token) & ENGLISH_WORDS if c[0] == token[0]}

        if candidates:
            corrected.append(
                min(candidates, key=lambda c: (abs(len(c) - len(token)), c))
            )
        else:
            corrected.append(token)

    return " ".join(corrected)


def expand_synonyms(tokens: list[str]) -> list[str]:
    expanded = list(tokens)

    for token in tokens:
        syns = wordnet.synsets(token)
        added = 0

        for synset in syns:
            if added >= MAX_SYNONYMS_PER_TOKEN:
                break

            for lemma in synset.lemmas():
                name = lemma.name().lower()
                if (
                    name != token
                    and name.isalpha()
                    and added < MAX_SYNONYMS_PER_TOKEN
                ):
                    expanded.append(name)
                    added += 1

    return expanded


class SearchHistory:
    _term_frequencies: Counter = Counter()

    @classmethod
    def record_query(cls, tokens: list[str]):
        cls._term_frequencies.update(tokens)

    @classmethod
    def get_top_terms(cls, n: int = 5) -> list[str]:
        return [term for term, _ in cls._term_frequencies.most_common(n)]


def boost_from_history(tokens: list[str]) -> list[str]:
    boosted = list(tokens)
    top_history_terms = SearchHistory.get_top_terms(5)

    for term in top_history_terms:
        boosted.extend([term] * HISTORY_BOOST_COUNT)

    return boosted


def refine_query(query_text: str) -> str:
    corrected = correct_spelling(query_text)
    tokens = preprocess_lotte(corrected)
    expanded = expand_synonyms(tokens)
    boosted = boost_from_history(expanded)
    SearchHistory.record_query(expanded)

    return " ".join(boosted)

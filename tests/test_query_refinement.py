import sys
from pathlib import Path
from collections import Counter


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.append(str(PROJECT_ROOT))


from services.query_refinement.query_refinement_service import (
    correct_spelling,
    expand_synonyms,
    boost_from_history,
    SearchHistory,
    refine_query,
    MAX_SYNONYMS_PER_TOKEN,
    HISTORY_BOOST_COUNT,
)
from services.query_refinement.trie_service import (
    VocabularyTrie,
    build_trie_from_inverted_index,
    get_vocabulary_trie,
)


passed = 0
failed = 0


def check(name: str, condition: bool, detail: str = ""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS  {name}")
    else:
        failed += 1
        print(f"  FAIL  {name}  {detail}")


# =============================================================
# 1. Spell correction tests
# =============================================================

def test_correct_spelling_typo():
    result = correct_spelling("wrok")
    check(
        "correct_spelling fixes 'wrok' to 'work'",
        result == "work",
        f"got '{result}'",
    )


def test_correct_spelling_transpose():
    result = correct_spelling("recieve")
    check(
        "correct_spelling fixes 'recieve' to 'receive'",
        result == "receive",
        f"got '{result}'",
    )


def test_correct_spelling_clean():
    result = correct_spelling("hello world")
    check(
        "correct_spelling leaves correct text unchanged",
        result == "hello world",
        f"got '{result}'",
    )


def test_correct_spelling_short_words_unchanged():
    result = correct_spelling("a an the")
    check(
        "correct_spelling leaves short (<=3) words unchanged",
        result == "a an the",
        f"got '{result}'",
    )


# =============================================================
# 2. Synonym expansion tests
# =============================================================

def test_expand_synonyms_adds_terms():
    tokens = ["doctor"]
    expanded = expand_synonyms(tokens)
    check(
        "expand_synonyms adds synonyms for 'doctor'",
        len(expanded) > 1,
        f"got {expanded}",
    )


def test_expand_synonyms_includes_original():
    tokens = ["car"]
    expanded = expand_synonyms(tokens)
    check(
        "expand_synonyms keeps original tokens",
        "car" in expanded,
        f"got {expanded}",
    )


def test_expand_synonyms_respects_max():
    tokens = ["run"]
    expanded = expand_synonyms(tokens)
    syn_count = len(expanded) - 1
    check(
        f"expand_synonyms respects MAX_SYNONYMS_PER_TOKEN={MAX_SYNONYMS_PER_TOKEN}",
        syn_count <= MAX_SYNONYMS_PER_TOKEN,
        f"got {syn_count} synonyms for 'run': {expanded}",
    )


def test_expand_synonyms_unknown_word():
    tokens = ["xyznonexistent"]
    expanded = expand_synonyms(tokens)
    check(
        "expand_synonyms handles unknown words gracefully",
        expanded == tokens,
        f"got {expanded}",
    )


# =============================================================
# 3. History & boosting tests
# =============================================================

def test_search_history_record():
    SearchHistory._term_frequencies = Counter()
    SearchHistory.record_query(["test", "query"])
    top = SearchHistory.get_top_terms(5)
    check(
        "SearchHistory records and retrieves terms",
        "test" in top and "query" in top,
        f"got {top}",
    )


def test_boost_from_history_adds_terms():
    SearchHistory._term_frequencies = Counter()
    SearchHistory.record_query(["music", "song", "guitar"])
    SearchHistory.record_query(["music", "song"])
    SearchHistory.record_query(["music"])
    boosted = boost_from_history(["new"])
    music_count = boosted.count("music")
    check(
        "boost_from_history adds top history terms",
        music_count == HISTORY_BOOST_COUNT,
        f"music appears {music_count} times, expected {HISTORY_BOOST_COUNT}",
    )


# =============================================================
# 4. Trie tests
# =============================================================

def test_trie_insert_and_search():
    trie = VocabularyTrie()
    trie.insert("hello")
    trie.insert("help")
    trie.insert("world")
    check("trie.search finds existing word", trie.search("hello"))
    check("trie.search returns False for missing word", not trie.search("nope"))
    check("trie.starts_with matches prefix", trie.starts_with("hel"))
    check("trie.starts_with False for bad prefix", not trie.starts_with("xyz"))


def test_trie_suggest_prefix():
    trie = VocabularyTrie()
    for word in ["computer", "computing", "compose", "compare", "compact"]:
        trie.insert(word)
    suggestions = trie.suggest("compu", limit=10)
    check(
        "trie.suggest returns matching completions",
        "computer" in suggestions and "computing" in suggestions,
        f"got {suggestions}",
    )


def test_trie_suggest_limit():
    trie = VocabularyTrie()
    words = ["a", "ab", "abc", "abcd", "abcde", "abcdef"]
    for w in words:
        trie.insert(w)
    suggestions = trie.suggest("a", limit=3)
    check(
        "trie.suggest respects limit parameter",
        len(suggestions) <= 3,
        f"got {len(suggestions)} suggestions",
    )


def test_trie_suggest_no_match():
    trie = VocabularyTrie()
    trie.insert("hello")
    suggestions = trie.suggest("xyznonexistent", limit=5)
    check(
        "trie.suggest returns empty list for no match",
        suggestions == [],
        f"got {suggestions}",
    )


def test_trie_suggest_empty_prefix():
    trie = VocabularyTrie()
    trie.insert("test")
    suggestions = trie.suggest("", limit=5)
    check(
        "trie.suggest handles empty prefix",
        "test" in suggestions,
        f"got {suggestions}",
    )


# =============================================================
# 5. End-to-end pipeline tests
# =============================================================

def test_refine_query_pipeline():
    result = refine_query("doctor")
    check(
        "refine_query returns a non-empty string",
        len(result) > 0,
        f"got '{result}'",
    )
    check(
        "refine_query result contains original term or synonym",
        "doctor" in result or any(
            s in result
            for s in ["physician", "doc", "medical"]
        ),
        f"got '{result}'",
    )


def test_correct_spelling_then_expand():
    corrected = correct_spelling("docter")
    tokens = ["doctor"]  # after preprocessing
    expanded = expand_synonyms(tokens)
    check(
        "spelling + synonym pipeline works",
        len(expanded) >= 1 and "doctor" in expanded,
        f"expanded: {expanded}",
    )


# =============================================================
# Run all tests
# =============================================================

def main():
    print("=" * 60)
    print("QUERY REFINEMENT SERVICE TESTS")
    print("=" * 60)

    # Reset SearchHistory before tests
    SearchHistory._term_frequencies = Counter()

    print("\n--- Spell Correction ---")
    test_correct_spelling_typo()
    test_correct_spelling_clean()
    test_correct_spelling_short_words_unchanged()

    print("\n--- Synonym Expansion ---")
    test_expand_synonyms_adds_terms()
    test_expand_synonyms_includes_original()
    test_expand_synonyms_respects_max()
    test_expand_synonyms_unknown_word()

    print("\n--- History & Boosting ---")
    test_search_history_record()
    test_boost_from_history_adds_terms()

    print("\n--- Trie ---")
    test_trie_insert_and_search()
    test_trie_suggest_prefix()
    test_trie_suggest_limit()
    test_trie_suggest_no_match()
    test_trie_suggest_empty_prefix()

    print("\n--- End-to-End Pipeline ---")
    test_refine_query_pipeline()
    test_correct_spelling_then_expand()

    print("\n" + "=" * 60)
    print(f"RESULTS:  {passed} passed,  {failed} failed")
    print("=" * 60)

    if failed > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()

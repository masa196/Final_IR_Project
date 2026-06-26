import json
from pathlib import Path
from functools import lru_cache

import joblib

PROJECT_ROOT = Path(__file__).resolve().parents[2]
INVERTED_INDEX_DIR = PROJECT_ROOT / "indices" / "lotte" / "inverted_index"


class TrieNode:
    __slots__ = ("children", "is_end")

    def __init__(self):
        self.children: dict[str, "TrieNode"] = {}
        self.is_end: bool = False


class VocabularyTrie:
    def __init__(self):
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        node = self.root
        for char in word:
            if char not in node.children:
                node.children[char] = TrieNode()
            node = node.children[char]
        node.is_end = True

    def search(self, word: str) -> bool:
        node = self._traverse(word)
        return node is not None and node.is_end

    def starts_with(self, prefix: str) -> bool:
        return self._traverse(prefix) is not None

    def suggest(self, prefix: str, limit: int = 10) -> list[str]:
        node = self._traverse(prefix)
        if node is None:
            return []
        results: list[str] = []
        self._dfs(node, prefix, results, limit)
        return results

    def _traverse(self, prefix: str) -> TrieNode | None:
        node = self.root
        for char in prefix:
            if char not in node.children:
                return None
            node = node.children[char]
        return node

    def _dfs(
        self,
        node: TrieNode,
        current_prefix: str,
        results: list[str],
        limit: int,
    ) -> None:
        if len(results) >= limit:
            return
        if node.is_end:
            results.append(current_prefix)
        for char in sorted(node.children.keys()):
            if len(results) >= limit:
                break
            self._dfs(
                node.children[char],
                current_prefix + char,
                results,
                limit,
            )


def build_trie_from_inverted_index(
    inverted_index_dir: Path = INVERTED_INDEX_DIR,
) -> VocabularyTrie:
    inverted_index_path = inverted_index_dir / "inverted_index.joblib"
    if not inverted_index_path.exists():
        return VocabularyTrie()

    inverted_index = joblib.load(inverted_index_path)
    trie = VocabularyTrie()
    for term in inverted_index.keys():
        trie.insert(term)
    return trie


@lru_cache(maxsize=1)
def get_vocabulary_trie() -> VocabularyTrie:
    return build_trie_from_inverted_index()

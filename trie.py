"""Standard (uncompressed) trie with frequencies, prefix counts, completion,
fuzzy search, delete-with-compaction, and JSON serialization.

This is the baseline structure; :mod:`radix` provides the path-compressed
variant with identical semantics so the two can be cross-checked.
"""

from __future__ import annotations

import json
from typing import Iterator

from levenshtein import ColumnState

_FORMAT = "trie/1"


class TrieNode:
    """One trie node: per-character children plus a terminal marker."""

    __slots__ = ("children", "terminal", "freq")

    def __init__(self) -> None:
        self.children: dict[str, "TrieNode"] = {}
        self.terminal: bool = False
        self.freq: int = 0


class Trie:
    """Character-edge trie mapping ``str -> int`` (frequency).

    Words must be non-empty strings; frequencies accumulate on repeated
    insertion.  ``delete`` removes an entry entirely (documented policy), and
    prunes now-purposeless nodes.
    """

    def __init__(self) -> None:
        self.root = TrieNode()
        self._size = 0

    # ------------------------------------------------------------------
    # insertion / deletion
    # ------------------------------------------------------------------
    def insert(self, word: str, freq: int = 1) -> None:
        """Insert ``word``, adding ``freq`` to its frequency (default 1)."""
        if not isinstance(word, str) or word == "":
            raise ValueError("words must be non-empty strings")
        if freq <= 0:
            raise ValueError("freq must be positive")
        node = self.root
        for ch in word:
            child = node.children.get(ch)
            if child is None:
                child = TrieNode()
                node.children[ch] = child
            node = child
        if not node.terminal:
            node.terminal = True
            self._size += 1
        node.freq += freq

    def delete(self, word: str) -> bool:
        """Remove ``word`` entirely; return ``True`` if it existed.

        Nodes left without purpose (no children and not terminal) are pruned
        on the way back up.
        """
        if not word:
            return False
        path: list[tuple[TrieNode, str]] = []
        node = self.root
        for ch in word:
            child = node.children.get(ch)
            if child is None:
                return False
            path.append((node, ch))
            node = child
        if not node.terminal:
            return False
        node.terminal = False
        node.freq = 0
        self._size -= 1
        # prune upward
        for parent, ch in reversed(path):
            target = parent.children[ch]
            if not target.terminal and not target.children:
                del parent.children[ch]
            else:
                break
        return True

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def count(self, word: str) -> int:
        """Exact frequency of ``word`` (0 when absent)."""
        node = self._node(word)
        return node.freq if node is not None and node.terminal else 0

    def count_prefix(self, prefix: str) -> int:
        """Total frequency of every key that starts with ``prefix``.

        An empty prefix sums the whole vocabulary.  ``prefix`` need not itself
        be a key.
        """
        node = self._node(prefix)
        if node is None:
            return 0
        return self._subtree_freq(node)

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        node = self._node(word)
        return node is not None and node.terminal

    def __len__(self) -> int:
        return self._size

    def words(self) -> list[str]:
        """All keys in lexicographic order."""
        return sorted(self._iter_words())

    def words_with_prefix(self, prefix: str) -> list[str]:
        """All keys starting with ``prefix``, lexicographic."""
        node = self._node(prefix)
        if node is None:
            return []
        return sorted(prefix + w for w in self._iter_words(node))

    def complete(self, prefix: str, limit: int | None = None) -> list[str]:
        """Autocomplete: keys starting with ``prefix`` sorted by descending
        frequency, then lexicographically."""
        node = self._node(prefix)
        if node is None:
            return []
        ranked = sorted(
            (self._subtree_freq_path(prefix, node)),
            key=lambda pair: (-pair[1], pair[0]),
        )
        words = [w for w, _ in ranked]
        return words if limit is None else words[:limit]

    def fuzzy(self, word: str, max_distance: int) -> list[tuple[str, int]]:
        """Keys within edit distance ``max_distance`` of ``word``.

        Returns ``(key, distance)`` pairs sorted by ``(distance, key)``.  The
        walk keeps one bounded-DP column per trie edge and prunes the whole
        subtree the moment no completion could land inside the bound.
        """
        if max_distance < 0:
            return []
        results: list[tuple[str, int]] = []
        state = ColumnState(word, max_distance)
        stack: list[tuple[TrieNode, str, ColumnState]] = [(self.root, "", state)]
        while stack:
            node, prefix, node_state = stack.pop()
            if node.terminal and prefix:
                cost = node_state.final_cost()
                if cost is not None:
                    results.append((prefix, cost))
            for ch in sorted(node.children, reverse=True):
                child = node.children[ch]
                child_state = node_state.clone()
                child_state.feed(ch)
                if not child_state.dead:
                    stack.append((child, prefix + ch, child_state))
        results.sort(key=lambda pair: (pair[1], pair[0]))
        return results

    # ------------------------------------------------------------------
    # statistics & serialization
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, object]:
        """Node/word counts and structural depth information."""
        nodes = 0
        max_depth = 0

        def walk(node: TrieNode, depth: int) -> None:
            nonlocal nodes, max_depth
            nodes += 1
            max_depth = max(max_depth, depth)
            for child in node.children.values():
                walk(child, depth + 1)

        walk(self.root, 0)
        return {
            "kind": "trie",
            "nodes": nodes,
            "words": self._size,
            "max_depth": max_depth,
            "compression_ratio": 1.0,  # uncompressed by definition
        }

    def dumps(self) -> str:
        """Serialize to a deterministic JSON document (sorted keys)."""
        def enc(node: TrieNode) -> dict:
            entry: dict[str, object] = {}
            if node.terminal:
                entry["f"] = node.freq
            if node.children:
                entry["c"] = {ch: enc(node.children[ch]) for ch in sorted(node.children)}
            return entry

        doc = {"format": _FORMAT, "root": enc(self.root)}
        return json.dumps(doc, ensure_ascii=False, sort_keys=True)

    @classmethod
    def loads(cls, text: str) -> "Trie":
        """Rebuild a trie previously produced by :meth:`dumps`."""
        doc = json.loads(text)
        if doc.get("format") != _FORMAT:
            raise ValueError(f"not a trie document: format={doc.get('format')!r}")
        trie = cls()

        def dec(entry: dict, node: TrieNode) -> None:
            if "f" in entry:
                node.terminal = True
                node.freq = int(entry["f"])
                trie._size += 1
            for ch, child_entry in entry.get("c", {}).items():
                child = TrieNode()
                node.children[ch] = child
                dec(child_entry, child)

        dec(doc["root"], trie.root)
        return trie

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(self.dumps())

    @classmethod
    def load(cls, path: str) -> "Trie":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.loads(fh.read())

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _node(self, key: str) -> TrieNode | None:
        node = self.root
        for ch in key:
            node = node.children.get(ch)  # type: ignore[assignment]
            if node is None:
                return None
        return node

    def _subtree_freq(self, node: TrieNode) -> int:
        total = node.freq if node.terminal else 0
        for child in node.children.values():
            total += self._subtree_freq(child)
        return total

    def _iter_words(self, node: TrieNode | None = None) -> Iterator[str]:
        """Yield keys below ``node`` (defaults to the root) as strings
        relative to that node, in lexicographic order.  Recursion depth is
        bounded by the longest key, and keys are ordinary short strings."""
        base = self.root if node is None else node

        def walk(n: TrieNode, prefix: str) -> Iterator[str]:
            if n.terminal:
                yield prefix
            for ch in sorted(n.children):
                yield from walk(n.children[ch], prefix + ch)

        yield from walk(base, "")

    def _subtree_freq_path(self, prefix: str, node: TrieNode) -> list[tuple[str, int]]:
        out: list[tuple[str, int]] = []

        def walk(n: TrieNode, path: str) -> None:
            if n.terminal:
                out.append((path, n.freq))
            for ch in sorted(n.children):
                walk(n.children[ch], path + ch)

        walk(node, prefix)
        return out

"""Compressed radix tree (path-compressed trie) with completion, fuzzy
search, delete with compaction, statistics, and JSON serialization.

Chains of single-child, non-terminal nodes are collapsed into one edge whose
label carries the whole substring — this is the path compression that makes
autocomplete walks skip straight over shared substrings.  Deleting a key
restores compaction by pruning dead nodes and re-merging exposed chains.
All subtree walks are iterative so deep trees never touch Python's recursion
limit.
"""

from __future__ import annotations

import json
from typing import Iterator

from levenshtein import ColumnState

_FORMAT = "radix-trie/1"


def common_prefix_len(a: str, b: str) -> int:
    """Length of the longest common prefix of ``a`` and ``b``."""
    n = min(len(a), len(b))
    i = 0
    while i < n and a[i] == b[i]:
        i += 1
    return i


class RadixNode:
    """A radix node: ``label`` is the substring on the edge from the parent."""

    __slots__ = ("label", "children", "terminal", "freq")

    def __init__(self, label: str = "") -> None:
        self.label = label
        self.children: dict[str, "RadixNode"] = {}  # keyed by label[0]
        self.terminal: bool = False
        self.freq: int = 0


class RadixTrie:
    """Path-compressed trie mapping ``str -> int`` (frequency).

    Semantics match :class:`trie.Trie` exactly — same insert/delete/complete/
    fuzzy results — while using one node per branch point instead of one per
    character.  Words must be non-empty; ``delete`` removes an entry entirely
    regardless of accumulated frequency.
    """

    def __init__(self) -> None:
        self.root = RadixNode("")
        self._size = 0

    # ------------------------------------------------------------------
    # insertion
    # ------------------------------------------------------------------
    def insert(self, word: str, freq: int = 1) -> None:
        """Insert ``word``, adding ``freq`` to its frequency (default 1).

        Insertion maintains path compression: every split point is a branch or
        a terminal, so no non-terminal node ever keeps a single child.
        """
        if not isinstance(word, str) or word == "":
            raise ValueError("words must be non-empty strings")
        if freq <= 0:
            raise ValueError("freq must be positive")
        node = self.root
        pos = 0
        while True:
            rest = word[pos:]
            if not rest:
                # the word ends exactly at `node`
                if not node.terminal:
                    node.terminal = True
                    self._size += 1
                node.freq += freq
                return
            child = node.children.get(rest[0])
            if child is None:
                leaf = RadixNode(rest)
                leaf.terminal = True
                leaf.freq = freq
                node.children[rest[0]] = leaf
                self._size += 1
                return
            common = common_prefix_len(child.label, rest)
            if common == len(child.label):
                pos += common
                node = child
                continue
            if common == len(rest):
                # the new word stops in the middle of this edge: split it
                intermediate = RadixNode(rest)
                intermediate.terminal = True
                intermediate.freq = freq
                child.label = child.label[common:]
                intermediate.children[child.label[0]] = child
                node.children[rest[0]] = intermediate
                self._size += 1
                return
            # partial overlap: introduce a branch point
            branch = RadixNode(child.label[:common])
            child.label = child.label[common:]
            branch.children[child.label[0]] = child
            leaf = RadixNode(rest[common:])
            leaf.terminal = True
            leaf.freq = freq
            branch.children[leaf.label[0]] = leaf
            node.children[rest[0]] = branch
            self._size += 1
            return

    # ------------------------------------------------------------------
    # deletion with compaction
    # ------------------------------------------------------------------
    def delete(self, word: str) -> bool:
        """Remove ``word`` entirely; return ``True`` if it existed.

        After removal the structure is repaired: purposeless nodes are pruned
        and any non-terminal node left with a single child is merged into it
        (node compaction), restoring the compressed invariant.
        """
        if not word:
            return False
        node = self.root
        path: list[tuple[RadixNode, str]] = []  # spine: (parent, child_key)
        pos = 0
        while pos < len(word):
            rest = word[pos:]
            child = node.children.get(rest[0])
            if child is None:
                return False
            common = common_prefix_len(child.label, rest)
            if common < len(child.label):
                # diverges, or the word would end inside an edge (no node)
                return False
            path.append((node, rest[0]))
            node = child
            pos += common
        if pos != len(word) or not node.terminal:
            return False
        node.terminal = False
        node.freq = 0
        self._size -= 1

        # walk the spine upward, pruning and re-merging
        current = node
        for parent, key in reversed(path):
            if not current.terminal and not current.children:
                del parent.children[key]
                current = parent
            elif not current.terminal and len(current.children) == 1:
                only = next(iter(current.children.values()))
                only.label = current.label + only.label
                parent.children[key] = only
                break  # parent's child count is unchanged: spine stays valid
            else:
                break
        return True

    # ------------------------------------------------------------------
    # queries
    # ------------------------------------------------------------------
    def count(self, word: str) -> int:
        """Exact frequency of ``word`` (0 when absent)."""
        node = self._exact(word)
        return node.freq if node is not None and node.terminal else 0

    def count_prefix(self, prefix: str) -> int:
        """Total frequency of every key starting with ``prefix``.

        Works when ``prefix`` ends inside an edge (the keys underneath all
        extend it).  An empty prefix sums the whole vocabulary.
        """
        located = self._locate_prefix(prefix)
        if located is None:
            return 0
        node, _skip = located
        return self._subtree_freq(node)

    def __contains__(self, word: object) -> bool:
        if not isinstance(word, str):
            return False
        node = self._exact(word)
        return node is not None and node.terminal

    def __len__(self) -> int:
        return self._size

    def words(self) -> list[str]:
        """All keys in lexicographic order."""
        return sorted(self._walk_conts(self.root, 0))

    def words_with_prefix(self, prefix: str) -> list[str]:
        """All keys starting with ``prefix``, lexicographic."""
        located = self._locate_prefix(prefix)
        if located is None:
            return []
        node, skip = located
        return sorted(prefix + c for c in self._walk_conts(node, skip))

    def complete(self, prefix: str, limit: int | None = None) -> list[str]:
        """Autocomplete: keys starting with ``prefix`` sorted by descending
        frequency, then lexicographically (the tie-break that keeps popular
        completions on top)."""
        located = self._locate_prefix(prefix)
        if located is None:
            return []
        node, skip = located
        ranked = sorted(
            self._walk_cont_freqs(node, skip, prefix),
            key=lambda pair: (-pair[1], pair[0]),
        )
        words = [w for w, _ in ranked]
        return words if limit is None else words[:limit]

    def fuzzy(self, word: str, max_distance: int) -> list[tuple[str, int]]:
        """Keys within edit distance ``max_distance`` of ``word``.

        Walks the radix edges while feeding one bounded-DP column per edge
        (see :mod:`levenshtein`); a subtree is pruned the moment its column's
        minimum exceeds the bound — no completion underneath can recover.
        Returns ``(key, distance)`` pairs sorted by ``(distance, key)``.
        An empty ``word`` matches every key with ``length <= max_distance``.
        """
        if max_distance < 0:
            return []
        results: list[tuple[str, int]] = []
        stack: list[tuple[RadixNode, str, ColumnState]] = [
            (self.root, "", ColumnState(word, max_distance))
        ]
        while stack:
            node, path, state = stack.pop()
            if node.terminal and path:
                cost = state.final_cost()
                if cost is not None:
                    results.append((path, cost))
            for key in sorted(node.children, reverse=True):
                child = node.children[key]
                child_state = state.clone()
                child_state.feed(child.label)
                if not child_state.dead:
                    stack.append((child, path + child.label, child_state))
        results.sort(key=lambda pair: (pair[1], pair[0]))
        return results

    # ------------------------------------------------------------------
    # statistics
    # ------------------------------------------------------------------
    def stats(self) -> dict[str, object]:
        """Node counts, edge-label volume, and the compression ratio.

        ``uncompressed_nodes`` is exactly how many nodes a plain character
        trie over the same keys needs (``1 + sum of label lengths``), so
        ``nodes / uncompressed_nodes`` below 1 measures the win from path
        compression.
        """
        nodes = 0
        label_chars = 0
        stack = [self.root]
        while stack:
            node = stack.pop()
            nodes += 1
            label_chars += len(node.label)
            stack.extend(node.children.values())
        uncompressed = label_chars + 1
        return {
            "kind": "radix",
            "nodes": nodes,
            "words": self._size,
            "label_chars": label_chars,
            "uncompressed_nodes": uncompressed,
            "compression_ratio": nodes / uncompressed if uncompressed else 1.0,
        }

    # ------------------------------------------------------------------
    # serialization
    # ------------------------------------------------------------------
    def dumps(self) -> str:
        """Serialize to a deterministic JSON document (sorted keys)."""
        def enc(node: RadixNode) -> dict:
            entry: dict[str, object] = {"l": node.label}
            if node.terminal:
                entry["f"] = node.freq
            if node.children:
                entry["c"] = {
                    key: enc(node.children[key]) for key in sorted(node.children)
                }
            return entry

        doc = {"format": _FORMAT, "root": enc(self.root)}
        return json.dumps(doc, ensure_ascii=False, sort_keys=True)

    @classmethod
    def loads(cls, text: str) -> "RadixTrie":
        """Rebuild a radix trie previously produced by :meth:`dumps`."""
        doc = json.loads(text)
        if doc.get("format") != _FORMAT:
            raise ValueError(f"not a radix-trie document: format={doc.get('format')!r}")
        trie = cls()

        def dec(entry: dict, node: RadixNode) -> None:
            node.label = entry["l"]
            if "f" in entry:
                node.terminal = True
                node.freq = int(entry["f"])
                trie._size += 1
            for key in sorted(entry.get("c", {})):
                child = RadixNode()
                node.children[key] = child
                dec(entry["c"][key], child)
                if child.label[:1] != key:
                    raise ValueError(
                        f"radix document: child label {child.label!r} does not "
                        f"match its key {key!r}"
                    )

        dec(doc["root"], trie.root)
        if trie.root.label:
            raise ValueError("radix document: root must carry an empty label")
        return trie

    def save(self, path: str) -> None:
        with open(path, "w", encoding="utf-8", newline="\n") as fh:
            fh.write(self.dumps())

    @classmethod
    def load(cls, path: str) -> "RadixTrie":
        with open(path, "r", encoding="utf-8") as fh:
            return cls.loads(fh.read())

    # ------------------------------------------------------------------
    # internals
    # ------------------------------------------------------------------
    def _exact(self, word: str) -> RadixNode | None:
        """Node holding ``word`` as a whole, or ``None`` (mid-edge = absent)."""
        if not word:
            return None
        node = self.root
        pos = 0
        while pos < len(word):
            rest = word[pos:]
            child = node.children.get(rest[0])
            if child is None:
                return None
            common = common_prefix_len(child.label, rest)
            if common < len(child.label):
                # diverges, or the word would end inside an edge (no node)
                return None
            pos += common
            node = child
        return node

    def _locate_prefix(self, prefix: str) -> tuple[RadixNode, int] | None:
        """Resolve ``prefix`` to ``(subtree_root, skip)``.

        ``skip`` is how many leading characters of ``subtree_root.label`` the
        prefix already consumed (0 when the prefix lands exactly on a node);
        continuations from there are ``label[skip:] + descendants``, so
        ``prefix + continuation`` is a full key.  ``None`` means no key has
        this prefix.
        """
        if prefix == "":
            return self.root, 0
        node = self.root
        pos = 0
        while pos < len(prefix):
            rest = prefix[pos:]
            child = node.children.get(rest[0])
            if child is None:
                return None
            common = common_prefix_len(child.label, rest)
            if common == len(child.label):
                pos += common
                node = child
                continue
            if common == len(rest):
                # prefix ends inside this edge — every word below extends it
                return child, common
            return None
        # prefix landed exactly on `node`: its label is part of the prefix,
        # so continuations must skip all of it
        return node, len(node.label)

    def _walk_conts(self, node: RadixNode, skip: int) -> Iterator[str]:
        """Yield continuation strings below ``node`` — its own label minus
        ``skip`` leading characters, then full descendant labels.  Iterative."""
        results: list[str] = []
        stack: list[tuple[RadixNode, str]] = [(node, node.label[skip:])]
        while stack:
            cur, prefix = stack.pop()
            if cur.terminal:
                results.append(prefix)
            for key in sorted(cur.children, reverse=True):
                child = cur.children[key]
                stack.append((child, prefix + child.label))
        return iter(results)

    def _walk_cont_freqs(
        self, node: RadixNode, skip: int, base: str
    ) -> list[tuple[str, int]]:
        """Yield ``(full_key, freq)`` pairs below ``node`` (iterative)."""
        results: list[tuple[str, int]] = []
        stack: list[tuple[RadixNode, str]] = [(node, node.label[skip:])]
        while stack:
            cur, prefix = stack.pop()
            if cur.terminal:
                results.append((base + prefix, cur.freq))
            for key in sorted(cur.children, reverse=True):
                child = cur.children[key]
                stack.append((child, prefix + child.label))
        return results

    def _subtree_freq(self, node: RadixNode) -> int:
        total = 0
        stack = [node]
        while stack:
            cur = stack.pop()
            if cur.terminal:
                total += cur.freq
            stack.extend(cur.children.values())
        return total

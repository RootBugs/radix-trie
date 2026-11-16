"""Shared deterministic fixtures for radix-trie tests.

Everything here is seeded so every run sees the same wordlist, frequencies,
and fuzzy-search corpus.
"""

from __future__ import annotations

import random

from levenshtein import levenshtein

PREFIXES = ["re", "un", "in", "pre", "dis", "over", "under", "inter"]
STEMS = ["set", "get", "run", "turn", "load", "form", "mark", "point", "play", "work"]
SUFFIXES = ["", "er", "ing", "s", "ed", "able", "ion"]

UNICODE_WORDS = [
    "café", "cafeteria", "cafés", "naïve", "naivety",
    "日本語", "日本", "こんにちは", "ありがとう",
    "ñandú", "ñandúes", "über", "übersetzen",
]


def build_wordlist(seed: int = 1234, count: int = 150) -> dict[str, int]:
    """A seeded wordlist of prefixed stems with seeded frequencies."""
    rng = random.Random(seed)
    words: dict[str, int] = {}
    while len(words) < count:
        word = rng.choice(PREFIXES) + rng.choice(STEMS) + rng.choice(SUFFIXES)
        if word and word not in words:
            words[word] = rng.randint(1, 100)
    return words


def clustered_wordlist(seed: int = 99, count: int = 200) -> dict[str, int]:
    """Keys sharing long common prefixes — the radix trie's best case."""
    rng = random.Random(seed)
    words: dict[str, int] = {}
    while len(words) < count:
        word = "common_prefix_segment_" + format(rng.getrandbits(32), "08x")
        if word not in words:
            words[word] = rng.randint(1, 50)
    return words


def fill(trie, words: dict[str, int]) -> None:
    for word, freq in words.items():
        trie.insert(word, freq)


def ref_complete(words: dict[str, int], prefix: str, limit: int | None = None):
    """Brute-force autocomplete: filter, sort by (-frequency, word)."""
    hits = [w for w in words if w.startswith(prefix)]
    hits.sort(key=lambda w: (-words[w], w))
    return hits if limit is None else hits[:limit]


def ref_fuzzy(words: dict[str, int], query: str, k: int):
    """Brute-force fuzzy search: full Levenshtein against every key."""
    out = [(w, levenshtein(query, w)) for w in words if levenshtein(query, w) <= k]
    out.sort(key=lambda pair: (pair[1], pair[0]))
    return out


def mutated(word: str, rng: random.Random) -> str:
    """One random single-character edit of ``word`` (delete/insert/replace)."""
    op = rng.randrange(3)
    i = rng.randrange(len(word))
    alphabet = "abcdefghijklmnopqrstuvwxyz"
    if op == 0:
        return word[:i] + word[i + 1:]
    if op == 1:
        return word[:i] + rng.choice(alphabet) + word[i:]
    return word[:i] + rng.choice(alphabet) + word[i + 1:]


def assert_compressed(testcase, trie) -> None:
    """The radix invariant: non-root, non-terminal nodes never have one child."""
    stack = [trie.root]
    while stack:
        node = stack.pop()
        for key, child in node.children.items():
            testcase.assertEqual(
                child.label[:1], key,
                f"child label {child.label!r} keyed by {key!r}",
            )
            testcase.assertTrue(child.label, "empty edge label below root")
            if not child.terminal:
                testcase.assertNotEqual(
                    len(child.children), 1,
                    f"uncompressed chain at label {child.label!r}",
                )
            stack.append(child)

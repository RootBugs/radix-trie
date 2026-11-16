"""Tests for bounded Levenshtein and trie-integrated fuzzy search.

The brute-force reference runs full Levenshtein against every key; the
bounded-DP search must return exactly the same set — finding everything
within ``k`` and nothing beyond it.
"""

from __future__ import annotations

import random
import unittest

from levenshtein import ColumnState, levenshtein, levenshtein_bounded
from radix import RadixTrie
from tests.fixtures import build_wordlist, fill, mutated, ref_fuzzy
from trie import Trie


class LevenshteinTests(unittest.TestCase):
    def test_known_values(self):
        self.assertEqual(levenshtein("", ""), 0)
        self.assertEqual(levenshtein("abc", "abc"), 0)
        self.assertEqual(levenshtein("", "abc"), 3)
        self.assertEqual(levenshtein("abc", ""), 3)
        self.assertEqual(levenshtein("kitten", "sitting"), 3)
        self.assertEqual(levenshtein("flaw", "lawn"), 2)
        self.assertEqual(levenshtein("a", "b"), 1)
        self.assertEqual(levenshtein("café", "cafe"), 1)

    def test_bounded_agrees_with_full_on_random_pairs(self):
        rng = random.Random(404)
        alphabet = "abcdef"
        for _ in range(200):
            a = "".join(rng.choice(alphabet) for _ in range(rng.randrange(9)))
            b = "".join(rng.choice(alphabet) for _ in range(rng.randrange(9)))
            exact = levenshtein(a, b)
            for k in range(0, 6):
                got = levenshtein_bounded(a, b, k)
                if exact <= k:
                    self.assertEqual(got, exact, (a, b, k))
                else:
                    self.assertIsNone(got, (a, b, k, exact))

    def test_bounded_edge_cases(self):
        self.assertIsNone(levenshtein_bounded("abc", "xyz", -1))
        self.assertEqual(levenshtein_bounded("same", "same", 0), 0)
        self.assertIsNone(levenshtein_bounded("ab", "abcd", 1))  # diff 2 > 1
        self.assertEqual(levenshtein_bounded("ab", "abcd", 2), 2)
        self.assertEqual(levenshtein_bounded("", "ab", 2), 2)
        self.assertIsNone(levenshtein_bounded("", "ab", 1))

    def test_column_state_prunes_and_reports_final_cost(self):
        state = ColumnState("abc", 1)
        state.feed("abc")
        self.assertEqual(state.final_cost(), 0)
        state = ColumnState("abc", 1)
        state.feed("xyz")
        self.assertTrue(state.dead)
        self.assertIsNone(state.final_cost())
        state = ColumnState("abc", 2)
        state.feed("xbc")
        self.assertFalse(state.dead)
        self.assertEqual(state.final_cost(), 1)

    def test_column_state_clone_is_independent(self):
        base = ColumnState("query", 6)
        base.feed("abc")
        twin = base.clone()
        twin.feed("def")
        self.assertEqual(base.final_cost(), levenshtein("query", "abc"))
        self.assertEqual(twin.final_cost(), levenshtein("query", "abcdef"))


class FuzzySearchTests(unittest.TestCase):
    def setUp(self):
        self.words = build_wordlist(seed=9, count=140)
        self.radix = RadixTrie()
        fill(self.radix, self.words)
        self.trie = Trie()
        fill(self.trie, self.words)

    def _queries(self, seed=1234) -> list[str]:
        rng = random.Random(seed)
        queries = ["", "run", "resetter", "zzzz", "xyz", "underplaying"]
        source = sorted(self.words)
        queries += [mutated(source[rng.randrange(len(source))], rng) for _ in range(25)]
        return queries

    def test_radix_fuzzy_matches_brute_force(self):
        for query in self._queries():
            for k in (0, 1, 2, 3):
                expected = ref_fuzzy(self.words, query, k)
                self.assertEqual(
                    self.radix.fuzzy(query, k), expected, f"query={query!r} k={k}"
                )

    def test_plain_trie_fuzzy_matches_brute_force(self):
        for query in self._queries(seed=555):
            for k in (0, 1, 2):
                expected = ref_fuzzy(self.words, query, k)
                self.assertEqual(
                    self.trie.fuzzy(query, k), expected, f"query={query!r} k={k}"
                )

    def test_finds_all_within_k_and_none_beyond(self):
        # "reset" exists in the corpus family: distance to neighbours is known
        query = "resett"  # one insertion away from "reset"
        hits = self.radix.fuzzy(query, 1)
        words_at_1 = {w for w, d in hits if d == 1}
        if "reset" in self.words:
            self.assertIn("reset", words_at_1)
        for w, d in hits:
            self.assertEqual(d, levenshtein(query, w))
            self.assertLessEqual(d, 1)
        # everything at distance <= 1 is present
        brute = {w for w in self.words if levenshtein(query, w) <= 1}
        self.assertEqual({w for w, d in hits}, brute)

    def test_k_zero_is_exact_membership(self):
        existing = sorted(self.words)[0]
        self.assertEqual(self.radix.fuzzy(existing, 0), [(existing, 0)])
        self.assertEqual(self.trie.fuzzy(existing, 0), [(existing, 0)])
        self.assertEqual(self.radix.fuzzy("not-a-word-" + existing, 0), [])

    def test_negative_k_returns_empty(self):
        self.assertEqual(self.radix.fuzzy("run", -1), [])
        self.assertEqual(self.trie.fuzzy("run", -1), [])

    def test_results_sorted_by_distance_then_word(self):
        hits = self.radix.fuzzy("runner", 3)
        self.assertEqual(hits, sorted(hits, key=lambda p: (p[1], p[0])))
        distances = [d for _, d in hits]
        self.assertEqual(distances, sorted(distances))

    def test_fuzzy_survives_serialization(self):
        clone = RadixTrie.loads(self.radix.dumps())
        query = "reformed"
        self.assertEqual(
            clone.fuzzy(query, 2), self.radix.fuzzy(query, 2)
        )

    def test_unrelated_word_finds_nothing_beyond_bound(self):
        hits = self.radix.fuzzy("qqqqqqqqqq", 2)
        self.assertEqual(hits, [])


if __name__ == "__main__":
    unittest.main()

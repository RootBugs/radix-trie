"""Tests for the standard (uncompressed) trie."""

from __future__ import annotations

import json
import os
import tempfile
import unittest

from tests.fixtures import UNICODE_WORDS, build_wordlist, fill, ref_complete
from trie import Trie


class InsertCountTests(unittest.TestCase):
    def test_insert_and_exact_count(self):
        t = Trie()
        t.insert("apple")
        t.insert("apple", 4)
        self.assertEqual(t.count("apple"), 5)
        self.assertEqual(t.count("app"), 0)
        self.assertEqual(len(t), 1)
        self.assertIn("apple", t)
        self.assertNotIn("app", t)

    def test_count_prefix_sums_frequencies(self):
        t = Trie()
        t.insert("car", 3)
        t.insert("card", 2)
        t.insert("care", 1)
        t.insert("dog", 7)
        self.assertEqual(t.count_prefix("car"), 6)
        self.assertEqual(t.count_prefix("car"), t.count("car") + t.count("card") + t.count("care"))
        self.assertEqual(t.count_prefix("ca"), 6)
        self.assertEqual(t.count_prefix(""), 13)
        self.assertEqual(t.count_prefix("zzz"), 0)
        self.assertEqual(t.count_prefix("cards"), 0)  # prefix longer than any key

    def test_bad_inputs_rejected(self):
        t = Trie()
        with self.assertRaises(ValueError):
            t.insert("")
        with self.assertRaises(ValueError):
            t.insert("ok", 0)
        with self.assertRaises(ValueError):
            t.insert("ok", -3)


class CompletionTests(unittest.TestCase):
    def test_complete_sorts_frequency_then_lexicographic(self):
        t = Trie()
        for w, f in [("ax", 5), ("ab", 5), ("az", 9), ("aq", 5), ("zz", 100)]:
            t.insert(w, f)
        self.assertEqual(t.complete("a"), ["az", "ab", "aq", "ax"])
        self.assertEqual(t.complete("a", limit=2), ["az", "ab"])
        self.assertEqual(t.complete("q"), [])

    def test_complete_matches_reference_on_wordlist(self):
        words = build_wordlist()
        t = Trie()
        fill(t, words)
        for prefix in ["", "re", "un", "pre", "reset", "dis", "xyz", "unders"]:
            self.assertEqual(t.complete(prefix), ref_complete(words, prefix), prefix)
            self.assertEqual(
                t.complete(prefix, limit=5), ref_complete(words, prefix, 5), prefix
            )

    def test_words_and_prefix_listing_sorted(self):
        words = build_wordlist(count=60)
        t = Trie()
        fill(t, words)
        self.assertEqual(t.words(), sorted(words))
        self.assertEqual(
            t.words_with_prefix("re"), sorted(w for w in words if w.startswith("re"))
        )


class DeleteTests(unittest.TestCase):
    def test_delete_removes_exactly_one_key(self):
        words = build_wordlist()
        t = Trie()
        fill(t, words)
        victim = sorted(words)[len(words) // 2]
        self.assertIn(victim, words)
        self.assertTrue(t.delete(victim))
        remaining = {w: f for w, f in words.items() if w != victim}
        self.assertEqual(len(t), len(remaining))
        self.assertNotIn(victim, t)
        self.assertEqual(t.count(victim), 0)
        for w, f in remaining.items():
            self.assertEqual(t.count(w), f, w)
        self.assertEqual(t.words(), sorted(remaining))

    def test_delete_missing_or_twice_returns_false(self):
        t = Trie()
        t.insert("alpha")
        self.assertFalse(t.delete("alph"))
        self.assertFalse(t.delete("alphabet"))
        self.assertTrue(t.delete("alpha"))
        self.assertFalse(t.delete("alpha"))
        self.assertEqual(len(t), 0)

    def test_delete_prunes_dead_nodes(self):
        t = Trie()
        t.insert("ab")
        t.insert("abc")
        self.assertTrue(t.delete("abc"))
        self.assertTrue(t.delete("ab"))
        self.assertEqual(t.stats()["nodes"], 1)  # only the root remains
        self.assertEqual(len(t), 0)

    def test_delete_partial_chain_keeps_siblings(self):
        t = Trie()
        t.insert("abcd")
        t.insert("abzz")
        self.assertTrue(t.delete("abzz"))
        self.assertEqual(t.words(), ["abcd"])
        self.assertEqual(t.count("abcd"), 1)


class StatsAndSerializationTests(unittest.TestCase):
    def test_stats_reports_nodes_depth_and_words(self):
        t = Trie()
        fill(t, {"a": 1, "ab": 1, "abc": 1})
        stats = t.stats()
        self.assertEqual(stats["kind"], "trie")
        self.assertEqual(stats["words"], 3)
        self.assertEqual(stats["nodes"], 4)  # root, a, b, c
        self.assertEqual(stats["max_depth"], 3)
        self.assertEqual(stats["compression_ratio"], 1.0)

    def test_dumps_loads_roundtrip(self):
        words = build_wordlist(count=80)
        t = Trie()
        fill(t, words)
        restored = Trie.loads(t.dumps())
        self.assertEqual(restored.words(), t.words())
        for w, f in words.items():
            self.assertEqual(restored.count(w), f)
            self.assertEqual(restored.count_prefix(w[:3]), t.count_prefix(w[:3]))
        self.assertEqual(restored.stats(), t.stats())
        self.assertEqual(restored.complete("re"), t.complete("re"))

    def test_save_load_roundtrip_file(self):
        t = Trie()
        fill(t, {"go": 2, "gone": 3})
        fd, path = tempfile.mkstemp(suffix=".json")
        self.addCleanup(os.remove, path)
        os.close(fd)
        t.save(path)
        restored = Trie.load(path)
        self.assertEqual(restored.words(), ["go", "gone"])
        self.assertEqual(restored.count("gone"), 3)

    def test_wrong_format_rejected(self):
        with self.assertRaises(ValueError):
            Trie.loads(json.dumps({"format": "radix-trie/1", "root": {}}))


class UnicodeTests(unittest.TestCase):
    def test_unicode_keys_insert_complete_delete(self):
        t = Trie()
        for i, w in enumerate(UNICODE_WORDS):
            t.insert(w, i + 1)
        self.assertEqual(len(t), len(UNICODE_WORDS))
        self.assertIn("café", t)
        self.assertIn("日本語", t)
        # café (freq 1... first two words share prefix "caf")
        completions = t.complete("caf")
        self.assertIn("café", completions)
        self.assertIn("cafeteria", completions)
        ordered = sorted(completions, key=lambda w: (-t.count(w), w))
        self.assertEqual(completions, ordered)
        self.assertTrue(t.delete("café"))
        self.assertNotIn("café", t)
        self.assertIn("cafeteria", t)

    def test_unicode_survives_serialization(self):
        t = Trie()
        fill(t, {w: 1 for w in UNICODE_WORDS})
        restored = Trie.loads(t.dumps())
        self.assertEqual(restored.words(), sorted(UNICODE_WORDS))


class StructureTests(unittest.TestCase):
    def test_random_ops_keep_all_keys_intact(self):
        # a plain trie has no compression invariant to break — deleting a
        # prefix of another key must never disturb its longer neighbour
        words = build_wordlist(seed=7, count=100)
        t = Trie()
        fill(t, words)
        victims = list(words)[:40]
        for v in victims:
            self.assertTrue(t.delete(v))
        expected = {w: f for w, f in words.items() if w not in set(victims)}
        self.assertEqual(t.words(), sorted(expected))
        for w, f in expected.items():
            self.assertEqual(t.count(w), f)


if __name__ == "__main__":
    unittest.main()

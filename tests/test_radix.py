"""Tests for the compressed radix trie: semantics, compaction, compression."""

from __future__ import annotations

import json
import os
import random
import tempfile
import unittest

from radix import RadixTrie, common_prefix_len
from tests.fixtures import (
    UNICODE_WORDS,
    assert_compressed,
    build_wordlist,
    clustered_wordlist,
    fill,
    ref_complete,
)
from trie import Trie


class BasicSemanticsTests(unittest.TestCase):
    def test_insert_count_complete_on_synthetic_branches(self):
        t = RadixTrie()
        t.insert("abcdef", 3)
        t.insert("abczz", 3)
        t.insert("abc", 10)
        self.assertEqual(len(t), 3)
        self.assertEqual(t.count("abc"), 10)
        self.assertEqual(t.count("ab"), 0)
        self.assertEqual(t.count_prefix("abc"), 16)  # 3 + 3 + 10
        self.assertEqual(t.count_prefix("ab"), 16)
        self.assertEqual(t.count_prefix("abcdef"), 3)
        self.assertEqual(t.count_prefix("abcd"), 3)  # prefix of abcdef
        self.assertEqual(t.count_prefix("abx"), 0)   # diverges inside an edge
        self.assertEqual(t.complete("abc"), ["abc", "abcdef", "abczz"])

    def test_common_prefix_len(self):
        self.assertEqual(common_prefix_len("abc", "abd"), 2)
        self.assertEqual(common_prefix_len("", "abd"), 0)
        self.assertEqual(common_prefix_len("abc", "abc"), 3)

    def test_mid_edge_lookup_is_not_a_key(self):
        t = RadixTrie()
        t.insert("alphabet")
        self.assertNotIn("alph", t)
        self.assertEqual(t.count("alph"), 0)
        self.assertIn("alphabet", t)
        # ...but prefix queries through the middle of an edge work
        self.assertEqual(t.count_prefix("alpha"), 1)
        self.assertEqual(t.words_with_prefix("alph"), ["alphabet"])

    def test_bad_inputs_rejected(self):
        t = RadixTrie()
        with self.assertRaises(ValueError):
            t.insert("")
        with self.assertRaises(ValueError):
            t.insert("x", 0)


class CompletionTests(unittest.TestCase):
    def test_complete_matches_reference_on_wordlist(self):
        words = build_wordlist()
        t = RadixTrie()
        fill(t, words)
        for prefix in ["", "r", "re", "reset", "unders", "dis", "zzz"]:
            self.assertEqual(t.complete(prefix), ref_complete(words, prefix), prefix)
            self.assertEqual(
                t.complete(prefix, limit=4), ref_complete(words, prefix, 4), prefix
            )

    def test_complete_tiebreak_is_lexicographic(self):
        t = RadixTrie()
        for w in ["zx", "za", "zm"]:
            t.insert(w, 7)
        t.insert("zz", 1)
        self.assertEqual(t.complete("z"), ["za", "zm", "zx", "zz"])

    def test_words_match_reference_and_trie(self):
        words = build_wordlist(count=120)
        r = RadixTrie()
        fill(r, words)
        plain = Trie()
        fill(plain, words)
        self.assertEqual(r.words(), sorted(words))
        self.assertEqual(r.words(), plain.words())


class CrossEngineEquivalenceTests(unittest.TestCase):
    """The compressed and uncompressed structures must agree on everything."""

    def test_counts_and_completions_identical(self):
        words = build_wordlist(seed=42)
        radix, plain = RadixTrie(), Trie()
        fill(radix, words)
        fill(plain, words)
        for w in words:
            self.assertEqual(radix.count(w), plain.count(w))
            self.assertEqual(radix.count_prefix(w[:4]), plain.count_prefix(w[:4]))
        for prefix in ["", "u", "un", "pre", "dis", "run", "qqq"]:
            self.assertEqual(
                radix.complete(prefix), plain.complete(prefix), prefix
            )
            self.assertEqual(radix.words_with_prefix(prefix),
                             plain.words_with_prefix(prefix), prefix)

    def test_delete_sequence_identical(self):
        words = build_wordlist(seed=43, count=90)
        radix, plain = RadixTrie(), Trie()
        fill(radix, words)
        fill(plain, words)
        for victim in sorted(words):
            self.assertEqual(radix.delete(victim), plain.delete(victim), victim)
            self.assertEqual(radix.words(), plain.words())
            self.assertEqual(len(radix), len(plain))


class CompressionTests(unittest.TestCase):
    def test_compression_ratio_on_clustered_keys(self):
        words = clustered_wordlist()
        plain = Trie()
        fill(plain, words)
        radix = RadixTrie()
        fill(radix, words)

        r_stats = radix.stats()
        p_stats = plain.stats()
        # far fewer nodes than the character trie
        self.assertLess(r_stats["nodes"], p_stats["nodes"])
        self.assertLess(r_stats["compression_ratio"], 1.0)
        # the documented identity: a plain trie needs exactly `label_chars + 1`
        self.assertEqual(r_stats["uncompressed_nodes"], p_stats["nodes"])
        self.assertEqual(r_stats["words"], len(words))

    def test_compressed_nodes_less_than_uncompressed_on_prefix_words(self):
        t = RadixTrie()
        words = ["international", "interstate", "internet", "interface"]
        fill(t, {w: 1 for w in words})
        stats = t.stats()
        self.assertLess(stats["nodes"], stats["uncompressed_nodes"])
        # hand-counted structure: root, "inter", "n" branch, "ational", "et",
        # "state", "face" — seven nodes instead of twenty-five
        self.assertEqual(stats["nodes"], 7)
        self.assertEqual(stats["uncompressed_nodes"], 25)

    def test_stats_on_single_word(self):
        t = RadixTrie()
        t.insert("word")
        stats = t.stats()
        self.assertEqual(stats["nodes"], 2)       # root + leaf
        self.assertEqual(stats["uncompressed_nodes"], 5)
        self.assertEqual(stats["compression_ratio"], 2 / 5)


class DeleteCompactionTests(unittest.TestCase):
    def test_delete_remerges_single_child_chain(self):
        t = RadixTrie()
        t.insert("abc")
        t.insert("abz")
        self.assertGreater(t.stats()["nodes"], 3)  # branch exists
        self.assertTrue(t.delete("abz"))
        assert_compressed(self, t)
        # whole edge collapsed back to one label
        self.assertEqual(t.stats()["nodes"], 2)
        self.assertEqual(t.words(), ["abc"])
        self.assertEqual(t.stats()["label_chars"], 3)

    def test_delete_middle_of_edge_leaves_valid_structure(self):
        t = RadixTrie()
        t.insert("ab")
        t.insert("abc")
        self.assertTrue(t.delete("ab"))
        assert_compressed(self, t)
        self.assertEqual(t.words(), ["abc"])
        self.assertEqual(t.count_prefix("ab"), 1)
        self.assertFalse(t.delete("ab"))

    def test_delete_missing_returns_false(self):
        t = RadixTrie()
        t.insert("alpha")
        self.assertFalse(t.delete("alphaa"))
        self.assertFalse(t.delete("lpha"))
        self.assertEqual(len(t), 1)

    def test_random_op_sequence_matches_reference_dict(self):
        rng = random.Random(2026)
        words = build_wordlist(seed=11, count=120)
        ref: dict[str, int] = {}
        t = RadixTrie()
        for step in range(300):
            action = rng.random()
            if action < 0.6:
                word = rng.choice(sorted(words))
                t.insert(word)
                ref[word] = ref.get(word, 0) + 1
            elif action < 0.9:
                word = rng.choice(sorted(ref)) if ref else "none"
                expected = word in ref
                self.assertEqual(t.delete(word), expected, f"step {step}")
                ref.pop(word, None)
            else:
                word = rng.choice(sorted(words))
                self.assertEqual(t.count(word), ref.get(word, 0), f"step {step}")
            if step % 25 == 0:
                assert_compressed(self, t)
        assert_compressed(self, t)
        self.assertEqual(t.words(), sorted(ref))
        for w, f in ref.items():
            self.assertEqual(t.count(w), f)
        self.assertEqual(t.complete("u"), ref_complete(ref, "u"))


class UnicodeTests(unittest.TestCase):
    def test_unicode_keys_complete_fuzzy_delete(self):
        t = RadixTrie()
        for i, w in enumerate(UNICODE_WORDS):
            t.insert(w, i + 1)
        self.assertEqual(len(t), len(UNICODE_WORDS))
        self.assertIn("café", t)
        caf = t.complete("caf")
        self.assertEqual(set(caf), {"café", "cafeteria", "cafés"})
        # frequency ordering: cafeteria (2) < cafés (3) < café (1) reversed:
        self.assertEqual(caf, sorted(caf, key=lambda w: (-t.count(w), w)))
        self.assertEqual(t.count_prefix("日本"), t.count("日本") + t.count("日本語"))
        self.assertTrue(t.delete("café"))
        self.assertNotIn("café", t)
        self.assertIn("cafeteria", t)
        assert_compressed(self, t)

    def test_unicode_serialization_roundtrip(self):
        t = RadixTrie()
        fill(t, {w: i + 1 for i, w in enumerate(UNICODE_WORDS)})
        restored = RadixTrie.loads(t.dumps())
        self.assertEqual(restored.words(), t.words())
        for w in UNICODE_WORDS:
            self.assertEqual(restored.count(w), t.count(w))


class SerializationTests(unittest.TestCase):
    def test_dumps_loads_roundtrip_preserves_everything(self):
        words = clustered_wordlist(seed=5, count=60)
        t = RadixTrie()
        fill(t, words)
        restored = RadixTrie.loads(t.dumps())
        self.assertEqual(restored.words(), t.words())
        self.assertEqual(restored.stats(), t.stats())
        self.assertEqual(restored.complete("common_prefix_segment_000"),
                         t.complete("common_prefix_segment_000"))
        for w, f in list(words.items())[:30]:
            self.assertEqual(restored.count(w), f)
        # both documents are byte-identical (deterministic JSON)
        self.assertEqual(restored.dumps(), t.dumps())

    def test_save_load_roundtrip_file(self):
        t = RadixTrie()
        fill(t, {"go": 2, "gone": 3, "gong": 7})
        fd, path = tempfile.mkstemp(suffix=".json")
        self.addCleanup(os.remove, path)
        os.close(fd)
        t.save(path)
        restored = RadixTrie.load(path)
        self.assertEqual(restored.words(), ["go", "gone", "gong"])
        self.assertEqual(restored.count("gong"), 7)

    def test_wrong_format_rejected(self):
        with self.assertRaises(ValueError):
            RadixTrie.loads(json.dumps({"format": "trie/1", "root": {}}))

    def test_corrupt_child_key_rejected(self):
        doc = json.loads(RadixTrie().dumps())
        # empty trie loads fine...
        self.assertEqual(RadixTrie.loads(json.dumps(doc)).words(), [])
        # ...but a mismatched child key is a structural error
        doc["root"]["c"] = {"z": {"l": "abc"}}
        with self.assertRaises(ValueError):
            RadixTrie.loads(json.dumps(doc))


if __name__ == "__main__":
    unittest.main()

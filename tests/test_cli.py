"""Tests for the radix-trie CLI."""

from __future__ import annotations

import contextlib
import io
import json
import os
import tempfile
import unittest

import cli


class CliTestCase(unittest.TestCase):
    def _wordlist(self, text: str) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt")
        self.addCleanup(os.remove, path)
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            fh.write(text)
        return path

    def _run(self, argv: list[str]) -> tuple[int, str, str]:
        out, err = io.StringIO(), io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            code = cli.main(argv)
        return code, out.getvalue(), err.getvalue()

    def _temp_path(self, suffix=".json") -> str:
        fd, path = tempfile.mkstemp(suffix=suffix)
        self.addCleanup(os.remove, path)
        os.close(fd)
        return path


class CliTests(CliTestCase):
    def setUp(self):
        self.words = self._wordlist(
            "# comment\n"
            "reset 10\n"
            "rerun 7\n"
            "unset 3\n"
            "under 2\n"
            "\n"
            "returner 5\n"
        )

    def test_build_stdout_emits_json_document(self):
        code, out, err = self._run(["build", "--wordlist", self.words])
        self.assertEqual(code, 0)
        doc = json.loads(out)
        self.assertEqual(doc["format"], "radix-trie/1")

    def test_build_to_file_then_complete_from_trie(self):
        target = self._temp_path()
        code, out, err = self._run(
            ["build", "--wordlist", self.words, "-o", target]
        )
        self.assertEqual(code, 0)
        self.assertIn("5 words", out)
        code, out, err = self._run(["complete", "re", "--trie", target])
        self.assertEqual(code, 0)
        self.assertEqual(out.split(), ["reset", "rerun", "returner"])

    def test_complete_from_wordlist_with_limit(self):
        code, out, err = self._run(
            ["complete", "re", "--wordlist", self.words, "--limit", "2"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.split(), ["reset", "rerun"])  # frequency order

    def test_complete_engine_trie_agrees(self):
        code, out, err = self._run(
            ["complete", "re", "--wordlist", self.words, "--engine", "trie"]
        )
        self.assertEqual(code, 0)
        self.assertEqual(out.split(), ["reset", "rerun", "returner"])

    def test_fuzzy_prints_distance_word_pairs(self):
        code, out, err = self._run(
            ["fuzzy", "resext", "--distance", "1", "--wordlist", self.words]
        )
        self.assertEqual(code, 0)
        lines = out.strip().splitlines()
        self.assertIn("1\treset", lines)
        for line in lines:
            dist, word = line.split("\t")
            self.assertEqual(dist, "1")
            self.assertIn(word, {"reset", "rerun", "unset", "under", "returner"})

    def test_fuzzy_from_serialized_trie(self):
        target = self._temp_path()
        self._run(["build", "--wordlist", self.words, "-o", target,
                   "--engine", "trie"])
        code, out, err = self._run(
            ["fuzzy", "unsett", "--distance", "1", "--trie", target,
             "--engine", "trie"]
        )
        self.assertEqual(code, 0)
        self.assertIn("1\tunset", out)

    def test_stats_reports_compression(self):
        code, out, err = self._run(["stats", "--wordlist", self.words])
        self.assertEqual(code, 0)
        data = json.loads(out)
        self.assertEqual(data["kind"], "radix")
        self.assertEqual(data["words"], 5)
        self.assertLess(data["nodes"], data["uncompressed_nodes"])

    def test_missing_source_errors(self):
        code, out, err = self._run(["stats"])
        self.assertEqual(code, 1)
        self.assertIn("--wordlist or --trie", err)

    def test_missing_file_errors(self):
        code, out, err = self._run(["stats", "--wordlist", "no-such-file-xyz.txt"])
        self.assertEqual(code, 1)
        self.assertIn("error:", err)

    def test_malformed_wordlist_reports_line_number(self):
        bad = self._wordlist("good\nbadword notanumber\n")
        code, out, err = self._run(["stats", "--wordlist", bad])
        self.assertEqual(code, 1)
        self.assertIn(":2:", err)

    def test_loading_wrong_engine_document_errors(self):
        # a trie/1 document cannot be loaded as a radix trie
        target = self._temp_path()
        self._run(["build", "--wordlist", self.words, "-o", target,
                   "--engine", "trie"])
        code, out, err = self._run(
            ["complete", "re", "--trie", target, "--engine", "radix"]
        )
        self.assertEqual(code, 1)
        self.assertIn("error:", err)


if __name__ == "__main__":
    unittest.main()

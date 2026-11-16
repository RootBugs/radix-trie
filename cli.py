"""Command-line interface for radix-trie.

Usage::

    python cli.py build   --wordlist WORDS.txt [-o trie.json] [--engine trie|radix]
    python cli.py complete PREFIX [--wordlist WORDS.txt | --trie trie.json]
                          [--limit N] [--engine trie|radix]
    python cli.py fuzzy   WORD --distance K [--wordlist WORDS.txt |
                          --trie trie.json] [--engine trie|radix]
    python cli.py stats   [--wordlist WORDS.txt | --trie trie.json] [--engine ...]

Wordlist format: one entry per line, ``word`` or ``word frequency``; blank
lines and ``#`` comments are ignored.
"""

from __future__ import annotations

import argparse
import json
import sys

from radix import RadixTrie
from trie import Trie


def load_wordlist(path: str, engine) -> object:
    """Build a trie/radix trie from a wordlist file.

    Raises :class:`ValueError` with the offending line on malformed input.
    """
    trie = engine()
    with open(path, "r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line or line.startswith("#"):
                continue
            parts = line.split()
            if len(parts) == 1:
                trie.insert(parts[0])
            elif len(parts) == 2:
                try:
                    freq = int(parts[1])
                except ValueError:
                    raise ValueError(
                        f"{path}:{lineno}: bad frequency {parts[1]!r}"
                    ) from None
                if freq <= 0:
                    raise ValueError(f"{path}:{lineno}: frequency must be positive")
                trie.insert(parts[0], freq)
            else:
                raise ValueError(f"{path}:{lineno}: expected 'word [freq]'")
    return trie


def get_trie(args: argparse.Namespace):
    """Resolve the trie from --trie (serialized) or --wordlist (built fresh)."""
    engine = RadixTrie if args.engine == "radix" else Trie
    if args.trie:
        return engine.load(args.trie)
    if args.wordlist:
        return load_wordlist(args.wordlist, engine)
    raise ValueError("provide --wordlist or --trie")


def _add_common(sub: argparse.ArgumentParser) -> None:
    sub.add_argument("--wordlist", default=None, help="wordlist to build from")
    sub.add_argument("--trie", default=None, help="serialized trie JSON to load")
    sub.add_argument(
        "--engine", choices=("radix", "trie"), default="radix",
        help="data structure to use (default: radix)",
    )


def _build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="radix-trie", description=__doc__)
    sub = ap.add_subparsers(dest="command", required=True)

    build = sub.add_parser("build", help="build from a wordlist and serialize")
    _add_common(build)
    build.add_argument("-o", "--out", default=None, help="write JSON here")

    complete = sub.add_parser("complete", help="autocomplete a prefix")
    complete.add_argument("prefix")
    complete.add_argument("--limit", type=int, default=10)
    _add_common(complete)

    fuzzy = sub.add_parser("fuzzy", help="bounded-edit-distance search")
    fuzzy.add_argument("word")
    fuzzy.add_argument("--distance", type=int, required=True)
    _add_common(fuzzy)

    stats = sub.add_parser("stats", help="structural statistics")
    _add_common(stats)
    return ap


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        trie = get_trie(args)
        return _dispatch(args, trie)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1


def _dispatch(args: argparse.Namespace, trie) -> int:
    if args.command == "build":
        if args.out:
            trie.save(args.out)
            print(f"wrote {args.out} ({len(trie)} words)")
        else:
            sys.stdout.write(trie.dumps())
            print()
        return 0

    if args.command == "complete":
        for word in trie.complete(args.prefix, limit=max(args.limit, 0)):
            print(word)
        return 0

    if args.command == "fuzzy":
        for word, dist in trie.fuzzy(args.word, args.distance):
            print(f"{dist}\t{word}")
        return 0

    if args.command == "stats":
        print(json.dumps(trie.stats(), indent=2, sort_keys=True))
        return 0

    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())

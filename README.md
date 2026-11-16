# radix-trie — Compressed Trie & Fuzzy Search

Autocomplete that feels instant. `radix-trie` ships **two structures with
identical semantics** — a plain character trie and a path-compressed radix
tree — plus a bounded-Levenshtein fuzzy search that prunes whole subtrees
while it walks. Insert, prefix counts, ranked completion, delete with node
compaction, and deterministic JSON serialization, all zero-dependency.

## Why it's special

- **Path compression you can measure.** Every chain of single-child,
  non-terminal nodes collapses into one edge label. `stats()` reports
  `uncompressed_nodes` — *exactly* what a plain trie over the same keys would
  cost (`1 + sum of label lengths`) — and the ratio against the real node
  count. On clustered keys the compressed tree is an order of magnitude
  smaller, and a test asserts the identity against the plain trie's own
  node count.
- **Fuzzy search is bounded DP, not `min` over everything.** The edge walk
  carries one banded DP column per edge; the moment the column minimum
  crosses `k`, the entire subtree is provably out of reach and gets pruned.
  The test oracle is brute-force full Levenshtein over every key — the trie
  must find *everything* within `k` and *nothing* beyond it, for both engines.
- **Delete restores the invariant.** Removing a key prunes dead nodes and
  re-merges the exposed single-child chain (`ab` + `c` → `abc`); a checker
  verifies the compressed invariant after randomized insert/delete storms.
- **Two engines, one contract.** `Trie` and `RadixTrie` are cross-checked
  word-for-word: counts, prefix sums, completions, and an entire delete
  sequence must agree after every single operation.

## Architecture

| Module | Responsibility |
|---|---|
| `trie.py` | Character-edge `Trie`: insert/delete/`count`/`count_prefix`/`complete`/`fuzzy`, stats, JSON roundtrip |
| `radix.py` | `RadixTrie`: path-compressed edges, mid-edge prefix resolution, compaction on delete, compression stats, JSON roundtrip |
| `levenshtein.py` | Full `levenshtein`, banded `levenshtein_bounded` with early exit, and the `ColumnState` the trie walks feed character by character |
| `cli.py` | `build`, `complete`, `fuzzy`, `stats` over wordlists or serialized tries, with `--engine trie\|radix` |

## Runnable example

```console
$ python -c "
from radix import RadixTrie

t = RadixTrie()
for word, freq in [('reset', 10), ('rerun', 7), ('returner', 5), ('unset', 3), ('under', 2)]:
    t.insert(word, freq)
print('complete re :', t.complete('re'))
print('fuzzy resext:', t.fuzzy('resext', 1))
print('count_prefix:', t.count_prefix('re'))
print('stats       :', {k: t.stats()[k] for k in ('nodes', 'uncompressed_nodes')})
t.delete('rerun')
print('after delete:', t.words())
print('roundtrip   :', RadixTrie.loads(t.dumps()).words() == t.words())
"
complete re : ['reset', 'rerun', 'returner']
fuzzy resext: [('reset', 1)]
count_prefix: 22
stats       : {'nodes': 8, 'uncompressed_nodes': 23}
after delete: ['reset', 'returner', 'under', 'unset']
roundtrip   : True
```

The CLI works on wordlists (`word [frequency]` per line) or serialized JSON:

```console
$ printf "reset 10\nrerun 7\nreturner 5\nunset 3\nunder 2\n" > words.txt
$ python cli.py complete re --wordlist words.txt --limit 2
reset
rerun
$ python cli.py fuzzy resext --distance 1 --wordlist words.txt
1	reset
$ python cli.py build --wordlist words.txt -o trie.json
wrote trie.json (5 words)
$ python cli.py stats --trie trie.json
{
  "compression_ratio": 0.347...,
  "kind": "radix",
  ...
}
```

## Semantics (documented policies)

- Words must be non-empty strings; frequencies must be positive and accumulate.
- `complete(prefix)` sorts by **descending frequency, then lexicographically**;
  `limit` truncates after ranking.
- `count_prefix(p)` sums frequencies of all keys with prefix `p` (empty prefix
  = whole vocabulary); `p` may end mid-edge or mid-path and need not be a key.
- `delete(word)` removes the entry entirely regardless of accumulated
  frequency and returns whether it existed.
- Fuzzy results are `(word, distance)` pairs sorted by `(distance, word)`.

## Test command

```
python -m unittest discover -s tests -v
```

Run from the project root. 63 tests cover reference-matching completions,
brute-force-matching fuzzy search, compression identities, compaction
invariants, unicode keys, and persistence roundtrips.

## License

MIT — see `LICENSE`.

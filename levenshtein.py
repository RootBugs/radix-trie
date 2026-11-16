"""Levenshtein edit distance: full DP and bounded (early-exit) DP.

The bounded variant powers fuzzy search: it answers "is the distance at most
``k``?" while computing at most a ``k``-wide band of the table, and returns
``None`` the moment no completion of either string could still land inside the
bound.  That early exit is what lets trie walks prune whole subtrees.
"""

from __future__ import annotations

INF = 1 << 30  # stands in for "beyond the bound" inside band tables


def levenshtein(a: str, b: str) -> int:
    """Classic O(len(a) * len(b)) edit distance (insert/delete/substitute = 1).

    Used as the brute-force oracle by the test suite and as the unbounded
    fallback when no bound is needed.
    """
    if a == b:
        return 0
    if not a:
        return len(b)
    if not b:
        return len(a)
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        current = [i] + [0] * len(b)
        for j, cb in enumerate(b, start=1):
            sub = previous[j - 1] + (0 if ca == cb else 1)
            current[j] = min(previous[j] + 1, current[j - 1] + 1, sub)
        previous = current
    return previous[len(b)]


def levenshtein_bounded(a: str, b: str, k: int) -> int | None:
    """Banded edit distance with early exit.

    Returns the exact distance when it is ``<= k``, otherwise ``None``.
    Only cells with ``|i - j| <= k`` are computed (outside the band no path of
    cost ``<= k`` can pass), and the whole computation aborts as soon as a
    column's minimum exceeds ``k`` — from that point on every later column must
    exceed ``k`` too, because each cell's three recurrence sources all sit in
    the current or next column.
    """
    if k < 0:
        return None
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > k:
        return None
    if not a:
        return lb if lb <= k else None
    if not b:
        return la if la <= k else None

    # column j of the table; j = 0 is the "empty trie prefix" column
    col = [i if i <= k else INF for i in range(la + 1)]
    if min(col) > k:
        return None
    for j in range(1, lb + 1):
        c = b[j - 1]
        lo = max(1, j - k)
        hi = min(la, j + k)
        new = [INF] * (la + 1)
        new[0] = j if j <= k else INF
        for i in range(lo, hi + 1):
            sub = col[i - 1] + (0 if a[i - 1] == c else 1)
            best = col[i] + 1          # delete from a-side (vertical move)
            if new[i - 1] + 1 < best:  # insert (horizontal move)
                best = new[i - 1] + 1
            if sub < best:
                best = sub
            new[i] = best
        col = new
        if min(col) > k:
            return None
    result = col[la]
    return result if result <= k else None


class ColumnState:
    """One DP column while walking a trie edge character by character.

    ``feed(ch)`` appends a character to the word-under-construction side of the
    table; ``minimum()`` reports the smallest cell so the walk can prune;
    ``final_cost()`` yields the distance once the whole word is consumed.
    """

    __slots__ = ("_col", "_query", "_k", "_j", "_dead")

    def __init__(self, query: str, k: int) -> None:
        self._query = query
        self._k = k
        self._j = 0
        self._dead = False
        self._col = [i if i <= k else INF for i in range(len(query) + 1)]
        if min(self._col) > k:
            self._dead = True

    @property
    def dead(self) -> bool:
        """True once no completion of the word can fall within ``k``."""
        return self._dead

    def feed(self, text: str) -> None:
        """Extend the word side by every character of ``text``."""
        if self._dead:
            return
        k = self._k
        la = len(self._query)
        for c in text:
            j = self._j + 1
            lo = max(1, j - k)
            hi = min(la, j + k)
            new = [INF] * (la + 1)
            new[0] = j if j <= k else INF
            col = self._col
            q = self._query
            for i in range(lo, hi + 1):
                sub = col[i - 1] + (0 if q[i - 1] == c else 1)
                best = col[i] + 1
                if new[i - 1] + 1 < best:
                    best = new[i - 1] + 1
                if sub < best:
                    best = sub
                new[i] = best
            self._col = new
            self._j = j
            if min(new) > k:
                self._dead = True
                return

    def clone(self) -> "ColumnState":
        """Copy for backtracking at branch points (cheap list copy)."""
        twin = ColumnState.__new__(ColumnState)
        twin._query = self._query
        twin._k = self._k
        twin._j = self._j
        twin._dead = self._dead
        twin._col = list(self._col)
        return twin

    def final_cost(self) -> int | None:
        """Distance for the word consumed so far, or ``None`` if ``> k``."""
        if self._dead:
            return None
        result = self._col[len(self._query)]
        return result if result <= self._k else None

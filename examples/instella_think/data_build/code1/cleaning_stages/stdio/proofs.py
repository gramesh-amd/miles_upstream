#!/usr/bin/env python3
"""Exhaustive checks of disputed labels (candidates from blocking_candidates.csv: >= 2 references
pass every other test and agree on a different output). Each check implements the statement
literally by exhaustive search, independent of every reference. Writes proofs.json:
{custom_id: [{test, label, brute, verdict, method}]}. verdict: label_correct | label_wrong | invalid_input.
"""
import functools, heapq, itertools, json, os

K = "klear-code-rlvr_filtered-request-"
S = "synthetic2-rlvr-code-compressed_filtered-request-"
out = {}


def rec(cid, test, label, brute, method, verdict=None):
    if verdict is None:
        verdict = "label_correct" if str(brute).strip() == str(label).strip() else "label_wrong"
    out.setdefault(cid, []).append(dict(test=test, label=str(label).strip(), brute=str(brute).strip(),
                                        verdict=verdict, method=method))


# --- joystick (CF 651A statement): exhaustive game tree over charger choices -------------------
@functools.lru_cache(None)
def joy(a, b):
    best = 0
    if b > 1:  # charge a, b discharges (b must not be at 1%)
        nb = b - 2
        best = max(best, 1 + (joy(a + 1, nb) if nb > 0 else 0))
    if a > 1:
        na = a - 2
        best = max(best, 1 + (joy(na, b + 1) if na > 0 else 0))
    return best


assert joy(3, 5) == 6 and joy(4, 4) == 5  # statement samples
for cid, t in ((K + "27-83", 8), (K + "5-30", 6)):
    rec(cid, t, "0", joy(1, 1), "exhaustive game tree; statement samples 3 5->6, 4 4->5 reproduced")


# --- minimal total cost to sort by swaps (cost w_i + w_j): Dijkstra over all permutations --------
def min_sort_cost(w):
    start, goal = tuple(w), tuple(sorted(w))
    dist = {start: 0}
    pq = [(0, start)]
    while pq:
        d, s = heapq.heappop(pq)
        if s == goal:
            return d
        if d > dist[s]:
            continue
        for i, j in itertools.combinations(range(len(s)), 2):
            t = list(s)
            t[i], t[j] = t[j], t[i]
            t = tuple(t)
            nd = d + s[i] + s[j]
            if nd < dist.get(t, 1 << 60):
                dist[t] = nd
                heapq.heappush(pq, (nd, t))


assert min_sort_cost([1, 5, 3, 4, 2]) == 7 and min_sort_cost([4, 3, 2, 1]) == 10
rec(K + "67-32", 11, "13", min_sort_cost([3, 5, 2, 1]), "Dijkstra over all 4! permutations; samples reproduced")


# --- Treasure Hunt (CF 979B) n=1: every legal single change per ribbon, max substring repeat -----
def beauty(s):
    # max occurrences of any substring = max char frequency (single letters dominate)
    return max(s.count(c) for c in set(s))


def best_after_one_change(s):
    letters = [chr(c) for c in range(65, 91)] + [chr(c) for c in range(97, 123)]
    return max(beauty(s[:i] + c + s[i + 1:]) for i in range(len(s)) for c in letters if c != s[i])


r = [best_after_one_change(x) for x in ("aaaaaaaaaa", "AAAAAAcAAA", "bbbbbbzzbb")]
win = ["Kuro", "Shiro", "Katie"][r.index(max(r))] if r.count(max(r)) == 1 else "Draw"
rec(S + "62-48", 4, "Shiro", win, f"all single changes enumerated; beauties {r}")


# --- Greg's grid (pairwise co-prime cells, value i+j): brute force small grids, then sieve -------
def greg_brute(n, m):
    vals = sorted({i + j for i in range(1, n + 1) for j in range(1, m + 1)})
    from math import gcd
    best = 0
    for k in range(1, len(vals) + 1):
        found = False
        for comb in itertools.combinations(vals, k):
            if all(gcd(a, b) == 1 for a, b in itertools.combinations(comb, 2)):
                found = True
                break
        if found:
            best = k
        else:
            break
    return best


def primepi(x):
    sieve = bytearray([1]) * (x + 1)
    sieve[0:2] = b"\x00\x00"
    for p in range(2, int(x ** 0.5) + 1):
        if sieve[p]:
            sieve[p * p::p] = bytearray(len(range(p * p, x + 1, p)))
    return sum(sieve)


# cells with equal values are never co-prime (value >= 2), so the answer only depends on the value set
assert all(greg_brute(n, m) == primepi(n + m) for n in range(1, 6) for m in range(1, 6))
assert greg_brute(3, 4) == 4
rec(S + "99-78", 7, "148934", primepi(2_000_000),
    "exhaustive subset search == pi(n+m) for all n,m<=5 (and sample 3 4->4); pi(2e6) by sieve")


# --- binary-string operations (CF 1430D): exhaustive search over all deletion choices ------------
@functools.lru_cache(None)
def ops(s):
    if not s:
        return 0
    best = 0
    for i in range(len(s)):
        t = s[:i] + s[i + 1:]
        if t:
            k = 1
            while k < len(t) and t[k] == t[0]:
                k += 1
            t = t[k:]
        best = max(best, 1 + ops(t))
    return best


assert ops("111010") == 3
# test input: t=5 cases (6,111000) (1,1) (1,1) (2,'18') (6,000010); '18' is not a binary string
for cid in (K + "50-3", K + "96-20"):
    rec(cid, 5, "2 1 1 1 2", " ".join(str(ops(s)) for s in ("111000", "1", "1")) + " ? " + str(ops("000010")),
        "case 4 is n=2 s='18' (not binary: violates the statement); other cases by exhaustive search",
        verdict="invalid_input")

# --- jogging 4-cycles: test has duplicate roads, violating 'no more than 1 road' ----------------
edges = [(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (1, 4), (2, 3)]
dups = len(edges) - len({frozenset(e) for e in edges})
rec(S + "63-92", 6, "8", f"{dups} duplicate roads", "constraint check", verdict="invalid_input")

# --- F(x) = sum |x - A_i|^K, smallest minimising integer x: exhaustive over the whole range ----------
def argmin_f(a, k):
    lo, hi = 1, 50000  # A_i <= 5e4, F is minimised inside [min A, max A]
    return min(range(lo, hi + 1), key=lambda x: (sum(abs(x - v) ** k for v in a), x))


assert argmin_f([6, 1, 7], 1) == 6  # statement sample
rec(S + "110-88", 1, "16668", argmin_f([1, 2, 50000], 3), "exhaustive over x in [1, 5e4]; sample reproduced")

json.dump(out, open(os.path.join(os.path.dirname(__file__), "proofs.json"), "w"), indent=1)
for k, v in out.items():
    for x in v:
        print(k.split("request-")[1], x["test"], x["verdict"], "label", x["label"], "brute", x["brute"])

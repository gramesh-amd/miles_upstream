#!/usr/bin/env python3
"""Exact and near-duplicate statement clusters over in-scope rows.

exact: identical body after lowercasing and collapsing whitespace/punctuation.
near : word-5-gram Jaccard >= 0.85 AND identical multiset of numbers in the body (so easy/hard
       versions that differ only in constraints are NOT merged) AND same test format kind.
Out: work/dup_clusters.pkl {row: cluster_id}, printed stats.
"""
import collections, pickle, re, sys
sys.path.insert(0, __import__("os").path.dirname(__file__))
from analyze import body_of

W = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/work"


def norm(b):
    return " ".join(re.sub(r"[^\w]+", " ", b.lower()).split())


def main():
    vs = pickle.load(open("/tmp/v7stdio_rows.pkl", "rb"))
    info = {}
    for v in vs:
        b = body_of(v["prompt"])
        n = norm(b)
        w = n.split()
        sh = {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}
        nums = tuple(sorted(re.findall(r"\d+", b)))
        info[v["row"]] = (n, sh, nums, v["kind"])
    parent = {r: r for r in info}

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a, b):
        parent[find(a)] = find(b)

    by_exact = collections.defaultdict(list)
    for r, (n, _, _, k) in info.items():
        by_exact[(n, k)].append(r)
    n_exact_pairs = 0
    for rs in by_exact.values():
        for r in rs[1:]:
            union(rs[0], r)
            n_exact_pairs += 1
    # candidate pairs through shared rare shingles
    df = collections.Counter(s for (_, sh, _, _) in info.values() for s in sh)
    idx = collections.defaultdict(list)
    for r, (_, sh, _, _) in info.items():
        rare = sorted(sh, key=lambda s: (df[s], s))[:8]
        for s in rare:
            if df[s] <= 30:
                idx[s].append(r)
    near_pairs = set()
    for rs in idx.values():
        for i in range(len(rs)):
            for j in range(i + 1, len(rs)):
                a, b = rs[i], rs[j]
                if find(a) == find(b):
                    continue
                A, B = info[a], info[b]
                if A[3] != B[3] or A[2] != B[2]:
                    continue
                jac = len(A[1] & B[1]) / max(1, len(A[1] | B[1]))
                if jac >= 0.85:
                    near_pairs.add((a, b, round(jac, 3)))
    for a, b, _ in near_pairs:
        union(a, b)
    cl = {r: find(r) for r in info}
    pickle.dump(dict(cluster=cl, near_pairs=sorted(near_pairs)), open(f"{W}/dup_clusters.pkl", "wb"))
    src = {v["row"]: v["src"] for v in vs}
    sizes = collections.Counter(cl.values())
    for s in ("klear", "synthetic2"):
        rows = [r for r in cl if src[r] == s]
        print(s, "rows", len(rows), "distinct exact", len({(info[r][0], info[r][3]) for r in rows}),
              "clusters after near", len({cl[r] for r in rows}))
    print("exact merges", n_exact_pairs, "near pairs", len(near_pairs), "cross-source clusters",
          sum(1 for c in set(cl.values()) if len({src[r] for r in cl if cl[r] == c}) > 1))


if __name__ == "__main__":
    main()

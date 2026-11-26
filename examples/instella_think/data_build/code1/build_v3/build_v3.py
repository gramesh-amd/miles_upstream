#!/usr/bin/env python3
"""Build final_v3 candidate mixes from final_v2 (no code execution here; all verification ran through the grader).

Steps (every row counted once, at its first rule):
  1. trim tests > 1 MB (row dropped if < 5 tests remain or a constant would pass); drop rows whose fastest verified
     reference needs > 4 s on some test (v7 stdio timings; 5 s limit)
  2. add restored stdio rows (grader-verified, >= 5 tests, not constant): image placeholders, multi-answer
     statements that Open-R1 grades by token compare / that have no checker anywhere
  3. add Open-R1 checker rows accepted by verify_checkers.py (they carry metadata.checker); remove the final_v1 rows
     they replace
  4. dedup over the union: same CF id, test overlap (stdio >= 50% of the smaller set and >= min(5, n) shared io
     pairs; function >= 80% and >= 5 shared asserts), MinHash 5-gram >= 0.7 with equal digits or >= 50% shared
     inputs. Keep the copy with more tests, then the final_v2 copy.
  5. mix: A none; B keep 1/3 of not-in-v6 AceCoder + S2-function rows (base-easy with p~0.82), stdio rows in the
     1-5/8 prefilter band or not in v6 drawn twice; C drop all not-in-v6 AceCoder + S2-function, stdio x2 as in B.

    python3 build_v3.py --mix B --checkers checkers/accepted_rows.jsonl --out DIR [--write]
"""
import argparse, ast, collections, csv, gzip, hashlib, json, os, pickle, random, re, sys
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.environ.get("INSTELLA_RL_ROOT", "/it-share-prj2-1/data/goramesh-instella-rl")
WORK = os.environ.get("CODE1_WORK", f"{ROOT}/out/v7_improve/data_audit")
V7 = f"{ROOT}/out/v7_data"
sys.path.insert(0, HERE)
from neardup import run as minhash_pairs
from testdup import keys as test_keys

ap = argparse.ArgumentParser()
ap.add_argument("--mix", default="B")
ap.add_argument("--checkers", default=os.path.join(WORK, "checkers_s0/accepted_rows.jsonl") + "," + os.path.join(WORK, "checkers_s1/accepted_rows.jsonl"))
ap.add_argument("--out", default=f"{V7}/final_v3_candidate")
ap.add_argument("--write", action="store_true")
ap.add_argument("--seed", type=int, default=0)
a = ap.parse_args()
rng = random.Random(a.seed)
S = collections.Counter()
dropped = []  # (custom_id, src, route, reason)

M = pd.read_csv(os.path.join(WORK, "master2.csv"), keep_default_na=False).set_index("cid")
date = {r["custom_id"]: (r["date"], r["ref"], r["ref_id"]) for r in csv.DictReader(open(f"{V7}/decontam/date_check.csv"))}
decon = set(open(f"{V7}/decontam/drop_list.txt").read().split())


def pref_of(cid):
    if cid not in M.index or M.at[cid, "pid"] == "":
        return None
    v = M.at[cid, "v6_pref"]
    return int(float(v)) if v != "" else None


def cf_id(r):
    m = r["metadata"]
    if m.get("openr1_id"):
        return "cf:" + m["openr1_id"]
    d = date.get(m["custom_id"])
    return f"{d[1]}:{d[2]}" if d and d[1] in ("cf", "codeforces") and d[2] else None


def const_passes(tests):
    outs = [" ".join(str(t["output"]).split()) for t in tests[:32]]
    return len(set(outs)) == 1


def const_floor(tests):
    outs = [" ".join(str(t["output"]).split()) for t in tests[:32]]
    return round(collections.Counter(outs).most_common(1)[0][1] / len(outs), 3)


# ---------- 0. final_v2
rows = [json.loads(l) for l in open(f"{V7}/final_v2/train.jsonl")]
refs = json.load(gzip.open(f"{V7}/final_v2/work/verifying_refs.json.gz", "rt"))
for r in rows:
    r["_origin"] = "final_v2"
S["final_v2"] = len(rows)

# ---------- 1. huge tests / slow references
R = pickle.load(open(f"{V7}/stdio/work/results.pkl", "rb"))
fast = {}
for (row, j, kind), rs in R.items():
    if rs and all(x.get("ex") or x.get("ws") or x.get("tk") for x in rs):
        mx = max(x.get("t", 0) for x in rs)
        fast[row] = min(fast.get(row, 99), mx)
keep = []
for r in rows:
    m = r["metadata"]
    cid = m["custom_id"]
    if m.get("lcb_v3_flag"):
        dropped.append((cid, m["src"], m["route"], "in_lcb_v3_window_2023-05_to_2024-07"))
        S["drop_lcb_v3_window"] += 1
        continue
    if m["route"] == "stdio":
        lab = json.loads(r["label"])
        big = [t for t in lab if len(t["input"]) + len(t["output"]) > 1_000_000]
        if big:
            lab2 = [t for t in lab if t not in big]
            if len(lab2) < 5 or const_passes(lab2):
                dropped.append((cid, m["src"], m["route"], "huge_tests_lt5_after_trim"))
                S["drop_huge"] += 1
                continue
            r["label"] = json.dumps(lab2, ensure_ascii=False)
            m["n_tests"] = len(lab2)
            m["v7_changes"] = sorted(set(m.get("v7_changes") or []) | {"v3_trimmed_huge_tests"})
            m["const_floor"] = const_floor(lab2)
            S["trim_huge_rows"] += 1
            S["trim_huge_tests"] += len(big)
        row = int(M.at[cid, "row"]) if cid in M.index else None
        if row is not None and fast.get(row, 0) > 4.0 and "openr1_statement" not in (m.get("v7_changes") or []):
            dropped.append((cid, m["src"], m["route"], f"slow_reference_{fast[row]:.1f}s"))
            S["drop_slow"] += 1
            continue
    keep.append(r)
rows = keep

# ---------- 2. restored stdio rows
C = {c["cid"]: c for c in (json.loads(l) for l in gzip.open(os.path.join(WORK, "restore_cands.jsonl.gz"), "rt"))}
V = {v["cid"]: v for v in (json.loads(l) for l in open(os.path.join(WORK, "restore_verify.jsonl")))}
TAG = {"image": "v3_restored_image_placeholder", "multi_noR1checker": "v3_restored_multi_answer_openr1_token_compare",
       "multi_nomap": "v3_restored_multi_answer_exact_compare"}
for cid, v in V.items():
    c = C[cid]
    g = c["group"]
    if g not in TAG:
        continue
    S[f"restore_cand_{g}"] += 1
    if not v.get("verified"):
        S[f"restore_unverified_{g}"] += 1
        continue
    if len(c["tests"]) < 5 or const_passes(c["tests"]):
        S[f"restore_lt5_or_const_{g}"] += 1
        continue
    if cid in decon:
        continue
    if g != "image":
        ever = M.at[cid, "v6_ever"] if cid in M.index else ""
        if not ((ever != "" and float(ever) > 0) or (v.get("other_refs_pass") or 0) >= 1):
            S[f"restore_no_exact_compare_evidence_{g}"] += 1
            continue
    d = date.get(cid)
    if d and d[0] >= "2024-08-01":
        S["restore_date"] += 1
        continue
    md = c["metadata"]
    md = md if isinstance(md, dict) else ast.literal_eval(md)
    md = dict(md)
    ref = c["refs"][v["ref"]]
    md.update(n_tests=len(c["tests"]), custom_id=cid, src=c["src"], route="stdio", verified=True,
              verifier_origins=[ref["origin"]], fixes=[], v7_changes=[TAG[g]],
              lcb_v3_flag=bool(d and "2023-05-01" <= d[0] < "2024-08-01"), const_floor=const_floor(c["tests"]),
              flags={"image_placeholder": g == "image", "multi_answer_statement": g != "image"},
              v3_restore_reason=c["cat"])
    md.update(c["opts"])
    r = {"prompt": [{"role": "user", "content": c["prompt"]}], "label": json.dumps(c["tests"], ensure_ascii=False),
         "metadata": md, "_origin": "restored"}
    refs[cid] = dict(kind="stdio", opts=c["opts"], prog=ref["code"], origin=ref["origin"])
    rows.append(r)
    S[f"restore_added_{g}"] += 1

# ---------- 3. checker rows
replaced = set()
cfiles = [p for p in a.checkers.split(",") if os.path.exists(p)]
if not cfiles:
    print("NOTE: no checker file", a.checkers)
for cf in cfiles:
    for l in open(cf):
        x = json.loads(l)
        r = x["row"]
        r["_origin"] = "checker"
        m = r["metadata"]
        if m["custom_id"] in decon:
            continue
        for rc in m.get("replaces_final_v1") or []:
            replaced.add(rc)
        refs[m["custom_id"]] = x["ref"]
        rows.append(r)
        S["checker_added"] += 1
if replaced:
    keep = []
    for r in rows:
        if r["_origin"] == "final_v2" and r["metadata"]["custom_id"] in replaced:
            dropped.append((r["metadata"]["custom_id"], r["metadata"]["src"], r["metadata"]["route"], "replaced_by_checker_row_with_more_tests"))
            S["drop_replaced_by_checker"] += 1
        else:
            keep.append(r)
    rows = keep

# ---------- 4. dedup over the union
par = list(range(len(rows)))


def f(x):
    while par[x] != x:
        par[x] = par[par[x]]
        x = par[x]
    return x


why = collections.Counter()


def u(i, j, k):
    if f(i) != f(j):
        par[f(i)] = f(j)
        why[k] += 1


byid = collections.defaultdict(list)
for i, r in enumerate(rows):
    k = cf_id(r)
    if k:
        byid[k].append(i)
for v in byid.values():
    for j in v[1:]:
        u(v[0], j, "same_cf_id")
K = [test_keys(r) for r in rows]
idx = collections.defaultdict(list)
for i, k in enumerate(K):
    for h in k:
        idx[h].append(i)
cnt = collections.Counter()
for h, v in idx.items():
    if len(v) <= 30:
        for x in range(len(v)):
            for y in range(x + 1, len(v)):
                cnt[v[x], v[y]] += 1
for (i, j), n in cnt.items():
    if rows[i]["metadata"]["route"] != rows[j]["metadata"]["route"]:
        continue
    m = min(len(K[i]), len(K[j]))
    if m < 3:
        continue
    if rows[i]["metadata"]["route"] == "stdio":
        if n >= 0.5 * m and n >= min(5, m):
            u(i, j, "test_overlap_stdio")
    elif n >= 0.8 * m and n >= 5:
        u(i, j, "test_overlap_function")
items = [("x", r["metadata"]["custom_id"], r["metadata"]["src"], r["metadata"]["route"], r["prompt"][0]["content"]) for r in rows]
for i, j, jac, same in minhash_pairs(items):
    ii = {" ".join(str(t.get("input", "")).split()) for t in json.loads(rows[i]["label"]) if isinstance(t, dict)}
    jj = {" ".join(str(t.get("input", "")).split()) for t in json.loads(rows[j]["label"]) if isinstance(t, dict)}
    share = len(ii & jj) / max(1, min(len(ii), len(jj))) if ii and jj else 0
    if same or share >= 0.5:
        u(i, j, "minhash_text")
comp = collections.defaultdict(list)
for i in range(len(rows)):
    comp[f(i)].append(i)
pri = {"final_v2": 0, "checker": 1, "restored": 2}
keep = []
for v in comp.values():
    v.sort(key=lambda i: (-len(json.loads(rows[i]["label"])), pri[rows[i]["_origin"]]))
    keep.append(v[0])
    win = rows[v[0]]["metadata"]["custom_id"]
    for i in v[1:]:
        m = rows[i]["metadata"]
        dropped.append((m["custom_id"], m["src"], m["route"], f"v3_duplicate_of:{win}"))
        S[f"dedup_drop_{rows[i]['_origin']}"] += 1
S.update({f"dedup_edges_{k}": n for k, n in why.items()})
rows = [rows[i] for i in sorted(keep)]

# ---------- 5. mix
for r in rows:
    r["metadata"]["v6_prefilter_n_correct"] = pref_of(r["metadata"]["custom_id"])


def easy_like(r):
    m = r["metadata"]
    return m["route"] == "function" and m["src"] in ("acecoder", "synthetic2") and pref_of(m["custom_id"]) is None


def stdio_band(r):
    m = r["metadata"]
    p = pref_of(m["custom_id"])
    return m["route"] == "stdio" and (p is None or p >= 1)


def mix(rows, name):
    out, copies = [], []
    E = [r for r in rows if easy_like(r)]
    if name == "A":
        keepE = set(id(r) for r in E)
    elif name == "B":
        rr = random.Random(a.seed)
        keepE = set(id(r) for r in rr.sample(E, len(E) // 3))
    else:
        keepE = set()
    for r in rows:
        if easy_like(r) and id(r) not in keepE:
            continue
        out.append(r)
    if name in ("B", "C"):
        for r in out:
            if stdio_band(r):
                copies.append(r)
    return out, copies


def stats(rows, copies):
    draws = rows + copies
    st = collections.Counter()
    for r in draws:
        m = r["metadata"]
        st[(m["src"], m["route"])] += 1
    n = len(draws)
    stdio = sum(v for (s, rt), v in st.items() if rt == "stdio")
    p0 = sum(1 for r in draws if pref_of(r["metadata"]["custom_id"]) == 0)
    e_easy = sum({"acecoder": .816, "klear": .754, "synthetic2": .82, "nemotron": .18}[r["metadata"]["src"]]
                 for r in draws if pref_of(r["metadata"]["custom_id"]) is None and r["_origin"] == "final_v2")
    return dict(unique_rows=len(rows), draws=n, stdio_share_draws=round(stdio / n, 3),
                stdio_unique=sum(1 for r in rows if r["metadata"]["route"] == "stdio"),
                pref0_share_draws=round(p0 / n, 3), est_base_easy_share_draws=round(e_easy / n, 3),
                by_src_route_draws={f"{s}|{rt}": v for (s, rt), v in sorted(st.items())})


report = {}
for name in "ABC":
    o, c = mix(rows, name)
    report[name] = stats(o, c)
print(json.dumps(dict(S), indent=1))
print(json.dumps(report, indent=1))

if a.write:
    out_rows, copies = mix(rows, a.mix)
    keepids = {r["metadata"]["custom_id"] for r in out_rows}
    for r in rows:
        if r["metadata"]["custom_id"] not in keepids:
            m = r["metadata"]
            dropped.append((m["custom_id"], m["src"], m["route"], f"mix_{a.mix}_downsampled_likely_base_easy"))
    os.makedirs(a.out, exist_ok=True)
    final = []
    for r in out_rows:
        r = {k: v for k, v in r.items() if not k.startswith("_")}
        final.append(r)
    for r in copies:
        r2 = {k: v for k, v in r.items() if not k.startswith("_")}
        r2 = json.loads(json.dumps(r2))
        m = r2["metadata"]
        m["oversample_of"] = m["custom_id"]
        m["custom_id"] = m["custom_id"] + "__os1"
        m["v7_changes"] = sorted(set(m.get("v7_changes") or []) | {"v3_oversample_copy"})
        final.append(r2)
    rng.shuffle(final)
    with open(os.path.join(a.out, "train.jsonl"), "w") as fo:
        for r in final:
            fo.write(json.dumps(r, ensure_ascii=False) + "\n")
    with open(os.path.join(a.out, "dropped_v3.csv"), "w") as fo:
        w = csv.writer(fo)
        w.writerow(["custom_id", "src", "route", "reason"])
        w.writerows(dropped)
    with gzip.open(os.path.join(a.out, "verifying_refs.json.gz"), "wt") as fo:
        json.dump({r["metadata"]["custom_id"]: refs.get(r["metadata"]["custom_id"]) for r in out_rows}, fo)
    json.dump(dict(steps=dict(S), mixes=report, chosen=a.mix, rows_written=len(final)),
              open(os.path.join(a.out, "stats.json"), "w"), indent=1)
    print("wrote", len(final), "rows to", a.out)

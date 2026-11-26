"""Assemble final_v1 from phase2.pkl: apply merges, the standard drops, MinHash dedup (with a grader-verified
test merge), the >= 5-test floor and the flags; write train.jsonl, dropped.csv, restored.csv, stats.json.

Drop order (each row is attributed to the first rule that removes it):
  P1 confirmed contamination (decontam/drop_list.txt)
  P2 non-generic paraphrase (decontam/paraphrase_list.csv, verdict paraphrase*, not *generic*)
  P4 test compares against a class the solution defines (__eq__ under the solution's control)
  P5 unverified: no reference passes every test (v7 sandbox runs, or now under the training grader)
  date: mapped problem dated on/after 2024-08-01 (decontam/date_check.csv)
  P7 MinHash 5-gram Jaccard >= 0.7 after LaTeX normalisation, with identical digit sequences
     (easy/hard variants differ in their constraints, so they stay apart); keep the copy with more tests;
     stdio losers' tests are merged into the winner when consistent and the winner's reference passes them
  P6 fewer than 5 tests in total

    python3 assemble.py stage1   -> /tmp/v7final/assemble1.pkl (+ phase-3 merge jobs, run through the grader)
    python3 assemble.py stage2   -> final_v1/{train.jsonl, dropped.csv, restored.csv, stats.json}
"""
import ast
import collections
import csv
import gzip
import hashlib
import json
import pickle
import random
import re
import sys
from concurrent.futures import ThreadPoolExecutor

import numpy as np

sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/scripts")
from analyze import body_of, norm_in  # noqa: E402
from sandbox import grader_eq  # noqa: E402
import gclient  # noqa: E402
from stdio_common import FINAL, V7, WORK, src_of  # noqa: E402

CUTOFF = "2024-08-01"
MAX_MERGED_TEST_BYTES = 1 << 20  # merged/uncapped tests above 1 MB are left out (request size, grading time)
EQ_CONFIRMED = {"rlvr_acecoder_filtered_filtered-request-165-54"}
BUILTIN = {"Fraction", "Decimal", "Counter", "OrderedDict", "defaultdict", "Point", "datetime", "date",
           "timedelta", "deque", "set", "frozenset", "namedtuple", "_v7_datetime", "_v7_timezone", "_v7_date",
           "_v7_Decimal", "_v7_Fraction", "_v7_timedelta"}


def text_of(rec):
    p = rec["prompt"]
    return p[0]["content"] if isinstance(p, list) else str(p)


def latex_norm(t):
    t = body_of(t).replace("$", " ")
    t = re.sub(r"\\(left|right|cdot|times|ldots|dots|cdots|quad|qquad|[,;!])", " ", t)
    t = re.sub(r"\\([A-Za-z]+)", r" \1 ", t)
    return re.sub(r"[{}^_\\]", "", t).lower()


def shingles(t):
    w = re.findall(r"[a-z0-9]+", t)
    return {" ".join(w[i:i + 5]) for i in range(max(1, len(w) - 4))}


def h64(s):
    return int.from_bytes(hashlib.blake2b(s.encode(), digest_size=8).digest(), "little") & 0x7FFFFFFF


def minhash_pairs(texts, thr=0.7, perms=128, bands=16):
    S = [shingles(t) for t in texts]
    rng = np.random.RandomState(0)
    P = 2 ** 31 - 1
    a = rng.randint(1, P, perms, dtype=np.int64)
    b = rng.randint(0, P, perms, dtype=np.int64)
    sig = np.stack([((np.array([h64(x) for x in s] or [0], dtype=np.int64)[:, None] * a + b) % P).min(0) for s in S])
    r = perms // bands
    cand = set()
    for bi in range(bands):
        bk = collections.defaultdict(list)
        for i, row in enumerate(sig[:, bi * r:(bi + 1) * r]):
            bk[row.tobytes()].append(i)
        for v in bk.values():
            if 1 < len(v) < 500:
                cand |= {(v[x], v[y]) for x in range(len(v)) for y in range(x + 1, len(v))}
    out = []
    for i, j in cand:
        jac = len(S[i] & S[j]) / max(1, len(S[i] | S[j]))
        if jac >= thr:
            out.append((i, j, jac))
    return out


def eq_hits(rec, tests):
    """compute.py P4 scan: a class named in a test comparison that the prompt itself defines."""
    blob = "\n".join(tests)
    cls = set(re.findall(r"==\s*([A-Z]\w*)\(", blob)) | set(re.findall(r"([A-Z]\w*)\([^()]*\)\s*==", blob))
    cls -= BUILTIN
    return sorted(c for c in cls if re.search(rf"class\s+{c}\b", text_of(rec)))


def stage1():
    P = pickle.load(open(f"{WORK}/phase2.pkl", "rb"))
    cands, res = P["cands"], P["res"]
    drop_list = {l.strip() for l in open(f"{V7}/decontam/drop_list.txt") if l.strip()}
    para = list(csv.DictReader(open(f"{V7}/decontam/paraphrase_list.csv")))
    pstrict = {r["custom_id"] for r in para if r["verdict"].startswith("paraphrase") and "generic" not in r["verdict"]}
    dates = {r["custom_id"]: r for r in csv.DictReader(open(f"{V7}/decontam/date_check.csv"))}
    gfail = json.load(open(f"{WORK}/grader_fail_class.json"))
    rows, drops, eqscan = {}, {}, []
    merged_stats = collections.Counter()
    for cid, c in cands.items():
        r = res[cid]
        rec = json.loads(json.dumps(c["rec"]))
        tests = list(c["tests"])
        changes = list(c["changes"])
        fixes = list(rec["metadata"].get("fixes", []))
        if c["kind"] == "assert" and len(set(tests)) < len(tests):
            fixes.append(f"dedupe_identical_asserts:{len(tests) - len(set(tests))}")
            tests = list(dict.fromkeys(tests))
            merged_stats["rows_identical_asserts_deduped"] += 1
        if r.get("verified"):
            n_dup = n_unc = 0
            for g, ok in zip(c["groups"], r["accepted"]):
                if ok:
                    small_t = [t for t in g["tests"] if len(str(t["input"])) + len(str(t["output"])) <= MAX_MERGED_TEST_BYTES]
                    merged_stats["merged_tests_skipped_gt1MB"] += len(g["tests"]) - len(small_t)
                    g = dict(g, tests=small_t)
                    tests += g["tests"]
                    if g["src"] == "uncapped":
                        n_unc += len(g["tests"])
                    else:
                        n_dup += len(g["tests"])
                merged_stats["groups_" + ("uncapped" if g["src"] == "uncapped" else "dup") + (
                    "_accepted" if ok else "_conflict" if g.get("conflict") else
                    "_nothing_new" if not g["tests"] else "_rejected_ref_fails")] += 1
            if n_dup:
                changes.append("merged_duplicate_tests")
                fixes.append(f"merge_duplicate_statement_tests:{n_dup}")
                merged_stats["rows_with_merged_dup_tests"] += 1
                merged_stats["merged_dup_tests"] += n_dup
            if n_unc:
                changes.append("added_uncapped_dolci_tests")
                fixes.append(f"add_uncapped_dolci_tests:{n_unc}")
                merged_stats["rows_with_uncapped_tests"] += 1
                merged_stats["uncapped_tests"] += n_unc
        rec["metadata"]["fixes"] = fixes
        rec["label"] = json.dumps(tests, ensure_ascii=False)
        rec["metadata"]["n_tests"] = len(tests)
        row = dict(cid=cid, src=src_of(cid), route=c["route"], kind=c["kind"], rec=rec, tests=tests, opts=c["opts"],
                   changes=changes, prog=(c["progs"][r["ref_index"]] if r.get("verified") else None),
                   losers=c.get("losers", []), groups=c["groups"], accepted=r.get("accepted"))
        if c["kind"] == "assert":
            hits = eq_hits(rec, tests)
            if hits:
                eqscan.append((cid, hits))
        reason = None
        if cid in drop_list:
            reason = "P1_confirmed_contamination"
        elif cid in pstrict:
            reason = "P2_nongeneric_paraphrase"
        elif cid in EQ_CONFIRMED:
            reason = "P4_solution_defined_eq"
        elif not r.get("verified"):
            why = r.get("why", "")
            if why == "no_reference_passes_under_training_grader":
                cls = gfail.get(cid, "other_route")
                reason = ("P4_strict_grader_rejects_solution_defined_objects" if cls == "strict_canon_rejects_solution_objects"
                          else "P5_unverified:training_grader_" + cls)
            elif "image_decorative_kept" in c["changes"]:
                reason = "P5_unverified:image_decorative_no_reference_passes"
            else:
                reason = "P5_unverified:" + why
        elif dates.get(cid, {}).get("date", "") >= CUTOFF:
            reason = "date_after_cutoff"
        if reason:
            drops[cid] = (row, reason)
        else:
            rows[cid] = row
    # P7 MinHash on the survivors
    ids = sorted(rows)
    norm = [latex_norm(text_of(rows[c]["rec"])) for c in ids]
    digits = ["".join(re.findall(r"\d", t)) for t in norm]
    pairs = minhash_pairs(norm)
    same = [(i, j, jac) for i, j, jac in pairs if digits[i] == digits[j]]
    par = list(range(len(ids)))

    def f(x):
        while par[x] != x:
            par[x] = par[par[x]]
            x = par[x]
        return x
    for i, j, _ in same:
        par[f(i)] = f(j)
    comp = collections.defaultdict(list)
    for i in range(len(ids)):
        comp[f(i)].append(i)
    mh_jobs, mh_losers = [], {}
    for g in comp.values():
        if len(g) < 2:
            continue
        g.sort(key=lambda k: (-len(rows[ids[k]]["tests"]), rows[ids[k]]["route"] != "stdio", ids[k]))
        win = rows[ids[g[0]]]
        groups = []
        if win["kind"] == "stdio":
            seen = {norm_in(t["input"]): t["output"] for t in win["tests"]}
            for k in g[1:]:
                lo = rows[ids[k]]
                if lo["kind"] != "stdio":
                    continue
                gt, conflict, local = [], False, {}
                for t in lo["tests"]:
                    key = norm_in(t["input"])
                    if key in seen or key in local:
                        if not grader_eq(t["output"], seen.get(key, local.get(key))):
                            conflict = True
                            break
                        continue
                    local[key] = t["output"]
                    gt.append(t)
                if conflict:
                    groups.append(dict(src=lo["cid"], tests=[], conflict=True))
                else:
                    seen.update(local)
                    groups.append(dict(src=lo["cid"], tests=gt, conflict=False))
        for k in g[1:]:
            mh_losers[ids[k]] = win["cid"]
        if any(x["tests"] for x in groups):
            mh_jobs.append((win["cid"], groups))
    pairs_info = dict(pairs=len(pairs), pairs_same_digits=len(same), pairs_digits_differ=len(pairs) - len(same),
                      examples_digits_differ=[(ids[i], ids[j], round(jac, 3)) for i, j, jac in pairs if digits[i] != digits[j]][:8],
                      examples_same=[(ids[i], ids[j], round(jac, 3)) for i, j, jac in same][:8],
                      by_src=collections.Counter(f"{rows[ids[i]]['src']}|{rows[ids[i]]['route']}~{rows[ids[j]]['src']}|{rows[ids[j]]['route']}"
                                                 for i, j, _ in same))
    pickle.dump(dict(rows=rows, drops=drops, mh_losers=mh_losers, mh_jobs=mh_jobs, pairs_info=pairs_info,
                     eqscan=eqscan, merged_stats=merged_stats), open(f"{WORK}/assemble1.pkl", "wb"))
    print("survivors", len(rows), "drops", collections.Counter(v[1].split(":")[0] for v in drops.values()))
    print("minhash", {k: v for k, v in pairs_info.items() if k != "by_src"}, dict(pairs_info["by_src"]))
    print("minhash losers", len(mh_losers), "merge jobs", len(mh_jobs))
    print("eq scan", eqscan)
    print(merged_stats)


def mh_verify(job):
    wcid, groups, row = job
    tests = row["tests"] + [t for g in groups for t in g["tests"]]
    res = gclient.run(row["kind"], row["prog"], tests, row["opts"])
    if res is None or not all(res[:len(row["tests"])]):
        return wcid, [False] * len(groups)
    acc, off = [], len(row["tests"])
    for g in groups:
        acc.append(bool(g["tests"]) and all(res[off:off + len(g["tests"])]))
        off += len(g["tests"])
    return wcid, acc


CONSTS = ["None", "zero", "empty_str", "empty_list", "false", "true", "echo"]


def front_load_discriminating_tests(rows):
    """Training grades the first CODE_MAX_TESTS=32 tests only. Where a row has more than 32 tests and one
    constant answer passes all of the first 32, move one test that constant fails to the front so the
    graded prefix still separates constants from solutions (only the order changes)."""
    audit = None
    c = collections.Counter()
    for cid, r in rows.items():
        t = r["tests"]
        if len(t) <= 32:
            continue
        c["rows_gt32"] += 1
        if r["kind"] == "stdio":
            groups = collections.defaultdict(list)
            for i, x in enumerate(t):
                groups[" ".join(str(x["output"]).split())].append(i)
            fails = {}
            for out, ii in groups.items():
                pref = [i for i in ii if i < 32]
                if len(pref) == 32:
                    fails[out] = next(i for i in range(32, len(t)) if i not in set(ii))
        else:
            if audit is None:
                audit = {a["custom_id"]: a for a in pickle.load(open("/tmp/v7fn/audit.pkl", "rb"))}
            a = audit.get(cid)
            if a is None:
                c["rows_gt32_no_stub_vectors"] += 1
                continue
            tix = {x: i for i, x in enumerate(a["tests"])}
            fails = {}
            for s in CONSTS:
                vec = [a["stubs"][s][tix[x]] == "1" if x in tix else False for x in t]
                if all(vec[:32]):
                    fails[s] = next((i for i in range(32, len(t)) if not vec[i]), None)
        move = sorted({i for i in fails.values() if i is not None})
        if move:
            r["tests"] = [t[i] for i in move] + [x for i, x in enumerate(t) if i not in set(move)]
            r["rec"]["label"] = json.dumps(r["tests"], ensure_ascii=False)
            r["rec"]["metadata"]["fixes"].append(f"front_load_discriminating_test:{len(move)}")
            r["changes"].append("reordered_tests_first32_discriminate")
            c["rows_reordered_first32"] += 1
    return {"first32_" + k: v for k, v in c.items()}


_AUDIT = None


def const_floor_of(r):
    """Best single-constant pass rate on the first 32 tests (what training grades). stdio: share of the most
    common expected output; assert: best of the 7 constant stubs from the v7 audit vectors (None when the row
    has no audit vectors, i.e. the S2 conversions, whose stubs were run in phase 1 and pass < 100%)."""
    global _AUDIT
    t = r["tests"][:32]
    if r["kind"] == "stdio":
        c = collections.Counter(" ".join(str(x["output"]).split()) for x in t)
        return round(c.most_common(1)[0][1] / len(t), 3)
    if _AUDIT is None:
        _AUDIT = {a["custom_id"]: a for a in pickle.load(open("/tmp/v7fn/audit.pkl", "rb"))}
    a = _AUDIT.get(r["cid"])
    if a is None:
        return None
    tix = {x: i for i, x in enumerate(a["tests"])}
    if not all(x in tix for x in t):
        return None
    return round(max(sum(a["stubs"][s][tix[x]] == "1" for x in t) / len(t) for s in CONSTS), 3)


def stage2():
    S = pickle.load(open(f"{WORK}/assemble1.pkl", "rb"))
    P = pickle.load(open(f"{WORK}/phase2.pkl", "rb"))
    rows, drops, mh_losers = S["rows"], S["drops"], S["mh_losers"]
    stats = collections.Counter()
    with ThreadPoolExecutor(16) as ex:
        acc = dict(ex.map(mh_verify, [(w, g, rows[w]) for w, g in S["mh_jobs"]]))
    for w, groups in S["mh_jobs"]:
        n = 0
        for g, ok in zip(groups, acc[w]):
            if ok:
                rows[w]["tests"] = rows[w]["tests"] + g["tests"]
                n += len(g["tests"])
            stats["minhash_groups_" + ("accepted" if ok else "conflict" if g["conflict"] else "rejected")] += 1
        if n:
            rows[w]["changes"].append("minhash_merged_tests")
            rows[w]["rec"]["metadata"]["fixes"].append(f"merge_minhash_duplicate_tests:{n}")
            rows[w]["rec"]["label"] = json.dumps(rows[w]["tests"], ensure_ascii=False)
            rows[w]["rec"]["metadata"]["n_tests"] = len(rows[w]["tests"])
            stats["minhash_rows_with_merged_tests"] += 1
            stats["minhash_merged_tests"] += n
    for cid, w in mh_losers.items():
        drops[cid] = (rows.pop(cid), f"P7_minhash_duplicate_of:{w}")
    for cid in list(rows):
        if len(rows[cid]["tests"]) < 5:
            drops[cid] = (rows.pop(cid), "P6_lt5_tests")
    dates = {r["custom_id"]: r for r in csv.DictReader(open(f"{V7}/decontam/date_check.csv"))}
    lcb = {c for c, r in dates.items() if str(r.get("window", "")).startswith("lcb_v3")}
    stats.update(front_load_discriminating_tests(rows))

    # ---- train.jsonl
    src_rows = [json.loads(l) for l in open(f"{V7}/input/code-train/dolci-code-train.jsonl")]
    ids = [json.loads(l) for l in open(f"{V7}/input/row_ids.jsonl")]
    v6 = {i["custom_id"]: (i, r) for i, r in zip(ids, src_rows)}
    out_ids = sorted(rows, key=lambda c: v6[c][0]["row"])
    refs_out = {}
    with open(f"{FINAL}/train.jsonl", "w") as f:
        for cid in out_ids:
            r = rows[cid]
            md = dict(r["rec"]["metadata"])
            md["dataset_source"] = v6[cid][1]["metadata"]["dataset_source"]
            md["src"] = r["src"]
            md["route"] = r["route"]
            md["v7_changes"] = sorted(set(r["changes"]))
            md["lcb_v3_flag"] = cid in lcb
            md["n_tests"] = len(r["tests"])
            md["const_floor"] = const_floor_of(r)
            f.write(json.dumps({"prompt": r["rec"]["prompt"], "label": r["rec"]["label"], "metadata": md},
                               ensure_ascii=False) + "\n")
            refs_out[cid] = dict(kind=r["kind"], opts=r["opts"], prog=r["prog"])
    with gzip.open(f"{FINAL}/work/verifying_refs.json.gz", "wt") as f:
        json.dump(refs_out, f)

    # ---- dropped.csv: every v6 row not in train.jsonl, with the first reason that removed it
    fn_drop = {r["custom_id"]: r["reason"] for r in csv.DictReader(open(f"{WORK}/fn/dropped.csv"))}
    st_drop = {r["custom_id"]: r["reason"] for r in csv.DictReader(open(f"{WORK}/stdio/dropped.csv"))}
    p1 = pickle.load(open(f"{WORK}/phase1.pkl", "rb"))
    small_ids = {r: c for r, c in ((v6[c][0]["row"], c) for c in v6)}
    conv_status = {small_ids[r]: o.get("status") for r, o in p1.items() if o["kind"] == "s2conv"}
    img_unver = {small_ids[r] for r, o in p1.items() if o["kind"] == "image" and not o["verified"]}
    loser_of = {}
    for cid, c in P["cands"].items():
        for lo in c.get("losers", []):
            loser_of[lo] = cid
    fdrops = dict(P["fdrops"])
    m0 = set(P["m0"])
    drows = []
    for cid, (i, _) in sorted(v6.items(), key=lambda kv: kv[1][0]["row"]):
        if cid in rows:
            continue
        if cid in drops:
            reason = drops[cid][1]
        elif cid in fdrops:
            reason = "F6_" + fdrops[cid]
        elif cid in loser_of:
            w = loser_of[cid]
            reason = "S7_duplicate_statement_of:" + w
            if w in drops:
                reason += f" (winner dropped: {drops[w][1].split(':')[0]})"
        elif cid in conv_status:
            reason = "S1_nonstdin_not_converted:" + str(conv_status[cid])
        elif cid in img_unver:
            reason = "S4_image_decorative_unverified"
        elif cid in fn_drop:
            reason = {"constant_passes_all": "F3_constant_passes_all"}.get(fn_drop[cid], "F_" + fn_drop[cid])
        elif cid in st_drop:
            rs = st_drop[cid]
            if rs.startswith("constant_output_floor"):
                rs = "S5_constant_output_passes_all"
            elif rs.startswith("image_placeholder"):
                rs = "S4_" + rs + "_not_decorative"
            elif rs.startswith("multi_answer"):
                rs = "S3_" + rs
            else:
                rs = "S_" + rs
            reason = rs
        elif cid in m0:
            continue
        else:
            reason = "unaccounted"
        drows.append(dict(custom_id=cid, src=i["src"], reason=reason))
    with open(f"{FINAL}/dropped.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["custom_id", "src", "reason"])
        w.writeheader()
        w.writerows(drows)

    # ---- restored.csv: rows v7 (function/stdio fixed.jsonl) dropped, and rows whose floor-pruned tests came back
    old = set()
    for fn in ("function", "stdio"):
        for l in open(f"{V7}/{fn}/fixed.jsonl"):
            old.add(json.loads(l)["metadata"]["custom_id"])
    rest = []
    for cid in out_ids:
        ch = set(rows[cid]["changes"])
        if cid not in old:
            why = ("image_decorative_verified" if "image_decorative_kept" in ch else
                   "s2_nonstdin_converted_to_function" if "s2_nonstdin_converted" in ch else
                   "binary_reward_floor_drop_restored" if "restored_binary_reward" in ch else
                   "duplicate_winner_changed")
            rest.append(dict(custom_id=cid, src=rows[cid]["src"], route=rows[cid]["route"], kind="row", reason=why))
        for t in ("restored_pruned_tests", "restored_thinned_tests"):
            if t in ch and cid in old:
                rest.append(dict(custom_id=cid, src=rows[cid]["src"], route=rows[cid]["route"], kind="tests", reason=t))
    with open(f"{FINAL}/restored.csv", "w", newline="") as f:
        w = csv.DictWriter(f, ["custom_id", "src", "route", "kind", "reason"])
        w.writeheader()
        w.writerows(rest)

    # ---- stats
    def key(src, route):
        return f"{src}|{route}"
    v6_by = collections.Counter()
    for cid, (i, r) in v6.items():
        lab = json.loads(r["label"])
        v6_by[key(i["src"], "function" if lab and isinstance(lab[0], str) else "stdio")] += 1
    fin_by = collections.Counter(key(rows[c]["src"], rows[c]["route"]) for c in rows)
    ntests = [len(rows[c]["tests"]) for c in rows]
    hist = collections.Counter(min(n, 33) for n in ntests)
    by_change = collections.Counter(ch for c in rows for ch in set(rows[c]["changes"]))
    by_change_route = collections.Counter(f"{ch}|{rows[c]['route']}" for c in rows for ch in set(rows[c]["changes"]))
    drop_by = collections.Counter(d["reason"].split(":")[0].split(" (")[0] for d in drows)
    drop_by_src = collections.Counter(f"{d['reason'].split(':')[0].split(' (')[0]}|{d['src']}" for d in drows)
    st = dict(v6_pool_by_src_route=dict(v6_by), v6_total=len(v6), final_by_src_route=dict(fin_by),
              final_total=len(rows), final_by_route=dict(collections.Counter(rows[c]["route"] for c in rows)),
              by_change=dict(by_change), by_change_route=dict(by_change_route), dropped_by_reason=dict(drop_by),
              dropped_by_reason_src=dict(drop_by_src), restored=dict(collections.Counter(f"{r['kind']}:{r['reason']}" for r in rest)),
              tests_per_row_hist={("33+" if k == 33 else str(k)): v for k, v in sorted(hist.items())},
              tests_per_row=dict(mean=float(np.mean(ntests)), median=float(np.median(ntests)), min=int(min(ntests)),
                                 max=int(max(ntests)), gt32=int(sum(n > 32 for n in ntests))),
              tests_per_row_by_route={rt: dict(mean=float(np.mean(v)), median=float(np.median(v)),
                                               p10=float(np.percentile(v, 10)), p90=float(np.percentile(v, 90)))
                                      for rt, v in ((rt, [len(rows[c]["tests"]) for c in rows if rows[c]["route"] == rt])
                                                    for rt in ("function", "stdio"))},
              lcb_v3_flagged=sum(c in lcb for c in rows), merge=dict(S["merged_stats"]), minhash=dict(
                  {k: v for k, v in S["pairs_info"].items() if k != "by_src"}, by_src=dict(S["pairs_info"]["by_src"]),
                  losers=len(mh_losers), **stats),
              phase2_notes=P["notes"], m0_klear_dups=len(P["m0"]), eq_scan=S["eqscan"],
              multi_answer_flagged_function=sum(bool(rows[c]["rec"]["metadata"].get("flags", {}).get("multi_answer_statement"))
                                                for c in rows))
    json.dump(st, open(f"{FINAL}/stats.json", "w"), indent=1, default=str)
    print(json.dumps({k: st[k] for k in ("final_total", "final_by_route", "final_by_src_route", "dropped_by_reason", "restored")}, indent=1))


if __name__ == "__main__":
    {"stage1": stage1, "stage2": stage2}[sys.argv[1]]()

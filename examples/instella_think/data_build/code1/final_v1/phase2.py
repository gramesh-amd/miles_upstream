"""Phase 2 (grader server): pick every candidate row's record, merge duplicate-statement tests, top up
rows under 5 tests from the uncapped Dolci tests, and re-verify every verified row under the training
grader's options (gclient). Unverified rows (no reference passed in the v7 sandbox runs) are not re-run.

Function route (fn_redecide.py output), per custom_id:
  * hand verdict U (underspecified, handcheck_blocking.csv) -> drop;
  * restoring the pruned tests left the row unverified (or without a recoverable interface) while the v7
    pruned version was verified -> keep the v7 version (fix `kept_v7_pruned_version`);
  * a signature line that changed only because the passing-reference set changed -> keep the v7 prompt.
Stdio route (stdio_redecide.py prededup.pkl + phase1.pkl), per duplicate cluster (stdio/work/dup_clusters.pkl):
  * winner: verified first, then most tests, then most verifying references;
  * every other copy's cleaned tests form one merge group. A group is dropped if any of its inputs already
    has a different expected output (or two of its own tests disagree); it is accepted only if the
    winner's verifying reference passes all of it;
  * winner still under 5 tests: uncapped Dolci tests (beyond the 32-test cap) are added one by one, each
    accepted only if the verifying reference passes it (the S6 rule, extended from < 3 to < 5).
S2 Python-literal rows converted in phase 1 become function-route rows unless their cluster has a kept copy.

    python3 phase2.py  ->  /tmp/v7final/phase2.pkl
"""
import collections
import csv
import json
import pickle
import re
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/scripts")
from analyze import norm_in  # noqa: E402
from sandbox import grader_eq  # noqa: E402  (pure string compare, no execution)
import gclient  # noqa: E402
from stdio_common import V7, WORK  # noqa: E402

PRI = {"v6": 0, "olmo": 1, "v6near": 2}
MAXREF = 4


def jl(path):
    return {json.loads(l)["metadata"]["custom_id"]: json.loads(l) for l in open(path)}


def function_candidates():
    old = jl(f"{V7}/function/fixed.jsonl")
    new = jl(f"{WORK}/fn/fixed.jsonl")
    hand = {r["custom_id"]: r["verdict"] for r in csv.DictReader(open(f"{V7}/function/handcheck_blocking.csv"))}
    audit = {a["custom_id"]: a for a in pickle.load(open("/tmp/v7fn/audit.pkl", "rb"))}
    fin_new = {d["custom_id"]: d for d in pickle.load(open(f"{WORK}/fn/final.pkl", "rb"))["final"]}
    fin_old = {d["custom_id"]: d for d in pickle.load(open("/tmp/v7fn/final.pkl", "rb"))["final"]}
    out, drops = {}, []
    for cid in sorted(set(old) | set(new)):
        o, n = old.get(cid), new.get(cid)
        if hand.get(cid) == "U":
            drops.append((cid, "underspecified_hand_checked"))
            continue
        changes = []
        if n is not None and (o is None or n["metadata"]["verified"] or not o["metadata"]["verified"]):
            rec, d = json.loads(json.dumps(n)), fin_new[cid]
            if o is None:
                changes.append("restored_binary_reward")
            elif o["label"] != n["label"]:
                changes.append("restored_pruned_tests")
            if o is not None and o["prompt"] != n["prompt"]:
                rec["prompt"] = o["prompt"]
                changes.append("kept_v7_prompt")
        else:
            rec, d = json.loads(json.dumps(o)), fin_old[cid]
            changes.append("kept_v7_pruned_version")
        a = audit[cid]
        tests = json.loads(rec["label"])
        tix = {t: i for i, t in enumerate(a["tests"])}
        order = []
        if d.get("verify_ref") is not None:
            order.append((d["verify_ref"], d.get("verify_suffix", "")))
        okr = [i for i, x in enumerate(a["refs"]) if x["prog"] == "ok"]
        for i in okr:
            v = a["refs"][i]["v"]
            if all(t in tix and v[tix[t]] == "1" for t in tests) and i != d.get("verify_ref"):
                order.append((i, ""))
        out[cid] = dict(route="function", rec=rec, tests=tests, kind="assert", opts={}, changes=changes,
                        verified_v7=bool(rec["metadata"]["verified"]),
                        progs=[a["ref_codes"][i] + s for i, s in order][:MAXREF], groups=[])
    return out, drops


def stdio_candidates():
    pre = pickle.load(open(f"{WORK}/stdio/prededup.pkl", "rb"))
    p1 = pickle.load(open(f"{WORK}/phase1.pkl", "rb"))
    small = pickle.load(open(f"{WORK}/stdio_rows_small.pkl", "rb"))
    A = pickle.load(open(f"{V7}/stdio/work/analysis.pkl", "rb"))
    CL = pickle.load(open(f"{V7}/stdio/work/dup_clusters.pkl", "rb"))["cluster"]
    old_kept = set(jl(f"{V7}/stdio/fixed.jsonl"))
    members = collections.defaultdict(list)
    for r, p in pre.items():
        members[p["cluster"]].append(r)
    out, notes = {}, collections.Counter()

    def status(r):
        if A[r]["image"]:
            return bool(p1.get(r, {}).get("verified")), p1.get(r, {}).get("ref")
        p = pre[r]
        return bool(p["verified"]), (p["verifiers"][0] if p["verifiers"] else None)

    for cl, rows in members.items():
        rows.sort(key=lambda r: (-status(r)[0], -pre[r]["n"], -len(pre[r]["verifiers"]), r))
        win = rows[0]
        ver, ref0 = status(win)
        rec = json.loads(json.dumps(pre[win]["rec"]))
        v = small[win]
        tests = json.loads(rec["label"])
        kind = "assert" if isinstance(tests[0], str) else "stdio"
        changes = []
        if A[win]["image"]:
            changes.append("image_decorative_kept")
        elif rec["metadata"]["custom_id"] not in old_kept:
            changes.append("restored_binary_reward")
        else:
            changes.append("restored_thinned_tests" if any(f.startswith("thin_constant_tests") for f in
                                                           _old_fixes(rec["metadata"]["custom_id"])) else None)
        changes = [c for c in changes if c]
        groups = []
        if kind == "stdio":
            seen = {}
            for t in tests:
                seen.setdefault(norm_in(t["input"]), t["output"])
            for r in rows[1:]:
                lt = json.loads(pre[r]["rec"]["label"])
                if not lt or isinstance(lt[0], str):
                    notes["merge_skipped_other_kind"] += 1
                    continue
                g, conflict, local = [], False, {}
                for t in lt:
                    k = norm_in(t["input"])
                    if k in seen or k in local:
                        if not grader_eq(t["output"], seen.get(k, local.get(k))):
                            conflict = True
                            break
                        continue
                    local[k] = t["output"]
                    g.append({"input": t["input"], "output": t["output"]})
                if conflict:
                    notes["merge_group_conflicting_output"] += 1
                    groups.append(dict(src="dup:" + pre[r]["custom_id"], tests=[], conflict=True))
                    continue
                seen.update(local)
                groups.append(dict(src="dup:" + pre[r]["custom_id"], tests=g, conflict=False))
            n_after = len(tests) + sum(len(g["tests"]) for g in groups)
            full = v.get("full_tests") or []
            if n_after < 5 and isinstance(full, list) and len(full) > len(v["label"]):
                for t in full[len(v["label"]):]:
                    k = norm_in(t["input"])
                    if k in seen:
                        continue
                    seen[k] = t["output"]
                    groups.append(dict(src="uncapped", tests=[{"input": t["input"], "output": t["output"]}], conflict=False))
                    if len(tests) + sum(len(g["tests"]) for g in groups) >= 32:
                        break
        refs = v["refs"]
        order = ([ref0] if ref0 is not None else []) + sorted(
            [j for j in range(len(refs)) if j != ref0 and j in set(pre[win]["verifiers"])],
            key=lambda j: (PRI.get(refs[j]["origin"], 3), j))
        out[rec["metadata"]["custom_id"]] = dict(
            route="stdio" if kind == "stdio" else "function", rec=rec, tests=tests, kind=kind,
            opts=gclient.stdio_opts(rec["metadata"]) if kind == "stdio" else {}, changes=changes,
            verified_v7=ver, progs=[refs[j]["code"] for j in order][:MAXREF], groups=groups, row=win,
            cluster=cl, losers=[pre[r]["custom_id"] for r in rows[1:]])
    kept_clusters = set(members)
    conv = {}
    for r, o in p1.items():
        if o["kind"] != "s2conv":
            continue
        if o.get("status") != "ok":
            notes["s2conv:" + o.get("status", "?")] += 1
            continue
        if CL.get(r) in kept_clusters:
            notes["s2conv:duplicate_of_kept_stdio_row"] += 1
            continue
        v = small[r]
        meta = dict(v["metadata"])
        sentence = f"Implement `def {o['name']}({o['args']})`."
        has_name = re.search(r"(?<![A-Za-z0-9_])" + re.escape(o["name"]) + r"(?![A-Za-z0-9_])", v["prompt"])
        prompt = v["prompt"] if has_name else v["prompt"].rstrip() + "\n\n" + sentence
        meta.update(n_tests=len(o["tests"]), custom_id=v["custom_id"], verified=True, verifier_origins=[v["refs"][o["ref"]]["origin"]],
                    fixes=["nonstdin_to_assert_with_signature"] + (["add_signature"] if not has_name else [])
                    + (["add_uncapped_dolci_tests"] if o.get("n_extra") else []))
        rec = dict(prompt=[{"role": "user", "content": prompt}], label=json.dumps(o["tests"], ensure_ascii=False), metadata=meta)
        conv[v["custom_id"]] = dict(route="function", rec=rec, tests=o["tests"], kind="assert", opts={},
                                    changes=["s2_nonstdin_converted"], verified_v7=True,
                                    progs=[v["refs"][o["ref"]]["code"]], groups=[], row=r, cluster=CL.get(r))
    return out, conv, notes


_OLDFX = None


def _old_fixes(cid):
    global _OLDFX
    if _OLDFX is None:
        _OLDFX = {k: v["metadata"].get("fixes", []) for k, v in jl(f"{V7}/stdio/fixed.jsonl").items()}
    return _OLDFX.get(cid, [])


def verify(item):
    cid, c = item
    if not c["verified_v7"]:
        return cid, dict(verified=False, why="unverified_in_v7_runs")
    gtests = [t for g in c["groups"] for t in g["tests"]]
    tried = 0
    for prog in c["progs"]:
        tried += 1
        res = gclient.run(c["kind"], prog, c["tests"] + gtests, c["opts"], stop_on_fail=not gtests)
        if res is None:
            return cid, dict(verified=False, why="grader_unreachable")
        if not all(res[:len(c["tests"])]):
            continue
        acc, off = [], len(c["tests"])
        for g in c["groups"]:
            ok = bool(g["tests"]) and all(res[off:off + len(g["tests"])])
            acc.append(ok)
            off += len(g["tests"])
        return cid, dict(verified=True, tried=tried, ref_index=tried - 1, accepted=acc)
    return cid, dict(verified=False, why="no_reference_passes_under_training_grader", tried=tried)


def main():
    fn, fdrops = function_candidates()
    st, conv, notes = stdio_candidates()
    dup = set(fn) & set(st)
    for cid in dup:
        del st[cid]
    notes["klear_assert_rows_in_both_files_dedup"] = len(dup)
    cands = {**fn, **st, **conv}
    print(len(fn), len(st), len(conv), notes, flush=True)
    res = {}
    items = sorted(cands.items(), key=lambda kv: -len(kv[1]["tests"]) - sum(len(g["tests"]) for g in kv[1]["groups"]))
    with ThreadPoolExecutor(16) as ex:
        for i, (cid, r) in enumerate(ex.map(verify, items)):
            res[cid] = r
            if (i + 1) % 1000 == 0:
                print(i + 1, flush=True)
    pickle.dump(dict(cands=cands, res=res, fdrops=fdrops, notes=dict(notes), m0=sorted(dup)), open(f"{WORK}/phase2.pkl", "wb"))
    c = collections.Counter((cands[k]["route"], r["verified"], r.get("why")) for k, r in res.items())
    for k, v in sorted(c.items(), key=str):
        print(k, v)


if __name__ == "__main__":
    main()

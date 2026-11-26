"""Phase 1 (grader server): rows that were never executed under v7.

  A. Decorative-image stdio rows that survive the normal cleaning (stdio_redecide.py): run up to 8
     references (v6 solved > OLMo > v6 near-miss) through the grader on the cleaned tests, stop at the
     first one that passes every test.
  B. S2 rows whose tests are Python-literal function calls (S1 drops "no_interface_or_format" and
     "conversion_unverified"): take the function name the references agree on (>= 2 references and >= 50%
     define it at top level), convert the label with analyze.to_asserts (the S1 converter), extend from the
     uncapped Dolci tests when < 5, derive the signature from the references (decide.signature_lines
     style), verify with the references that define that name, and run the 7 constant stubs.

    python3 phase1.py   ->  /tmp/v7final/phase1.pkl, /tmp/v7final/stdio_rows_small.pkl
"""
import ast
import collections
import pickle
import sys
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/scripts")
sys.path.insert(1, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function/scripts")
from analyze import to_asserts, norm_in  # noqa: E402
import testlib as tl  # noqa: E402
import gclient  # noqa: E402
from stdio_common import V7, WORK  # noqa: E402

SW = f"{V7}/stdio/work"
PRI = {"v6": 0, "olmo": 1, "v6near": 2}
CONVERTIBLE = {"fn_args_wrapped_out", "pylist_of_lines", "fn_args_list_repr", "fn_args_json"}
STUBS = {"None": "None", "zero": "0", "empty_str": "''", "empty_list": "[]", "false": "False", "true": "True",
         "echo": "a[0] if a else None"}


def ref_order(refs, only=None):
    idx = [j for j in range(len(refs)) if only is None or j in only]
    return sorted(idx, key=lambda j: (PRI.get(refs[j]["origin"], 3), j))


def first_pass(kind, refs, order, tests, opts, limit=8):
    tried = 0
    for j in order[:limit]:
        tried += 1
        res = gclient.run(kind, refs[j]["code"], tests, opts, stop_on_fail=True)
        if res is not None and all(res):
            return j, tried
    return None, tried


def top_defs(code):
    try:
        tree = ast.parse(code)
    except Exception:
        return {}
    return {n.name: n for n in tree.body if isinstance(n, ast.FunctionDef)}


def image_job(job):
    row, v, rec = job
    import json
    tests = json.loads(rec["label"])
    opts = gclient.stdio_opts(rec["metadata"])
    j, tried = first_pass("stdio", v["refs"], ref_order(v["refs"]), tests, opts)
    return row, dict(kind="image", verified=j is not None, ref=j, tried=tried, n=len(tests))


def conv_job(job):
    row, v, a = job
    refs = v["refs"]
    out = dict(kind="s2conv", fmt=a["fmt"])
    names = collections.Counter()
    defs = {}
    for j, r in enumerate(refs):
        for n, fd in top_defs(r["code"]).items():
            names[n] += 1
            defs.setdefault(n, []).append((j, fd))
    if not names:
        return row, dict(out, status="no_reference_defines_a_function")
    name, cnt = names.most_common(1)[0]
    if not (len(refs) >= 2 and cnt >= max(2, 0.5 * len(refs))):
        return row, dict(out, status="references_disagree_on_name", top=(name, cnt, len(refs)))
    out["name"] = name
    base = to_asserts(a["fmt"], v["label"], name)
    if not base:
        return row, dict(out, status="tests_not_convertible")
    seen = set(base)
    base = list(dict.fromkeys(base))
    extra = []
    full = v.get("full_tests") or []
    if len(base) < 5 and isinstance(full, list) and len(full) > len(v["label"]):
        ex = to_asserts(a["fmt"], full[len(v["label"]):], name) or []
        for t in ex:
            if t not in seen and len(base) + len(extra) < 32:
                seen.add(t)
                extra.append(t)
    sigs = collections.Counter(tl.render_args(fd.args) for _, fd in defs[name])
    args = sigs.most_common(1)[0][0]
    order = ref_order(refs, {j for j, _ in defs[name]})
    tests = base + extra
    verified, kept_extra, tried = None, [], 0
    for j in order[:8]:
        tried += 1
        res = gclient.run("assert", refs[j]["code"], tests, stop_on_fail=False)
        if res is None or not all(res[:len(base)]):
            continue
        verified = j
        kept_extra = [t for t, ok in zip(extra, res[len(base):]) if ok]
        break
    out.update(args=args, n_base=len(base), n_extra_cand=len(extra), tried=tried)
    if verified is None:
        return row, dict(out, status="no_reference_passes_converted_tests")
    final = base + kept_extra
    stub_full = []
    for k, ret in STUBS.items():
        prog = f"def {name}(*a, **k):\n    return {ret}\n"
        res = gclient.run("assert", prog, final, stop_on_fail=True)
        if res is not None and all(res):
            stub_full.append(k)
    return row, dict(out, status="ok" if not stub_full else "constant_passes_all", ref=verified,
                     tests=final, n_extra=len(kept_extra), stub_full=stub_full)


def main():
    vs = pickle.load(open("/tmp/v7stdio_rows.pkl", "rb"))
    A = pickle.load(open(f"{SW}/analysis.pkl", "rb"))
    pre = pickle.load(open(f"{WORK}/stdio/prededup.pkl", "rb"))
    dec = pickle.load(open(f"{WORK}/stdio/decisions.pkl", "rb"))
    need = set(pre) | {r for r, d in dec.items() if d["status"] == "dropped"}
    small = {v["row"]: {k: v[k] for k in ("row", "custom_id", "src", "kind", "label", "full_tests", "refs", "prompt",
                                          "metadata", "olmo_passrate")} for v in vs if v["row"] in need}
    pickle.dump(small, open(f"{WORK}/stdio_rows_small.pkl", "wb"))
    del vs
    img_jobs = [(image_job, (r, small[r], pre[r]["rec"])) for r in pre if A[r]["image"]]
    conv_rows = [r for r, d in dec.items() if d["status"] == "dropped" and d["reason"].startswith("nonstdin_tests")]
    conv_jobs = [(conv_job, (r, small[r], A[r])) for r in conv_rows if A[r]["fmt"] in CONVERTIBLE]
    print("image candidates", len(img_jobs), "s2 conversion candidates", len(conv_jobs), "of", len(conv_rows), flush=True)
    res = {}
    with ThreadPoolExecutor(16) as ex:
        for i, (row, o) in enumerate(ex.map(lambda fj: fj[0](fj[1]), img_jobs + conv_jobs)):
            res[row] = o
            if (i + 1) % 100 == 0:
                print(i + 1, flush=True)
    for r in conv_rows:
        if r not in res:
            res[r] = dict(kind="s2conv", fmt=A[r]["fmt"], status="format_not_convertible")
    pickle.dump(res, open(f"{WORK}/phase1.pkl", "wb"))
    c = collections.Counter((o["kind"], o.get("status", o.get("verified"))) for o in res.values())
    for k, v in sorted(c.items(), key=str):
        print(k, v)


if __name__ == "__main__":
    main()

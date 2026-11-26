"""Sanity check of final_v3_candidate through the grader: random rows (all restored/checker rows sampled first) with
their verifying reference must pass every graded test; stubs (empty program, most common output) must not.
Checker rows go through /test_program_checker; function rows through the strict assert route."""
import collections, gzip, json, random, sys
sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v1/scripts")
sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/final_v2/scripts")
import gclient as gc
import gclient2 as gc2

D = sys.argv[1]
rows = [json.loads(l) for l in open(f"{D}/train.jsonl")]
refs = json.load(gzip.open(f"{D}/verifying_refs.json.gz", "rt"))
rows = [r for r in rows if "oversample_of" not in r["metadata"]]
rng = random.Random(0)
new = [r for r in rows if any(c.startswith("v3_restored") or c == "openr1_checker" for c in r["metadata"].get("v7_changes", []))]
old = [r for r in rows if r not in new]
sample = rng.sample(new, min(30, len(new))) + rng.sample(old, 20)
out = collections.Counter()
for r in sample:
    m = r["metadata"]
    ref = refs.get(m["custom_id"])
    tests = json.loads(r["label"])[:32]
    if not ref:
        out["no_ref"] += 1
        continue
    chk = m.get("checker")
    if m["route"] == "function":
        res = gc.run("assert", ref["prog"], tests)
        stub = gc.run("assert", "", tests)
        stubs = [stub]
    else:
        opts = gc.stdio_opts(m)
        run = (lambda p: gc2.run_checker(p, tests, chk)) if chk else (lambda p: gc.run("stdio", p, tests, opts))
        res = run(ref["prog"])
        common = collections.Counter(" ".join(t["output"].split()) for t in tests).most_common(1)[0][0]
        stubs = [run(""), run("print(" + json.dumps(common) + ")")]
    kind = "new" if r in new else "old"
    out[f"{kind}_ref_pass"] += bool(res) and all(res)
    out[f"{kind}_n"] += 1
    out[f"{kind}_stub_pass"] += any(s and all(s) for s in stubs)
print(dict(out))
json.dump(dict(out), open(f"{D}/sanity.json", "w"), indent=1)

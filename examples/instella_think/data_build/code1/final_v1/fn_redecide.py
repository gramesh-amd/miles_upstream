"""Function route, binary-reward re-decide: re-runs function/scripts/decide.py + finalize.py on the
existing audit (per-reference and per-stub verdict vectors) with the partial-credit floor rules removed.

Patched relative to the v7 function pipeline:
  * decide.plan step 4: no stub-floor pruning (best stub >= 50%); "a constant passes every test" still drops;
  * decide.finish step 6: no "< 3 meaningful (non-stub-passable) tests" drop (the >= 5 total floor is applied
    later, in assemble.py);
  * nothing is executed locally: the blocking-test probes are skipped (their only use is evidence; the hand
    verdicts in handcheck_blocking.csv still apply), and the signature-stub arity check is reused from the
    v7 decide.pkl when the same sentence was checked there, else accepted only when every signature comes
    from references (the references are then re-run through the grader server in verify.py);
  * outputs go to /tmp/v7final/fn (finalize writes fixed.jsonl / dropped.csv / final.pkl there).

    python fn_redecide.py
"""
import collections
import pickle
import sys
import types

FN = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function/scripts"
OUT = "/tmp/v7final/fn"
sys.path.insert(0, FN)


def load_patched(name, patches):
    src = open(f"{FN}/{name}.py").read()
    for old, new in patches:
        assert src.count(old) == 1, (name, old)
        src = src.replace(old, new)
    mod = types.ModuleType(name)
    mod.__file__ = f"{FN}/{name}.py"
    return mod, src


decide, dsrc = load_patched("decide", [
    ("    if best >= 0.5:\n        pruned", "    if False:\n        pruned"),
    ("    if len(meaningful) < 3:\n        d[\"drop\"] = \"lt3_meaningful_tests\"",
     "    if False:\n        d[\"drop\"] = \"lt3_meaningful_tests\""),
    ("        if ex:\n            for i in pp[\"near\"][:6]:", "        if False:\n            for i in pp[\"near\"][:6]:"),
])
exec(compile(dsrc, decide.__file__, "exec"), decide.__dict__)
sys.modules["decide"] = decide


def _no_exec(*a, **k):
    return {"prog": "not_run", "res": []}


decide.sandbox = types.SimpleNamespace(run=_no_exec)
OLD = {d["row"]: d for d in pickle.load(open("/tmp/v7fn/decide.pkl", "rb"))}
_CUR = {}


def _arity(out, details):
    old = OLD.get(_CUR["row"]) or {}
    osig = old.get("sig") or {}
    if osig.get("details") == details and "arity_ok" in osig:
        return osig["arity_ok"], osig.get("arity_bad", [])
    return all(x["from_refs"] and x["kind"] in ("def", "class") for x in details), ["not_executed:from_refs"]


decide.arity_verdict = _arity


def finish(job):
    _CUR["row"] = job[0]["row"]
    d = decide.finish(job)
    old = OLD.get(d["row"])
    if old and old.get("evidence", {}).get("blocking") and d.get("evidence", {}).get("blocking"):
        d["evidence"]["blocking"] = old["evidence"]["blocking"]
    return d


if __name__ == "__main__":
    audit = pickle.load(open("/tmp/v7fn/audit.pkl", "rb"))
    rows = {r["row"]: r for r in pickle.load(open("/tmp/v7fn/rows.pkl", "rb"))}
    out = [finish((a, rows[a["row"]])) for a in audit]
    pickle.dump(out, open(f"{OUT}/decide.pkl", "wb"))
    c = collections.Counter((d["src"], "drop:" + str(d["drop"])) for d in out)
    for k, v in sorted(c.items()):
        print(k, v)
    fin, fsrc = load_patched("finalize", [
        ('OUT = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function"', f'OUT = "{OUT}"'),
        ('pickle.load(open(f"{WORK}/decide.pkl", "rb"))', f'pickle.load(open("{OUT}/decide.pkl", "rb"))'),
        ('open(f"{WORK}/final.pkl", "wb")', f'open("{OUT}/final.pkl", "wb")'),
    ])
    fin.__dict__["__name__"] = "__main__"
    exec(compile(fsrc, fin.__file__, "exec"), fin.__dict__)

"""Why do rows that the v7 sandbox verified fail under the training grader? For each such assert row, re-run
its first reference with strict canonicalisation (training), without it (module name tmp_sol), and as __main__.
Failing only under strict = the test compares against an object the solution defines (P4 class).

    python3 classify_grader_fails.py  ->  /tmp/v7final/grader_fail_class.json
"""
import json
import pickle

import gclient

P = pickle.load(open("/tmp/v7final/phase2.pkl", "rb"))
out = {}
for c, r in P["res"].items():
    if r.get("why") != "no_reference_passes_under_training_grader":
        continue
    cc = P["cands"][c]
    if cc["kind"] != "assert" or not cc["progs"]:
        out[c] = "other_route"
        continue
    prog = cc["progs"][0]
    s = gclient.run("assert", prog, cc["tests"])
    body = lambda strict, name: {"program": prog, "tests": cc["tests"], "max_execution_time": 5, "strict": strict, "name": name}
    m = gclient._post("/test_program", body(False, "tmp_sol"), len(cc["tests"]))
    if s is not None and all(s):
        out[c] = "flaky_passes_on_rerun"
    elif m is not None and all(m):
        out[c] = "strict_canon_rejects_solution_objects"
    else:
        out[c] = "fails_without_strict_too"
json.dump(out, open("/tmp/v7final/grader_fail_class.json", "w"), indent=1)
import collections
print(collections.Counter(out.values()))

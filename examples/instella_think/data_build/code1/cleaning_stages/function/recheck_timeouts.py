"""Re-runs, at low concurrency, every reference whose audit vector has a timeout on rows that no reference
fully passes, so machine load cannot turn a correct reference into a failure. Patches audit.pkl in place.

    python3 recheck_timeouts.py [workers]
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

import sandbox
from audit import code_char

WORK = "/tmp/v7fn"


def job(args):
    row, i, code, tests = args
    out = sandbox.run(code, tests, timeout=5.0)
    return row, i, out["prog"], "".join(code_char(r, out["prog"]) for r in out["res"])


if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 8
    audit = pickle.load(open(f"{WORK}/audit.pkl", "rb"))
    by = {a["row"]: a for a in audit}
    jobs = []
    for a in audit:
        oi = list(dict.fromkeys(a["orig_ix"]))
        fi = [c["ix"] for c in a["prep"]["final"]]
        okv = [x["v"] for x in a["refs"] if x["prog"] == "ok"]
        if any(all(v[j] == "1" for j in fi) for v in okv) and any(all(v[j] == "1" for j in oi) for v in okv):
            continue
        for i, x in enumerate(a["refs"]):
            if "T" in x["v"] or x["prog"] == "crash":
                jobs.append((a["row"], i, a["ref_codes"][i], a["tests"]))
    print("rechecking", len(jobs), "reference runs", flush=True)
    changed = 0
    with ProcessPoolExecutor(workers) as ex:
        for row, i, prog, v in ex.map(job, jobs, chunksize=1):
            x = by[row]["refs"][i]
            if v != x["v"] or prog != x["prog"]:
                changed += 1
                x["v_first"] = x["v"]
                x["v"], x["prog"] = v, prog
    pickle.dump(audit, open(f"{WORK}/audit.pkl", "wb"))
    print("changed", changed)

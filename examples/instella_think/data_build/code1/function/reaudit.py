"""Re-audits only the rows whose static test preparation changed (e.g. after extending testlib), and
patches audit.pkl in place.

    python3 reaudit.py [workers]
"""
import pickle
import sys
from concurrent.futures import ProcessPoolExecutor

from audit import audit_row, prep

WORK = "/tmp/v7fn"

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 16
    rows = {r["row"]: r for r in pickle.load(open(f"{WORK}/rows.pkl", "rb"))}
    audit = pickle.load(open(f"{WORK}/audit.pkl", "rb"))
    todo = []
    for i, a in enumerate(audit):
        if "error" in a:
            todo.append(i)
            continue
        new = [c["t"] for c in prep(rows[a["row"]])["final"]]
        old = [c["t"] for c in a["prep"]["final"]]
        if new != old:
            todo.append(i)
    print("re-auditing", len(todo), flush=True)
    with ProcessPoolExecutor(workers) as ex:
        for i, r in zip(todo, ex.map(audit_row, [rows[audit[i]["row"]] for i in todo])):
            audit[i] = r
    pickle.dump(audit, open(f"{WORK}/audit.pkl", "wb"))
    print("done")

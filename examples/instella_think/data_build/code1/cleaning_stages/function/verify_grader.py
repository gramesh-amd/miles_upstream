"""Final check through the training grader's own execution path (code_exec_server.Runner.assert_tests:
one fresh sandboxed `python3 -I` per test, program + "\\n\\n" + test as __main__, exit 0 = pass).

    python3 verify_grader.py [workers] [stub_sample]

For every kept verified row: the recorded passing reference must pass every final test.
On a sample of kept rows: the None/0/''/[]/False/True/echo stubs must stay under 50%, and a program that
only calls sys.exit(0) shows the main-guard hole (it passes every assert under the current grader).
Writes verify_grader.json.
"""
import json
import pickle
import random
import sys
import tempfile

sys.path.insert(0, "/tmp/wt_lux/examples/instella_think")
import code_exec_server as ces  # noqa: E402
from audit import stub_programs  # noqa: E402
import testlib as tl  # noqa: E402

WORK = "/tmp/v7fn"
OUT = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/function"

if __name__ == "__main__":
    workers = int(sys.argv[1]) if len(sys.argv) > 1 else 24
    nstub = int(sys.argv[2]) if len(sys.argv) > 2 else 600
    fin = pickle.load(open(f"{WORK}/final.pkl", "rb"))["final"]
    audit = {a["row"]: a for a in pickle.load(open(f"{WORK}/audit.pkl", "rb"))}
    runner = ces.Runner(workers, 2048, sys.executable, tempfile.mkdtemp(prefix="v7ver-"))
    from concurrent.futures import ThreadPoolExecutor

    def ref_check(d):
        if d["verify_ref"] is None:
            return d["row"], None
        code = audit[d["row"]]["ref_codes"][d["verify_ref"]] + d.get("verify_suffix", "")
        res = runner.assert_tests(code, d["tests_final"], 5.0, name="__main__")
        return d["row"], res

    ver = [d for d in fin if d["verified"]]
    out = {"ref_rows": len(ver), "ref_all_pass": 0, "ref_fail_rows": []}
    with ThreadPoolExecutor(workers) as ex:
        for row, res in ex.map(ref_check, ver):
            if res is not None and all(res):
                out["ref_all_pass"] += 1
            else:
                out["ref_fail_rows"].append(row)
    print("reference all-pass via grader route", out["ref_all_pass"], "/", out["ref_rows"], flush=True)

    rng = random.Random(0)
    sample = rng.sample(fin, min(nstub, len(fin)))

    def stub_check(d):
        fns = []
        for t in d["tests_final"]:
            tr = tl.parse(t)
            if tr is not None:
                for f in tl.called_free(tr):
                    if f not in fns and not f.startswith("_v7_"):
                        fns.append(f)
        best = 0.0
        for k, prog in stub_programs(fns).items():
            if k == "empty_program":
                continue
            res = runner.assert_tests(prog, d["tests_final"], 5.0, name="__main__")
            best = max(best, sum(res) / len(res))
        ex_res = runner.assert_tests("import sys\nif __name__ == '__main__':\n    sys.exit(0)\n", d["tests_final"],
                                     5.0, name="__main__")
        return best, sum(ex_res) / len(ex_res)

    floors, exits = [], []
    with ThreadPoolExecutor(max(1, workers // 4)) as ex:
        for best, e in ex.map(stub_check, sample):
            floors.append(best)
            exits.append(e)
    out["stub_sample"] = len(sample)
    out["stub_ge50"] = sum(f >= 0.5 for f in floors)
    out["stub_mean_floor"] = sum(floors) / len(floors)
    out["exit0_full_pass"] = sum(e >= 1.0 for e in exits)
    print(json.dumps({k: v for k, v in out.items() if k != "ref_fail_rows"}), "fails", out["ref_fail_rows"][:20])
    json.dump(out, open(f"{OUT}/verify_grader.json", "w"), indent=1)

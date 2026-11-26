"""Checks sandbox.run (fork-per-test batch harness) against the training grader's own per-test route,
code_exec_server.Runner.assert_tests (one fresh `python3 -I` per test, program + test as __main__).

    python3 fidelity.py <audit pkl> <n_pairs> [workers]
"""
import pickle
import random
import sys
import tempfile

sys.path.insert(0, "/tmp/wt_lux/examples/instella_think")
import code_exec_server as ces  # noqa: E402

if __name__ == "__main__":
    path, n = sys.argv[1], int(sys.argv[2])
    workers = int(sys.argv[3]) if len(sys.argv) > 3 else 8
    a = pickle.load(open(path, "rb"))
    rng = random.Random(1)
    pairs = []
    for r in a:
        for i, x in enumerate(r["refs"]):
            pairs.append((r, i))
    rng.shuffle(pairs)
    pairs = pairs[:n]
    runner = ces.Runner(workers, 2048, sys.executable, tempfile.mkdtemp(prefix="v7fid-"))
    agree = tot = 0
    rows_agree = 0
    bad = []
    for r, i in pairs:
        code = r["ref_codes"][i]
        got = runner.assert_tests(code, r["tests"], 5.0, name="__main__")
        mine = [1 if c == "1" else 0 for c in r["refs"][i]["v"]]
        a_ = sum(x == y for x, y in zip(got, mine))
        agree += a_
        tot += len(got)
        rows_agree += got == mine
        if got != mine:
            bad.append((r["custom_id"], i, "".join(map(str, got)), r["refs"][i]["v"]))
    print(f"test verdicts agree {agree}/{tot}; program vectors identical {rows_agree}/{len(pairs)}")
    for b in bad[:20]:
        print(b)

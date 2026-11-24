"""Tests for the deterministic parts of the v8 data build; no sandbox, no network.

    python3 examples/instella_think/data_v8/test_data_v8.py
"""
import ast
import collections
import json
import logging
import os
import pickle
import re
import subprocess
import sys
import tempfile
import unittest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import assemble  # noqa: E402
import common  # noqa: E402
import convert_eval_pkl  # noqa: E402
import stdio_verify as sv  # noqa: E402


def fake_echo_run(code, row, timeout=5.0, max_tests=common.MAX_TESTS):
    """Runs only echo programs: looks each test's input up in the program's table, exact compare."""
    table = ast.literal_eval(re.search(r"^T = (.*)$", code, re.M).group(1))
    return [int(table.get(t["input"].strip(), "") == t["output"]) for t in row["label"]], {}


class StableHash(unittest.TestCase):
    def test_value_is_pinned(self):
        # published eval hashes and dedup lists depend on these exact values
        self.assertEqual(common.h64("abc"), 0x5995D533D814BBD8)
        self.assertEqual(common.h64("two sum"), 0x566ED24C84734DC1)

    def test_independent_of_hash_seed(self):
        cmd = [sys.executable, "-c", f"import sys; sys.path.insert(0, {HERE!r}); import common; print(common.h64('x y z'))"]
        outs = {subprocess.run(cmd, env={**os.environ, "PYTHONHASHSEED": s}, capture_output=True, text=True).stdout
                for s in ("0", "1", "2")}
        self.assertEqual(len(outs), 1)
        self.assertEqual(int(outs.pop()), common.h64("x y z"))
        self.assertTrue(0 <= common.h64("x y z") < 1 << 64)


class ConvertEvalPkl(unittest.TestCase):
    def test_round_trip_and_refuses_globals(self):
        ev = [{"eval_set": "lcb", "eval_id": "1873_A", "platform": "codeforces", "title": "A", "text": "dropped",
               "io": [(common.h64("1\x002"), 13)], "calls": [("f", common.h64("[1]\x002"), 12)]}]
        with tempfile.TemporaryDirectory() as d:
            with open(f"{d}/eval.pkl", "wb") as f:
                pickle.dump(ev, f)
            self.assertEqual(convert_eval_pkl.convert(f"{d}/eval.pkl", f"{d}/out.jsonl"), 1)
            row, = common.read_jsonl(f"{d}/out.jsonl")
            self.assertNotIn("text", row)
            self.assertEqual(row["io"], [[common.h64("1\x002"), 13]])
            self.assertEqual(row["calls"], [["f", common.h64("[1]\x002"), 12]])
            with open(f"{d}/bad.pkl", "wb") as f:
                pickle.dump([{"io": collections.OrderedDict()}], f)
            with self.assertRaises(pickle.UnpicklingError):
                convert_eval_pkl.convert(f"{d}/bad.pkl", f"{d}/bad.jsonl")


class Redact(unittest.TestCase):
    def test_token_never_logged(self):
        rec = logging.LogRecord("code_reward", logging.WARNING, "", 0, "failed: %s", ("url /abc123/test_program",), None)
        common._Redact("abc123").filter(rec)
        self.assertEqual(rec.getMessage(), "failed: url /<token>/test_program")


class Dedup(unittest.TestCase):
    words = " ".join(f"w{i}" for i in range(400))

    def rows(self, carry, new):
        return ([{"cid": f"c{i}", "stmt": s} for i, s in enumerate(carry)],
                [{"cid": f"n{i}", "stmt": s} for i, s in enumerate(new)])

    def test_contained_statement_is_a_dup(self):
        carry, new = self.rows([self.words], [self.words[:1500], "something else entirely " * 40])
        dups = assemble.find_dups(carry, new, common.h64)
        self.assertEqual(set(dups), {"n0"})
        self.assertEqual(dups["n0"][0], "c0")
        self.assertGreaterEqual(dups["n0"][1], 0.5)

    def test_short_statement_never_matches(self):
        carry, new = self.rows([self.words], [" ".join(self.words.split()[:10])])
        self.assertEqual(assemble.find_dups(carry, new, common.h64), {})

    def test_boilerplate_grams_are_ignored(self):
        boiler = " ".join(f"b{i}" for i in range(200))
        carry, new = self.rows([boiler + f" own{i} " * 3 for i in range(assemble.DEDUP_BIG + 1)], [boiler])
        self.assertEqual(assemble.find_dups(carry, new, common.h64), {})


class Tests(unittest.TestCase):
    def test_constant_output_thinning(self):
        tests = [{"input": "x" * (i + 1), "output": "YES"} for i in range(18)] + \
                [{"input": f"n{i}", "output": str(i)} for i in range(6)]
        f0, _ = sv.const_floor(tests)
        self.assertEqual(f0, 0.75)
        thinned, f1 = sv.thin(tests)
        self.assertLess(f1, 0.5)
        yes = [t for t in thinned if t["output"] == "YES"]
        self.assertEqual(len(yes), 5)
        # the shortest inputs of the dominant answer go first
        self.assertEqual(sorted(len(t["input"]) for t in yes), [14, 15, 16, 17, 18])

    def test_select_tests_is_seeded(self):
        tests = [{"input": "x" * ((i * 37) % 101), "output": str(i)} for i in range(100)]
        a, b = sv.select_tests(tests), sv.select_tests(tests)
        self.assertEqual(a, b)
        self.assertEqual(len(a), sv.SELECT_K)
        big = sorted(range(100), key=lambda i: -len(tests[i]["input"]))[:8]
        self.assertEqual(a[:8], [tests[i] for i in big])
        self.assertNotEqual(a, sv.select_tests(tests, seed=1))
        self.assertEqual(sv.select_tests(tests[:20]), tests[:20])


class BugFixes(unittest.TestCase):
    def setUp(self):
        self.real_run = common.run

    def tearDown(self):
        common.run = self.real_run

    def test_echo_handles_repeated_inputs(self):
        tests = [{"input": "3\n", "output": "1 2"}, {"input": "3", "output": "2 1"}, {"input": "4\n", "output": "x"},
                 {"input": "3\n", "output": "3 0"}]
        self.assertEqual(sv.echo_batches(tests), [[0, 2], [1], [3]])
        common.run = fake_echo_run
        one_table, _ = fake_echo_run(sv.echo_program(tests), {"label": tests})
        self.assertFalse(all(one_table))
        self.assertEqual(sv.run_echo(tests, {}, sv.TRAIN_T), [1, 1, 1, 1])
        perturbed = sv.run_echo(tests, {}, sv.TRAIN_T, perturb=True)
        self.assertEqual(perturbed, [0, 0, 0, 0])

    def test_echo_sandbox_error_is_not_a_verdict(self):
        common.run = lambda *a, **k: (None, {})
        self.assertIsNone(sv.run_echo([{"input": "1", "output": "1"}], {}, sv.TRAIN_T))

    def test_ref_filter_sandbox_error_is_retryable(self):
        tests = [{"input": str(i), "output": str(i)} for i in range(10)]
        common.run = lambda *a, **k: (None, {})
        kept, why = sv.ref_filter("print(1)", [1] * 5 + [0] * 5, tests, {})
        self.assertIsNone(kept)
        self.assertEqual(why, "sandbox_error")
        self.assertTrue(why.startswith(sv.RETRYABLE))

    def test_ref_filter_rules(self):
        tests = [{"input": str(i), "output": str(i)} for i in range(10)]
        common.run = lambda code, row, timeout=5.0, **k: ([1] * len(row["label"]), {})
        kept, fixes = sv.ref_filter("p", [1] * 8 + [0] * 2, tests, {})
        self.assertEqual(kept, tests[:8])
        self.assertEqual(fixes, ["drop_tests_no_reference_finishes_5s:2"])
        common.run = lambda code, row, timeout=5.0, **k: ([0] * len(row["label"]), {})
        self.assertEqual(sv.ref_filter("p", [1] * 7 + [0] * 3, tests, {}), (None, "reference_passes_lt_80pct"))


class Assemble(unittest.TestCase):
    def write(self, path, rows):
        with open(path, "w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")

    def row(self, cid, src, stmt):
        return {"prompt": [{"role": "user", "content": stmt}], "label": "[]",
                "metadata": {"custom_id": cid, "src": src, "statement": stmt}}

    def build(self, d, carry, shards, fn):
        os.makedirs(f"{d}/build")
        self.write(f"{d}/carry.jsonl", carry)
        for name, rows in shards.items():
            self.write(f"{d}/build/verified_{name}.jsonl", rows)
        self.write(f"{d}/build/fn_rows.jsonl", fn)

    def run_assemble(self, d, *extra):
        cmd = [sys.executable, f"{HERE}/assemble.py", "--build-dir", f"{d}/build", "--carry", f"{d}/carry.jsonl",
               "--out-dir", f"{d}/out", *extra]
        return subprocess.run(cmd, capture_output=True, text=True)

    def test_sorted_order_ignores_input_order(self):
        carry = [self.row(f"acecoder_{i}", "acecoder", f"carry problem {i} " * 3) for i in range(6)] + \
                [self.row(f"klear_{i}", "klear", f"klear problem {i} " * 3) for i in range(4)]
        new = [self.row(f"v8_x_{i}", "v8_x", f"new problem {i} " * 3) for i in range(8)]
        fn = [self.row(f"v8_leetcode_{i}", "leetcode", f"fn problem {i} " * 3) for i in range(3)]
        outs = []
        for flip in (False, True):
            with tempfile.TemporaryDirectory() as d:
                c, n = (carry[::-1], new[::-1]) if flip else (carry, new)
                self.build(d, c, {"a.s0of2": n[:4], "a.s1of2": n[4:]} if not flip else {"a.s0of3": n}, fn[::-1] if flip else fn)
                p = self.run_assemble(d)
                self.assertEqual(p.returncode, 0, p.stderr)
                with open(f"{d}/out/train_pre_decontam.jsonl") as f:
                    outs.append(f.read())
        self.assertEqual(outs[0], outs[1])
        self.assertEqual(len(outs[0].splitlines()), 21)

    def test_duplicate_ids_are_explicit(self):
        with tempfile.TemporaryDirectory() as d:
            a, b = self.row("v8_x_1", "v8_x", "first copy " * 5), self.row("v8_x_1", "v8_x", "second copy " * 5)
            self.build(d, [self.row("klear_0", "klear", "carry " * 5)], {"a.s0of2": [a], "a.s1of2": [b]}, [])
            p = self.run_assemble(d)
            self.assertNotEqual(p.returncode, 0)
            self.assertIn("v8_x_1", p.stderr)
            p = self.run_assemble(d, "--duplicates", "last")
            self.assertEqual(p.returncode, 0, p.stderr)
            rows = list(common.read_jsonl(f"{d}/out/train_pre_decontam.jsonl"))
            self.assertEqual([r["metadata"]["statement"] for r in rows if r["metadata"]["custom_id"] == "v8_x_1"],
                             [b["metadata"]["statement"]])

    def test_legacy_hash_needs_a_pinned_list(self):
        with tempfile.TemporaryDirectory() as d:
            self.build(d, [self.row("klear_0", "klear", "carry " * 5)], {}, [])
            self.assertNotEqual(self.run_assemble(d, "--hash", "legacy").returncode, 0)


if __name__ == "__main__":
    unittest.main()

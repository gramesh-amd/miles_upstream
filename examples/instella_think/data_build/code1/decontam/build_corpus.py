#!/usr/bin/env python3
"""Build normalised train / eval / reference corpora for decontamination.

Outputs (work/):
  train.pkl  list of dicts: custom_id, row, src, text, tests_label, tests_full, funcs
  eval.pkl   list of dicts: eval_set, eval_id, part, text, title, date, platform, io, calls, asserts, funcs, sets
  ref_cf.pkl / ref_lc.pkl  dated reference problems (open-r1/codeforces, LeetCodeDataset) for the date check
"""
import ast, glob, gzip, hashlib, json, os, pickle, re, sys
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, "/tmp/wt_lux/examples/instella_think")
from prepare_code_data import parse_tests  # noqa: E402

OUT = "/it-share-prj2-1/data/goramesh-instella-rl/out"
V7 = f"{OUT}/v7_data"
WORK = f"{V7}/decontam/work"
EB = "/tmp/evalbuild"
EV2 = f"{OUT}/v5_assessment/evals_v2"
os.makedirs(WORK, exist_ok=True)


# ---------------------------------------------------------------- helpers
def canon(v):
    """Container-insensitive canonical repr (tuple/list unified, sets sorted)."""
    if isinstance(v, (list, tuple)):
        return "[" + ",".join(canon(x) for x in v) + "]"
    if isinstance(v, (set, frozenset)):
        return "{" + ",".join(sorted(canon(x) for x in v)) + "}"
    if isinstance(v, dict):
        return "{" + ",".join(sorted(canon(k) + ":" + canon(x) for k, x in v.items())) + "}"
    if isinstance(v, float) and v == int(v) and abs(v) < 1e15:
        return repr(int(v))
    return repr(v)


def lit(node):
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in WRAPPERS \
            and len(node.args) == 1 and not node.keywords:
        return ast.literal_eval(node.args[0])
    return ast.literal_eval(node)


WRAPPERS = {"set", "sorted", "list", "tuple", "frozenset", "str", "int", "float", "round", "abs", "len", "bool"}


def call_of(node):
    """Unwrap set(f(x)) etc. Return ast.Call of the tested function or None."""
    for _ in range(3):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in WRAPPERS \
                and len(node.args) >= 1 and isinstance(node.args[0], ast.Call):
            node = node.args[0]
        else:
            break
    if isinstance(node, ast.Call):
        f = node.func
        if isinstance(f, ast.Name):
            return node, f.id
        if isinstance(f, ast.Attribute):
            return node, f.attr
    return None, None


def call_key(callnode, expected):
    try:
        args = [lit(a) for a in callnode.args]
        kw = sorted((k.arg, canon(lit(k.value))) for k in callnode.keywords)
    except Exception:
        return None
    a = canon(args) + (repr(kw) if kw else "")
    return a, canon(expected)


def parse_assert(s):
    """'assert f(args) == exp' -> (fname, args_canon, exp_canon) or None."""
    try:
        tree = ast.parse(s.strip())
    except Exception:
        return None
    if not tree.body or not isinstance(tree.body[0], ast.Assert):
        return None
    t = tree.body[0].test
    try:
        if isinstance(t, ast.Compare) and len(t.ops) == 1:
            op, left, right = t.ops[0], t.left, t.comparators[0]
            if isinstance(op, (ast.Eq, ast.Is)):
                c, name = call_of(left)
                if c is None:
                    c, name = call_of(right)
                    left, right = right, left
                if c is not None:
                    k = call_key(c, lit(right))
                    return (name,) + k if k else None
            if isinstance(op, (ast.Lt, ast.LtE)) and isinstance(left, ast.Call) and getattr(left.func, "id", "") == "abs":
                inner = left.args[0]
                if isinstance(inner, ast.BinOp) and isinstance(inner.op, ast.Sub):
                    c, name = call_of(inner.left)
                    if c is not None:
                        k = call_key(c, lit(inner.right))
                        return (name,) + k if k else None
        if isinstance(t, ast.Call):
            fn = getattr(t.func, "attr", getattr(t.func, "id", ""))
            if fn == "isclose" and len(t.args) >= 2:
                c, name = call_of(t.args[0])
                if c is not None:
                    k = call_key(c, lit(t.args[1]))
                    return (name,) + k if k else None
            c, name = call_of(t)
            if c is not None:
                k = call_key(c, True)
                return (name,) + k if k else None
        if isinstance(t, ast.UnaryOp) and isinstance(t.op, ast.Not):
            c, name = call_of(t.operand)
            if c is not None:
                k = call_key(c, False)
                return (name,) + k if k else None
    except Exception:
        return None
    return None


def norm_assert(s):
    return re.sub(r"\s+", "", s.strip())


def norm_io(s):
    s = str(s).replace("\r", "")
    return "\n".join(l.rstrip() for l in s.strip().split("\n"))


def sh(s):
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little")


def io_key(i, o):
    """Stable hash of a normalised (stdin, stdout) pair plus the input length (for the triviality filter)."""
    return sh(i + "\x00" + o), len(i)


def call_hash(c):
    """(fname, args_canon, exp_canon) -> (fname, hash(args, exp), len(args)+len(exp))."""
    return c[0], sh(c[1] + "\x00" + c[2]), len(c[1]) + len(c[2])


SIG = re.compile(r"def\s+([A-Za-z_]\w*)\s*\(([^)]*)\)")
BT = re.compile(r"`([A-Za-z_]\w*)\s*\(([^`)]*)\)?")


def params_of(argstr):
    out = []
    for p in argstr.split(","):
        p = p.strip()
        if not p or p in ("self", "/", "*"):
            continue
        p = p.lstrip("*").split(":")[0].split("=")[0].strip()
        if re.match(r"^[A-Za-z_]\w*$", p):
            out.append(p)
    return tuple(out)


def funcs_from_text(text):
    fs = {}
    for m in SIG.finditer(text):
        fs.setdefault(m.group(1), params_of(m.group(2)))
    for m in BT.finditer(text):
        fs.setdefault(m.group(1), params_of(m.group(2) or ""))
    return fs


def doctest_calls(prompt, entry):
    """'>>> f(x)\\n y' examples from HumanEval docstrings."""
    out = []
    lines = prompt.split("\n")
    for i, l in enumerate(lines):
        l = l.strip()
        if l.startswith(">>>") and i + 1 < len(lines):
            k = parse_assert(f"assert {l[3:].strip()} == {lines[i + 1].strip()}")
            if k:
                out.append(k)
    for m in re.finditer(rf"({entry}\([^\n]*?\))\s*(?:==>|=>|➞|==|should return|returns?)\s*([^\n]+)", prompt):
        k = parse_assert(f"assert {m.group(1)} == {m.group(2).strip().rstrip('.')}")
        if k:
            out.append(k)
    return out


def literal_lists_from_test(test):
    """HE+/MBPP+ harness: inputs = [...]; results = [...]."""
    try:
        tree = ast.parse(test)
    except Exception:
        return [], []
    vals = {}
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and len(node.targets) == 1 and isinstance(node.targets[0], ast.Name) \
                and node.targets[0].id in ("inputs", "results"):
            try:
                vals[node.targets[0].id] = ast.literal_eval(node.value)
            except Exception:
                pass
    return vals.get("inputs", []), vals.get("results", [])


# ---------------------------------------------------------------- train
def build_train():
    rows = [json.loads(l) for l in open(f"{V7}/input/code-train/dolci-code-train.jsonl")]
    ids = [json.loads(l) for l in open(f"{V7}/input/row_ids.jsonl")]
    want = {i["custom_id"] for i in ids}
    full = {}
    for f in sorted(glob.glob(os.path.expanduser(
            "~/.cache/huggingface/hub/datasets--allenai--Dolci-Think-RL-7B/snapshots/*/data/*.parquet"))):
        t = pq.read_table(f, columns=["custom_id", "ground_truth"]).to_pandas()
        t = t[t.custom_id.isin(want)]
        for cid, gt in zip(t.custom_id, t.ground_truth):
            full[cid] = parse_tests(list(gt))[0]
    out = []
    for r, i in zip(rows, ids):
        text = r["prompt"][0]["content"]
        lab = json.loads(r["label"]) if isinstance(r["label"], str) else r["label"]
        tf = full.get(i["custom_id"])
        funcs = funcs_from_text(text)
        for t in (lab if isinstance(lab, list) else []):
            if isinstance(t, str):
                k = parse_assert(t)
                if k and k[0] not in funcs:
                    funcs[k[0]] = ()
        out.append(dict(custom_id=i["custom_id"], row=i["row"], src=i["src"], text=text,
                        tests_label=lab, tests_full=tf if isinstance(tf, list) else None, funcs=funcs))
    print("train rows", len(out), "with full tests", sum(o["tests_full"] is not None for o in out))
    return out


# ---------------------------------------------------------------- eval
def lcb_items():
    L = [json.loads(l) for f in sorted(glob.glob(f"{EB}/raw/lcb_test*.jsonl")) for l in open(f)]
    sets = {}
    for s in ("lcb_v3", "lcb_v5", "lcb_v6"):
        for l in open(f"{EV2}/{s}.jsonl"):
            sets.setdefault(json.loads(l)["metadata"]["task_id"], []).append(s)
    items = []
    for x in L:
        qid = x["question_id"]
        mem = sets.get(qid, [])
        tests = []
        for s in mem:
            p = f"{EB}/tests/{s}/{qid}.json"
            if os.path.exists(p):
                tests = json.load(open(p))
                break
        tests = tests + json.loads(x["public_test_cases"])
        io, calls = [], []
        meta = json.loads(x["metadata"] or "{}")
        fname = meta.get("func_name")
        for t in tests:
            if x["platform"] == "leetcode" and fname:
                try:
                    args = [json.loads(a) for a in t["input"].split("\n") if a.strip()]
                    calls.append((fname, canon(args), canon(json.loads(t["output"]))))
                except Exception:
                    pass
            else:
                io.append((norm_io(t["input"]), norm_io(t["output"])))
        funcs = {}
        if x["starter_code"]:
            funcs = {k: v for k, v in funcs_from_text(x["starter_code"]).items() if k != "__init__"}
        items.append(dict(eval_set="lcb", eval_id=qid, part="statement", text=x["question_content"],
                          title=x["question_title"], date=x["contest_date"][:10], platform=x["platform"],
                          contest_id=x["contest_id"], io=list(set(io)), calls=list(set(calls)), asserts=[],
                          funcs=funcs, sets=mem or ["lcb_other"]))
    return items


def cf_items():
    cf = pd.read_parquet(f"{EB}/raw/cf_verifiable_test.parquet")
    items = []
    for d in cf.to_dict("records"):
        pid = d["id"]
        text = "\n".join(str(d[k]) for k in ("description", "input_format", "output_format", "note") if d[k] and d[k] != "None")
        p = f"{EB}/tests/codeforces/{pid.replace('/', '_')}.json"
        tests = json.load(open(p)) if os.path.exists(p) else []
        try:
            tests += ast.literal_eval(d["examples"]) if d["examples"] and d["examples"] != "None" else []
        except Exception:
            pass
        io = list({(norm_io(t["input"]), norm_io(t["output"])) for t in tests})
        date = pd.to_datetime(int(d["contest_start"]), unit="s").strftime("%Y-%m-%d")
        items.append(dict(eval_set="codeforces", eval_id=pid, part="statement", text=text, title=d["title"],
                          date=date, platform="codeforces", contest_id=str(d["contest_id"]), io=io, calls=[],
                          asserts=[], funcs={}, sets=["codeforces"]))
    return items


def he_items():
    d = pd.read_parquet(f"{EB}/raw/humanevalplus.parquet")
    items = []
    for r in d.to_dict("records"):
        ep = r["entry_point"]
        inputs, results = literal_lists_from_test(r["test"])
        calls = []
        for a, b in zip(inputs, results):
            try:
                calls.append((ep, canon(list(a)), canon(b)))
            except Exception:
                pass
        calls += doctest_calls(r["prompt"], ep)
        f = funcs_from_text(r["prompt"])
        items.append(dict(eval_set="humanevalplus", eval_id=r["task_id"], part="statement", text=r["prompt"],
                          title=ep, date=None, platform="humaneval", contest_id=None, io=[], calls=list(set(calls)),
                          asserts=[], funcs={ep: f.get(ep, ())}, sets=["humanevalplus"]))
    return items


def mbpp_items():
    d = pd.read_parquet(f"{EB}/raw/mbppplus.parquet")
    items = []
    for r in d.to_dict("records"):
        tl = r["test_list"]
        tl = list(tl) if not isinstance(tl, str) else re.findall(r"'(assert .*?)'(?:\n|\]| )", tl)
        names = {k[0] for k in (parse_assert(a) for a in tl) if k}
        allf = funcs_from_text(r["code"])
        funcs = {k: v for k, v in allf.items() if k in names} or allf
        ep = next(iter(funcs), None)
        inputs, results = literal_lists_from_test(r["test"])
        calls = [k for k in (parse_assert(a) for a in tl) if k]
        for a, b in zip(inputs, results):
            try:
                calls.append((ep, canon(list(a)), canon(b)))
            except Exception:
                pass
        sig = f"def {ep}({', '.join(funcs.get(ep, ()))}):" if ep else ""
        items.append(dict(eval_set="mbppplus", eval_id=f"mbppplus/{r['task_id']}", part="statement",
                          text=r["prompt"] + "\n" + sig, title=ep, date=None, platform="mbpp", contest_id=None, io=[],
                          calls=list(set(calls)), asserts=[norm_assert(a) for a in tl], funcs=funcs, sets=["mbppplus"]))
    return items


def pro_items(name):
    d = pd.read_parquet(f"{EB}/raw/{name}.parquet")
    items = []
    for r in d.to_dict("records"):
        tl = [l for l in r["test_code"].split("\n") if l.strip().startswith("assert")]
        calls = [k for k in (parse_assert(a) for a in tl) if k]
        for part, txt in (("raw", r["raw_problem"]), ("new", r["new_problem"])):
            items.append(dict(eval_set=name, eval_id=f"{name}/{r['id']}", part=part, text=txt, title=None, date=None,
                              platform=name, contest_id=None, io=[], calls=list(set(calls)) if part == "new" else [],
                              asserts=[norm_assert(a) for a in tl] if part == "new" else [],
                              funcs=funcs_from_text(txt), sets=[name]))
    return items


def crux_items():
    items = []
    seen = {json.loads(l)["metadata"]["task_id"] for l in open(f"{EV2}/cruxeval_o.jsonl")}
    for l in open(f"{EB}/raw/cruxeval.jsonl"):
        d = json.loads(l)
        if d["id"] not in seen:
            continue
        calls = []
        try:
            args = ast.literal_eval("(" + d["input"] + ",)")
            calls.append(("f", canon(list(args)), canon(ast.literal_eval(d["output"]))))
        except Exception:
            pass
        items.append(dict(eval_set="cruxeval", eval_id=d["id"], part="code", text=d["code"], title=None, date=None,
                          platform="cruxeval", contest_id=None, io=[], calls=calls,
                          asserts=[norm_assert(f"assert f({d['input']}) == {d['output']}")], funcs={},
                          sets=["cruxeval_o", "cruxeval_i"]))
    return items


# ---------------------------------------------------------------- references for the date check
def ref_cf():
    fs = sorted(glob.glob("/tmp/decontam_ref/cf/data/*.parquet"))
    cols = ["id", "contest_id", "contest_name", "contest_start", "index", "title", "description",
            "input_format", "output_format", "note", "rating"]
    out = []
    for f in fs:
        t = pq.read_table(f, columns=cols).to_pandas()
        for d in t.to_dict("records"):
            text = "\n".join(str(d[k]) for k in ("description", "input_format", "output_format", "note")
                             if d[k] is not None and str(d[k]) != "None")
            date = pd.to_datetime(int(d["contest_start"]), unit="s").strftime("%Y-%m-%d") if d["contest_start"] else None
            out.append(dict(ref="cf", ref_id=d["id"], title=d["title"], date=date, text=text,
                            contest=d["contest_name"], rating=d["rating"]))
    dedup = {o["ref_id"]: o for o in out}
    return list(dedup.values())


def ref_lc():
    out = []
    for f in ("LeetCodeDataset-train.jsonl", "LeetCodeDataset-test.jsonl"):
        for l in open(f"/tmp/decontam_ref/lc/{f}"):
            d = json.loads(l)
            out.append(dict(ref="leetcode", ref_id=d["task_id"], title=d["task_id"], date=d["estimated_date"],
                            text=d["problem_description"], contest=None, rating=d["difficulty"]))
    return out


if __name__ == "__main__":
    only_eval = sys.argv[1:] == ["eval"]
    if not only_eval:
        tr = build_train()
        pickle.dump(tr, open(f"{WORK}/train.pkl", "wb"))
    ev = lcb_items() + cf_items() + he_items() + mbpp_items() + pro_items("humanevalpro") + pro_items("mbpppro") + crux_items()
    import collections
    c = collections.Counter()
    for e in ev:
        for s in e["sets"]:
            c[(s, e["part"])] += 1
    print("eval items", len(ev), sorted(c.items()))
    print("eval with io", sum(bool(e["io"]) for e in ev), "with calls", sum(bool(e["calls"]) for e in ev))
    for e in ev:
        e["io"] = list({io_key(i, o) for i, o in e["io"]})
        e["calls"] = list({call_hash(c) for c in e["calls"]})
    pickle.dump(ev, open(f"{WORK}/eval.pkl", "wb"))
    if not only_eval and os.path.exists("/tmp/decontam_ref/cf/data"):
        rc = ref_cf()
        print("ref cf", len(rc))
        pickle.dump(rc, open(f"{WORK}/ref_cf.pkl", "wb"))
    if not only_eval:
        rl = ref_lc()
        print("ref lc", len(rl))
        pickle.dump(rl, open(f"{WORK}/ref_lc.pkl", "wb"))

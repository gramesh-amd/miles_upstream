"""Shared helpers for the final_v2 build (Open-R1 Codeforces join on top of final_v1)."""
import hashlib
import json
import os
import re
import sys

V7 = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data"
FINAL1 = f"{V7}/final_v1"
FINAL2 = f"{V7}/final_v2"
WORK = "/tmp/v7v2"
OPENR1 = "/tmp/datacand/raw/open-r1__codeforces"
CUTOFF = "2024-08-01"
MAX_TESTS_TOTAL = 32          # recovered rows: own tests first, then duplicate-copy and Open-R1 official tests, up to 32
MAX_TEST_BYTES = 1 << 20      # final_v1 rule: no test over 1 MB

sys.path.insert(0, f"{V7}/stdio/scripts")
sys.path.insert(0, f"{FINAL1}/scripts")
os.environ.setdefault("V7_GRADER_URL", "http://127.0.0.1:18988/testtoken")


def latex_dollars(t):
    """Open-R1 statements mark math with $$$...$$$ (Codeforces) or $$...$$ (formula images converted to LaTeX).
    Our rows use single $...$ (the TACO/APPS style of the Klear rows)."""
    if not t:
        return ""
    t = t.replace("$$$", "$").replace("$$", "$")
    return t.replace("\r\n", "\n").strip()


def sha(s):
    return hashlib.sha1(s.encode("utf-8", "replace")).hexdigest()


def jdump(o, path):
    with open(path, "w") as f:
        json.dump(o, f, indent=1, default=str)


# Known Dolci prompt wrappers (analyze.KLEAR_HEAD / TAILS)
PREFIXES = ["Solve the following coding problem using the programming language python:\n\n"]
SUFFIXES = [
    r"\n*The input will be given via stdin and the output should be printed to stdout by your code\.\n\nNow solve the problem by providing the code\.\s*$",
    r"\n*The input will be stdin and you should print your solution to stdout\n\n\nNow solve the problem and return the code\.\s*$",
    r"\n*Write Python code to solve the problem\. Present the code in \n```python\nYour code\n```\nat the end\.\s*$",
]


def split_wrapper(p):
    pre = next((x for x in PREFIXES if p.startswith(x)), "")
    core = p[len(pre):]
    suf = ""
    for s in SUFFIXES:
        m = re.search(s, core)
        if m:
            suf = core[m.start():]
            core = core[:m.start()]
            break
    return pre, core, suf


def prompt_format(core):
    if "-----Input-----" in core:
        return "dash"
    if re.search(r"(^|\n)Problem Description:\n", core) and "Input Specification:" in core:
        return "s2cf"
    if re.search(r"\n\nInput\n\n", core):
        return "cc"
    return "other"


def load_server_module():
    """The current code_exec_server.py (sandbox.py imports an older copy under the same name)."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("ces_v7wt", "/tmp/wt_v7/examples/instella_think/code_exec_server.py")
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m

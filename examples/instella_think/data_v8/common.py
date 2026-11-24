"""Shared helpers for the v8 code RL data build (see the README section "Code RL data (v8 mix)").

Sandbox calls go through code_reward.py's own request path (split_checker, train_options, _post), so
a row is verified exactly as training will grade it. code_reward reads CODE_EXEC_API_URL at import;
it is imported on the first run() call, so the offline steps (assemble, decontam, finalize) never
need it or a sandbox.
"""
import base64
import gzip
import hashlib
import io
import json
import logging
import os
import pickle
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

HERE = os.path.dirname(os.path.abspath(__file__))
RECIPE = os.path.dirname(HERE)

# Must match the training launch: rows never carry more tests than the reward grades.
MAX_TESTS = 128

# The stdio wrapper of the carry-over Klear rows, reused for every new stdio row.
KLEAR_HEAD = "Solve the following coding problem using the programming language python:\n\n"
KLEAR_TAIL = ("\n\nThe input will be given via stdin and the output should be printed to stdout by your code."
              "\n\nNow solve the problem by providing the code.")


def stdio_prompt(statement: str) -> str:
    return KLEAR_HEAD + statement.strip() + KLEAR_TAIL


def starter_prompt(statement: str, starter: str, instruction: str) -> str:
    return f"{statement.strip()}\n\n{instruction}\n```python\n{starter.rstrip()}\n```"


def statement_of(row: dict) -> str:
    m = row["metadata"]
    if m.get("statement"):
        return m["statement"]
    c = row["prompt"][-1]["content"]
    head = KLEAR_HEAD.rstrip("\n")
    return c[len(head):] if c.startswith(head) else c


# ---------------------------------------------------------------- hashing and tokens
def h64(s: str) -> int:
    """Unsigned 64-bit blake2b; the eval corpus's hidden-test hashes use the same function."""
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "replace"), digest_size=8).digest(), "little")


_LATEX_CMD = re.compile(r"\\(?:texttt|mathrm|text|textbf|mathbf|operatorname)\{([^}]*)\}")
_LATEX = re.compile(r"\\[A-Za-z]+")
_WORD = re.compile(r"[a-z0-9]+")


def tok(s: str) -> list:
    """Lowercase word tokens with LaTeX commands stripped (the decontamination tokenizer)."""
    s = _LATEX_CMD.sub(r" \1 ", s)
    s = _LATEX.sub(" ", s)
    return _WORD.findall(s.lower())


# ---------------------------------------------------------------- tests and code
def clean_tests(tests):
    """Dedupe on (stripped input, normalised output); keep only str/str pairs."""
    seen, out = set(), []
    for t in tests or []:
        if not isinstance(t, dict):
            continue
        i, o = t.get("input"), t.get("output")
        if isinstance(i, list):
            i = "\n".join(map(str, i))
        if isinstance(o, list):
            o = "\n".join(map(str, o))
        if not isinstance(i, str) or not isinstance(o, str):
            continue
        k = (i.strip(), "\n".join(x.strip() for x in o.strip().split("\n")))
        if k in seen:
            continue
        seen.add(k)
        out.append({"input": i, "output": o})
    return out


def is_python(code) -> bool:
    if not code:
        return False
    try:
        compile(code, "<s>", "exec")
        return True
    except Exception:
        return False


class DataOnlyUnpickler(pickle.Unpickler):
    """Plain containers and scalars only: refusing every global means nothing in the payload can be
    instantiated or called."""

    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"refusing global {module}.{name}")


def decode_hardtests(s: str):
    """HardTests test_cases: base64(zlib(pickle(JSON string)))."""
    obj = DataOnlyUnpickler(io.BytesIO(zlib.decompress(base64.b64decode(s.encode())))).load()
    return json.loads(obj) if isinstance(obj, (str, bytes)) else obj


# ---------------------------------------------------------------- statement flags
_RE_INTERACTIVE = re.compile(r"(?i)\binteractive problem\b|\bthis is an interactive\b|\bflush the output\b|fflush\(stdout\)|\bsys\.stdout\.flush\(\)|\binteractor\b")
_RE_MULTI = re.compile(r"(?i)if there are (several|multiple|many) (possible |valid |correct |optimal )?(answers|solutions)|print any (of them|one of them|valid|such)|output any (of them|one of them|valid|such)|you (may|can) (print|output) any|any of (them|the answers) (is|will be) (accepted|correct)|any (valid|correct) answer|any of the (valid|possible|correct) answers")
_RE_FLOAT_TOL = re.compile(r"(?i)(absolute|relative) (or (absolute|relative) )?error|10\^?\{?-\d+\}?|1e-\d|\bprecision\b.*\b(error|exceed)")
_RE_IMAGE = re.compile(r"(?i)<image>|!\[[^\]]*\]\(|\.png\b|\.jpg\b|\.gif\b|see the (picture|image|figure)|shown in the (picture|image|figure)|<img")
_RE_FN = re.compile(r"(?i)you don't need to read input|do not need to read input|complete the function|your task is to complete|don't need to (read|print)")
_RE_FILEIO = re.compile(r"(?i)\binput\.txt\b|\boutput\.txt\b|freopen")


def text_flags(stmt: str) -> dict:
    s = stmt or ""
    return {
        "interactive": bool(_RE_INTERACTIVE.search(s)),
        "multi_answer_text": bool(_RE_MULTI.search(s)),
        "float_tol_text": bool(_RE_FLOAT_TOL.search(s)),
        "image": bool(_RE_IMAGE.search(s)),
        "fn_statement": bool(_RE_FN.search(s)),
        "file_io": bool(_RE_FILEIO.search(s)),
    }


# ---------------------------------------------------------------- files
def read_jsonl(path):
    """Rows of a .jsonl or .jsonl.gz file."""
    with (gzip.open(path, "rt") if path.endswith(".gz") else open(path)) as f:
        for line in f:
            yield json.loads(line)


def load_union(union_path: str, clusters_path: str):
    """The problem union (one representative pid per cluster) and every member pid of its clusters,
    by cluster id."""
    import pandas as pd

    u = pd.read_csv(union_path)
    f = (pd.read_parquet(clusters_path, columns=["pid", "source", "cluster"]) if clusters_path.endswith(".parquet")
         else pd.read_csv(clusters_path, usecols=["pid", "source", "cluster"]))
    f = f[f.cluster.isin(set(u.cluster))]
    members = f.groupby("cluster").pid.apply(list).to_dict()
    return u, f, members


# ---------------------------------------------------------------- sandbox
_cr = None


def _reward():
    global _cr
    if _cr is None:
        if not os.environ.get("CODE_EXEC_API_URL"):
            raise SystemExit("set CODE_EXEC_API_URL to a code_exec_server.py URL (http://host:port/<token>)")
        os.environ.setdefault("CODE_MAX_TESTS", str(MAX_TESTS))
        os.environ.setdefault("CODE_EXEC_TIMEOUT", "5")
        os.environ.setdefault("CODE_STRICT_GRADING", "1")
        sys.path.insert(0, RECIPE)
        import code_reward
        code_reward.logger.addFilter(_Redact(code_reward._API_URL.rsplit("/", 1)[-1]))
        _cr = code_reward
    return _cr


class _Redact(logging.Filter):
    """Request errors quote the URL, whose last path segment is the sandbox token."""

    def __init__(self, secret):
        super().__init__()
        self.secret = secret

    def filter(self, record):
        if self.secret:
            record.msg, record.args = record.getMessage().replace(self.secret, "<token>"), ()
        return True


def run(code: str, row: dict, timeout: float = 5.0, max_tests: int = MAX_TESTS):
    """Per-test 0/1 list exactly as training grades `code` on `row`, with a per-test time limit of
    `timeout` (None if the sandbox failed), plus the server's outcome flags."""
    cr = _reward()
    label = row["label"]
    tests = json.loads(label) if isinstance(label, str) else label
    meta = row.get("metadata") or {}
    tests, checker = cr.split_checker(tests, meta)
    if tests is None:
        return None, {}
    flags = {}
    if checker:
        tests = tests[:max_tests]
        body = {"program": code, "tests": tests, "max_execution_time": timeout, "checker": checker}
        res = cr._post(cr._API_URL, "/test_program_checker", body, len(tests),
                       min(600.0, (timeout + cr._CHECKER_TIMEOUT) * len(tests) + 30.0), flags)
        return res, flags
    opts = cr.train_options(meta)
    if isinstance(tests[0], dict):
        tests = [{"input": t.get("input", ""), "output": t.get("output", "")} for t in tests][:max_tests]
        body = {"program": code, "tests": tests, "max_execution_time": timeout}
        body.update({k: v for k, v in opts.items() if k in cr._STDIO_OPTS})
        path = "/test_program_stdio"
    else:
        tests = [str(t) for t in tests][:max_tests]
        body = {"program": code, "tests": tests, "max_execution_time": timeout}
        body.update({k: v for k, v in opts.items() if k in cr._ASSERT_OPTS})
        path = "/test_program"
    res = cr._post(cr._API_URL, path, body, len(tests), min(600.0, timeout * len(tests) + 30.0), flags)
    return res, flags


def pmap(fn, items, workers=16):
    with ThreadPoolExecutor(workers) as ex:
        return list(ex.map(fn, items))

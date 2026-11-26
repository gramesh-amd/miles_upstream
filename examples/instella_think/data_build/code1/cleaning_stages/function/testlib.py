"""Static analysis and rewriting of assert-style tests (no execution here)."""
import ast
import builtins
import importlib
import re

BUILTINS = set(dir(builtins)) | {"__name__", "__file__"}

# Free names in tests that are standard-library modules or symbols. A test that uses one of them
# without importing it only passes when the candidate program happens to import it under that name.
MODULES = {"math", "re", "json", "string", "sys", "collections", "itertools", "functools", "random", "os",
           "time", "datetime", "copy", "heapq", "bisect", "operator", "decimal", "fractions", "statistics",
           "typing", "enum", "dataclasses", "uuid", "hashlib", "base64", "struct", "io", "textwrap",
           "unicodedata", "cmath", "numbers", "types", "abc", "calendar", "array", "logging", "inspect",
           "zlib", "binascii", "secrets", "contextlib", "weakref", "pathlib", "urllib", "html", "csv",
           "ipaddress", "socket", "threading", "queue", "pprint", "traceback", "warnings", "locale",
           "asyncio", "errno", "sqlite3", "shutil", "tempfile", "glob", "fnmatch", "platform", "getpass",
           "signal", "pickle", "gzip", "bz2", "lzma", "zipfile", "tarfile", "configparser", "argparse",
           "contextvars", "concurrent", "multiprocessing", "xml", "email", "mimetypes", "http", "difflib",
           "codecs", "keyword", "tokenize", "ast", "gc", "atexit", "colorsys", "graphlib", "hmac", "shlex",
           "sched", "selectors", "ssl", "subprocess", "unittest", "doctest", "timeit", "cProfile", "pdb"}
SYMBOLS = {"datetime": "datetime", "date": "datetime", "timedelta": "datetime", "timezone": "datetime",
           "Decimal": "decimal", "Fraction": "fractions", "OrderedDict": "collections", "Counter": "collections",
           "defaultdict": "collections", "deque": "collections", "namedtuple": "collections",
           "List": "typing", "Dict": "typing", "Tuple": "typing", "Optional": "typing", "Any": "typing",
           "Union": "typing", "Set": "typing", "Callable": "typing", "reduce": "functools",
           "partial": "functools", "Enum": "enum", "IntEnum": "enum", "Path": "pathlib", "sqrt": "math",
           "pi": "math", "inf": "math", "isclose": "math", "gcd": "math", "ceil": "math", "floor": "math",
           "log": "math", "permutations": "itertools", "combinations": "itertools", "product": "itertools",
           "chain": "itertools", "accumulate": "itertools", "heappush": "heapq", "heappop": "heapq",
           "dataclass": "dataclasses", "UUID": "uuid", "ascii_letters": "string", "ascii_lowercase": "string",
           "ascii_uppercase": "string", "digits": "string", "punctuation": "string"}


def parse(t):
    try:
        return ast.parse(t)
    except (SyntaxError, ValueError, RecursionError, MemoryError):
        return None


def defined_names(tree):
    d = set()
    for n in ast.walk(tree):
        if isinstance(n, (ast.Import, ast.ImportFrom)):
            for a in n.names:
                d.add((a.asname or a.name).split(".")[0])
        elif isinstance(n, ast.Name) and isinstance(n.ctx, (ast.Store, ast.Del)):
            d.add(n.id)
        elif isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            d.add(n.name)
        elif isinstance(n, ast.arg):
            d.add(n.arg)
        elif isinstance(n, ast.ExceptHandler) and n.name:
            d.add(n.name)
    return d


def top_assigned(tree):
    d = set()
    for st in tree.body:
        for n in ast.walk(st):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Store):
                d.add(n.id)
        if isinstance(st, (ast.FunctionDef, ast.ClassDef)):
            d.add(st.name)
        if isinstance(st, (ast.Import, ast.ImportFrom)):
            for a in st.names:
                d.add((a.asname or a.name).split(".")[0])
    return d


def free_names(tree):
    loads = [n.id for n in ast.walk(tree) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)]
    d = defined_names(tree)
    out = []
    for x in loads:
        if x not in d and x not in BUILTINS and x not in out:
            out.append(x)
    return out


def called_free(tree):
    free = set(free_names(tree))
    out = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id in free and n.func.id not in out:
            out.append(n.func.id)
    return out


def has_assert(tree):
    return any(isinstance(n, ast.Assert) for n in ast.walk(tree)) or any(
        isinstance(n, ast.Raise) for n in ast.walk(tree))


def norm_key(tree):
    try:
        return ast.dump(tree, annotate_fields=False, include_attributes=False)
    except RecursionError:
        return None


# ---- stdlib aliasing --------------------------------------------------------------
def _attrs_used(tree, name):
    out = set()
    for n in ast.walk(tree):
        if isinstance(n, ast.Attribute) and isinstance(n.value, ast.Name) and n.value.id == name:
            out.add(n.attr)
    return out


def _called(tree, name):
    return any(isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == name for n in ast.walk(tree))


def import_plan(tree, name):
    """Returns ('import', module) or ('from', module) for a stdlib free name, or None."""
    attrs = _attrs_used(tree, name)
    called = _called(tree, name)
    if name in MODULES and not called:
        try:
            m = importlib.import_module(name)
        except Exception:
            m = None
        if m is not None and all(hasattr(m, a) for a in attrs):
            return ("import", name)
    if name in SYMBOLS:
        try:
            obj = getattr(importlib.import_module(SYMBOLS[name]), name)
        except Exception:
            return None
        if all(hasattr(obj, a) for a in attrs):
            return ("from", SYMBOLS[name])
    return None


class _Rename(ast.NodeTransformer):
    def __init__(self, m):
        self.m = m

    def visit_Name(self, node):
        if node.id in self.m:
            return ast.copy_location(ast.Name(id=self.m[node.id], ctx=node.ctx), node)
        return node


def alias_stdlib(t, names):
    """Rewrites free stdlib names to private aliases imported at the top of the test, so the test no
    longer depends on how (or whether) the program imported them, and cannot rebind program globals."""
    tree = parse(t)
    plans = {}
    for n in names:
        p = import_plan(tree, n)
        if p:
            plans[n] = p
    if not plans:
        return t, []
    m = {n: f"_v7_{n}" for n in plans}
    new = _Rename(m).visit(tree)
    ast.fix_missing_locations(new)
    lines = []
    for n, (kind, mod) in plans.items():
        lines.append(f"import {mod} as _v7_{n}" if kind == "import" else f"from {mod} import {n} as _v7_{n}")
    return "\n".join(lines) + "\n" + ast.unparse(new), sorted(plans)


# ---- float / order-insensitive rewrites ----------------------------------------------
CLOSE_HELPER = '''def _v7_close(a, b):
    import math as _m
    if isinstance(b, float) or (isinstance(a, float) and isinstance(b, int) and not isinstance(b, bool)):
        return isinstance(a, (int, float)) and _m.isclose(a, b, rel_tol=1e-9, abs_tol=1e-9)
    if isinstance(b, (list, tuple)):
        return type(a) is type(b) and len(a) == len(b) and all(_v7_close(x, y) for x, y in zip(a, b))
    if isinstance(b, dict):
        return isinstance(a, dict) and a.keys() == b.keys() and all(_v7_close(a[k], b[k]) for k in b)
    return a == b
'''


def _lit(node):
    try:
        return True, ast.literal_eval(node)
    except Exception:
        return False, None


def _has_float(v, depth=0):
    if depth > 6:
        return False
    if isinstance(v, float):
        return v == v and v not in (float("inf"), float("-inf")) and not v.is_integer()
    if isinstance(v, (list, tuple, set, frozenset)):
        return any(_has_float(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return any(_has_float(x, depth + 1) for x in v.values())
    return False


def _has_set(v, depth=0):
    if isinstance(v, (set, frozenset)):
        return True
    if depth > 6:
        return False
    if isinstance(v, (list, tuple)):
        return any(_has_set(x, depth + 1) for x in v)
    if isinstance(v, dict):
        return any(_has_set(x, depth + 1) for x in v.values())
    return False


def eq_parts(tree):
    """For a single `assert A == B` test (optionally with a message): (A, B, literal_side) or None."""
    if tree is None or len(tree.body) != 1 or not isinstance(tree.body[0], ast.Assert):
        return None
    t = tree.body[0].test
    if not (isinstance(t, ast.Compare) and len(t.ops) == 1 and isinstance(t.ops[0], ast.Eq)):
        return None
    a, b = t.left, t.comparators[0]
    okb, vb = _lit(b)
    oka, va = _lit(a)
    if okb and not oka:
        return a, b, vb
    if oka and not okb:
        return b, a, va
    return None


def float_rewrite(t):
    tree = parse(t)
    p = eq_parts(tree)
    if not p:
        return None
    expr, lit, v = p
    if not _has_float(v) or _has_set(v):
        return None
    return CLOSE_HELPER + f"assert _v7_close({ast.unparse(expr)}, {ast.unparse(lit)})"


def order_rewrite(t):
    tree = parse(t)
    p = eq_parts(tree)
    if not p:
        return None
    expr, lit, v = p
    if not isinstance(v, list) or len(v) < 2:
        return None
    return f"assert sorted({ast.unparse(expr)}, key=repr) == sorted({ast.unparse(lit)}, key=repr)"


ANY_ORDER_RE = re.compile(r"(in any order|any order|order (of [\w\s]+ )?(does not|doesn't|do not|don't) matter|"
                          r"regardless of (their |the )?order|no particular order|order is not important|"
                          r"order is irrelevant|can be returned in any|in arbitrary order|order is arbitrary)", re.I)
# The narrower test used for rewriting: the any-order phrase has to be about the returned/output list.
OUTPUT_ORDER_RE = re.compile(
    r"(return(ed|s)?\b[^.]{0,80}?\b(in any order|in arbitrary order)|"
    r"(output|result|returned|answer)[^.]{0,100}?(in any order|(does not|doesn't|do not|don't) matter|"
    r"can be (in )?any order|arbitrary order|is not important)|"
    r"order of (the )?(output|result|returned|combinations|pairs|words|names|elements|characters|items|urls|"
    r"emojis|dependencies|integers|tickers|states|unique|missing|returned)[^.]{0,60}?(does not|doesn't) matter)", re.I)
_IN_EACH_RE = re.compile(r"order of [^.]{0,40}\b(in each|in the tuples?|within)\b", re.I)


def output_order_free(prompt):
    m = OUTPUT_ORDER_RE.search(prompt)
    return bool(m) and not _IN_EACH_RE.search(m.group(0))


MULTI_ANS_RE = re.compile(r"(any (one )?(of them|valid|correct|such|possible) (answer|solution|one|output|sequence|"
                          r"arrangement|permutation|order|path|string|array|list|pair|configuration|assignment)|"
                          r"if there are (multiple|several|many) (possible |valid |correct )?(answers|solutions|ways|"
                          r"outputs|options)|(multiple|several) (valid|correct|possible) (answers|solutions|outputs)|"
                          r"return any|output any|print any|any of them)", re.I)


# ---- prompt examples reused as tests ----------------------------------------------------
_WS = re.compile(r"\s+")


def _norm(s):
    return _WS.sub("", s).replace('"', "'")


def example_leak(t, prompt_norm, fns):
    """True when a test's call arguments and expected value appear together in the prompt."""
    tree = parse(t)
    p = eq_parts(tree)
    if not p:
        return False
    expr, lit, _ = p
    if not (isinstance(expr, ast.Call) and isinstance(expr.func, ast.Name) and expr.func.id in fns):
        return False
    args = ",".join(ast.unparse(a) for a in expr.args)
    a = _norm(args)
    e = _norm(ast.unparse(lit))
    if len(a) < 3:
        return False
    i = prompt_norm.find(a)
    while i >= 0:
        win = prompt_norm[i + len(a): i + len(a) + 400]
        if e in win or (e in ("True", "False") and e.lower() in win.lower()):
            return True
        i = prompt_norm.find(a, i + 1)
    return False


# ---- signatures ---------------------------------------------------------------------------
def call_shapes(trees, fn):
    pos, kws = set(), set()
    star = False
    for tree in trees:
        for n in ast.walk(tree):
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name) and n.func.id == fn:
                if any(isinstance(a, ast.Starred) for a in n.args) or any(k.arg is None for k in n.keywords):
                    star = True
                pos.add(len([a for a in n.args if not isinstance(a, ast.Starred)]))
                kws |= {k.arg for k in n.keywords if k.arg}
    return pos, kws, star


def ref_def(code, fn):
    tree = parse(code)
    if tree is None:
        return None
    for st in tree.body:
        if isinstance(st, (ast.FunctionDef, ast.AsyncFunctionDef)) and st.name == fn:
            return st
    return None


def ref_class(code, cls):
    tree = parse(code)
    if tree is None:
        return None
    for st in tree.body:
        if isinstance(st, ast.ClassDef) and st.name == cls:
            return st
    return None


def render_args(a, drop_self=False):
    """Parameter list without annotations; long defaults become `...`."""
    parts = []
    posonly = list(a.posonlyargs)
    args = posonly + list(a.args)
    if drop_self and args:
        args = args[1:]
        if posonly:
            posonly = posonly[1:]
    nd = len(a.defaults)
    allpos = posonly + list(a.args) if not drop_self else args
    defaults = [None] * (len(allpos) - nd) + list(a.defaults)
    defaults = defaults[-len(args):] if args else []
    for i, x in enumerate(args):
        d = defaults[i] if i < len(defaults) else None
        if d is None:
            parts.append(x.arg)
        else:
            s = ast.unparse(d)
            parts.append(f"{x.arg}={s}" if len(s) <= 24 else f"{x.arg}=...")
        if posonly and i == len(posonly) - 1:
            parts.append("/")
    if a.vararg:
        parts.append("*" + a.vararg.arg)
    elif a.kwonlyargs:
        parts.append("*")
    for x, d in zip(a.kwonlyargs, a.kw_defaults):
        if d is None:
            parts.append(x.arg)
        else:
            s = ast.unparse(d)
            parts.append(f"{x.arg}={s}" if len(s) <= 24 else f"{x.arg}=...")
    if a.kwarg:
        parts.append("**" + a.kwarg.arg)
    return ", ".join(parts)


def arity_ok(a, pos, kws, drop_self=False):
    args = list(a.posonlyargs) + list(a.args)
    if drop_self:
        args = args[1:]
    names = [x.arg for x in args] + [x.arg for x in a.kwonlyargs]
    nreq = len(args) - len(a.defaults)
    for p in pos:
        if p > len(args) and not a.vararg:
            return False
    if pos and min(pos) + len(kws & {x.arg for x in args[min(pos):]}) < nreq and not kws:
        return False
    if kws - set(names) and not a.kwarg:
        return False
    return True

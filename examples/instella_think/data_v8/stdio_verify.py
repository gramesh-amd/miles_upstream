#!/usr/bin/env python3
"""Stage 2 of the stdio build: decide, verify on the sandbox and emit training rows for stage-1 candidates.

    CODE_EXEC_API_URL=http://host:port/<token> python3 stdio_verify.py WORK/build/cands_openr1.jsonl.gz \
        --out-dir WORK/build [--workers 32] [--shard k/n]

Per candidate:
  1. statement drops: short / non-English / interactive / image (kept only if decorative and verified) /
     another language requested / file I/O / function-style statement / non-stdio input mode / < 3 tests
  2. tests: select_tests (the 8 largest inputs + seeded random, at most 32)
  3. grader options: float_tol from the statement's tolerance when outputs contain decimals (1e-6 when the
     statement names none), case_insensitive from the statement; CodeContests+ float / yes-no checker classes
  4. a statement allowing several answers needs a checker, else drop
  5. checker (open-r1 generated_checker as is; HardTests output_judging_function through HT_ADAPTER):
     echoing the label must pass (A1, else drop). Flags only: R1 empty output passes a test with a non-empty
     label, R2 the most common label passes > 10% of the others, R3 a perturbed label passes >= 50%
  6. reference: up to MAX_REF_TRIES references at TRAIN_T; if none passes every test, the best one re-runs its
     failures at SLOW_T. Tests it only fails by time are dropped; if it still fails some, the problem is kept
     with only the passed tests when >= 80% pass (and >= 3 remain), else dropped. Without any reference,
     open-r1 / HardTests / CodeContests+ rows are kept with verified=False and their upstream validation
     (UPSTREAM_VALIDATION); other sources drop
  7. constant-output floor: if one answer covers > FLOOR_KEEP of the tests, thin that group (shortest inputs
     first) to < 50% keeping >= 8 tests, else drop
  8. probes: no constant program may pass every test; a wrong-answer submission that does gets up to 2
     unselected tests it fails and the reference passes, else the row is flagged; the reference is re-checked
     on the final tests. A probe the sandbox fails to run counts as not passing, so the row is kept without it
Writes OUT/verified_<name>[.s<k>of<n>].jsonl and OUT/dropped_<name>[...].csv. Resumable by pid: a re-run
skips pids already kept or dropped, except drops caused by the sandbox (RETRYABLE), which are retried; the
drop file is append-only, so the last line for a pid is its decision. A cut-off last line (a killed run) is
ignored when scanning and removed from this shard's own files before appending. --check verifies nothing
and exits 1 if some pid is still dropped only for a retryable reason.
"""
import argparse
import collections
import csv
import glob
import gzip
import io
import json
import os
import random
import re
import sys
import threading
import time
import zlib
from concurrent.futures import ThreadPoolExecutor

import common
from common import stdio_prompt, text_flags

sys.set_int_max_str_digits(0)
# Training's per-test limit, and the margin a reference gets before a test counts as too slow for it.
TRAIN_T, SLOW_T = 5.0, 20.0
MAX_REF_TRIES = 6
SELECT_K = 32
# A 0.5 floor would drop every two-answer (YES/NO) problem. The reward is all-tests binary, so a constant
# answer only scores by passing every test, which the constant-program probes rule out.
FLOOR_KEEP = 0.75
MIN_THINNED = 8
REF_KEEP_FRAC = 0.8
WA_CHUNK = 64
RETRYABLE = ("sandbox_error", "exception")
# How each upstream validated its labels; rows without a passing Python reference are kept on this basis.
UPSTREAM_VALIDATION = {
    "openr1": "open-r1/codeforces: generated-test outputs kept only where multiple correct human solutions agree; "
              "checkers kept once they accepted the real correct solutions (dataset card)",
    "hardtests": "HardTests: outputs from up to 8 human oracle programs, kept where two agree on >90% of inputs "
                 "(arXiv 2505.24098 sec. 3.4)",
    "ccplus": "CodeContests+: tests and checkers selected by TPR/TNR on human correct/incorrect submissions",
}
FLOATY = re.compile(r"^-?\d+\.\d+$")

HT_ADAPTER = '''
import sys
{judge}

def _main():
    inp = open(sys.argv[1]).read()
    ref = open(sys.argv[2]).read()
    cand = open(sys.argv[3]).read()
    try:
        ok = bool(output_judging_function(inp, cand, ref))
    except Exception:
        ok = False
    print(1 if ok else 0)

_main()
'''

# ---------------------------------------------------------------- statement analysis
RX = dict(
    interactive=re.compile(r"\binteractive (problem|task)\b|this is an interactive|\binteraction\b.{0,80}\bflush|\bflush\(\)|fflush\(stdout\)|System\.out\.flush", re.I | re.S),
    image=re.compile(r"<image>|!\[[^\]]*\]\(|\[image\]|<img\b|\bimage(s)? (below|above)\b|\bpicture (below|above)\b|\bshown (in the|on the) (picture|image|figure)\b|\bsee (the )?(picture|figure|image)\b", re.I),
    multi=re.compile(
        r"\b(print|output|return|write)\s+(out\s+)?any\b"
        r"|\bany\s+(one\s+)?of\s+them\s+(will\s+be|is|are|would\s+be)\s+(accepted|correct|considered|fine|ok)"
        r"|\bany\s+(valid|correct|possible|suitable|optimal|acceptable)\s+(answer|solution|option|output|one|sequence|assignment|arrangement|permutation|configuration|order|string|pair|set|way|tree|graph|path|coloring|colouring|partition|split)"
        r"|\b(multiple|several|more\s+than\s+one)\s+(possible\s+|valid\s+|correct\s+|optimal\s+|suitable\s+)?(answers|solutions)\b"
        r"|\bany\s+(answer|solution)\s+(is|will\s+be)\s+(accepted|correct)|\bspecial\s+judge\b|\bchecker\b"
        r"|driver\s+code\s+will\s+print\s+1\s+if\s+your\s+answer\s+is\s+correct"
        r"|\b(print|output)\w*\b[^.\n]{0,60}\bin\s+any\s+order|\bin\s+any\s+order\s+(is|are|will\s+be)\s+(accepted|allowed|fine)"
        r"|order\s+(of\s+\w+\s+)?(does\s+not|doesn'?t)\s+matter"
        r"|\b(accept|accepts|satisfied\s+with|fine\s+with|happy\s+with)\s+any\s+(of\s+(them|those|these)|answer|solution)", re.I),
    multi_unique=re.compile(r"(lexicographically|smallest|largest|minimum|maximum|minimal|maximal|first|earliest|least|greatest)", re.I),
    float_stmt=re.compile(
        r"(absolute|relative)\s+(or\s+(absolute|relative)\s+)?error"
        r"|error\s+(of\s+)?(at\s+most|less\s+than|not\s+(exceed|more|greater)|doesn'?t\s+exceed|does\s+not\s+exceed|should\s+not|must\s+not|up\s+to|within)"
        r"|within\s+(an?\s+)?(error|tolerance|precision)"
        r"|\b(precision|accuracy)\s+of\s+(at\s+least\s+)?(10|1e)"
        r"|differ(s|ence)?\s+(from\s+the\s+(correct|jury'?s?)\s+(answer|value)\s+)?(by\s+)?(at\s+most|no\s+more|less\s+than|not\s+more)"
        r"|\bwill\s+be\s+(accepted|considered\s+correct)\s+if"
        r"|error\s+(of\s+)?[\d.eE^\-−{}\s]+\s+or\s+less|error\s+greater\s+than|error\s+(is\s+)?(allowed|acceptable|permitted|tolerated)"
        r"|(absolute|relative)\s+tolerance", re.I),
    case_stmt=re.compile(
        r"in\s+any\s+(letter\s+)?(case|register)"
        r"|case[- ]insensitive|case\s+does\s+not\s+matter|case\s+doesn'?t\s+matter"
        r"|(upper|lower)\s*(case)?\s+or\s+(upper|lower)\s*case"
        r"|(each|every)\s+letter\s+(can|may)\s+be\s+(printed|output)"
        r"|will\s+(all\s+)?be\s+(recognized|recognised|considered|treated)\s+as\s+(a\s+)?(positive|negative|correct|valid|identical)", re.I),
    other_lang=re.compile(
        r"\b(write|implement|solve|code)\b[^.\n]{0,40}\bin\s+(C\+\+|C#|Java(?!Script)|JavaScript|Haskell|Ruby|Rust|Go(lang)?|Kotlin|Scala|PHP|Perl|Swift|Clojure|Elixir|Fortran|COBOL|Brainf\*?ck|assembly|SQL|Bash|shell)\b"
        r"|\bthis kata (is|must be solved) in\b|\btranslat(e|ion) (this|the) kata\b|\bonly (C\+\+|Java|C#|JavaScript) (is|are) (allowed|accepted)\b", re.I),
)
TOL_RX = re.compile(
    r"10\s*\^\s*\{?\s*[-−–]\s*(\d+)\s*\}?"          # 10^{-6}, 10^-6
    r"|10\s*\*\*\s*\(?\s*[-−–]\s*(\d+)"                # 10**-6
    r"|\b1\s*[eE]\s*[-−–]\s*(\d+)"                   # 1e-6
    r"|\b10\s*[-−–]\s*(\d+)\b"                        # 10-6 (superscript lost)
    r"|\b(0\.0*1)\b")                                 # 0.000001
NONENG = re.compile(r"[\u0400-\u04ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]")
IMG_TAG = re.compile(r"<image>|\[Image\]|!\[[^\]]*\]\([^)]*\)", re.I)
IMG_FIG = re.compile(r"\b(figure|picture|shown below|image below|illustrat|see the image|pictured|the drawing)\b", re.I)


def tolerance(body):
    """The statement's float tolerance: a value, "unspecified" (tolerance wording without a number) or None."""
    m = RX["float_stmt"].search(body)
    if not m:
        return None
    win = body[max(0, m.start() - 250): m.end() + 250]
    best = None
    for t in TOL_RX.finditer(win):
        if t.group(5):
            v = float(t.group(5))
        else:
            k = int(next(g for g in t.groups()[:4] if g))
            if not (1 <= k <= 12):
                continue
            v = 10.0 ** -k
        if v < 1:
            best = v if best is None else max(best, v)
    return best if best is not None else "unspecified"


def has_multi(body):
    """Sentences that grant answer freedom, minus "print any character" and "print the smallest of them"."""
    hits = []
    for m in RX["multi"].finditer(body):
        s = body[max(0, body.rfind(".", 0, m.start()) + 1): body.find(".", m.end()) if body.find(".", m.end()) > 0 else len(body)]
        if re.search(r"(print|output|return|display|write)\s+(out\s+)?any\s+(of\s+the\s+)?(integer|number|element|character|letter|key|other)\b", s, re.I) and not re.search(r"answer|solution", s, re.I):
            continue
        if re.search(r"print\s+all\s+of\s+them|print\s+(them\s+)?all\b", s, re.I) and not re.search(r"any\s+order|print\s+any", s, re.I):
            continue
        if RX["multi_unique"].search(s) and not re.search(r"(print|output|return)\s+any", s, re.I):
            continue
        hits.append(s.strip()[:200])
    return hits


def non_english(body):
    letters = re.findall(r"[^\W\d_]", body)
    ne = len(NONENG.findall(body))
    eng_words = len(re.findall(r"\b(the|of|and|to|is|in|that|each|given|number|print|output|input)\b", body, re.I))
    return ne / max(1, len(letters)) > 0.3 and eng_words < 15


def decorative_image(t):
    """Every image tag sits on its own line or in the Note / Explanation section, and the statement never
    points at a figure for required information."""
    if IMG_FIG.search(t):
        return False
    note = re.search(r"\n\s*(Note|Explanation)\b", t)
    for m in IMG_TAG.finditer(t):
        if note and m.start() > note.start():
            continue
        ls = t.rfind("\n", 0, m.start()) + 1
        le = t.find("\n", m.end())
        le = len(t) if le < 0 else le
        if t[ls:le].strip() != m.group(0).strip():
            return False
    return bool(IMG_TAG.search(t))


# ---------------------------------------------------------------- tests and probes
def select_tests(tests, k=SELECT_K, n_big=8, seed=0):
    """The n_big largest-input tests, then seeded-random others, capped at k."""
    if len(tests) <= k:
        return list(tests)
    idx = sorted(range(len(tests)), key=lambda i: -len(tests[i]["input"]))
    big, rest = idx[:n_big], idx[n_big:]
    random.Random(seed).shuffle(rest)
    return [tests[i] for i in big + rest[:k - n_big]]


def constant_programs(tests):
    """Trivial programs: empty, common constants, the first test's output and the most common output."""
    progs = {"empty": "pass\n"}
    for c in ["0", "1", "-1", "YES", "NO", "Yes", "No"]:
        progs[f"const_{c}"] = f"print({c!r})\n"
    if tests:
        progs["first_output"] = f"import sys\nsys.stdout.write({tests[0]['output']!r})\n"
        mc = collections.Counter(t["output"].strip() for t in tests).most_common(1)[0][0]
        progs["mode_output"] = f"import sys\nsys.stdout.write({mc!r})\n"
    return progs


def mutate(o):
    toks = o.split()
    if not toks:
        return "0"
    last = toks[-1]
    toks[-1] = str(int(last) + 1) if re.fullmatch(r"-?\d+", last) else last + "x"
    return " ".join(toks)


def echo_program(tests, perturb=False):
    table = {t["input"].strip(): mutate(t["output"]) if perturb else t["output"] for t in tests}
    return "import sys\nT = " + repr(table) + "\nsys.stdout.write(T.get(sys.stdin.read().strip(), ''))\n"


def echo_batches(tests):
    """Test indices split so that no batch repeats a stripped input: an echo program maps input -> output,
    and a checker row may hold the same input with two different valid outputs."""
    batches, seen = [], collections.Counter()
    for i, t in enumerate(tests):
        k = seen[t["input"].strip()]
        seen[t["input"].strip()] += 1
        if k == len(batches):
            batches.append([])
        batches[k].append(i)
    return batches


def run_echo(tests, md, timeout, perturb=False):
    """Per-test result of a program that prints each test's label (perturbed), or None on a sandbox error."""
    res = [0] * len(tests)
    for b in echo_batches(tests):
        sub = [tests[i] for i in b]
        r, _ = common.run(echo_program(sub, perturb), {"label": sub, "metadata": md}, timeout)
        if r is None:
            return None
        for i, x in zip(b, r, strict=True):
            res[i] = x
    return res


def norm_out(o):
    return "\n".join(ln.rstrip() for ln in str(o).strip().split("\n"))


def const_floor(tests):
    g = collections.defaultdict(list)
    for i, t in enumerate(tests):
        g[norm_out(t["output"])].append(i)
    groups = sorted(g.values(), key=len, reverse=True)
    return len(groups[0]) / len(tests), groups


def thin(tests):
    idx = list(range(len(tests)))
    while True:
        f, gs = const_floor([tests[i] for i in idx])
        if f < 0.5 or len(idx) <= 3:
            return [tests[i] for i in idx], f
        dom = [idx[k] for k in gs[0]]
        idx.remove(min(dom, key=lambda i: len(tests[i]["input"])))


def passes_all(res):
    return res is not None and len(res) > 0 and all(res)


def ref_filter(ref, res, tests, md):
    """Re-run a reference's failing tests at SLOW_T. Returns (kept tests, fixes) or (None, drop reason)."""
    fails = [i for i, x in enumerate(res) if not x]
    slow, _ = common.run(ref, {"label": [tests[i] for i in fails], "metadata": md}, SLOW_T)
    if slow is None:
        return None, "sandbox_error"
    still = [i for i, x in zip(fails, slow, strict=True) if not x]
    if still and len(tests) - len(still) < REF_KEEP_FRAC * len(tests):
        return None, "reference_passes_lt_80pct"
    out = []
    if len(fails) > len(still):
        out.append(f"drop_tests_no_reference_finishes_5s:{len(fails) - len(still)}")
    if still:
        out.append(f"drop_tests_reference_fails_80pct_rule:{len(still)}")
    kept = [t for i, t in enumerate(tests) if i not in set(fails)]
    if len(kept) < 3:
        return None, "lt3_tests_after_reference_check"
    return kept, out


def decide(c):
    st = c["statement"]
    tf = text_flags(st)
    if len(st) < 80:
        return None, "empty_statement"
    if non_english(st):
        return None, "non_english"
    if RX["interactive"].search(st) or tf["interactive"]:
        return None, "interactive"
    decorative, fixes = False, list(c.get("fixes") or [])
    if RX["image"].search(st) or tf["image"]:
        if not decorative_image(st):
            return None, "image"
        decorative = True
        fixes.append("image_decorative_kept")
    if RX["other_lang"].search(st):
        return None, "other_language_request"
    if tf["file_io"]:
        return None, "file_io"
    if tf["fn_statement"]:
        return None, "function_style_statement"
    if c["meta"].get("input_mode") not in (None, "stdio"):
        return None, "input_mode_" + str(c["meta"].get("input_mode"))

    tests = [t for t in c["tests"] if isinstance(t.get("input"), str) and isinstance(t.get("output"), str)]
    if len(tests) < 3:
        return None, "lt3_tests"
    all_tests = tests
    tests = select_tests(tests)
    picked = {(t["input"], t["output"]) for t in tests}

    md = {}
    tol = tolerance(st)
    floaty = any(FLOATY.match(tok) for t in tests for tok in t["output"].split())
    if tol is not None and floaty:
        md["float_tol"] = tol if isinstance(tol, float) else 1e-6
        if tol == "unspecified":
            md["float_tol_unspecified_value"] = True
    if RX["case_stmt"].search(st):
        md["case_insensitive"] = True
    cc = (c["meta"].get("union") or {}).get("checker") if c["source"] == "ccplus" else None
    if cc == "float" and "float_tol" not in md and floaty:
        md["float_tol"] = tol if isinstance(tol, float) else 1e-6
        md["float_tol_from_ccplus_checker"] = True
    if cc == "yesno_ci":
        md["case_insensitive"] = True

    checker = None
    if c.get("checker"):
        checker = c["checker"] if c["checker_kind"] == "openr1_generated" else HT_ADAPTER.format(judge=c["checker"])
    multi = has_multi(st)
    if multi and not checker:
        return None, "multi_answer_no_checker"
    if checker:
        md["checker"] = checker
        md.pop("float_tol", None)
        md.pop("case_insensitive", None)

    def row_of(ts):
        return {"label": ts, "metadata": md}

    flags = []
    if checker:
        # the echo program embeds every input, so a slow run gets SLOW_T before it counts as a rejection
        a1 = run_echo(tests, md, TRAIN_T)
        if not passes_all(a1):
            a1 = run_echo(tests, md, SLOW_T)
        if a1 is None:
            return None, "sandbox_error"
        if not all(a1):
            return None, "checker_rejects_label"
        nonempty = [t for t in tests if t["output"].strip()]
        r1, _ = common.run("pass\n", row_of(nonempty), TRAIN_T) if nonempty else ([0], {})
        if r1 is None or any(r1):
            flags.append("checker_accepts_empty_on_some_tests")
        _, gs = const_floor(tests)
        mode = tests[gs[0][0]]["output"]
        others = [t for t in tests if norm_out(t["output"]) != norm_out(mode)]
        if others:
            r2, _ = common.run("import sys\nsys.stdout.write(" + repr(mode) + ")\n", row_of(others), TRAIN_T)
            if r2 is None or sum(r2) > 0.1 * len(others):
                flags.append("checker_accepts_common_output")
        r3 = run_echo(tests, md, TRAIN_T, perturb=True)
        if r3 is None or sum(r3) >= 0.5 * len(tests):
            flags.append("checker_accepts_perturbed_label")

    verifier = None
    if c["refs"]:
        best = None
        for k, ref in enumerate(c["refs"][:MAX_REF_TRIES]):
            res, _ = common.run(ref, row_of(tests), TRAIN_T)
            if res is None:
                continue
            if all(res):
                verifier = k
                break
            if best is None or sum(res) > sum(best[1]):
                best = (k, res)
        if verifier is None:
            if best is None:
                return None, "sandbox_error"
            kept, info = ref_filter(c["refs"][best[0]], best[1], tests, md)
            if kept is None:
                return None, info
            tests, verifier = kept, best[0]
            fixes += info
    elif decorative:
        return None, "image_decorative_no_reference"
    elif c["source"] in UPSTREAM_VALIDATION:
        flags.append("label_validated_upstream_only")
    else:
        return None, "no_reference"

    f0, _ = const_floor(tests)
    if f0 > FLOOR_KEEP:
        thinned, f1 = thin(tests)
        if f1 >= 0.5 or len(thinned) < MIN_THINNED:
            return None, f"constant_output_floor_{f0:.2f}"
        tests = thinned
        fixes.append("thin_constant_tests")
    floor, _ = const_floor(tests)

    def const_passes(ts):
        for name, prog in constant_programs(ts).items():
            res, _ = common.run(prog, row_of(ts), TRAIN_T)
            if passes_all(res):
                return name
        return None

    hit = const_passes(tests)
    if hit:
        return None, f"constant_program_passes:{hit}"
    r = row_of(tests)
    pool = [t for t in all_tests if (t["input"], t["output"]) not in picked]
    added, wa_pass = 0, 0
    for w in c["wa"]:
        res, _ = common.run(w, r, TRAIN_T)
        if not passes_all(res):
            continue
        killers = []
        if verifier is not None:
            for i in range(0, len(pool), WA_CHUNK):
                chunk = pool[i:i + WA_CHUNK]
                wr, _ = common.run(w, row_of(chunk), TRAIN_T)
                if wr is None:
                    continue
                fails = [t for t, x in zip(chunk, wr, strict=True) if not x]
                if fails:
                    vr, _ = common.run(c["refs"][verifier], row_of(fails), TRAIN_T)
                    killers += [t for t, x in zip(fails, vr or [], strict=False) if x]
                if len(killers) >= 2:
                    break
        if not killers or len(tests) + 2 > common.MAX_TESTS:
            wa_pass += 1
            continue
        tests = tests + killers[:2]
        pool = [t for t in pool if t not in killers[:2]]
        added += len(killers[:2])
        r = row_of(tests)
    if added:
        fixes.append(f"add_tests_failing_wrong_answer_submission:{added}")
        hit = const_passes(tests)
        if hit:
            return None, f"constant_program_passes:{hit}"
        floor, _ = const_floor(tests)
    if wa_pass:
        flags.append(f"wrong_answer_submission_passes:{wa_pass}/{len(c['wa'])}")
    if verifier is not None:
        res, _ = common.run(c["refs"][verifier], r, TRAIN_T)
        if res is None:
            return None, "sandbox_error"
        if not all(res):
            kept, info = ref_filter(c["refs"][verifier], res, tests, md)
            if kept is None:
                return None, info
            tests = kept
            fixes += [x + "_final" for x in info]
            hit = const_passes(tests)
            if hit:
                return None, f"constant_program_passes:{hit}"
            floor, _ = const_floor(tests)
            if floor > FLOOR_KEEP:
                return None, f"constant_output_floor_{floor:.2f}"

    cid = "v8_" + re.sub(r"[^A-Za-z0-9]+", "_", c["pid"]).strip("_")
    meta = {"rm_type": "code", "split": "train", "custom_id": cid, "src": "v8_" + c["source"], "route": "stdio",
            "n_tests": len(tests), "verified": verifier is not None, "v8_flags": flags, "fixes": fixes,
            "const_floor": round(floor, 3), "upstream_pid": c["pid"], "cluster": c["cluster"], "statement": st,
            "verifier_ref": verifier, "n_wa_probes": len(c["wa"]),
            **({} if verifier is not None else {"label_validation": UPSTREAM_VALIDATION[c["source"]]}),
            **md,
            **{k: c["meta"].get(k) for k in ("date", "rating", "platform", "url", "time_limit", "difficulty")
               if c["meta"].get(k) is not None}}
    if checker:
        meta["checker_source"] = ("open-r1/codeforces generated_checker" if c["checker_kind"] == "openr1_generated"
                                  else "HardTests output_judging_function (adapter)")
    u = c["meta"].get("union") or {}
    meta["lcb_v3_flag"] = bool(u.get("lcb_v3_window"))
    meta["date_eff"] = u.get("date_eff")
    rec = {"prompt": [{"role": "user", "content": stdio_prompt(st)}],
           "label": json.dumps(tests, ensure_ascii=False), "metadata": meta}
    return rec, None


def shard_files(out_dir, kind, name, ext):
    """<kind>_<name>.<ext> and its shards <kind>_<name>.s<k>of<n>.<ext>, nothing else."""
    return (glob.glob(os.path.join(out_dir, f"{kind}_{name}.{ext}"))
            + glob.glob(os.path.join(out_dir, f"{kind}_{name}.s*of*.{ext}")))


def complete_lines(path):
    """The file's newline-terminated lines. A last line without one is a row whose write was cut off
    (a killed run, or a sibling shard appending right now), so it is left out."""
    with open(path, "rb") as f:
        for ln in f:
            if ln.endswith(b"\n"):
                yield ln


def drop_partial_tail(path):
    """Cuts a cut-off last line before this run appends to the file, so the next row starts on its own line."""
    if not os.path.exists(path):
        return
    with open(path, "rb+") as f:
        end = f.seek(0, os.SEEK_END)
        pos = end
        while pos > 0:
            step = min(pos, 1 << 20)
            f.seek(pos - step)
            i = f.read(step).rfind(b"\n")
            if i >= 0:
                pos = pos - step + i + 1
                break
            pos -= step
        if pos < end:
            f.truncate(pos)


def drops(path):
    """(pid, reason) of a dropped_*.csv; a cut-off last row (no reason) is left out."""
    text = b"".join(complete_lines(path)).decode()
    for r in csv.DictReader(io.StringIO(text, newline="")):
        if r.get("pid") and r.get("reason") is not None:
            yield r["pid"], r["reason"]


def unresolved(out_dir, name):
    """{pid: last reason} of pids neither verified nor dropped for a non-retryable reason."""
    kept = set()
    for p in shard_files(out_dir, "verified", name, "jsonl"):
        kept |= {json.loads(ln)["metadata"]["upstream_pid"] for ln in complete_lines(p)}
    final, retry = set(), {}
    for p in shard_files(out_dir, "dropped", name, "csv"):
        for pid, reason in drops(p):
            if reason.startswith(RETRYABLE):
                retry[pid] = reason
            else:
                final.add(pid)
    return {pid: reason for pid, reason in retry.items() if pid not in kept and pid not in final}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("cands", help="stdio_cands.py output (cands_<name>.jsonl.gz)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=32)
    ap.add_argument("--limit", type=int, default=0, help="stop after this many candidates (0 = all)")
    ap.add_argument("--shard", default="", help="k/n: only pids with crc32(pid) %% n == k")
    ap.add_argument("--check", action="store_true",
                    help="verify nothing; exit 1 if any pid's drops are all retryable (sandbox_error / exception)")
    a = ap.parse_args()
    name = os.path.basename(a.cands).replace("cands_", "").replace(".jsonl.gz", "")
    sk, sn = map(int, a.shard.split("/")) if a.shard else (0, 1)
    suffix = f".s{sk}of{sn}" if a.shard else ""
    out_p = os.path.join(a.out_dir, f"verified_{name}{suffix}.jsonl")
    drop_p = os.path.join(a.out_dir, f"dropped_{name}{suffix}.csv")
    os.makedirs(a.out_dir, exist_ok=True)
    if a.check:
        left = unresolved(a.out_dir, name)
        for pid, reason in sorted(left.items())[:20]:
            print(f"unresolved {pid}: {reason}")
        print(f"{name}: {len(left)} rows still dropped by a sandbox error or exception", flush=True)
        return 1 if left else 0
    for p in (out_p, drop_p):
        drop_partial_tail(p)
    done = set()
    for p in shard_files(a.out_dir, "verified", name, "jsonl"):
        done |= {json.loads(ln)["metadata"]["upstream_pid"] for ln in complete_lines(p)}
    for p in shard_files(a.out_dir, "dropped", name, "csv"):
        done |= {pid for pid, reason in drops(p) if not reason.startswith(RETRYABLE)}
    fo = open(out_p, "a")
    new_drop = not os.path.exists(drop_p) or os.path.getsize(drop_p) == 0
    fd = open(drop_p, "a", newline="")
    w = csv.DictWriter(fd, ["pid", "source", "reason"])
    if new_drop:
        w.writeheader()
    lock = threading.Lock()
    stats = collections.Counter()
    sem = threading.Semaphore(a.workers * 2)
    t0 = time.time()

    def work(c):
        try:
            rec, info = decide(c)
        except Exception as e:
            rec, info = None, f"exception:{type(e).__name__}:{str(e)[:80]}"
        with lock:
            if rec is None:
                w.writerow({"pid": c["pid"], "source": c["source"], "reason": info})
                fd.flush()
                stats["drop:" + info.split(":")[0]] += 1
            else:
                fo.write(json.dumps(rec, ensure_ascii=False) + "\n")
                fo.flush()
                stats["kept"] += 1
            n = sum(stats.values())
            if n % 200 == 0:
                print(f"{time.time() - t0:.0f}s {n} {dict(stats.most_common(12))}", flush=True)
        sem.release()

    n = 0
    with ThreadPoolExecutor(a.workers) as ex, gzip.open(a.cands, "rt") as f:
        for line in f:
            c = json.loads(line)
            if c["pid"] in done or zlib.crc32(c["pid"].encode()) % sn != sk:
                continue
            n += 1
            if a.limit and n > a.limit:
                break
            sem.acquire()
            ex.submit(work, c)
    fo.close()
    fd.close()
    print("FINAL", dict(stats), flush=True)


if __name__ == "__main__":
    sys.exit(main())

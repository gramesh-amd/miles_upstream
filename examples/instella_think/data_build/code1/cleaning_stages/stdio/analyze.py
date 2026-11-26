#!/usr/bin/env python3
"""Statement-side analysis for every in-scope row: boilerplate-stripped body, defect flags, parsed
statement samples, test-format class, and candidate test sets (original / converted / augmented).

Out: work/analysis.pkl  {row: dict}
"""
import ast, json, pickle, re, collections, sys

sys.path.insert(0, "/it-share-prj2-1/data/goramesh-instella-rl/out/v5_assessment/litreview/prompt_audit")
from stdio_format import classify as fmt_classify, lit  # v5 serialization classifier (re-used)

KLEAR_HEAD = "Solve the following coding problem using the programming language python:"
TAILS = [
    "Write Python code to solve the problem. Present the code in \n```python\nYour code\n```\nat the end.",
    "Now solve the problem and return the code.",
    "Now solve the problem by providing the code.",
]


def body_of(p):
    b = p
    if b.startswith(KLEAR_HEAD):
        b = b[len(KLEAR_HEAD):]
    for t in TAILS:
        i = b.rfind(t)
        if i >= 0:
            b = b[:i]
    # Klear format footer
    b = re.sub(r"The input will be given via stdin and the output should be printed to stdout by your code\.\s*$", "", b.strip())
    b = re.sub(r"The input will be stdin and you should print your solution to stdout\s*$", "", b.strip())
    return b.strip()


# ---------------- flags ----------------
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


def tolerance(body):
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
    hits = []
    for m in RX["multi"].finditer(body):
        s = body[max(0, body.rfind(".", 0, m.start()) + 1): body.find(".", m.end()) if body.find(".", m.end()) > 0 else len(body)]
        if re.search(r"(print|output|return|display|write)\s+(out\s+)?any\s+(of\s+the\s+)?(integer|number|element|character|letter|key|other)\b", s, re.I) and not re.search(r"answer|solution", s, re.I):
            continue  # "print any character": not answer freedom
        if re.search(r"print\s+all\s+of\s+them|print\s+(them\s+)?all\b", s, re.I) and not re.search(r"any\s+order|print\s+any", s, re.I):
            continue
        if RX["multi_unique"].search(s) and not re.search(r"(print|output|return)\s+any", s, re.I):
            continue  # "if there are multiple answers, print the smallest"
        hits.append(s.strip()[:200])
    return hits


NONENG = re.compile(r"[\u0400-\u04ff\u4e00-\u9fff\u3040-\u30ff\uac00-\ud7af\u0600-\u06ff]")


def lang_flags(body):
    letters = re.findall(r"[^\W\d_]", body)
    ne = len(NONENG.findall(body))
    eng_words = len(re.findall(r"\b(the|of|and|to|is|in|that|each|given|number|print|output|input)\b", body, re.I))
    return dict(nonascii_frac=round(ne / max(1, len(letters)), 3), eng_words=eng_words,
                non_english=(ne / max(1, len(letters)) > 0.3 and eng_words < 15))


# ---------------- statement samples ----------------
IN_M = re.compile(r"^[-#*=\s]*(sample|example|test)?\s*(case\s*)?(input|in)(\s*(example|sample))?\s*[#№]?\s*(\d+)?\s*(:|\.)?\s*[-=*\s]*$", re.I)
OUT_M = re.compile(r"^[-#*=\s]*(sample|example|test)?\s*(case\s*)?(output|out)(\s*(example|sample))?\s*[#№]?\s*(\d+)?\s*(for the sample input\s*\d*)?\s*(:|\.)?\s*[-=*\s]*$", re.I)
IN_INLINE = re.compile(r"^\s*(sample\s+|example\s+)?input\s*\d*\s*:\s*(\S.*)$", re.I)
OUT_INLINE = re.compile(r"^\s*(sample\s+|example\s+)?output\s*\d*\s*:\s*(\S.*)$", re.I)
STOP = re.compile(r"^[-#*=\s]*(note|notes|explanation|hint|hints|constraints|scoring|subtask|example|examples|sample|input|output|test case|in the first|in the second|for the first|for the second|in sample|in example|illustration|clarification)\b", re.I)


def _prose(line):
    w = re.findall(r"[A-Za-z]{2,}", line)
    return len(w) >= 6 and len(line) > 40


def parse_samples(body):
    lines = [l.rstrip() for l in body.replace("\r", "").split("\n")]
    marks = []
    for i, l in enumerate(lines):
        s = l.strip().strip("`")
        if IN_M.match(s) and not re.search(r"format|specification", s, re.I):
            marks.append((i, "in", None))
        elif OUT_M.match(s) and not re.search(r"format|specification", s, re.I):
            marks.append((i, "out", None))
        elif IN_INLINE.match(s):
            marks.append((i, "in", IN_INLINE.match(s).group(2)))
        elif OUT_INLINE.match(s):
            marks.append((i, "out", OUT_INLINE.match(s).group(2)))
    samples = []
    for k, (i, typ, inl) in enumerate(marks):
        if typ != "in" or k + 1 >= len(marks) or marks[k + 1][1] != "out":
            continue
        j, _, oinl = marks[k + 1]
        inp = ([inl] if inl else []) + lines[i + 1: j]
        nxt = marks[k + 2][0] if k + 2 < len(marks) else len(lines)
        out = [oinl] if oinl else []
        started = bool(oinl)
        for t in range(j + 1, nxt):
            l = lines[t]
            if not l.strip():
                if started:
                    # blank line ends the block unless the next non-blank line is still data
                    nb = next((lines[u] for u in range(t + 1, nxt) if lines[u].strip()), "")
                    if not nb or STOP.match(nb.strip()) or _prose(nb):
                        break
                out.append(l)
                continue
            if STOP.match(l.strip()) and started:
                break
            if _prose(l) and started:
                break
            started = True
            out.append(l)
        clean = lambda xs: "\n".join(x for x in xs if not x.strip().startswith("```")).strip("\n")
        si, so = clean(inp), clean(out)
        if not so.strip() or any(_prose(x) for x in si.split("\n")):
            continue
        samples.append(dict(input=si.strip("\n") + "\n", output=so.strip()))
    return samples


def norm_in(s):
    return " ".join(str(s).split())


# ---------------- S2 conversions ----------------
STOPNAMES = set("""print input int str len max min sum range list dict set tuple sorted map abs pow round open
return if for while and or not f g h S P A B C D E F G H N M K X Y Z T Q a b c d e i j k n m p q r s t u v w x y z
O log sqrt gcd lcm mod sin cos tan exp floor ceil isPrime example Example""".split())


def fn_candidates(body, refs):
    names = collections.Counter(m.group(1) for m in re.finditer(r"\b([A-Za-z_]\w*)\s*\(", body))
    names.update({m.group(1): 3 for m in re.finditer(r"\bdef\s+([A-Za-z_]\w*)\s*\(", body)})
    names.update({m.group(1): 2 for m in re.finditer(r"(?:function|method)\s+`?([A-Za-z_]\w*)`?", body)})
    names.update({m.group(1): 2 for m in re.finditer(r"`([A-Za-z_]\w*)`", body)})
    ref_defs = collections.Counter()
    for r in refs:
        for m in re.finditer(r"^\s*def\s+([A-Za-z_]\w*)\s*\(", r["code"], re.M):
            ref_defs[m.group(1)] += 1
    cands = [(n, c, ref_defs[n]) for n, c in names.items() if n not in STOPNAMES and len(n) > 1 and ref_defs[n] > 0]
    cands.sort(key=lambda x: (-x[2], -x[1]))
    return cands


def _maybe_json(e):
    if isinstance(e, str):
        try:
            return json.loads(e)
        except Exception:
            return e
    return e


def _assert_expr(call, exp):
    if isinstance(exp, float):
        return f"assert abs({call} - {exp!r}) <= 1e-6 * max(1.0, abs({exp!r}))"
    return f"assert {call} == {exp!r}"


def to_asserts(fmt, tests, name):
    out = []
    for t in tests:
        vi, vo = lit(str(t["input"])), lit(str(t["output"]))
        if not isinstance(vi, list) or vo is None and str(t["output"]).strip() not in ("None", "null"):
            return None
        args = vi
        if fmt in ("fn_args_json",) or (fmt == "fn_args_list_repr" and all(isinstance(e, str) for e in vi)):
            if all(isinstance(e, str) for e in vi) and all(_maybe_json(e) is not e for e in vi) \
                    and any(e.strip()[:1] in '"[{' for e in vi):
                args = [_maybe_json(e) for e in vi]
        if fmt in ("fn_args_wrapped_out", "pylist_of_lines", "fn_args_list_repr") and isinstance(vo, list) and len(vo) == 1:
            exp = vo[0]
        else:
            exp = vo
        if isinstance(exp, str) and fmt == "fn_args_json":
            exp = _maybe_json(exp)
        call = f"{name}({', '.join(repr(a) for a in args)})"
        out.append(_assert_expr(call, exp))
    return out


def to_stdin(tests):
    out = []
    for t in tests:
        vi, vo = lit(str(t["input"])), lit(str(t["output"]))
        if not (isinstance(vi, list) and all(isinstance(e, str) for e in vi)):
            return None
        if isinstance(vo, list):
            o = "\n".join(str(e) for e in vo)
        else:
            o = str(t["output"])
        out.append({"input": "\n".join(vi) + "\n", "output": o})
    return out


IO_FMT = re.compile(r"-----\s*input|^\s*input\s*(format|specification)?\s*:?\s*$|\bfirst line\b|standard input|\bstdin\b|\bread from\b", re.I | re.M)


def main():
    vs = pickle.load(open("/tmp/v7stdio_rows.pkl", "rb"))
    res = {}
    for v in vs:
        body = body_of(v["prompt"])
        a = dict(row=v["row"], custom_id=v["custom_id"], src=v["src"], kind=v["kind"], body_len=len(body))
        a.update(lang_flags(body))
        a["interactive"] = bool(RX["interactive"].search(body))
        a["image"] = bool(RX["image"].search(body))
        a["image_tags"] = len(re.findall(r"<image>|!\[[^\]]*\]\(|<img\b", body, re.I))
        a["empty_stmt"] = len(re.sub(r"\s+", " ", body)) < 60
        a["other_lang"] = (m.group(0) if (m := RX["other_lang"].search(body)) else None)
        a["multi_hits"] = has_multi(body)
        a["tol"] = tolerance(body)
        a["case_stmt"] = (m.group(0) if (m := RX["case_stmt"].search(body)) else None)
        a["samples"] = parse_samples(body) if v["kind"] == "stdio" else []
        a["io_fmt"] = bool(IO_FMT.search(body))
        cands = {}
        if v["kind"] == "stdio":
            fmt = fmt_classify(v["prompt"], v["label"])
            a["fmt"] = fmt
            if fmt in ("pylist_of_lines", "fn_args_list_repr", "fn_args_wrapped_out", "fn_args_json"):
                if fmt in ("pylist_of_lines", "fn_args_list_repr") and a["io_fmt"]:
                    st = to_stdin(v["label"])
                    if st:
                        cands["to_stdin"] = ("stdio", st)
                fc = fn_candidates(body, v["refs"])
                a["fn_cands"] = fc[:3]
                if fc:
                    asr = to_asserts(fmt, v["label"], fc[0][0])
                    if asr:
                        cands["to_assert"] = ("assert", asr)
                if fmt == "pylist_of_lines" and "to_stdin" not in cands:
                    st = to_stdin(v["label"])
                    if st:
                        cands["to_stdin"] = ("stdio", st)
        else:
            a["fmt"] = "assert"
        a["conv_cands"] = cands
        res[v["row"]] = a
    pickle.dump(res, open("/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/stdio/work/analysis.pkl", "wb"))
    c = collections.Counter()
    for a in res.values():
        s = a["src"]
        for k in ("interactive", "image", "empty_stmt", "non_english"):
            c[(s, k)] += bool(a[k])
        c[(s, "image_tags")] += a["image_tags"] > 0
        c[(s, "other_lang")] += bool(a["other_lang"])
        c[(s, "multi")] += bool(a["multi_hits"])
        c[(s, "tol")] += a["tol"] is not None
        c[(s, "case")] += bool(a["case_stmt"])
        c[(s, "has_samples")] += bool(a["samples"])
        c[(s, "fmt_" + a["fmt"])] += 1
        for k in a["conv_cands"]:
            c[(s, "cand_" + k)] += 1
    for k in sorted(c):
        print(k, c[k])


if __name__ == "__main__":
    main()

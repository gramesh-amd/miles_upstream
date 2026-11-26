#!/usr/bin/env python3
"""How many training prompts contain their own hidden tests verbatim (statement examples reused as tests).

stdio test is 'in prompt' if the whitespace-normalised input AND output both occur in the whitespace-normalised
prompt (input >= 3 non-space chars, to avoid '1' / 'YES' trivially matching).
assert test is 'in prompt' if the call expression f(args) and the expected literal both occur (whitespace removed),
or the whole assert string occurs.
Output: work/prompt_tests.parquet (per row) and a printed summary.
"""
import ast, pickle, re
import pandas as pd

W = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam/work"


def ws(s):
    return re.sub(r"\s+", "", str(s))


def split_assert(a):
    try:
        t = ast.parse(a.strip()).body[0].test
    except Exception:
        return None
    if isinstance(t, ast.Compare) and len(t.ops) == 1:
        return ast.get_source_segment(a.strip(), t.left), ast.get_source_segment(a.strip(), t.comparators[0])
    return None


def main():
    train = pickle.load(open(f"{W}/train.pkl", "rb"))
    rows = []
    for t in train:
        p = ws(t["text"])
        for which in ("label", "full"):
            tests = t["tests_label"] if which == "label" else (t["tests_full"] or [])
            n = k = 0
            for x in tests:
                n += 1
                if isinstance(x, dict):
                    i, o = ws(x["input"]), ws(x["output"])
                    if len(i) >= 3 and i in p and o in p:
                        k += 1
                elif isinstance(x, str):
                    if ws(x) in p:
                        k += 1
                        continue
                    sp = split_assert(x)
                    if sp and sp[0] and sp[1] and ws(sp[0]) in p and ws(sp[1]) in p:
                        k += 1
            rows.append(dict(custom_id=t["custom_id"], src=t["src"], which=which, n_tests=n, n_in_prompt=k))
    df = pd.DataFrame(rows)
    df["any"] = df.n_in_prompt > 0
    df["all"] = (df.n_in_prompt == df.n_tests) & (df.n_tests > 0)
    df["frac"] = df.n_in_prompt / df.n_tests.clip(lower=1)
    df.to_parquet(f"{W}/prompt_tests.parquet")
    s = df.groupby(["which", "src"]).agg(rows=("any", "size"), rows_any=("any", "sum"), rows_all=("all", "sum"),
                                         tests=("n_tests", "sum"), tests_in_prompt=("n_in_prompt", "sum"))
    tot = df.groupby("which").agg(rows=("any", "size"), rows_any=("any", "sum"), rows_all=("all", "sum"),
                                  tests=("n_tests", "sum"), tests_in_prompt=("n_in_prompt", "sum"))
    tot["src"] = "ALL"
    s = pd.concat([s.reset_index(), tot.reset_index()])
    s["pct_rows_any"] = (100 * s.rows_any / s.rows).round(1)
    s["pct_rows_all"] = (100 * s.rows_all / s.rows).round(1)
    s["pct_tests"] = (100 * s.tests_in_prompt / s.tests).round(1)
    print(s.to_string(index=False))
    s.to_csv(f"{W}/prompt_tests_summary.csv", index=False)


if __name__ == "__main__":
    main()

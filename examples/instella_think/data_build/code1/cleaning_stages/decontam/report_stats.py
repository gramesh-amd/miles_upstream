#!/usr/bin/env python3
"""Numbers for REPORT.md: per-eval-set verdict counts, date check table, prompt-test reuse."""
import pickle
import pandas as pd

D = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/decontam"
W = f"{D}/work"
c = pd.read_csv(f"{D}/contaminated.csv")
c["base"] = c.verdict.str.replace("_raw_subproblem", "", regex=False)
c["raw"] = c.verdict.str.endswith("_raw_subproblem")
SETS = ["lcb_v3", "lcb_v5", "lcb_v6", "humanevalplus", "mbppplus", "humanevalpro", "mbpppro", "cruxeval_o", "cruxeval_i", "codeforces"]
N = {"lcb_v3": 612, "lcb_v5": 279, "lcb_v6": 175, "humanevalplus": 164, "mbppplus": 378, "humanevalpro": 164,
     "mbpppro": 378, "cruxeval_o": 800, "cruxeval_i": 800, "codeforces": 422}
rows = []
for s in SETS:
    g = c[c.eval_set == s]
    d = {"eval_set": s, "n_items": N[s], "pairs_checked": len(g)}
    for v in ("contaminated", "paraphrase", "paraphrase_generic", "false_positive"):
        x = g[(g.base == v) & ~g.raw]
        d[f"{v}_items"] = x.eval_id.nunique()
        d[f"{v}_rows"] = x.custom_id.nunique()
    x = g[(g.base == "contaminated") & g.raw]
    d["raw_sub_contaminated_items"] = x.eval_id.nunique()
    rows.append(d)
t = pd.DataFrame(rows)
t.to_csv(f"{D}/per_eval_counts.csv", index=False)
print(t.to_string(index=False))

# date check: best dated reference per train row (13-gram >= .2, or tfidf >= .6 and 8-gram >= .3)
refs = pickle.load(open(f"{W}/refs.pkl", "rb"))
train = pickle.load(open(f"{W}/train.pkl", "rb"))
p = pd.read_parquet(f"{W}/ref_pairs.parquet")
p = p[(p.ng13 >= 0.2) | ((p.tfidf >= 0.6) & (p.ng8 >= 0.3))]
p = p.sort_values(["ng13", "tfidf"], ascending=False).drop_duplicates("ti")
m = p.assign(ref=[refs[i]["ref"] for i in p.ei], ref_id=[refs[i]["ref_id"] for i in p.ei],
             date=[refs[i]["date"] for i in p.ei], src=[train[i]["src"] for i in p.ti],
             custom_id=[train[i]["custom_id"] for i in p.ti])
m.to_parquet(f"{W}/ref_mapped.parquet")
win = {"lcb_v3 (2023-05..2024-07)": ("2023-05-01", "2024-07-31"), "lcb_v5 (2024-08..2025-02)": ("2024-08-01", "2025-02-28"),
       "lcb_v6 (2025-02..2025-05)": ("2025-02-01", "2025-05-31")}
m["window"] = ""
for k, (a, b) in win.items():
    m.loc[(m.date >= a) & (m.date <= b), "window"] += k + ";"
m[["custom_id", "src", "ref", "ref_id", "date", "window", "ng13", "ng8", "tfidf"]].to_csv(f"{D}/date_check.csv", index=False)
print("\nmapped rows", len(m), m.groupby(["src", "ref"]).size().to_dict())
print("year range", m.date.min(), m.date.max(), " >=2023-05:", (m.date >= "2023-05-01").sum())
print(m[m.window != ""][["custom_id", "ref_id", "date", "window"]].to_string(index=False))
train = pickle.load(open(f"{W}/train.pkl", "rb"))
stdio_srcs = pd.Series([t["src"] for t in train]).value_counts().to_dict()
print("rows per src", stdio_srcs)
print(m.groupby(m.date.str[:4]).size().to_dict())

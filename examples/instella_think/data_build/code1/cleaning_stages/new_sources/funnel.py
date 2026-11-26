#!/usr/bin/env python3
"""Per-source funnel (cutoff -> decontam -> dedup vs v7 -> >=5 tests -> not interactive -> checker if multi-answer),
cross-source overlap, and the recommended union. Reads work/{corpus,overlap,text_hits,eval_items}.parquet."""
import json
import numpy as np
import pandas as pd

N = "/it-share-prj2-1/data/goramesh-instella-rl/out/v7_data/new_sources"
CUT = "2024-08-01"
C = pd.read_parquet(f"{N}/work/corpus.parquet").drop(columns=["statement"])
O = pd.read_parquet(f"{N}/work/overlap.parquet")
H = pd.read_parquet(f"{N}/work/text_hits.parquet", columns=["uid", "eval_set", "eval_id", "det", "cov"])
E = pd.read_parquet(f"{N}/work/eval_items.parquet", columns=["item", "eval_set", "eval_id", "date"])
edate = dict(zip(E.item, E.date))

HEADLINE = {"lcb", "codeforces", "openr1_cf_test", "humanevalplus", "mbppplus", "humanevalpro", "mbpppro", "cruxeval"}
H = H[H.eval_set.isin(HEADLINE)].copy()
H["edate"] = (H.eval_set + "|" + H.eval_id).map(edate)
H["kind"] = np.where((H.eval_set == "lcb") & (H.edate < CUT), "lcb_pre", np.where(H.eval_set == "lcb", "lcb_post", H.eval_set))
O["src"] = O.uid.map(dict(zip(C.pid, C.source))).fillna("ours:" + O.uid.str.split(":").str[1])
idh = O[O.id_hit != ""][["uid", "id_hit"]].assign(kind=lambda d: np.where(d.id_hit.str.contains("lcb\\|"), "lcb_id", "cf_id"))
TH = pd.read_parquet(f"{N}/work/test_hits.parquet")
TH = TH[TH.shared >= 4].copy()                      # v7: false positives share 1-3 edge-case tests
TH["edate"] = TH["item"].map(edate)
TH["kind"] = np.where(TH["item"].str.startswith("lcb|"), np.where(TH.edate < CUT, "lcb_pre", "lcb_post"), "codeforces") + "_tests"
TH.to_csv(f"{N}/test_overlap_hits.csv", index=False)
hit_kind = pd.concat([H[["uid", "kind"]], idh[["uid", "kind"]], TH[["uid", "kind"]]])
O = O.merge(hit_kind.groupby("uid").kind.agg(lambda s: ",".join(sorted(set(s)))).rename("hit"), left_on="uid",
            right_index=True, how="left")
cl = O.groupby("cluster")
O["cl_hit"] = cl.hit.transform(lambda s: ",".join(sorted(set(",".join(s.dropna()).split(",")) - {""})))
O["cl_v7"] = cl.src.transform(lambda s: s.str.startswith("ours:v7").any())
O["cl_v7_stdio"] = cl.src.transform(lambda s: (s == "ours:v7_stdio").any())
O["cl_dolci"] = cl.src.transform(lambda s: s.str.startswith("ours:dolci").any())
O["cl_size"] = cl.uid.transform("size")
D = C.merge(O[["uid", "cluster", "hit", "cl_hit", "cl_v7", "cl_v7_stdio", "cl_dolci", "nsamp_nb", "ng8_max", "ng13_max"]],
            left_on="pid", right_on="uid")

# dates: own / v5 cluster / source-vintage bound, then propagated through the new clusters (latest dated member)
D["date_u"] = D.date_upper.where(D.date_upper.notna() & (D.date_upper != "<NA>"))
cmax = D.groupby("cluster").date_u.transform("max")
D["date_eff"] = D.date_u.fillna(cmax)
D.loc[D.date_u.isna() & D.date_eff.notna(), "date_source"] = "new_cluster"
D["dated"] = D.date_eff.notna()
D["pre"] = D.date_eff < CUT
D["lcb_v3_window"] = (D.date_eff >= "2023-05-01") & (D.date_eff < CUT)

D["stdio"] = ~D.fn_style & D.f_stdin_format
D["contam"] = D.cl_hit != ""
D["contam_strict_lcbpre_only"] = D.cl_hit.isin(["lcb_pre"])
D["n_t"] = D.n_tests_known
D["ge5"] = D.n_t >= 5
D["interactive_any"] = D.interactive | D.file_io
D["multi"] = D.multi_answer_text | D.needs_checker
ck = D.checker.fillna("None")
D["checker_py"] = ck.isin(["generated", "judge_fn"])
D["checker_opt"] = ck.isin(["yesno_ci", "float"])          # grader options case_insensitive / float_tol
D["checker_cpp"] = ck.eq("custom")                          # CC+ testlib C++: not runnable by our Python route
D["checker_ok"] = D.checker_py | D.checker_opt
D["ok_multi"] = ~D.multi | D.checker_ok
D.to_parquet(f"{N}/work/funnel_rows.parquet")

w = D.sample_weight


def wsum(m):
    return int(round((w * m).sum()))


rows = []
for s, g in D.groupby("source"):
    ws = g.sample_weight
    f = lambda m: int(round((ws * m).sum()))  # noqa: E731
    st = g.stdio
    pre = st & g.pre
    dc = pre & ~g.contam
    dd = dc & ~g.cl_v7
    t5 = dd & g.ge5
    ni = t5 & ~g.interactive_any
    fin = ni & g.ok_multi
    fin_img = fin & ~g.image
    m = st & g.multi
    rows.append(dict(
        source=s, raw=f(pd.Series(True, index=g.index)), stdio=f(st), dated_pct=round(100 * (g.dated & st).sum() / max(st.sum(), 1), 1),
        pre=f(pre), undated=f(st & ~g.dated), contam_any=f(st & g.contam), contam_pre=f(pre & g.contam),
        contam_lcbpre_only=f(pre & g.contam_strict_lcbpre_only), after_decontam=f(dc), dup_v7=f(dc & g.cl_v7),
        dup_v7_stdio=f(dc & g.cl_v7_stdio), in_dolci_any=f(dc & g.cl_dolci), after_dedup=f(dd), ge5=f(t5),
        tests_known_pct=round(100 * (g.n_t.notna() & st).sum() / max(st.sum(), 1), 1), med_tests=g.loc[st, "n_t"].median(),
        not_interactive=f(ni), final=f(fin), final_uniq_clusters=g.loc[fin, "cluster"].nunique(),
        final_no_image=f(fin_img), image_pct=round(100 * (g.image & st).mean() / max(st.mean(), 1e-9), 1),
        interactive_pct=round(100 * (g.interactive_any & st).sum() / max(st.sum(), 1), 1),
        multi=f(m), multi_pct=round(100 * m.sum() / max(st.sum(), 1), 1),
        chk_py=f(m & g.checker_py), chk_opt=f(m & g.checker_opt), chk_cpp=f(m & g.checker_cpp),
        chk_cov_pct=round(100 * (m & g.checker_ok).sum() / max(m.sum(), 1), 1),
        lcb_v3_window_final=f(fin & g.lcb_v3_window),
    ))
T = pd.DataFrame(rows).sort_values("final", ascending=False)
T.to_csv(f"{N}/funnel.csv", index=False)
pd.set_option("display.width", 250)
print(T.to_string(index=False))

# hits per source x kind (stdio rows, before cutoff filter)
hk = D[D.stdio & D.contam].assign(k=lambda d: d.cl_hit.str.split(",")).explode("k")
print(pd.crosstab(hk.source, hk.k).to_string())
pd.crosstab(hk.source, hk.k).to_csv(f"{N}/contam_by_source.csv")

# cross-source overlap among each source's survivors: share of A's final clusters that contain a B row (any stage)
fin = D[D.stdio & D.pre & ~D.contam & ~D.cl_v7 & D.ge5 & ~D.interactive_any & D.ok_multi]
srcs = list(T.source)
mem = D.groupby("cluster").source.agg(set)
M = pd.DataFrame(0.0, index=srcs, columns=srcs)
for a in srcs:
    ca = fin[fin.source == a].cluster.unique()
    if len(ca) == 0:
        continue
    for b in srcs:
        M.loc[a, b] = round(100 * np.mean([b in mem[c] for c in ca]), 1)
M.to_csv(f"{N}/overlap_matrix.csv")
print(M.to_string())

# recommended union (priority order; one representative per cluster)
PRIO = ["openr1_cf_verifiable", "ccplus", "hardtests", "taco_verified", "deepcoder_taco", "deepcoder_primeintellect",
        "deepmind_code_contests", "synthetic2_rl", "eurus2_rl", "rstar_seed", "klear_codesub", "skywork_or1",
        "nemotron_rl_cc", "baai_taco", "intellect3_code", "rstar_synthetic_rl"]
seen, out = set(), []
for s in PRIO:
    g = fin[fin.source == s]
    cs = g.cluster.unique()
    new = [c for c in cs if c not in seen]
    seen.update(cs)
    out.append(dict(source=s, final_clusters=len(cs), new_unique=len(new), cum=len(seen)))
U = pd.DataFrame(out)
U.to_csv(f"{N}/union_cumulative.csv", index=False)
print(U.to_string(index=False))

# recommended set: strong-test sources only
REC = ["openr1_cf_verifiable", "ccplus", "hardtests", "taco_verified", "deepcoder_taco", "deepcoder_primeintellect"]
r = fin[fin.source.isin(REC)].copy()
r["prio"] = r.source.map({s: i for i, s in enumerate(REC)})
# CC+: require 1x TPR/TNR >= 0.9 where known (CC+ paper's "verified"); HardTests: dated AtCoder/CF only
r = r[~((r.source == "ccplus") & ~((r.tpr1 >= 0.9) & (r.tnr1 >= 0.9)))]
r = r.sort_values(["prio"]).drop_duplicates("cluster")
r[["pid", "source", "cluster", "date_eff", "date_source", "n_t", "multi", "checker", "image", "lcb_v3_window",
   "rating", "proxy_solve", "cl_dolci"]].to_csv(f"{N}/recommended_union.csv.gz", index=False)
print("recommended union", len(r), r.source.value_counts().to_dict(), "image", int(r.image.sum()),
      "lcb_v3_window", int(r.lcb_v3_window.sum()), "in_dolci_cluster", int(r.cl_dolci.sum()))
json.dump(dict(n=len(r), by_source=r.source.value_counts().to_dict(), image=int(r.image.sum()),
               lcb_v3_window=int(r.lcb_v3_window.sum()), in_dolci=int(r.cl_dolci.sum())),
          open(f"{N}/recommended_union_summary.json", "w"), indent=1)

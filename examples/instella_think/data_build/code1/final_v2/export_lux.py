"""Write final_v2/lux/checker_candidates.jsonl.gz: every checker group (Open-R1 generated_checker rows that passed the
static checks and decontamination in prepare.py / decontam.py / groups.py), one JSON object per CF problem:
  pid, custom_id (row id), copies, src, task(s), prompt (unchanged Dolci prompt, or the Open-R1 text for image copies),
  checker (Python source), own_tests (winner's tests first, then merged copies), n_win_own, official (Open-R1 official
  tests with new inputs), refs (ordered: v6, OLMo by Dolci reward, accepted CF Python submissions, v6 near-miss),
  cf_wa (wrong CF submissions), mapping (prepare.mapping_check of the winner), metadata (v6), date, rating,
  v1_dups / v1_dup_max_tests (final_v1 duplicates found by CF id, S7 cluster or MinHash), login_node_pilot (the 6
  groups verified on the login node before the scope change, for comparison only).

    python3 export_lux.py
"""
import gzip
import json
import os
import pickle

from v2common import FINAL2, WORK


def main():
    G = pickle.load(open(f"{WORK}/groups.pkl", "rb"))
    R = pickle.load(open(f"{WORK}/cands.pkl", "rb"))
    pilot = pickle.load(open(f"{WORK}/checker_pilot_verify.pkl", "rb"))
    os.makedirs(f"{FINAL2}/lux", exist_ok=True)
    n = 0
    with gzip.open(f"{FINAL2}/lux/checker_candidates.jsonl.gz", "wt") as f:
        for pid, g in sorted(G.items()):
            if g["route"] != "checker":
                continue
            p = pilot.get(pid)
            pv = None if p is None else (p.get("skipped") or p["verify"].get("why") or "verified")
            obj = dict(pid=pid, custom_id=g["cid"], copies=g["copies"], src=g["src"], tasks=g["tasks"], prompt=g["prompt"],
                       checker=g["checker"], own_tests=g["own_tests"], n_win_own=g["n_win_own"], official=g["official"],
                       refs=[dict(origin=r["origin"], code=r["code"]) for r in g["refs"]], cf_wa=g["cf_wa"],
                       mapping=R[g["cid"]]["mapping"], metadata=g["metadata"], date=g["date"], rating=g["rating"],
                       image_copy=g["image_copy"], v1_dups=g["v1_dups"], v1_dup_max_tests=g["v1_dup_max_tests"],
                       login_node_pilot=pv)
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")
            n += 1
    print("checker groups", n, "rows covered", sum(len(g["copies"]) for g in G.values() if g["route"] == "checker"))


if __name__ == "__main__":
    main()

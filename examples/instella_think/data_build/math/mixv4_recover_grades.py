#!/usr/bin/env python3
"""Recover the per-prompt pass rates that were ALREADY measured on our DPO
checkpoint, instead of re-running a 57-node-hour grading pass.

PROVENANCE. data/dolci_math_filtered_deduped_nomrt.jsonl (17249 prompts) is the
surviving output of an Olmo-3-style offline difficulty filter run on OUR DPO
checkpoint over a 30180-prompt pool: 8 rollouts per prompt at T=1.0/top_p=1.0,
drop pass_rate > 0.625. The per-prompt grades were used as a filter and then
dropped from the shipped file, which carries only dataset_source. They survive
in the bucket under
  goramesh/instella-backup/data/olmo3_rl_research_data/backup_20260515_0029/
as 7 shards of *_generations.jsonl, each line
  {idx, label, rewards:[8 floats], pass_rate, completions:[8 strings]}
`idx` is LOCAL to the shard; global index = shard_start + idx. The shard name
encodes start and count: ..._shard_<start>_<count>_generations.jsonl.

WHY NOT JUST READ THE CHECKPOINTS. Only 3 of 7 shards have the compact
.checkpoint.json; the other 4 exist only as multi-GB generations files. So the
big files are streamed once, the four small fields kept, and the completions
discarded.

VERIFIER CAVEAT, carried forward deliberately: these grades come from
olmo3_math_verify (whole-response rfind("\\boxed")), while training rewards with
--rm-type deepscaler, which scores 0 unless the response closes its reasoning
and grades only the text after the closer. deepscaler is STRICTLY STRICTER, so
every pass_rate here is an UPPER BOUND on the training reward. The AIME eval
decode measured 14.98% of samples failing exactly that way (`no_closer`). Treat
these numbers as optimistic and do not quote them as training pass rates.
"""
import json
import os
import re
import subprocess
import sys

ROOT = os.environ.get("INSTELLA_RL_ROOT", "/it-share-prj2-1/data/goramesh-instella-rl")
OUT = f"{ROOT}/out/analysis_1012/dolci_filter"
CREDS = os.path.expanduser("~/.oci_s3_creds.env")
BUCKET_DIR = ("goramesh/instella-backup/data/olmo3_rl_research_data/"
              "backup_20260515_0029")
SHARDS = [(0, 4312), (4312, 4312), (8624, 4312), (12936, 4311),
          (17247, 4311), (21558, 4311), (25869, 4311)]
GRADES = f"{OUT}/grades_30k.jsonl"


def fetch(key, dest):
    subprocess.run([sys.executable, os.path.join(os.path.dirname(os.path.abspath(__file__)), "oci_ls.py"), CREDS,
                    "--get", key, dest], check=True,
                   stdout=subprocess.DEVNULL)


os.makedirs(OUT, exist_ok=True)
# The pool the grades index into (gidx = line number); build_mix_v4.py joins on it.
POOL = f"{OUT}/pool_30k.jsonl"
if not os.path.exists(POOL):
    fetch("goramesh/instella-backup/data/olmo3_rl_research_data/dolci_think_rl_math_30k.jsonl", POOL)
done_shards = set()
if os.path.exists(GRADES):
    for line in open(GRADES):
        done_shards.add(json.loads(line)["shard"])
    print(f"resuming: {len(done_shards)} shards already extracted")

with open(GRADES, "a") as out:
    for start, count in SHARDS:
        if start in done_shards:
            continue
        name = f"dolci_think_rl_math_30k_filtered_olmo3_shard_{start}_{count}_generations.jsonl"
        tmp = f"{OUT}/_tmp_gen.jsonl"
        print(f"  shard {start:>6} ({count} prompts): downloading...", flush=True)
        fetch(f"{BUCKET_DIR}/{name}", tmp)
        n = 0
        for line in open(tmp, errors="replace"):
            try:
                d = json.loads(line)
            except json.JSONDecodeError:
                continue
            if "pass_rate" not in d or "idx" not in d:
                continue
            out.write(json.dumps({
                "shard": start,
                "gidx": start + int(d["idx"]),
                "label": d.get("label"),
                "rewards": d.get("rewards"),
                "pass_rate": d["pass_rate"],
            }) + "\n")
            n += 1
        out.flush()
        os.remove(tmp)          # discard the multi-GB completions
        print(f"    extracted {n} graded prompts (expected <= {count})", flush=True)

tot = sum(1 for _ in open(GRADES))
print(f"\ntotal graded prompts recovered: {tot}")
assert tot > 0, "recovered nothing - the generations schema must have changed"

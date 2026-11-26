#!/usr/bin/env python3
"""Build data/mix_v5_math.jsonl = mix_v4 + the eval's answer-format instruction.

THE ONLY CHANGE FROM mix_v4 IS THE PROMPT SUFFIX, and it is the single highest-
certainty fix we found in arm F.

WHY. The reward is `deepscaler`, which returns 0 unless the response contains a
reasoning closer and then grades only the text AFTER it. Measured over arm F:

    closer rate      66.7% -> 98.5%   (x1.477)
    raw reward       0.390 -> 0.582   (x1.492)

Those are the same multiplier. Essentially the entire reward gain of the first
150 rollouts was the model learning to emit `</think>`, not learning to solve
maths -- conditional on emitting a closer, accuracy did not improve.

And it bought nothing on the benchmark, because the gap is one-sided: 0 of
25,162 training prompts carried a format instruction, while EVERY eval prompt
ends with "Present the answer in LaTex format: \\boxed{Your answer}". The model
already formatted correctly on AIME at baseline, so ~150 rollouts of four-node
time went into a skill the eval never exercises.

Appending the eval's own suffix to the training prompts makes reward measure
reasoning from step 0 and puts training and eval in the same format regime.

DELIBERATELY NOT CHANGING THE MIX. It is tempting to also drop the "spent"
sources, but per-domain contribution says that framing was wrong: dapo_17k_en
sits near the structural ceiling yet is the second most productive source per
rollout spent (+1.05pp of the overall +4.0pp from 11.9% of rollouts). More
importantly, arm F's recipe flaws are now identified and concrete, so changing
the recipe AND the data at once would make arm G uninterpretable. Data held
constant; recipe fixed.
"""
import json
import os

ROOT = os.environ.get("INSTELLA_RL_ROOT", "/it-share-prj2-1/data/goramesh-instella-rl")
SRC = os.environ.get("SRC", f"{ROOT}/data/mix_v4_math.jsonl")
DST = os.environ.get("DST", f"{ROOT}/data/mix_v5_math.jsonl")

# Byte-identical to the suffix in data/aime-2024-olmes.jsonl. Any divergence
# here re-creates the train/eval format gap this script exists to close, so it
# is asserted against the real eval file rather than retyped from memory.
SUFFIX = "\n\nPresent the answer in LaTex format: \\boxed{Your answer}"
ev = json.loads(open(f"{ROOT}/data/aime-2024-olmes.jsonl").readline())
assert ev["prompt"][0]["content"].endswith(SUFFIX), \
    "eval suffix changed; update SUFFIX so train and eval stay in the same regime"

n = added = already = 0
with open(DST, "w") as out:
    for line in open(SRC):
        r = json.loads(line)
        c = r["prompt"][0]["content"]
        n += 1
        if c.rstrip().endswith("\\boxed{Your answer}"):
            already += 1
        else:
            r["prompt"][0]["content"] = c.rstrip() + SUFFIX
            added += 1
        out.write(json.dumps(r) + "\n")

print(f"read  {n} prompts from {os.path.basename(SRC)}")
print(f"  suffix appended : {added}")
print(f"  already present : {already}")
print(f"-> {DST}")

# Verify the result against the eval format and the rm_type trap. metadata.rm_type
# OUTRANKS --rm-type, so a stray one would silently grade with a different
# verifier than training is configured for.
chk = [json.loads(l) for l in open(DST)]
bad_fmt = sum(1 for r in chk
              if not r["prompt"][0]["content"].endswith("\\boxed{Your answer}"))
bad_rm = sum(1 for r in chk if "rm_type" in (r.get("metadata") or {}))
bad_lab = sum(1 for r in chk if not str(r.get("label", "")).strip())
print(f"\nverify: {len(chk)} rows | missing suffix {bad_fmt} | "
      f"metadata.rm_type {bad_rm} | empty label {bad_lab}")
assert bad_fmt == 0 and bad_rm == 0 and bad_lab == 0, "mix_v5 failed verification"
print("mix_v5 OK")
print("\nsample prompt tail:")
print("  " + repr(chk[0]["prompt"][0]["content"][-90:]))

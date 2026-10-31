#!/usr/bin/env bash
# Print the nodes of a job, one per line. This build of Spur has no
# `scontrol show hostnames`, so expand 'n-[001-003,007]' here.
#
#   cluster/nodes.sh <JOBID>
set -uo pipefail
JOBID="${1:?usage: nodes.sh <JOBID>}"
python3 -c '
import re, sys
out = []
for m in re.finditer(r"([^,\[\]]+)(?:\[([^\]]*)\])?", sys.argv[1]):
    prefix, body = m.group(1), m.group(2)
    if not prefix and not body:
        continue
    if not body:
        out.append(prefix); continue
    for part in body.split(","):
        if "-" in part:
            lo, hi = part.split("-", 1)
            out += [f"{prefix}{i:0{len(lo)}d}" for i in range(int(lo), int(hi) + 1)]
        else:
            out.append(f"{prefix}{part}")
print("\n".join(out))
' "$(squeue -j "$JOBID" -h -o "%N" 2>/dev/null)"

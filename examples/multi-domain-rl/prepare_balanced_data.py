#!/usr/bin/env python3
"""Prepare domain-balanced training data for multi-domain RL.

Reads domain-tagged JSONL and produces a round-robin interleaved output
so that consecutive prompts cycle through domains. This ensures every
rollout batch contains a roughly equal mix of domains regardless of
batch size.

Usage:
    python prepare_balanced_data.py \
        --input /path/to/train_math_code_if.jsonl \
        --output /path/to/balanced_train.jsonl \
        [--domain-key rm_type] \
        [--seed 42]
"""

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description="Interleave JSONL by domain for balanced batching")
    parser.add_argument("--input", required=True, help="Input JSONL with metadata.rm_type")
    parser.add_argument("--output", required=True, help="Output balanced JSONL")
    parser.add_argument("--domain-key", default="rm_type", help="Metadata key for domain")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    by_domain: dict[str, list] = defaultdict(list)
    with open(args.input) as f:
        for line in f:
            row = json.loads(line)
            domain = row.get("metadata", {}).get(args.domain_key, "unknown")
            by_domain[domain].append(row)

    for domain in by_domain:
        random.shuffle(by_domain[domain])

    domains = sorted(by_domain.keys())
    print(f"Domains: {', '.join(f'{d}={len(by_domain[d])}' for d in domains)}")

    iters = {d: iter(by_domain[d]) for d in domains}
    exhausted = set()
    interleaved = []

    while len(exhausted) < len(domains):
        for d in domains:
            if d in exhausted:
                continue
            try:
                interleaved.append(next(iters[d]))
            except StopIteration:
                exhausted.add(d)

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w") as f:
        for row in interleaved:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"Wrote {len(interleaved)} samples to {args.output}")

    verify: dict[str, int] = defaultdict(int)
    for row in interleaved:
        verify[row.get("metadata", {}).get(args.domain_key, "unknown")] += 1
    print(f"Verification: {dict(verify)}")


if __name__ == "__main__":
    main()

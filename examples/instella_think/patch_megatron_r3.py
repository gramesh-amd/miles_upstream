#!/usr/bin/env python3
"""Port miles' rollout routing replay (R3) hooks into the FarSkip Megatron. Idempotent.

    python3 examples/instella_think/patch_megatron_r3.py [MEGATRON_PATH]

USE_ROUTING_REPLAY=1 (run_code.sh's default) needs two hooks the Primus pin lacks; without them
no Replay is registered and the first train step dies in register_replay_list_moe with
"IndexError: list index out of range". These are the same two hooks the image's newer
/root/Megatron-LM carries:
  1. TopKRouter.__init__ registers one Replay per router (router.py).
  2. topk_routing_with_score_function sends its top-k through the replay manager (moe_utils.py),
     so the trainer reuses the expert choices the rollout engine made.
Both are no-ops unless --use-rollout-routing-replay enables the manager.
"""

import ast
import os
import sys

MARK = "# R3 (miles rollout routing replay)"


def patch(path, scope, anchor, insert, after):
    src = open(path).read()
    if MARK in src:
        print(f"  v already patched: {path}")
        return
    lines = src.splitlines(keepends=True)
    # First anchor line after the (unique) scope line, i.e. inside that function.
    scopes = [i for i, l in enumerate(lines) if l.startswith(scope)]
    if len(scopes) != 1:
        sys.exit(f"  x scope found {len(scopes)} times (want 1) in {path}: {scope!r}")
    hit = next((i for i in range(scopes[0], len(lines)) if lines[i].rstrip("\n") == anchor), None)
    if hit is None:
        sys.exit(f"  x anchor not found after {scope!r} in {path}: {anchor!r}")
    at = hit + 1 if after else hit
    lines[at:at] = insert
    new = "".join(lines)
    ast.parse(new)  # fail loud rather than write a file that breaks `import megatron`
    open(path, "w").write(new)
    print(f"  v patched: {path}")


def main():
    megatron = sys.argv[1] if len(sys.argv) > 1 else os.environ["MEGATRON_PATH"]
    moe = os.path.join(megatron, "megatron", "core", "transformer", "moe")

    # 1. End of TopKRouter.__init__: the last statement of its expert-bias if/else.
    patch(
        os.path.join(moe, "router.py"),
        scope="class TopKRouter(",
        anchor="            self.ga_steps = None",
        insert=[
            "\n",
            f"        {MARK}: one Replay per router; inert unless the manager is enabled.\n",
            "        from miles.utils.replay_base import routing_replay_manager\n",
            "\n",
            '        routing_replay_manager.register_to_module(self, "routing_replay")\n',
        ],
        after=True,
    )
    # 2. topk_routing_with_score_function: wrap compute_topk before its first use.
    patch(
        os.path.join(moe, "moe_utils.py"),
        scope="def topk_routing_with_score_function(",
        anchor='    if score_function == "softmax":',
        insert=[
            f"    {MARK}: record or replay the rollout's expert choices.\n",
            "    from miles.utils.replay_base import routing_replay_manager\n",
            "\n",
            "    compute_topk = routing_replay_manager.get_topk_fn(compute_topk, return_probs=True)\n",
            "\n",
        ],
        after=False,
    )
    print("Megatron R3 hooks applied OK.")


if __name__ == "__main__":
    main()

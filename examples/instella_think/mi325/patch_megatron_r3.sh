#!/bin/bash
# Port the two miles R3 (rollout routing replay) hooks into the FarSkip Megatron.
#
# WHY. --use-rollout-routing-replay makes the trainer reuse the expert sets the
# sglang rollout actually routed to, instead of re-deriving them from its own
# logits. On this model that is worth ~4.3x on the train/rollout logprob gap
# (mi325 A/B, R3 off vs on: tis_abs 0.0500 -> 0.0116, train_rollout_kl 0.00627 ->
# 0.00041, tis_clipfrac 7.9e-4 -> 1.15e-6 -- three functionals of the same
# divergence moving by linear/quadratic/tail powers, which is what must happen if
# the distribution's scale really shrank and cannot happen from re-weighting).
# ~77% of the post-fp32-LM-head gap was MoE routing.
#
# THE PROBLEM. miles drives R3 through `routing_replay_manager`, but the hooks that
# actually register a Replay per router and route top-k through the manager live in
# MEGATRON, not in miles. The Primus FarSkip fork carries FarSkip but never had
# them, so `replay_list` stayed EMPTY and register_replay_list_moe died on
# replay_list[0] with "IndexError: list index out of range" (MI355 run
# 20260820_054007). The image's newer /root/Megatron-LM HAS both hooks but has no
# FarSkip, so -- following what MI355 did -- we port the hooks into the FarSkip
# fork rather than switching Megatron trees.
#
# The upstream hooks cannot be copied verbatim: /root/Megatron-LM is newer and
# wraps them in scaffolding this fork lacks (`is_mtp_layer`, `RouterReplay`,
# `config.moe_enable_routing_replay`, and an `is_mtp`/`router_replay` signature on
# topk_routing_with_score_function). So we insert the minimal equivalent.
#
# Both hooks are INERT when R3 is off: register_to_module() early-returns unless
# the manager is enabled, and get_topk_fn()'s wrapper falls through to the original
# top-k when `manager.enabled` is False.
#
# NOTE on the fused path: topk_routing_with_score_function returns EARLY when
# fused=True (TE's fused_topk_with_score_function), which would bypass R3 entirely.
# That path is taken from config.moe_router_fusion, which is False for this run --
# the script asserts it so a future config change cannot silently disable R3.
#
# Idempotent. Run in EVERY node's container, AFTER apply_farskip_training.sh
# (which builds /root/megatron_farskip).
set -euo pipefail

WORK="${MEGATRON_PATH:-/root/megatron_farskip}"
ROUTER="${WORK}/megatron/core/transformer/moe/router.py"
MOE_UTILS="${WORK}/megatron/core/transformer/moe/moe_utils.py"
for f in "$ROUTER" "$MOE_UTILS"; do
    [ -f "$f" ] || { echo "ERROR: $f not found (run apply_farskip_training.sh first)" >&2; exit 1; }
done

python3 - "$ROUTER" "$MOE_UTILS" <<'PY'
import sys

router_path, moe_utils_path = sys.argv[1], sys.argv[2]

# ---------------------------------------------------------------- hook 1: router
# Register one Replay per TopKRouter, at the end of TopKRouter.__init__. The
# anchor is the final statement of that __init__ followed by the next method.
ROUTER_ANCHOR = """            self.global_tokens_per_expert = None
            self.ga_steps = None

    def _maintain_float32_expert_bias(self):"""

ROUTER_PATCHED = """            self.global_tokens_per_expert = None
            self.ga_steps = None

        # miles rollout routing replay (R3): register this router so miles'
        # routing_replay_manager can record the rollout's expert picks and replay
        # them during training. register_to_module early-returns unless the manager
        # is enabled, so this is a no-op without --use-rollout-routing-replay.
        # One Replay is created per router in construction order, which is the
        # global layer order register_replay_list_moe indexes by.
        from miles.utils.replay_base import routing_replay_manager

        routing_replay_manager.register_to_module(self, "routing_replay")

    def _maintain_float32_expert_bias(self):"""

src = open(router_path).read()
if "routing_replay_manager.register_to_module" in src:
    print("  router.py: already patched")
elif ROUTER_ANCHOR in src:
    open(router_path, "w").write(src.replace(ROUTER_ANCHOR, ROUTER_PATCHED, 1))
    print("  router.py: patched (one Replay registered per TopKRouter)")
else:
    sys.exit("  ERROR: router.py anchor not found; TopKRouter.__init__ has changed")

# ------------------------------------------------------------- hook 2: moe_utils
# Route the local compute_topk through the manager. Anchor on the end of the
# existing compute_topk definition.
MOE_ANCHOR = """        else:
            return torch.topk(scores, k=topk, dim=1)

    if score_function == "softmax":"""

MOE_PATCHED = """        else:
            return torch.topk(scores, k=topk, dim=1)

    # miles rollout routing replay (R3): send top-k through the manager so that in
    # replay stages the trainer REUSES the rollout engine's expert picks instead of
    # re-deriving them from its own logits. return_probs=True because this
    # compute_topk returns (values, indices), which the callers below unpack.
    # The wrapper returns the original function unchanged when the manager is
    # disabled, so this costs nothing when R3 is off.
    from miles.utils.replay_base import routing_replay_manager

    compute_topk = routing_replay_manager.get_topk_fn(compute_topk, return_probs=True)

    if score_function == "softmax":"""

src = open(moe_utils_path).read()
if "routing_replay_manager.get_topk_fn" in src:
    print("  moe_utils.py: already patched")
elif MOE_ANCHOR in src:
    open(moe_utils_path, "w").write(src.replace(MOE_ANCHOR, MOE_PATCHED, 1))
    print("  moe_utils.py: patched (compute_topk routed through the manager)")
else:
    sys.exit("  ERROR: moe_utils.py anchor not found; compute_topk has changed")
PY

echo "=== verifying the patched fork imports and the hooks are reachable ==="
WORK="$WORK" PYTHONPATH="${WORK}:${PRIMUS_ROOT:-/workspace/repos/Primus-Instella-Internal/training}:${PYTHONPATH:-}" python3 - <<'PY'
import ast
import os
import sys

# Must come from the environment, not a literal: the heredoc is quoted so the
# shell does not interpolate, and hardcoding the default would patch
# $MEGATRON_PATH while verifying /root/megatron_farskip -- reporting success for
# a tree nobody touched.
work = os.environ["WORK"]
checks = []

src = open(f"{work}/megatron/core/transformer/moe/router.py").read()
checks.append(("router.py parses", lambda: ast.parse(src) is not None))
checks.append(("register_to_module inside TopKRouter.__init__",
               lambda: "routing_replay_manager.register_to_module" in src))

src2 = open(f"{work}/megatron/core/transformer/moe/moe_utils.py").read()
checks.append(("moe_utils.py parses", lambda: ast.parse(src2) is not None))
checks.append(("compute_topk wrapped", lambda: "get_topk_fn(compute_topk" in src2))
# The fused path returns before compute_topk is ever called, so R3 only applies
# when moe_router_fusion is False. Assert the ordering we rely on still holds.
checks.append(("wrap happens before the score_function dispatch",
               lambda: src2.index("get_topk_fn(compute_topk") < src2.index('if score_function == "softmax":')))

ok = True
for name, fn in checks:
    try:
        good = bool(fn())
    except Exception as exc:  # noqa: BLE001
        good = False
        name = f"{name}  ({exc})"
    print(f"  {'OK  ' if good else 'FAIL'} {name}")
    ok &= good
sys.exit(0 if ok else 1)
PY

echo "done. Enable with USE_ROUTING_REPLAY=1 and verify with bringup_verify.sh"

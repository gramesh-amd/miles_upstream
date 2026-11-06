"""Log MoE router statistics during training, from inside the trainer.

Wired in with
    --custom-megatron-before-train-step-hook-path router_log_hook.log_router_stats
miles calls it as
    hook(args, rollout_id, step_id, model, optimizer, opt_param_scheduler)
(miles/backends/megatron_utils/model.py), so it has the live Megatron model.
Enable with ROUTER_LOG=1 in run_async.sh.

WHY NOT THE ROLLOUT LOG HOOK. `--custom-rollout-log-function-path` runs in the
ROLLOUT worker, which holds the SGLang engine, not the trainer -- it cannot see
the training-side router. And miles never calls Megatron's own
`track_moe_metrics`, so `--moe-per-layer-logging` populates a tracker nothing
reads. This hook is the only trainer-side seam.

WHAT IT RECORDS, and why these quantities. MiMo-V2.6 saw expert-load CV go
0.78 -> 2.0 and cold experts 0.5% -> 22% within 20 steps once the router was
trainable. Instella-MoE's own paper S2.8 traced a post-training regression to
load-balancing left on, measuring top-1 expert flip at 54% (vs 5% healthy) and
top-6 Jaccard at 0.42 (vs 0.94).

With ROUTER_OFF=1 (aux loss 0, no load balancing, frozen expert bias) the task
gradient still moves the router weights, and nothing pulls load back into
balance. Token routing is not visible from the weights alone, so this logs the
router weights and expert bias and their drift from step 0; token-level routing
has to be measured offline against saved checkpoints.

SAFETY. Everything is wrapped: a logging hook must never be able to kill a
45-hour run. Rank 0 only, no collectives, no .item() on anything that might be
sharded in a way that needs a sync.
"""

import json
import os
import time

import torch

_REF = {}  # name -> flattened reference weight at first call
_PATH = os.environ.get("ROUTER_LOG_PATH", "router_log.jsonl")
_EVERY = int(os.environ.get("ROUTER_LOG_EVERY", "5"))


def _is_rank0():
    try:
        return (not torch.distributed.is_initialized()) or torch.distributed.get_rank() == 0
    except Exception:  # noqa: BLE001
        return True


def _router_tensors(model):
    """Yield (name, tensor) for router weights and expert biases."""
    mods = model if isinstance(model, (list, tuple)) else [model]
    for m in mods:
        try:
            named = m.named_parameters()
        except Exception:  # noqa: BLE001
            continue
        for name, p in named:
            low = name.lower()
            if "router" not in low and "expert_bias" not in low:
                continue
            # router.weight, router.expert_bias, local_experts routing gates
            if p is None or not torch.is_tensor(p.data):
                continue
            yield name, p.data.detach()


def log_router_stats(args, rollout_id, step_id, model, optimizer, opt_param_scheduler):
    try:
        if not _is_rank0():
            return
        if _EVERY <= 0 or (int(step_id) % _EVERY != 0):
            return

        rec = {"rollout_id": int(rollout_id), "step_id": int(step_id), "t": time.time(), "tensors": {}}
        n_t = 0
        for name, t in _router_tensors(model):
            f = t.float().flatten()
            if f.numel() == 0:
                continue
            n_t += 1
            ref = _REF.get(name)
            entry = {
                "numel": int(f.numel()),
                "l2": float(torch.linalg.vector_norm(f).item()),
                "absmax": float(f.abs().max().item()),
                "mean": float(f.mean().item()),
                "std": float(f.std().item()) if f.numel() > 1 else 0.0,
            }
            if ref is None:
                # Keep a CPU copy of step-0 weights so drift is measurable. Router
                # weights are tiny (num_experts x hidden), so this is cheap.
                _REF[name] = f.cpu().clone()
            else:
                r = ref.to(f.device)
                d = f - r
                entry["drift_l2"] = float(torch.linalg.vector_norm(d).item())
                rn = float(torch.linalg.vector_norm(r).item())
                entry["drift_rel"] = entry["drift_l2"] / rn if rn > 0 else 0.0
                denom = (torch.linalg.vector_norm(f) * torch.linalg.vector_norm(r)).item()
                entry["cos_to_ref"] = float((f @ r).item() / denom) if denom > 0 else 1.0
            rec["tensors"][name] = entry

        rec["n_router_tensors"] = n_t
        if n_t == 0:
            rec["warning"] = (
                "no router/expert_bias parameters matched -- the name " "filter needs updating for this model"
            )
        os.makedirs(os.path.dirname(_PATH) or ".", exist_ok=True)
        with open(_PATH, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
    except Exception as e:  # noqa: BLE001
        # Never let logging kill training.
        try:
            with open(_PATH + ".err", "a") as fh:
                fh.write(f"{time.time()} {type(e).__name__}: {e}\n")
        except Exception:  # noqa: BLE001
            pass

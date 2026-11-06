#!/usr/bin/env python3
"""Ghost-proof verification that a Ray worker really joined the cluster.

`ray status` is not usable for this. On this cluster it listed three nodes under
"Active" for five days while two of them were dead -- their containers had been
SIGKILLed and the GCS entry was never reaped:

    Active:
     1 node_7d1bfbd43b94...   <- real, 8 GPUs
     1 node_717a97c19368...   <- dead since 2026-10-04
     1 node_683a4c4a80db...   <- dead since 2026-10-04
    Total Usage: 0.0/8.0 GPU          <- the giveaway: 3 "active" nodes, 8 GPUs

So a check of the form "ray status shows N nodes" passes on a cluster of ghosts.
This script instead requires all three of the following, which no stale entry can
satisfy together:

  1. IDENTITY  -- the GCS entry for the exact NodeID minted by the raylet we just
                  started is Alive=True. A ghost has a different, older NodeID, so
                  re-joining a node that previously died does not inherit its pass.
  2. RESOURCES -- that NodeID advertises the expected GPU count. Dead nodes keep
                  their autoscaler line but lose their resources (GPU=None above),
                  which is why the totals disagreed with the node count.
  3. EXECUTION -- a CPU-only task pinned to that NodeID with
                  NodeAffinitySchedulingStrategy(soft=False) runs and reports back
                  its own node id and hostname. This is the part a ghost can never
                  fake: it requires a live raylet to lease a worker. soft=False
                  means the task fails rather than silently landing on the head.

Usage:
    bringup_ray_check.py --gcs <head-ip>:6379 --node-id <id> \
                         --expect-ip <node-ip> [--expect-gpus 8] [--timeout 120]
    bringup_ray_check.py --gcs <head-ip>:6379 --list
"""
import argparse
import socket
import sys
import time


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcs", required=True, help="head GCS address, host:port")
    ap.add_argument("--node-id", help="NodeID the worker's own raylet reported")
    ap.add_argument("--expect-ip")
    ap.add_argument("--expect-gpus", type=float, default=8.0)
    ap.add_argument("--timeout", type=float, default=120.0)
    ap.add_argument("--list", action="store_true", help="just dump the node table")
    args = ap.parse_args()

    import ray

    ray.init(address=args.gcs, log_to_driver=False, configure_logging=False)

    # ---------------------------------------------------------------- node table
    deadline = time.time() + args.timeout
    target = None
    while True:
        nodes = ray.nodes()
        target = next((n for n in nodes if n["NodeID"] == args.node_id), None) if args.node_id else None
        if args.list or target is not None or time.time() > deadline:
            break
        time.sleep(3)

    alive = [n for n in nodes if n["Alive"]]
    ghosts = [n for n in nodes if not n["Alive"]]
    print(f"cluster {args.gcs}: {len(alive)} alive, {len(ghosts)} stale/ghost entries")
    for n in sorted(nodes, key=lambda x: (not x["Alive"], x["NodeManagerAddress"])):
        print("  {:<6} {:<16} GPU={:<5} CPU={:<6} {}".format(
            "ALIVE" if n["Alive"] else "GHOST",
            n["NodeManagerAddress"],
            str(n["Resources"].get("GPU")),
            str(n["Resources"].get("CPU")),
            n["NodeID"][:16]))
    total_gpus = sum(n["Resources"].get("GPU", 0) for n in alive)
    print(f"  live GPU total across alive nodes: {total_gpus}")
    if ghosts:
        print("  NOTE: ghost entries above are what `ray status` misreports as Active.")
    if args.list:
        return 0

    if not args.node_id:
        print("ERROR: --node-id is required unless --list", file=sys.stderr)
        return 2

    # ---------------------------------------------------------------- 1. identity
    if target is None:
        print(f"ERROR: NodeID {args.node_id} never appeared in the GCS within "
              f"{args.timeout:.0f}s. The raylet started locally but did not register.",
              file=sys.stderr)
        return 1
    if not target["Alive"]:
        print(f"ERROR: NodeID {args.node_id} is registered but Alive=False.", file=sys.stderr)
        return 1
    print(f"OK   identity: {args.node_id[:16]} registered and Alive=True")

    if args.expect_ip and target["NodeManagerAddress"] != args.expect_ip:
        print(f"ERROR: node registered as {target['NodeManagerAddress']}, expected "
              f"{args.expect_ip}. A wrong --node-ip-address makes cross-node NCCL "
              f"setup pick an unroutable interface.", file=sys.stderr)
        return 1
    print(f"OK   address: advertised as {target['NodeManagerAddress']}")

    # --------------------------------------------------------------- 2. resources
    got = target["Resources"].get("GPU")
    if got != args.expect_gpus:
        print(f"ERROR: node advertises GPU={got}, expected {args.expect_gpus}. "
              f"A worker that joined without --num-gpus, or with HIP_VISIBLE_DEVICES "
              f"set, under-reports here and the trainer will silently place fewer "
              f"ranks than you asked for.", file=sys.stderr)
        return 1
    print(f"OK   resources: GPU={got}, CPU={target['Resources'].get('CPU')}")

    # --------------------------------------------------------------- 3. execution
    from ray.util.scheduling_strategies import NodeAffinitySchedulingStrategy

    @ray.remote(num_cpus=1, num_gpus=0)
    def probe():
        import os
        return (socket.gethostname(), os.getpid(),
                ray.get_runtime_context().get_node_id())

    strategy = NodeAffinitySchedulingStrategy(node_id=args.node_id, soft=False)
    try:
        host, pid, seen = ray.get(
            probe.options(scheduling_strategy=strategy).remote(), timeout=args.timeout)
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: could not execute a task on {args.node_id[:16]}: "
              f"{type(exc).__name__}: {exc}\n"
              f"       The GCS entry exists but no worker can be leased -- this is "
              f"exactly the ghost case.", file=sys.stderr)
        return 1
    if seen != args.node_id:
        print(f"ERROR: probe ran on {seen[:16]}, not {args.node_id[:16]}", file=sys.stderr)
        return 1
    print(f"OK   execution: task ran on {host} (pid {pid}) on node {seen[:16]}")

    print(f"\nRAY_JOIN_VERIFIED {args.expect_ip or target['NodeManagerAddress']} "
          f"node={args.node_id[:16]} gpus={got}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

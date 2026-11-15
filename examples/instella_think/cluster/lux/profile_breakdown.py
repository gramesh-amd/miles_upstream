#!/usr/bin/env python3
"""GPU time split by kernel category from torch-profiler traces (trainer and sglang).

    profile_breakdown.py [--top 25] <trace file or dir> ...

Reads Chrome traces (*.json.gz / *.json: miles TrainProfiler, sglang /start_profile), groups them by
kind (train / EXTEND / DECODE, from the file name), and prints markdown:
- per category: kernel time and its share, summed over the GPUs of the group;
- per GPU, busy time (union of kernel intervals) next to summed kernel time: summed > busy means
  overlapping kernels or the ROCm 7.2 profiler's inflated durations (rocm-systems#13882), so the
  shares are only as good as that ratio is close to 1;
- exposed communication: collective time not overlapped by any compute kernel;
- for trainer traces, the split by miles phase (miles/<timer> ranges: log_probs, actor_train,
  update_weights, ...), each kernel counted in the innermost range containing its start;
- the top kernels and the top unclassified ones (to extend CATEGORIES against real names).
GEMM kernels launched from an op whose name looks like MoE (grouped/expert/moe) count as MoE.
"""
import argparse, collections, glob, gzip, json, os, re

CATEGORIES = [  # first match wins; case-insensitive, on the kernel name
    ("communication", r"nccl|rccl|cross_device_reduce|all_?reduce|all_?gather|reduce_?scatter|all_?to_?all|alltoall|mscclpp|mori"),
    ("attention", r"fmha|flash|attn|attention|_fwd_kernel|_bwd_kernel|_fwd_grouped_kernel|kernel_stage[12]|mla|paged|decode_kernel|extend_kernel"),
    ("rope", r"rope|rotary"),
    ("moe", r"moe|fmoe|expert|topk|top_k|permute|sorting|align_block|grouped_gemm|gmm|routing"),
    ("gemm", r"^cijk_|hipblaslt|tensile|gemm|matmul|wvsplitk|_mm_|_mm\b|dot_|bmm|ck_tile.*gemm|addmm"),
    ("norm", r"norm|rms"),
    ("softmax/loss", r"softmax|log_prob|logsumexp|cross_entropy|entropy"),
    ("memory", r"memcpy|memset|copy|fill|cat_|concat|index|gather|scatter|transpose|contiguous"),
    ("elementwise/other", r"elementwise|vectorized|reduce|act_and_mul|silu|gelu|mul|add|sigmoid|triton_"),
]
CAT_RE = [(c, re.compile(p, re.I)) for c, p in CATEGORIES]
MOE_OP = re.compile(r"grouped|expert|moe|gmm|fused_experts", re.I)
COMPUTE = {"attention", "rope", "moe", "gemm", "norm", "softmax/loss", "elementwise/other"}


def classify(name, op):
    for c, r in CAT_RE:
        if r.search(name):
            if c == "gemm" and op and MOE_OP.search(op):
                return "moe"
            return c
    return "unclassified"


def kind_of(path):
    b = os.path.basename(path)
    for k in ("EXTEND", "DECODE"):
        if f"-{k}" in b:
            return k
    return "train" if "train" in b else "other"


def union(iv):
    iv = sorted(iv)
    out = []
    for s, e in iv:
        if out and s <= out[-1][1]:
            out[-1][1] = max(out[-1][1], e)
        else:
            out.append([s, e])
    return out


def length(iv):
    return sum(e - s for s, e in iv)


def intersect(a, b):  # both unions (sorted, disjoint)
    i = j = 0
    out = 0.0
    while i < len(a) and j < len(b):
        s, e = max(a[i][0], b[j][0]), min(a[i][1], b[j][1])
        if s < e:
            out += e - s
        if a[i][1] < b[j][1]:
            i += 1
        else:
            j += 1
    return out


def load(path):
    op = gzip.open if path.endswith(".gz") else open
    with op(path, "rt") as f:
        return json.load(f).get("traceEvents", [])


def analyse(path):
    ev = load(path)
    ops = {}
    for e in ev:
        if e.get("ph") == "X" and e.get("cat") == "cpu_op":
            x = (e.get("args") or {}).get("External id")
            if x is not None:
                ops[x] = e.get("name", "")
    ranges = [(e["ts"], e["ts"] + e["dur"], e["name"], e["cat"]) for e in ev
              if e.get("ph") == "X" and e.get("cat") in ("gpu_user_annotation", "user_annotation")
              and str(e.get("name", "")).startswith("miles/")]
    # GPU-side copies of the ranges share the kernels' clock; CPU ranges only as a fallback.
    gpu_ranges = [(s, t, n) for s, t, n, c in ranges if c == "gpu_user_annotation"] or [(s, t, n) for s, t, n, _ in ranges]
    kern = collections.defaultdict(list)  # device -> [(ts, end, name, cat, phase)]
    for e in ev:
        if e.get("ph") != "X" or e.get("cat") != "kernel":
            continue
        a = e.get("args") or {}
        name = e.get("name", "")
        cat = classify(name, ops.get(a.get("External id")))
        ts, end = e["ts"], e["ts"] + e.get("dur", 0)
        phase = None
        best = None
        for s, t, n in gpu_ranges:
            if s <= ts < t and (best is None or t - s < best):
                best, phase = t - s, n
        kern[a.get("device", 0)].append((ts, end, name, cat, phase))
    return kern


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--top", type=int, default=25)
    a = ap.parse_args()
    files = []
    for p in a.paths:
        files += sorted(glob.glob(os.path.join(p, "**", "*.json*"), recursive=True)) if os.path.isdir(p) else [p]
    files = [f for f in files if f.endswith((".json", ".json.gz")) and "trace" in os.path.basename(f)]
    groups = collections.defaultdict(list)
    for f in files:
        groups[kind_of(f)].append(f)
    for kind, fs in sorted(groups.items()):
        cat_t = collections.Counter(); name_t = collections.Counter(); name_n = collections.Counter(); name_c = {}
        unc = collections.Counter(); phase_cat = collections.defaultdict(collections.Counter)
        gpus = []
        for f in fs:
            for dev, ks in analyse(f).items():
                if not ks:
                    continue
                busy = union([(s, e) for s, e, *_ in ks])
                comm = union([(s, e) for s, e, _, c, _ in ks if c == "communication"])
                comp = union([(s, e) for s, e, _, c, _ in ks if c in COMPUTE])
                total = sum(e - s for s, e, *_ in ks)
                wall = max(e for _, e, *_ in ks) - min(s for s, *_ in ks)
                gpus.append((os.path.basename(f)[:60], dev, wall, length(busy), total, length(comm),
                             length(comm) - intersect(comm, comp)))
                for s, e, n, c, ph in ks:
                    d = e - s
                    cat_t[c] += d; name_t[n] += d; name_n[n] += 1; name_c.setdefault(n, c)
                    phase_cat[ph or "(no miles range)"][c] += d
                    if c == "unclassified":
                        unc[n] += d
        tot = sum(cat_t.values()) or 1
        print(f"\n## {kind}: {len(fs)} trace files, {len(gpus)} GPU timelines\n")
        print("| category | kernel time (s, all GPUs) | share |\n|---|---|---|")
        for c, t in cat_t.most_common():
            print(f"| {c} | {t / 1e6:.2f} | {100 * t / tot:.1f}% |")
        W = sum(g[2] for g in gpus); B = sum(g[3] for g in gpus); T = sum(g[4] for g in gpus)
        C = sum(g[5] for g in gpus); X = sum(g[6] for g in gpus)
        print(f"\nPer-GPU totals (summed over GPUs): wall {W / 1e6:.2f} s, busy {B / 1e6:.2f} s "
              f"({100 * B / max(W, 1):.0f}% of wall), summed kernel time {T / 1e6:.2f} s "
              f"(sum/busy = {T / max(B, 1):.2f}; >1.1 means overlap or inflated durations), "
              f"communication {C / 1e6:.2f} s of which exposed {X / 1e6:.2f} s ({100 * X / max(B, 1):.1f}% of busy).")
        if len(phase_cat) > 1:
            print("\n| miles phase | kernel time (s) | top categories |\n|---|---|---|")
            for ph, cc in sorted(phase_cat.items(), key=lambda x: -sum(x[1].values())):
                t = sum(cc.values())
                tops = ", ".join(f"{c} {100 * v / t:.0f}%" for c, v in cc.most_common(4))
                print(f"| {ph} | {t / 1e6:.2f} | {tops} |")
        print(f"\nTop {a.top} kernels:\n\n| kernel | category | calls | time (s) | share |\n|---|---|---|---|---|")
        for n, t in name_t.most_common(a.top):
            print(f"| `{n[:90]}` | {name_c[n]} | {name_n[n]} | {t / 1e6:.2f} | {100 * t / tot:.1f}% |")
        if unc:
            print(f"\nUnclassified ({100 * sum(unc.values()) / tot:.1f}% of kernel time), top 10:")
            for n, t in unc.most_common(10):
                print(f"- `{n[:120]}` {t / 1e6:.3f} s")


if __name__ == "__main__":
    main()

#!/bin/bash
# Bring one GPU node into the Instella RL Ray cluster, idempotently and loudly.
#
#   bringup_node.sh <host> [--check-only] [--teardown [--remove]] [--allow-busy]
#
# Run this from a host that can ssh to <host> (not from inside a container -- the
# containers have no ssh client or keys). The miles checkout holding this script must
# sit on a share every node mounts: the container bind-mounts it at /root/miles and
# runs the patch scripts from there (KIT below).
#
# Required env:
#   SHARE_HOST    host path of the shared work dir; mounted at /workspace in the container
#                 (checkpoints, data, hf_home, repos/Primus-Instella-Internal)
#   RAY_HEAD_IP   routable IP of the head node (with AS_HEAD=1, this node's own IP)
# Optional env:
#   MILES_HOST    host path of this miles checkout (default $SHARE_HOST/miles)
#   EXPECT_NET_IF interface NCCL is pinned to (default eno0, the default route on the
#                 MI325X pod this was written for)
#   CTR, IMAGE, NUM_GPUS, PII_ROOT, RAY_GCS_PORT, RAY_DASHBOARD_PORT
#
# DESIGN RULE: a half-provisioned node that joins the cluster is worse than one
# that fails to join, because the failure then surfaces hours later as a NCCL
# timeout or a silent accuracy regression on one eighth of the ranks. Every phase
# therefore ends in a check, and the first failed check exits non-zero before the
# next phase runs. Nothing is "best effort".
#
# Modes:
#   (default)     provision: container -> patches -> verify -> ray join -> verify
#   --check-only  verify only; makes NO changes anywhere. Safe on a live node.
#   --teardown    ray stop (so the node does not become a ghost) then stop the
#                 container. --remove also deletes it.
#   --allow-busy  skip the "someone else is using these GPUs" guard. See phase 1.
set -euo pipefail

# ------------------------------------------------------------------- parameters
CTR="${CTR:-instella_rl}"
IMAGE="${IMAGE:-rocm/sgl-dev:miles-rocm10-mi30x-20260930}"
RAY_HEAD_IP="${RAY_HEAD_IP:-}"
RAY_GCS_PORT="${RAY_GCS_PORT:-6379}"
RAY_DASHBOARD_PORT="${RAY_DASHBOARD_PORT:-8265}"
NUM_GPUS="${NUM_GPUS:-8}"
SHARE_HOST="${SHARE_HOST:-}"
EXPECT_NET_IF="${EXPECT_NET_IF:-eno0}"
# In-container paths: the share is at /workspace and this miles checkout at /root/miles.
PII_ROOT="${PII_ROOT:-/workspace/repos/Primus-Instella-Internal}"
KIT="/root/miles/examples/instella_think/mi325"
SSH="ssh -n -o BatchMode=yes -o ConnectTimeout=20"
# GPUs busier than this, or holding more VRAM than this, mean the node is in use.
BUSY_UTIL_PCT="${BUSY_UTIL_PCT:-10}"
BUSY_VRAM_PCT="${BUSY_VRAM_PCT:-5}"

MODE=provision
REMOVE=0
ALLOW_BUSY=0
HOST=""
while [ $# -gt 0 ]; do
    case "$1" in
        --check-only) MODE=check ;;
        --teardown)   MODE=teardown ;;
        --remove)     REMOVE=1 ;;
        --allow-busy) ALLOW_BUSY=1 ;;
        -h|--help)    sed -n '2,36p' "$0"; exit 0 ;;
        -*)           echo "unknown flag $1" >&2; exit 2 ;;
        *)            HOST="$1" ;;
    esac
    shift
done
[ -n "$HOST" ] || { echo "usage: $0 <host> [--check-only] [--teardown] [--allow-busy]" >&2; exit 2; }
[ -n "$SHARE_HOST" ] || { echo "set SHARE_HOST to the shared work dir every node mounts (see --help)" >&2; exit 2; }
[ -n "$RAY_HEAD_IP" ] || { echo "set RAY_HEAD_IP to the head node IP; with AS_HEAD=1, this node's own IP (see --help)" >&2; exit 2; }
MILES_HOST="${MILES_HOST:-${SHARE_HOST}/miles}"

# ----------------------------------------------------------------------- output
step() { printf '\n\033[1;36m== %s\033[0m\n' "$*"; }
ok()   { printf '   \033[32mok\033[0m   %s\n' "$*"; }
info() { printf '   ..   %s\n' "$*"; }
warn() { printf '   \033[1;33mWARN\033[0m %s\n' "$*" >&2; }
die()  { printf '\n\033[1;31mFATAL on %s: %s\033[0m\n' "$HOST" "$*" >&2; exit 1; }

rexec() { $SSH "$HOST" "$1"; }                                     # on the host
cexec() { $SSH "$HOST" "docker exec $CTR bash -lc $(printf '%q' "$1")"; }  # in the container
hexec() { $SSH "$RAY_HEAD_IP" "docker exec $CTR bash -lc $(printf '%q' "$1")"; }  # in the head's container

# `docker inspect` on a missing container exits non-zero but still emits a blank
# line on stdout, so the obvious `$(... || echo absent)` yields "\nabsent" and no
# case arm matches. Normalise instead of pattern-matching the noise.
ctr_state() {
    local s
    s="$(rexec "docker inspect -f '{{.State.Status}}' $CTR 2>/dev/null || true" | tr -d '[:space:]')"
    printf '%s' "${s:-absent}"
}

# Run a command, indent its output, and fail on ITS status. Piping straight into
# sed would report sed's status instead, so every non-zero exit would be swallowed
# -- which in this script means continuing past a failed patch.
run_show() { # run_show <runner> <label> <command>
    local runner="$1" label="$2" cmd="$3" out rc
    info "$label"
    set +e; out="$("$runner" "$cmd" 2>&1)"; rc=$?; set -e
    [ -n "$out" ] && printf '%s\n' "$out" | sed 's/^/        /'
    return $rc
}

echo "bringup_node.sh  host=$HOST  mode=$MODE  container=$CTR"
echo "                 head=${RAY_HEAD_IP}:${RAY_GCS_PORT}  gpus=${NUM_GPUS}"

# ============================================================ 0. host preflight
step "0. host preflight"
rexec true 2>/dev/null || die "cannot ssh to $HOST"
HOSTNAME_REMOTE="$(rexec 'hostname')"
ok "ssh ok, hostname=$HOSTNAME_REMOTE"

# The Ray node IP is NOT `hostname -i` (that is 127.0.1.1 here, via /etc/hosts) and
# NOT the name you ssh to. The containers run --network host, so Ray binds whatever
# the host's default route uses (eno0 on every node of the pod this was written for),
# which is why Ray reports addresses that look nothing like the hostnames you ssh to.
# Derive it on the node, never guess.
read -r NODE_IF NODE_IP <<<"$(rexec 'ip -o -4 route get 1 | grep -oP "(?<=dev )\S+|(?<=src )\S+" | tr "\n" " "')"
[ -n "${NODE_IP:-}" ] || die "could not determine the default-route source address"
ok "ray node ip = $NODE_IP (via $NODE_IF)"
# NCCL_SOCKET_IFNAME is pinned to this interface. Left to auto-select, NCCL also
# binds docker0/br-* and cross-node collectives stall.
[ "$NODE_IF" = "$EXPECT_NET_IF" ] || die "default route is on '$NODE_IF', but NCCL_SOCKET_IFNAME is expected to be $EXPECT_NET_IF; refusing to create a container that would mispin NCCL (set EXPECT_NET_IF if this pod differs)"

rexec "test -e /dev/kfd && test -e /dev/dri" || die "/dev/kfd or /dev/dri missing -- no ROCm on this node"
ok "/dev/kfd and /dev/dri present"
rexec "test -d $SHARE_HOST" || die "$SHARE_HOST not mounted on $HOST (the /workspace bind would be empty)"
ok "share $SHARE_HOST visible"
rexec "test -f $MILES_HOST/examples/instella_think/mi325/bringup_verify.sh" \
    || die "$MILES_HOST on $HOST is not a miles checkout with examples/instella_think/mi325 (set MILES_HOST)"
ok "miles checkout $MILES_HOST visible"
rexec "docker image inspect $IMAGE >/dev/null 2>&1" || die "image $IMAGE not present; pull it first"
ok "image $IMAGE present"

# ==================================================== 1. is this node actually free
# These nodes sit in another group's Slurm reservation and we use them only while
# idle. Slurm is NOT a usable signal here: it reported State=IDLE+RESERVED,
# CPUAlloc=0 on nodes that were in fact running an 8-GPU job, because that job was
# started as a bare docker container outside Slurm. Ask the GPUs instead.
if [ "$MODE" != "teardown" ]; then
step "1. node-is-idle guard"
# THREE samples, take the MINIMUM of the per-sample maxima. A single
# `rocm-smi --showuse` read is an instantaneous counter that spikes for
# unrelated reasons -- querying the device is itself enough to register
# activity. One sample rejected eight genuinely idle nodes here (reported 100%,
# with "foreign containers: none"; five follow-up samples read 31%, 0, 0, 0, 0
# and rocm-smi --showpids listed no KFD processes at all). A real workload holds
# the GPUs across all three samples; a transient does not.
MAXV=999
for _probe in 1 2 3; do
    _b="$(rexec "rocm-smi --showuse --showmemuse 2>/dev/null | grep -oP '(?<=GPU use \(%\): )\d+|(?<=\(VRAM%\): )\d+' | tr '\n' ' '")"
    _m=0; for v in $_b; do [ "$v" -gt "$_m" ] && _m="$v"; done
    [ "$_m" -lt "$MAXV" ] && MAXV="$_m"
    [ "$MAXV" -le "$BUSY_UTIL_PCT" ] && break   # already clearly idle, stop probing
    sleep 3
done
# KFD process list, but only processes actually HOLDING MEMORY. The columns are
# PID / PROCESS NAME / GPU(s) / VRAM USED / SDMA USED / CU OCCUPANCY, tab
# separated. Merely having the device open is not enough to call a node busy:
# idle nodes here routinely show several python processes with VRAM USED = 0,
# and gating on their mere presence rejected a node whose eight GPUs were
# provably free. A workload that would actually contend with ours holds VRAM.
KFD="$(rexec "rocm-smi --showpids 2>/dev/null | awk -F'\t' '\$1 ~ /^[0-9]+\$/ && \$4+0 > 0 {print \$1\"(\"\$4\"MB)\"}' | tr '\n' ' '" || true)"
# device-metrics-exporter and fleet-* are AMD's monitoring sidecars: they run on
# most nodes, hold no GPU memory and do no compute, so counting them as another
# workload refuses perfectly idle nodes (one was rejected twice on exactly this
# while reading 0% GPU, zero VRAM holders and Slurm IDLE). Anything else still
# counts as foreign.
FOREIGN="$(rexec "docker ps --format '{{.Names}}' | grep -v '^${CTR}\$' | grep -vE '^(device-metrics-exporter|fleet-)' | tr '\n' ' '" || true)"
info "min-of-3 max GPU util/VRAM across all 8 GPUs: ${MAXV}%"
info "KFD processes holding GPUs: ${KFD:-<none>}"
info "other running containers: ${FOREIGN:-<none>}"
if [ "$MAXV" -gt "$BUSY_UTIL_PCT" ] || [ -n "${FOREIGN// /}" ] || [ -n "${KFD// /}" ]; then
    rexec "scontrol show node $HOSTNAME_REMOTE 2>/dev/null | grep -E 'State=|ReservationName' | sed 's/^/        slurm: /'" || true
    if [ "$ALLOW_BUSY" = 1 ] || [ "$MODE" = check ]; then
        # --check-only changes nothing, so report and keep going.
        printf '   \033[33mwarn\033[0m node looks BUSY (continuing: %s)\n' \
            "$([ "$MODE" = check ] && echo '--check-only makes no changes' || echo '--allow-busy given')"
    else
        die "node is IN USE (max GPU/VRAM ${MAXV}%, foreign containers: ${FOREIGN:-none}).
       Joining it to Ray would advertise ${NUM_GPUS} GPUs that are not actually free: the
       trainer would place ranks here and contend with, or OOM against, the other
       workload. Note Slurm reports these nodes IDLE regardless (see above) because
       the other job runs outside Slurm, so Slurm cannot be used as the guard.
       Re-run with --allow-busy only if you know the other containers are idle."
    fi
else
    ok "node is idle"
fi
fi

# ============================================================== teardown and exit
if [ "$MODE" = "teardown" ]; then
    step "teardown"
    STATE="$(ctr_state)"
    if [ "$STATE" = "running" ]; then
        # ORDER MATTERS. `ray stop` deregisters the raylet from the GCS so the head
        # marks the node dead immediately. Killing the container first (what a
        # reclaimed Slurm allocation does with SIGKILL -> "Exited (137)") leaves the
        # GCS entry behind, and `ray status` then lists the node under Active for
        # days. Those are the ghosts this cluster already has two of.
        info "ray stop --force (deregisters the raylet so it does not become a ghost)"
        cexec 'ray stop --force' 2>&1 | sed 's/^/        /' || true
        sleep 3
        info "docker stop $CTR"
        rexec "docker stop -t 30 $CTR" >/dev/null && ok "container stopped"
    else
        ok "container is $STATE, nothing to stop"
    fi
    if [ "$REMOVE" = 1 ]; then
        rexec "docker rm -f $CTR" >/dev/null 2>&1 && ok "container removed" || true
        info "NOTE: removing the container discards /sgl-workspace/sglang and"
        info "      /root/megatron_farskip, so the next bringup re-runs all patches"
        info "      (~minutes). Prefer plain --teardown if you expect the node back."
    fi
    step "verify the head no longer counts this node"
    run_show hexec "cluster node table as the head sees it" \
        "python3 $KIT/bringup_ray_check.py --gcs ${RAY_HEAD_IP}:${RAY_GCS_PORT} --list" || true
    echo
    ok "teardown complete for $HOST ($NODE_IP)"
    exit 0
fi

# ============================================================== 2. container
step "2. container"
STATE="$(ctr_state)"
info "current state: $STATE"

# The docker run below reproduces the known-good reference container exactly.
# Choices that are not obvious, and why:
#   --privileged is NOT set. The reference runs privileged=false; --device=/dev/kfd
#     --device=/dev/dri --group-add video is the whole ROCm requirement. Least
#     privilege, and it is what the known-good node uses.
#   --group-add video gives access to the /dev/dri render nodes. Without it HIP
#     enumerates zero agents for a non-root user and Ray reports GPU=0.
#   --cap-add SYS_PTRACE and --security-opt seccomp=unconfined / label=disable match
#     the reference; rocgdb and the ROCm profilers need them, and seccomp=unconfined
#     is standard for HSA.
#   HIP_VISIBLE_DEVICES is deliberately NOT set. miles_docker_run.sh passes it
#     because it was written for per-GPU-subset eval jobs; the live reference
#     container does not have it in its environment at all. On a training node it
#     would be actively harmful -- Ray's accelerator probe honours it, so the node
#     would advertise fewer than 8 GPUs and the trainer would place fewer ranks
#     than requested with no error.
#   -e NCCL_SOCKET_IFNAME, HF_HOME, HF_HUB_OFFLINE are the only three variables
#     the reference passes at run time; everything else in its environment is baked
#     into the image (verified by diffing the container env against the image env).
#   -v binds the SHARED dirs, never a node-local path like /var/tmp, which would give
#     each node a different, mostly empty /workspace.
#   sleep infinity is the reference container's actual command
#     (docker inspect --format '{{.Path}} {{.Args}}' -> "sleep [infinity]"). The
#     container is a persistent shell target; work is started with docker exec.
read -r -d '' DOCKER_RUN <<EOF || true
docker run -d --name $CTR \
  --network host --ipc=host --shm-size 32g \
  --device=/dev/kfd --device=/dev/dri --group-add video \
  --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined --security-opt label=disable \
  -e HF_HOME=/workspace/hf_home \
  -e HF_HUB_OFFLINE=1 \
  -e NCCL_SOCKET_IFNAME=$NODE_IF \
  -v $SHARE_HOST:/workspace \
  -v $MILES_HOST:/root/miles \
  $IMAGE \
  sleep infinity
EOF

case "$STATE" in
    running) ok "already running" ;;
    exited|created|paused)
        [ "$MODE" = check ] && die "container is '$STATE', not running (--check-only makes no changes)"
        info "starting existing container (keeps its patched /sgl-workspace and /root/megatron_farskip)"
        rexec "docker start $CTR" >/dev/null || die "docker start $CTR failed"
        ok "started" ;;
    absent)
        [ "$MODE" = check ] && die "no container named $CTR on this node (--check-only makes no changes)"
        info "creating container"
        echo "$DOCKER_RUN" | sed 's/^/        /'
        rexec "$DOCKER_RUN" >/dev/null || die "docker run failed"
        ok "created" ;;
    *) die "unexpected container state '$STATE'" ;;
esac
rexec "docker inspect -f '{{.State.Running}}' $CTR" | grep -q true || die "container is not running after start"

# A container that was created earlier with the wrong flags will start happily and
# then fail much later, so compare it against the reference instead of trusting it.
step "3. container config matches the reference"
chk() { # chk <label> <docker-inspect-format> <expected>
    local got; got="$(rexec "docker inspect -f '$2' $CTR")"
    [ "$got" = "$3" ] || die "container $1 is '$got', expected '$3'.
       This container was created with different flags. Fix with:
         ssh -n $HOST 'docker rm -f $CTR' && $0 $HOST"
    ok "$1 = $got"
}
chk "image"       '{{.Config.Image}}'              "$IMAGE"
chk "network"     '{{.HostConfig.NetworkMode}}'    "host"
chk "ipc"         '{{.HostConfig.IpcMode}}'        "host"
chk "shm-size"    '{{.HostConfig.ShmSize}}'        "34359738368"
chk "privileged"  '{{.HostConfig.Privileged}}'     "false"
chk "group-add"   '{{.HostConfig.GroupAdd}}'       "[video]"
chk "command"     '{{.Path}} {{.Args}}'            "sleep [infinity]"
for m in "$SHARE_HOST:/workspace" "$MILES_HOST:/root/miles"; do
    rexec "docker inspect -f '{{range .Mounts}}{{.Source}}:{{.Destination}} {{end}}' $CTR" \
        | grep -qF "$m" || die "missing bind mount $m"
    ok "mount $m"
done
rexec "docker inspect -f '{{range .HostConfig.Devices}}{{.PathOnHost}} {{end}}' $CTR" \
    | grep -q '/dev/kfd' || die "/dev/kfd not passed through"
ok "devices /dev/kfd /dev/dri"
# HIP_VISIBLE_DEVICES must be absent, or Ray under-reports the GPU count.
if rexec "docker inspect -f '{{range .Config.Env}}{{println .}}{{end}}' $CTR" | grep -q '^HIP_VISIBLE_DEVICES='; then
    die "HIP_VISIBLE_DEVICES is set in this container; a training node must see all $NUM_GPUS GPUs"
fi
ok "HIP_VISIBLE_DEVICES unset (node will advertise all $NUM_GPUS GPUs)"
# Same image id as the head, or the patches and the Ray version can differ.
HEAD_IMG="$($SSH "$RAY_HEAD_IP" "docker inspect -f '{{.Image}}' $CTR")"
NODE_IMG="$(rexec "docker inspect -f '{{.Image}}' $CTR")"
# .Image is NOT portable across docker image stores. Nodes on the containerd
# snapshotter report the MANIFEST digest; nodes on classic overlay2 report the
# CONFIG digest. The same image therefore compares unequal between a containerd
# node (104d848a) and an overlay2 node (29ed1db6), which is a pure
# false positive -- both resolve to RepoDigest 104d848a, created at the same
# instant, with the same 87 layers. Compare the content instead: the repo digest
# if the image came from a registry, else the rootfs layer chain, which is what
# actually determines whether the patches and Ray versions match.
img_fingerprint () {  # $1 = host-exec function name is not portable, so pass cmd
    $1 "docker inspect -f '{{range .RepoDigests}}{{.}}{{end}}|{{range .RootFS.Layers}}{{.}}{{end}}' \
        \$(docker inspect -f '{{.Image}}' $CTR)" 2>/dev/null | tr -d ' \r'
}
HEAD_FP="$(img_fingerprint "$SSH $RAY_HEAD_IP")"
NODE_FP="$(rexec "docker inspect -f '{{range .RepoDigests}}{{.}}{{end}}|{{range .RootFS.Layers}}{{.}}{{end}}' \
    \$(docker inspect -f '{{.Image}}' $CTR)" 2>/dev/null | tr -d ' \r')"
if [ -n "$HEAD_FP" ] && [ "$HEAD_FP" = "$NODE_FP" ]; then
    ok "image content identical to the head's (repo digest + layer chain match)"
elif [ "$HEAD_IMG" != "$NODE_IMG" ]; then
    # THREE different digests currently share the tag miles-rocm10-mi30x-20260930:
    # the head is on sglang 57e7d0e848 (dev65) while every fresh node in this
    # fleet is on 6e25d1bd57 (dev66). 57e7d0e848 is an ANCESTOR of 6e25d1bd57;
    # the single intervening commit ("fix: initialize Inkling safely on ROCm",
    # #37294) touches only swa_memory_pool.py and inkling_common/attn.py, neither
    # reachable from InstellaMoE, and all 17 FarSkip patches apply cleanly to
    # both trees. So this is benign HERE, but it is still a real version skew and
    # the next tag push could make it not benign -- hence opt-in, not automatic.
    echo "    head: $HEAD_IMG"
    echo "    node: $NODE_IMG"
    [ "${ALLOW_IMAGE_MISMATCH:-0}" = "1" ] || die "image id differs from the head's; Ray refuses mismatched versions and the patches are built for one tree.
       Re-pull the head's digest, or set ALLOW_IMAGE_MISMATCH=1 after confirming the
       sglang commits are ancestor-related and phase 5 passes on this node."
    warn "image id differs from the head's, continuing because ALLOW_IMAGE_MISMATCH=1"
else
    ok "image id identical to the head's container"
fi

# ================================================================== 4. patches
# All five are container-local work. /workspace and /root/miles come from the share
# and are already identical everywhere; /sgl-workspace/sglang and
# /root/megatron_farskip live in this container's overlay and start unpatched.
if [ "$MODE" = provision ]; then
step "4. patches"
# IMPORTANT: there are FIVE steps, not three. /root/megatron_farskip does not exist
# in the image (verified: `docker run --rm $IMAGE ls /root` lists only
# Megatron-LM, go, miles). It is built by apply_farskip_training.sh, and both
# patch_megatron_r3.sh and patch_megatron_mcore_data.sh exit 1 without it. Running
# only the three "patch" scripts on a fresh node leaves the node with no FarSkip
# Megatron at all.
run_patch() { # run_patch <label> <command>
    run_show cexec "$1" "$2" || die "$1 exited non-zero"
    ok "$1 returned 0 (post-conditions re-checked independently in phase 5)"
}
# Idempotent: apply_farskip_training.sh does `rm -rf /root/megatron_farskip` and
# rebuilds, which would throw away the R3 and mcore_data patches. Only run it when
# the tree is actually missing or not the one python resolves.
if cexec "test -d /root/megatron_farskip/megatron/core && PYTHONPATH=$PII_ROOT/training python3 -c 'import megatron.core as m,sys; sys.exit(0 if m.__file__.startswith(\"/root/megatron_farskip\") else 1)'" 2>/dev/null; then
    ok "megatron_farskip already built and resolving (skipping apply_farskip_training.sh)"
else
    run_patch "apply_farskip_training.sh (builds /root/megatron_farskip; several minutes)" \
        "PII_ROOT=$PII_ROOT bash $KIT/apply_farskip_training.sh"
fi
run_patch "apply_farskip.sh (sglang rollout side)" \
    "bash $KIT/sglang_port/apply_farskip.sh"
# Must follow apply_farskip.sh: it edits extend_attention.py, which that script
# does not touch, and its anchor is the stock `if Lq == 576:` block. Without it a
# node dies at the first absorbed prefill -- Instella's Lq = kv_lora_rank +
# qk_rope_head_dim = 512 + 32 = 544 is not in the stock whitelist, so BLOCK_DMODEL
# pads to next_power_of_2(544) = 1024 and the query tile needs
# 64 * 1024 * 2 = 131072 B against gfx942's 65536 B of LDS:
#   OutOfResources: Required: 131072, Hardware limit: 65536
run_patch "patch_triton_mla_lds.sh (gfx942 LDS fix for Instella's Lq=544)" \
    "bash $KIT/patch_triton_mla_lds.sh"
run_patch "patch_megatron_r3.sh (routing replay hooks)" \
    "bash $KIT/patch_megatron_r3.sh"
run_patch "patch_megatron_mcore_data.sh (resume tolerance)" \
    "bash $KIT/patch_megatron_mcore_data.sh"
fi

# ==================================================== 5. independent verification
step "5. independent verification of every post-condition"
run_show cexec "greps and runtime imports, NOT the patch scripts' self-report" \
    "PII_ROOT=$PII_ROOT bash $KIT/bringup_verify.sh" \
    || die "post-condition verification failed (see above). NOT joining Ray: a node that
       joins with an unpatched sglang or Megatron fails hours later as a NCCL
       timeout or a silent accuracy regression on 1/N of the ranks."
ok "all post-conditions verified"

# ================================================================= 6. ray join
step "6. ray"
# --head bootstraps a NEW cluster on this node instead of joining an existing
# one. Needed because running several independent experiments at once means
# several independent Ray clusters, and the worker path below hard-requires a
# head that is already up.
if [ "${AS_HEAD:-0}" = "1" ]; then
    if [ "$(cexec 'ray status >/dev/null 2>&1 && echo yes || echo no')" = yes ]; then
        ok "a raylet is already running here; not restarting the head"
    else
        [ "$MODE" = check ] && die "no raylet running here (--check-only makes no changes)"
        run_show cexec "starting head" \
            "ray start --head --node-ip-address=${NODE_IP} --port=${RAY_GCS_PORT} --dashboard-host=0.0.0.0 --dashboard-port=${RAY_DASHBOARD_PORT} --num-gpus ${NUM_GPUS} --disable-usage-stats" \
            || die "ray start --head failed"
        sleep 8
    fi
    RAY_HEAD_IP="${NODE_IP}"
    hexec() { $SSH "$RAY_HEAD_IP" "docker exec $CTR bash -lc $(printf '%q' "$1")"; }
fi
HEAD_ALIVE="$($SSH "$RAY_HEAD_IP" "docker exec $CTR bash -lc 'ray status >/dev/null 2>&1 && echo yes || echo no'")"
[ "$HEAD_ALIVE" = yes ] || die "the Ray head on $RAY_HEAD_IP is not responding; start it before adding workers"
ok "head ${RAY_HEAD_IP}:${RAY_GCS_PORT} is up (dashboard ${RAY_DASHBOARD_PORT})"

# `ray status` is NOT node-local: it resolves a cluster address from
# /tmp/ray/ray_current_cluster and will happily report success by talking to a
# DIFFERENT cluster this container joined previously. That is not theoretical --
# one node still held a pointer to an older cluster's head, so this check passed,
# the join was skipped, and the node silently never joined its intended cluster.
# The run then sat forever in Ray scheduling on 16 of 24 GPUs looking alive.
# Test for a raylet process on THIS node instead; it cannot be satisfied remotely.
# pgrep matches ZOMBIES. One node held a "[raylet] <defunct>" left behind when its
# previous arm died; pgrep -x reported it, bringup declared the node joined and
# exited 0 while the node was in no cluster at all. A defunct process has no
# command line either, which is why the gcs-address check below read as empty.
# Count only raylets in a non-Z state.
JOINED_LOCAL="$(cexec "ps -eo stat=,comm= | awk '\$2 == \"raylet\" && \$1 !~ /Z/ {n++} END {print (n > 0) ? \"yes\" : \"no\"}'")"

# A local raylet proves a raylet exists, NOT that it belongs to the cluster we
# were asked to join. One node had a raylet left over from a previous arm's head at
# another head; bringup reported "already running", skipped the join and exited
# 0, so the node never joined its intended head and that arm stayed at 16 of 24
# GPUs. The raylet's own command line carries --gcs-address, which is the
# authoritative answer and is readable without touching the head.
if [ "$JOINED_LOCAL" = yes ]; then
    CUR_GCS="$(cexec "ps -eo args= | grep -m1 '[r]aylet' | sed -n 's/.*--gcs-address=\\([0-9.]*:[0-9]*\\).*/\\1/p'")"
    # Fail CLOSED on an unreadable address. Treating "cannot tell" as "fine" is
    # what let the defunct-raylet case through; rejoining a node that was
    # already correct is harmless, leaving one silently unjoined is not.
    if [ "$CUR_GCS" != "${RAY_HEAD_IP}:${RAY_GCS_PORT}" ]; then
        warn "raylet here reports gcs-address '${CUR_GCS:-<unreadable>}', expected ${RAY_HEAD_IP}:${RAY_GCS_PORT}; restarting ray to rejoin"
        [ "$MODE" = check ] && die "wrong or unverifiable cluster (--check-only makes no changes)"
        cexec "ray stop --force >/dev/null 2>&1 || true"
        sleep 6
        JOINED_LOCAL=no
    else
        ok "raylet already joined to ${CUR_GCS}"
    fi
fi

if [ "$JOINED_LOCAL" = no ]; then
    [ "$MODE" = check ] && die "no raylet running in the container on this node (--check-only makes no changes)"
    info "ray start --address=${RAY_HEAD_IP}:${RAY_GCS_PORT} --node-ip-address=${NODE_IP} --num-gpus ${NUM_GPUS}"
    # --node-ip-address is given explicitly. Ray would usually infer the right
    #   address from the route to the GCS, but on these hosts /etc/hosts maps the
    #   hostname to 127.0.1.1 and there is a second routable interface (eno1), so
    #   the inference is not something to rely on.
    # --num-gpus is given explicitly. Ray's ROCm probe does find all 8, but stating
    #   it means a node that comes up with GPUs masked or a broken amdgpu driver
    #   fails at `ray start` instead of quietly joining with GPU=0.
    # The GCS port must be 6379 -- the head was started with the default, as
    #   `--gcs_server_port=6379` on its gcs_server process confirms. 8265 is the
    #   dashboard/job-submission port, not a join address.
    run_show cexec "joining" \
        "ray start --address=${RAY_HEAD_IP}:${RAY_GCS_PORT} --node-ip-address=${NODE_IP} --num-gpus ${NUM_GPUS} --disable-usage-stats" \
        || die "ray start failed"
    sleep 5
else
    ok "a raylet is already running in this container"
fi

# The NodeID minted by THIS raylet. Everything downstream is checked against it, so
# a stale GCS entry for a previous incarnation of this node cannot satisfy the test.
NODE_ID="$(cexec "python3 -c \"import ray; ray.init(address='auto', log_to_driver=False, configure_logging=False); print('NODEID=' + ray.get_runtime_context().get_node_id())\" 2>/dev/null | grep -oP '(?<=NODEID=)[0-9a-f]+'")"
[ -n "$NODE_ID" ] || die "could not read this raylet's NodeID; it is not attached to a cluster"
ok "this raylet's NodeID = $NODE_ID"

step "7. ghost-proof join verification (asked from the head, not from the worker)"
run_show hexec "identity + resources + a real task pinned to this NodeID" \
    "python3 $KIT/bringup_ray_check.py --gcs ${RAY_HEAD_IP}:${RAY_GCS_PORT} --node-id ${NODE_ID} --expect-ip ${NODE_IP} --expect-gpus ${NUM_GPUS}" \
    || die "the node did not verifiably join (see above)"

echo
printf '\033[1;32mNODE READY: %s (%s) node=%s gpus=%s\033[0m\n' "$HOST" "$NODE_IP" "${NODE_ID:0:16}" "$NUM_GPUS"
echo "yield it again with: $0 $HOST --teardown"

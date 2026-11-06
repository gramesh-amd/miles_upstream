# Code RL on Lux (OLCF)

Lux has no Docker and no Spur, so the Crusoe tooling in `cluster/` is replaced here; the
recipe itself (`run_code.sh`, `run_async.sh`) runs unchanged. Run from the login node in
`~/miles` with `L=examples/instella_think/cluster/lux`.

| Crusoe | Lux |
|---|---|
| `bringup.sh` (docker pull + run) | `fetch.sbatch` builds the SIF once; `node.sbatch` + `node_agent.sh` keep an Apptainer instance up per node |
| `on_node.sh` (srun + docker exec) | `$L/on_node.sh [-b] [-p] <JOBID> <NODE\|-> '<cmd>'` |
| `code_exec.sh start [bench]` (ssh to a CPU host) | `sandbox.sbatch` on its own node; writes `~/.miles/code-exec-host` |
| `/shared_nfs/...` | `$LUX_ROOT=/lustre/orion/ven136/scratch/$USER/instella` (`~/.instella-env.sh`) |

Why the agent: Spur's containers lived under dockerd, outside Slurm. Under Apptainer every
process belongs to a Slurm step and dies with it, so `node_agent.sh` runs in one step that
lasts the job and executes the command files `on_node.sh` drops in its mailbox. The container
runs with `--fakeroot` (root inside, like Docker, so the sglang/mbridge/Megatron patches can
write into the image) and a writable overlay on `/mnt/bb/<jobid>`, so `env_setup.sh` is
needed once per job.

**Keep long runs in the foreground of their command.** Under fakeroot each `apptainer exec` has
its own `faked` daemon; a process left in the background after its command returns hangs on
its first file operation once that daemon exits. `run_code.sh` backgrounds the run, so follow
it with a wait (below).

## One run, one job: `run.sbatch`

The usual way to run. The arguments are the run spec, the same `VAR=VALUE` overrides
`cluster/launch.sh` takes on Crusoe; node 0 of the job runs the sandboxes and the rest are GPU
nodes (`-N 2` for the 1-node layout, `-N 4` for the 3-node recipe). From `~/miles`:

```bash
sbatch -A ven136 -N 4 -t 24:00:00 -o ~/logs/%x-%j.out examples/instella_think/cluster/lux/run.sbatch \
  CONTEXT_LEN=32768 DRGRPO=1 DRGRPO_DIVISOR=17400 OVERLONG_FILTER=1 SAVE_TAG=code32k_drgrpo
```

The job sets up every container, starts the sandboxes and the Ray cluster, runs the spec and ends
when the run does. Metrics go to W&B live (team `genaiamd` unless the spec sets `WANDB_ENTITY`;
`WANDB_MODE=offline` logs locally). W&B alone uses the ORNL proxy (`WANDB_HTTPS_PROXY`), so
traffic between nodes stays direct. Per job, `$LUX_ROOT/shared/jobs/<jobid>/` holds the spec, each
node's setup log, the sandbox log, `run.log` (the run's own log) and `run.status`. Prerequisites:
the once-only steps below (SIF, checkpoint, data, Megatron) and `wandb login` on the login node.

Cross-node collectives use RCCL's socket transport over `bond0` (the container's verbs stack
cannot drive the Pollara NICs); see `/lustre/orion/ven136/proj-shared/models/scripts/netenv.sh`
for the RoCE setup if weight sync becomes the bottleneck.

## Steps by hand (debugging)

```bash
L=examples/instella_think/cluster/lux; X="--export=ALL,LUX_DIR=$PWD/$L"
# Once: SIF (~10 min), DPO checkpoint, train data, benchmarks.
sbatch -A ven136 -o ~/logs/%x-%j.out $X $L/fetch.sbatch
# Per run: GPU node(s) and the sandbox node.
sbatch -A ven136 -N 1 -t 06:00:00 -o ~/logs/%x-%j.out $X $L/node.sbatch
sbatch -A ven136 -t 06:00:00 -o ~/logs/%x-%j.out $X $L/sandbox.sbatch
J=<node.sbatch job>
# Once per user (-p: through the ORNL proxy), then once per job:
$L/on_node.sh -p $J - 'cd $USER_HOME/miles && bash examples/instella_think/investigate_megatron.sh'
$L/on_node.sh $J - 'cd $USER_HOME/miles && bash examples/instella_think/cluster/env_setup.sh'
# Once per user: HF -> torch_dist.
$L/on_node.sh $J - 'cd $USER_HOME/miles && source /opt/venv/bin/activate && PHASE=prepare bash examples/instella_think/run_async.sh'
# Preflights, then the run (log path is printed in the -b log):
$L/on_node.sh $J - 'cd $USER_HOME/miles && source /opt/venv/bin/activate && CHECK_ONLY=1 SINGLE_NODE=1 bash examples/instella_think/run_code.sh'
$L/on_node.sh -b $J - 'cd $USER_HOME/miles && source /opt/venv/bin/activate && SINGLE_NODE=1 <VARS> bash examples/instella_think/run_code.sh; sleep 5; while pgrep -f run-frozen >/dev/null; do sleep 30; done'
```

Stop with `$L/on_node.sh $J - 'source /opt/venv/bin/activate; RAY_ADDRESS=http://127.0.0.1:8265 ray job stop <id>'`,
then `scancel` both jobs: nodes are charged until the job ends.


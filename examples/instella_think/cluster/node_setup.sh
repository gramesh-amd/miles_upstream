#!/bin/bash
# Per-node bring-up of the miles training container. Runs ON each allocated compute node
# (bringup.sh srun's it), driven by env vars: IMAGE PORT USER_NAME HOME_DIR NAME SLURM_JOB_ID.
#
# It pulls the image, starts the container with GPUs, creates a uid/gid-matched user with
# the GPU groups, launches an sshd on $PORT and records readiness. Spur's epilog removes the
# container at job end by its spur_job_id label. The host docker socket cannot be mounted
# (the docker-authz plugin refuses it), so there is no docker-in-docker and no watchdog.
set -uo pipefail
export SPUR_CONTROLLER_ADDR="${SPUR_CONTROLLER_ADDR:-http://crs-m2m-cpu-spur-005.crusoe.amd.com:6817}"

CNAME="${NAME}-${SLURM_JOB_ID}-$(hostname -s)"
STATUS_DIR="${HOME_DIR}/.miles/status-${SLURM_JOB_ID}"
mkdir -p "$STATUS_DIR"

# Node-local volume that also backs the Docker root. Caches belong here, not on NFS: the
# shared home volume runs near full and returns EDQUOT. Everything here dies with the node.
SCRATCH="/mnt/m2m_nobackup/${USER_NAME}"
mkdir -p "$SCRATCH"/{hf,wandb,triton,logs} 2>/dev/null || true

echo ">>> [$(hostname)] pulling ${IMAGE} ..."
docker pull "$IMAGE"

docker rm -f "$CNAME" >/dev/null 2>&1 || true
# Spur's prolog/epilog manage containers by this label: unlabeled ones can be removed as
# orphans mid-job, and are not cleaned up when the job ends.
SPUR_LABEL="spur_job_id=${SPUR_JOB_ID:-$SLURM_JOB_ID}"
echo ">>> [$(hostname)] starting container ${CNAME} ..."
docker run -d --name "$CNAME" --label "$SPUR_LABEL" \
  --network host --ipc=host --cap-add SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --device=/dev/kfd --device=/dev/dri \
  --group-add video --group-add render \
  -v "${HOME_DIR}:${HOME_DIR}" -v /shared_nfs:/shared_nfs \
  -v "${SCRATCH}:/scratch" \
  -e HF_HOME=/scratch/hf \
  -e WANDB_DIR=/scratch/wandb \
  -e TRITON_CACHE_DIR=/scratch/triton \
  --entrypoint /usr/bin/sleep \
  "$IMAGE" infinity \
  || { echo "FATAL: [$(hostname)] docker run failed; not marking ready" >&2; exit 1; }

echo ">>> [$(hostname)] configuring sshd + user inside container ..."
docker exec -e USER_NAME="$USER_NAME" -e HOME_DIR="$HOME_DIR" "$CNAME" bash -lc '
  set -e
  if ! command -v sshd >/dev/null 2>&1; then
    ( (apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y openssh-server) \
      || yum install -y openssh-server || microdnf install -y openssh-server ) >/tmp/sshd-install.log 2>&1 || true
  fi
  mkdir -p /run/sshd
  ssh-keygen -A
  HUID=$(stat -c %u "$HOME_DIR"); HGID=$(stat -c %g "$HOME_DIR")
  getent group "$HGID" >/dev/null || groupadd -g "$HGID" "hostgrp$HGID"
  getent passwd "$USER_NAME" >/dev/null || useradd -o -u "$HUID" -g "$HGID" -d "$HOME_DIR" -M -s /bin/bash "$USER_NAME"
  usermod -p "*" "$USER_NAME"
  # GPU device groups keep their host NUMERIC gids inside the container -> join by number
  KFD=$(stat -c %g /dev/kfd 2>/dev/null); DRI=$(stat -c %g /dev/dri/card0 2>/dev/null)
  for g in "$KFD" "$DRI"; do
    [ -n "$g" ] || continue
    gn=$(getent group "$g" | cut -d: -f1)
    [ -z "$gn" ] && { gn="grp$g"; groupadd -g "$g" "$gn" 2>/dev/null || true; }
    usermod -aG "$gn" "$USER_NAME" 2>/dev/null || true
  done
  getent group render >/dev/null && usermod -aG render "$USER_NAME" 2>/dev/null || true
  getent group video  >/dev/null && usermod -aG video  "$USER_NAME" 2>/dev/null || true
  command -v sudo >/dev/null && { echo "$USER_NAME ALL=(ALL) NOPASSWD:ALL" >/etc/sudoers.d/"$USER_NAME"; chmod 440 /etc/sudoers.d/"$USER_NAME"; }
  # megatron-core is an editable install under /root (mode 700); let non-root
  # traverse into it so `import megatron.core` / mbridge work as the login user.
  [ -d /root/Megatron-LM ] && chmod o+x /root 2>/dev/null || true
  # aiter/sglang/triton-custom editable installs live under /sgl-workspace, owned
  # by root. aiter copies its prebuilt flydsl JIT cache from there into ~/.aiter,
  # so the login user needs read/traverse access or it floods with Errno 13.
  [ -d /sgl-workspace ] && chmod -R o+rX /sgl-workspace 2>/dev/null || true
  # Processes here run as root, so anything they write into the bind-mounted
  # NFS home lands root-owned and the login user cannot even delete it. These
  # mounts are NFSv3 (no default ACLs), so a umask of 002 is the only inheritance
  # mechanism left.
  echo "umask 0002" > /etc/profile.d/00-umask.sh
  chmod 644 /etc/profile.d/00-umask.sh
  grep -q "^umask 0002" /etc/bash.bashrc 2>/dev/null || echo "umask 0002" >> /etc/bash.bashrc
'

echo ">>> [$(hostname)] launching sshd on port ${PORT} ..."
docker exec "$CNAME" /usr/sbin/sshd -e -p "$PORT" \
  -o PermitRootLogin=prohibit-password -o PubkeyAuthentication=yes \
  -o PasswordAuthentication=no -o UsePAM=no 2>&1 || echo "WARN: sshd start returned non-zero on $(hostname)"

[ "$(docker inspect -f '{{.State.Running}}' "$CNAME" 2>/dev/null)" = true ] \
  || { echo "FATAL: [$(hostname)] container $CNAME is not running; not marking ready" >&2; exit 1; }

echo "$(hostname)" > "${STATUS_DIR}/$(hostname -s).ready"
echo ">>> [$(hostname)] READY  ->  ssh -p ${PORT} ${USER_NAME}@$(hostname)"

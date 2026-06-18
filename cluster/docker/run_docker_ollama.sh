#!/usr/bin/env bash
# OPTIONAL Docker path for running Ollama on the cluster.
#
# Prefer the native install (../install_ollama.sh + ../slurm_extract.sbatch):
# it needs no admin rights and avoids Docker's network entirely. Use Docker
# only if your group's workflow requires it.
#
# IMPORTANT (per CS support): Docker's DEFAULT bridge subnet collides with the
# cluster's "red" network range and causes unpredictable failures. You MUST
# move Docker onto one of the reserved ranges:
#
#     docker1  192.168.152.0/24
#     docker2  192.168.153.0/24
#     docker3  192.168.154.0/24
#
# Two ways to do that, depending on what access you have:
#
#   (a) Daemon-wide (needs control of the docker daemon / dockerd config):
#       copy daemon.json into /etc/docker/daemon.json (or point dockerd at it)
#       and restart the daemon. This sets default-address-pools to a reserved
#       range so EVERY container/network you create stays off the red range.
#
#   (b) Per-network (no daemon changes — what this script does): create a user
#       bridge network pinned to a reserved subnet and attach the container to
#       it. This is the safest option when you "have no admin privileges".
set -euo pipefail

# Pick the reserved range to use (docker1/2/3). Override with DOCKER_SUBNET.
DOCKER_SUBNET="${DOCKER_SUBNET:-192.168.152.0/24}"
DOCKER_GATEWAY="${DOCKER_GATEWAY:-192.168.152.1}"
NET_NAME="${NET_NAME:-flood-ollama-net}"

OLLAMA_PORT="${OLLAMA_PORT:-11434}"
OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"
# Persist models on the host so they survive container restarts.
MODELS_DIR="${OLLAMA_MODELS:-$HOME/.ollama}"
mkdir -p "$MODELS_DIR"

# --- 1. reserved-range network (idempotent) --------------------------------
if ! docker network inspect "$NET_NAME" >/dev/null 2>&1; then
  echo "==> Creating docker network $NET_NAME on reserved subnet $DOCKER_SUBNET"
  docker network create \
    --driver bridge \
    --subnet "$DOCKER_SUBNET" \
    --gateway "$DOCKER_GATEWAY" \
    "$NET_NAME"
else
  echo "==> Reusing docker network $NET_NAME"
fi

# --- 2. GPU flag (only if the NVIDIA container runtime is available) --------
GPU_FLAG=""
if docker info 2>/dev/null | grep -qi nvidia; then
  GPU_FLAG="--gpus all"
  echo "==> NVIDIA runtime detected; passing --gpus all"
else
  echo "==> No NVIDIA docker runtime; running CPU-only (slow)."
fi

# --- 3. run the official Ollama image on the reserved network --------------
echo "==> Starting Ollama container (port $OLLAMA_PORT, models at $MODELS_DIR)"
docker run -d --rm \
  --name flood-ollama \
  --network "$NET_NAME" \
  $GPU_FLAG \
  -p "127.0.0.1:${OLLAMA_PORT}:11434" \
  -v "$MODELS_DIR:/root/.ollama" \
  ollama/ollama

echo "==> Waiting for the server..."
for i in $(seq 1 60); do
  if curl -fs "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1; then
    echo "==> Ready. Pulling model $OLLAMA_MODEL"
    docker exec flood-ollama ollama pull "$OLLAMA_MODEL"
    echo
    echo "Now run the extractor against it:"
    echo "    python ../extract_floods.py --csv ontario_floods_export.csv --port $OLLAMA_PORT --model $OLLAMA_MODEL"
    echo "Stop the server when done:  docker stop flood-ollama"
    exit 0
  fi
  sleep 1
done

echo "ERROR: Ollama container did not become ready." >&2
docker logs flood-ollama >&2 || true
exit 1

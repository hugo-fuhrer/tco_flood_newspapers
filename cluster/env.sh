# Shared environment for the cluster flood-extraction job. `source` this from
# your shell, from start_ollama.sh, and from the SLURM script so the binary,
# model cache, and port are configured consistently.
#
#   source env.sh
#
# Override any of these by exporting them BEFORE sourcing.

# --- Ollama install + model cache -----------------------------------------
export OLLAMA_HOME="${OLLAMA_HOME:-$HOME/ollama}"
export PATH="$OLLAMA_HOME/bin:$PATH"

# Models are multi-GB. Keep them off a small home quota if you have scratch.
# Set CLUSTER_SCRATCH to your scratch path; otherwise this falls back to $HOME.
export CLUSTER_SCRATCH="${CLUSTER_SCRATCH:-$HOME}"
export OLLAMA_MODELS="${OLLAMA_MODELS:-$CLUSTER_SCRATCH/.ollama/models}"
mkdir -p "$OLLAMA_MODELS"

# --- Networking -------------------------------------------------------------
# On a SHARED compute node the default port 11434 may already be taken by
# another user's Ollama. Derive a per-job port from the SLURM job id so two
# jobs on the same node don't collide. Falls back to 11434 outside SLURM.
if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export OLLAMA_PORT="${OLLAMA_PORT:-$((11434 + SLURM_JOB_ID % 1000))}"
else
  export OLLAMA_PORT="${OLLAMA_PORT:-11434}"
fi
# Bind to loopback only — we never want to expose the model server on the
# cluster's network (and it sidesteps the docker/red-network range issue).
export OLLAMA_HOST="${OLLAMA_HOST:-127.0.0.1:$OLLAMA_PORT}"

# --- Model -----------------------------------------------------------------
export OLLAMA_MODEL="${OLLAMA_MODEL:-llama3.1:8b}"

# Keep one model resident and limit parallelism so we fit a single GPU's VRAM.
export OLLAMA_KEEP_ALIVE="${OLLAMA_KEEP_ALIVE:-30m}"
export OLLAMA_NUM_PARALLEL="${OLLAMA_NUM_PARALLEL:-1}"

echo "[env] OLLAMA_HOME=$OLLAMA_HOME"
echo "[env] OLLAMA_MODELS=$OLLAMA_MODELS"
echo "[env] OLLAMA_HOST=$OLLAMA_HOST  (port $OLLAMA_PORT)"
echo "[env] OLLAMA_MODEL=$OLLAMA_MODEL"

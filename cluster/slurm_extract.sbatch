#!/usr/bin/env bash
#SBATCH --job-name=flood-extract
#SBATCH --output=flood-extract-%j.out
#SBATCH --error=flood-extract-%j.err
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=08:00:00
# NOTE: partition/qos names vary by cluster. Run `sinfo` and check
# https://support.cs.toronto.edu/systems/slurmresource.html, then uncomment:
##SBATCH --partition=gpunodes
##SBATCH --qos=normal
#
# Submit with:   sbatch slurm_extract.sbatch
# Monitor with:  squeue --me   ;   tail -f flood-extract-<jobid>.out
#
# This job:
#   1. configures the home-dir Ollama install + a per-job port,
#   2. starts a local Ollama server on the allocated GPU node,
#   3. ensures the model is present (uses the pre-pulled shared cache; the
#      compute node has NO internet, so pull on a login node first),
#   4. runs extract_floods.py over the exported Ontario-flood CSV,
#   5. shuts the server down.
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

# --- 1. environment --------------------------------------------------------
# shellcheck disable=SC1091
source env.sh

echo "==> Node: $(hostname)   GPU(s): ${CUDA_VISIBLE_DEVICES:-none}   Job: ${SLURM_JOB_ID:-NA}"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || echo "(no nvidia-smi)"

# Activate your Python env (venv or conda) that has dspy + tqdm installed.
# Edit to match how you created it (see README).
if [[ -d "$HOME/flood_env" ]]; then
  # shellcheck disable=SC1091
  source "$HOME/flood_env/bin/activate"
fi

# --- 2. start Ollama -------------------------------------------------------
export OLLAMA_PIDFILE="$HOME/.ollama_${SLURM_JOB_ID:-local}.pid"
bash start_ollama.sh

# Always stop the server when the job ends (success, failure, or scancel).
cleanup() {
  if [[ -f "$OLLAMA_PIDFILE" ]]; then
    echo "==> Stopping Ollama (pid $(cat "$OLLAMA_PIDFILE"))"
    kill "$(cat "$OLLAMA_PIDFILE")" 2>/dev/null || true
    rm -f "$OLLAMA_PIDFILE"
  fi
}
trap cleanup EXIT

# --- 3. ensure model present (no internet on compute nodes) ----------------
if ! ollama list 2>/dev/null | grep -q "${OLLAMA_MODEL%%:*}"; then
  echo "==> Model $OLLAMA_MODEL not in cache; attempting pull (needs internet)..."
  ollama pull "$OLLAMA_MODEL" || {
    echo "ERROR: model $OLLAMA_MODEL missing and pull failed." >&2
    echo "Pre-pull it on a login node: 'source env.sh && bash start_ollama.sh && ollama pull $OLLAMA_MODEL'" >&2
    exit 1
  }
fi

# --- 4. run extraction -----------------------------------------------------
INPUT_CSV="${INPUT_CSV:-ontario_floods_export.csv}"
OUT_JSONL="${OUT_JSONL:-flood_details.jsonl}"
OUT_CSV="${OUT_CSV:-flood_details.csv}"

echo "==> Extracting from $INPUT_CSV with model $OLLAMA_MODEL on port $OLLAMA_PORT"
python extract_floods.py \
  --csv "$INPUT_CSV" \
  --out "$OUT_JSONL" \
  --out-csv "$OUT_CSV" \
  --model "$OLLAMA_MODEL" \
  --port "$OLLAMA_PORT" \
  ${EXTRA_ARGS:-}

echo "==> Done. Output: $OUT_CSV"

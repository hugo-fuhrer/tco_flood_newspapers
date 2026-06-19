#!/usr/bin/env bash
# Start a local Ollama server bound to loopback on $OLLAMA_PORT, wait until it
# answers, and leave it running in the background. Safe to source or execute.
# Writes the server PID to $OLLAMA_PIDFILE so the SLURM script can stop it.
#
#   source env.sh && bash start_ollama.sh
#
# Returns 0 once the server responds; non-zero if it never comes up.
set -euo pipefail

HERE="$(cd "$(dirname "${BASH_SOURCE[0]:-$0}")" && pwd)"
# shellcheck disable=SC1091
source "$HERE/env.sh"

OLLAMA_PIDFILE="${OLLAMA_PIDFILE:-$HOME/.ollama_${SLURM_JOB_ID:-local}.pid}"
LOG="${OLLAMA_LOG:-$HOME/ollama_${SLURM_JOB_ID:-local}.log}"

# Already up on this port? Reuse it.
if curl -fs "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1; then
  echo "[ollama] already serving on port $OLLAMA_PORT — reusing it."
  exit 0
fi

echo "[ollama] starting server on $OLLAMA_HOST (log: $LOG)"
nohup ollama serve >"$LOG" 2>&1 &
echo $! > "$OLLAMA_PIDFILE"

# Wait up to ~60s for the API to respond.
for i in $(seq 1 60); do
  if curl -fs "http://127.0.0.1:$OLLAMA_PORT/api/version" >/dev/null 2>&1; then
    echo "[ollama] ready after ${i}s — $(curl -fs http://127.0.0.1:$OLLAMA_PORT/api/version)"
    exit 0
  fi
  sleep 1
done

echo "[ollama] ERROR: server did not become ready in 60s. Last log lines:" >&2
tail -n 20 "$LOG" >&2 || true
exit 1

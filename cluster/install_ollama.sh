#!/usr/bin/env bash
# Install Ollama into your home directory on the UofT CS cluster — no admin /
# sudo required. Run this once on a login/utility node (e.g. comps0), NOT on a
# compute node. The official `curl ... | sh` installer needs root; instead we
# unpack the official static tarball under $HOME.
#
#   bash install_ollama.sh
#
# Re-running upgrades in place. Models live under $OLLAMA_MODELS (see env.sh).
set -euo pipefail

# Where to put the binaries. Override with OLLAMA_HOME=... before running.
OLLAMA_HOME="${OLLAMA_HOME:-$HOME/ollama}"
OLLAMA_VERSION="${OLLAMA_VERSION:-}"   # e.g. "v0.5.7"; empty = latest
ARCH="$(uname -m)"

case "$ARCH" in
  x86_64|amd64) PKG="ollama-linux-amd64.tgz" ;;
  aarch64|arm64) PKG="ollama-linux-arm64.tgz" ;;
  *) echo "Unsupported arch: $ARCH" >&2; exit 1 ;;
esac

if [[ -n "$OLLAMA_VERSION" ]]; then
  URL="https://github.com/ollama/ollama/releases/download/${OLLAMA_VERSION}/${PKG}"
else
  URL="https://ollama.com/download/${PKG}"
fi

echo "==> Installing Ollama into $OLLAMA_HOME"
echo "    arch=$ARCH  package=$PKG"
echo "    url=$URL"
mkdir -p "$OLLAMA_HOME"

tmp="$(mktemp -d)"
trap 'rm -rf "$tmp"' EXIT

echo "==> Downloading (login nodes have internet; compute nodes do not)..."
if command -v curl >/dev/null 2>&1; then
  curl -fL --retry 4 --retry-delay 2 -o "$tmp/$PKG" "$URL"
else
  wget -t 5 -O "$tmp/$PKG" "$URL"
fi

echo "==> Unpacking..."
tar -xzf "$tmp/$PKG" -C "$OLLAMA_HOME"

# The tarball contains bin/ollama and lib/. Make sure bin/ollama is on PATH.
BIN="$OLLAMA_HOME/bin/ollama"
[[ -x "$BIN" ]] || BIN="$(find "$OLLAMA_HOME" -name ollama -type f -perm -u+x | head -n1)"
if [[ -z "$BIN" || ! -x "$BIN" ]]; then
  echo "ERROR: ollama binary not found after unpacking $PKG" >&2
  exit 1
fi

echo "==> Installed: $("$BIN" --version 2>/dev/null || echo "$BIN")"
echo
echo "Next steps:"
echo "  1. source $(dirname "$0")/env.sh       # puts ollama on PATH, sets model/cache dirs"
echo "  2. ollama --version                    # confirm it runs"
echo "  3. On a login node with internet, pre-pull your model so the offline"
echo "     compute node can use it from the shared cache:"
echo "         source env.sh && bash start_ollama.sh && ollama pull \"\$OLLAMA_MODEL\""

#!/usr/bin/env bash
# Session setup for the Claude sandbox. Idempotent. Run from the repo root: bash work/setup.sh
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
pip install -q --break-system-packages pandas pyarrow numpy scipy scikit-learn pybaseball requests 2>/dev/null || true
# Chadwick tools (cwevent, cwdaily) for Retrosheet parsing
if ! command -v "$HOME/.local/bin/cwevent" >/dev/null 2>&1; then
  mkdir -p "$HOME/ext" && cd "$HOME/ext"
  [ -d chadwick ] || git clone -q --depth 1 https://github.com/chadwickbureau/chadwick
  cd chadwick && autoreconf -i >/dev/null 2>&1 && ./configure -q --prefix="$HOME/.local" >/dev/null && make -s -j8 >/dev/null && make -s install >/dev/null
fi
# Retrosheet event files (only needed to rebuild data/retro; the built parquet is committed)
if [ "${RETRO:-0}" = "1" ] && [ ! -d "$ROOT/ext/retrosheet/seasons" ]; then
  mkdir -p "$ROOT/ext" && cd "$ROOT/ext"
  git clone -q --depth 1 --filter=blob:none --no-checkout https://github.com/chadwickbureau/retrosheet
  cd retrosheet && git sparse-checkout set --no-cone $(for y in $(seq 2010 2025); do echo "seasons/$y/"; done) reference/ && git checkout -q HEAD
fi
echo "setup ok"

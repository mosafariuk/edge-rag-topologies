#!/usr/bin/env bash
# Run the full topology matrix from ONE probe location. Copy probe.py + this file to each VM.
#   REGION=sydney ./run_matrix.sh
# Fill the URLs below (or export them) before running. Leave a URL empty to skip that topology.
set -euo pipefail
REGION="${REGION:?set REGION, e.g. sydney}"
N="${N:-300}"
OUT="results/${REGION}.csv"

CENTRAL_URL="${CENTRAL_URL:-}"              # https://central.example.com  (baseline/server.py, POOL=1)
CENTRAL_UNPOOLED_URL="${CENTRAL_UNPOOLED_URL:-}"   # same server with POOL=0 (optional)
EDGE_UNPOOLED_URL="${EDGE_UNPOOLED_URL:-}"  # https://edge-rag-unpooled.<acct>.workers.dev
EDGE_POOLED_URL="${EDGE_POOLED_URL:-}"      # https://edge-rag-pooled.<acct>.workers.dev
EDGE_REPLICA_URL="${EDGE_REPLICA_URL:-}"    # https://edge-rag-replica.<acct>.workers.dev
VECTORIZE_URL="${VECTORIZE_URL:-}"          # optional vendor-default stack

run() { [ -n "$2" ] && python3 probe.py --target "$2" --label "$1" --region "$REGION" --n "$N" --mode both --out "$OUT" || echo "[skip] $1"; }

# Interleave topologies in two passes so time-of-day drift affects all of them equally.
for pass in 1 2; do
  echo "=== pass $pass ==="
  run centralized          "$CENTRAL_URL"
  run edge-pooled          "$EDGE_POOLED_URL"
  run edge-replica         "$EDGE_REPLICA_URL"
  run edge-unpooled        "$EDGE_UNPOOLED_URL"
  run centralized-unpooled "$CENTRAL_UNPOOLED_URL"
  run vectorize            "$VECTORIZE_URL"
done
echo "done -> $OUT ; copy it back and run: python3 analyze.py results/*.csv"

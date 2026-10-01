#!/usr/bin/env bash
# Run the full topology matrix from ONE probe location. Copy probe.py + this file to each VM.
#   REGION=sydney ROT=1 ./run_matrix.sh
# ROT rotates the topology order so that concurrent regions do not all hit the CPU-bound
# centralized baseline at the same moment. Leave a URL empty to skip that topology.
set -uo pipefail
REGION="${REGION:?set REGION, e.g. sydney}"
N="${N:-300}"
ROT="${ROT:-0}"
INTERVAL="${INTERVAL:-0.5}"
OUT="results/${REGION}.csv"

LABELS=(centralized edge-pooled edge-replica edge-unpooled central-tunnel)
URLS=("${CENTRAL_URL:-}" "${EDGE_POOLED_URL:-}" "${EDGE_REPLICA_URL:-}" "${EDGE_UNPOOLED_URL:-}" "${CENTRAL_TUNNEL_URL:-}")
K=${#LABELS[@]}

for pass in 1 2; do
  echo "=== $(date -u +%FT%TZ) pass $pass (rotation $ROT)"
  for j in $(seq 0 $((K-1))); do
    i=$(( (j + ROT) % K ))
    if [ -n "${URLS[$i]}" ]; then
      python3 probe.py --target "${URLS[$i]}" --label "${LABELS[$i]}" --region "$REGION" --n "$N" \
        --mode both --interval "$INTERVAL" --out "$OUT" || echo "[warn] ${LABELS[$i]} failed"
    else
      echo "[skip] ${LABELS[$i]}"
    fi
  done
done
echo "=== $(date -u +%FT%TZ) MATRIX DONE -> $OUT"

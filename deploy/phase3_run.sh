#!/usr/bin/env bash
# Phase 3: retrieval-tier benchmark on real embeddings (1M MS MARCO passages, bge-base-en-v1.5).
# Runs end-to-end on one VM under nohup; every step skips work already done, so re-running resumes.
#   nohup bash phase3_run.sh > phase3.log 2>&1 &
# Expects bench.py and embed_chunked.py next to it. Host used: GCP c3-standard-22, us-central1-a.
set -euo pipefail
cd "$(dirname "$0")"
N=1000000; Q=1000; WORKERS=4; THREADS=5
export PYTHONUNBUFFERED=1  # live progress in the nohup log
MSMARCO=https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz

# 1. dependencies
if ! command -v docker >/dev/null; then sudo apt-get update -q && sudo apt-get install -yq docker.io python3-venv; fi
if [ ! -x venv/bin/python ]; then
  python3 -m venv venv
  venv/bin/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
  venv/bin/pip install -q sentence-transformers numpy "psycopg[binary]"
fi

# 2. data: first N passages (pid 0..N-1, same prefix as the Phase 2 100k corpus), first Q dev.small queries
[ -f collection.tsv ] || curl -sSL "$MSMARCO" | tar xz
[ -f passages.txt ] || head -n "$N" collection.tsv | cut -f2 > passages.txt
[ -f queries.txt ] || head -n "$Q" queries.dev.small.tsv | cut -f2 > queries.txt
# embed_chunked.py drops blank lines; ids must stay aligned with pids
[ "$(grep -c . passages.txt)" -eq "$N" ] && [ "$(grep -c . queries.txt)" -eq "$Q" ]

# 3. embeddings (CPU; raw query text, no instruction prefix, matching the Phase 2 Worker)
[ -f queries.npy ] || OMP_NUM_THREADS=$THREADS venv/bin/python embed_chunked.py --device cpu \
  --passages queries.txt --limit "$Q" --chunk "$Q" --outdir qchunks --cache queries.npy
if [ ! -f corpus.npy ]; then
  for w in $(seq 0 $((WORKERS-1))); do
    OMP_NUM_THREADS=$THREADS venv/bin/python embed_chunked.py --device cpu --batch 64 \
      --passages passages.txt --limit "$N" --outdir chunks --cache corpus.npy \
      --offset "$w" --stride "$WORKERS" > "embed_w$w.log" 2>&1 &
  done
  wait
  venv/bin/python embed_chunked.py --merge-only --passages passages.txt --limit "$N" --outdir chunks --cache corpus.npy
fi

# 4. pgvector (same image family as July; sized for 88 GB RAM)
if ! sudo docker ps --format '{{.Names}}' | grep -q '^phase3_pg$'; then
  sudo docker rm -f phase3_pg 2>/dev/null || true
  sudo docker run -d --name phase3_pg --shm-size=24g -p 55432:5432 \
    -e POSTGRES_USER=bench -e POSTGRES_PASSWORD=bench -e POSTGRES_DB=bench \
    pgvector/pgvector:pg17 postgres -c shared_buffers=16GB -c effective_cache_size=64GB \
    -c max_worker_processes=24 -c max_parallel_workers=16 -c max_parallel_workers_per_gather=4 \
    -c work_mem=64MB -c max_wal_size=16GB -c checkpoint_timeout=30min -c jit=off
  until sudo docker exec phase3_pg pg_isready -U bench -d bench >/dev/null 2>&1; do sleep 2; done
  sleep 5
fi

# 5. benchmark grid (bench.py HNSW_BUILDS x ef_search, IVF_BUILDS x probes, exact seqscan baseline)
RESUME=""; [ -f results_phase3/results.csv ] && [ "$(wc -l < results_phase3/results.csv)" -gt 1 ] && RESUME="--resume"  # header alone = load never finished
venv/bin/python bench.py --vectors corpus.npy --query-vectors queries.npy --queries "$Q" --k 10 \
  --out results_phase3 --build-workers 8 --mwm 16GB --warmup 50 --repeats 5 --seqscan-queries 200 $RESUME \
  --note "GCP c3-standard-22 (22 vCPU, 88 GB), us-central1-a, pd-ssd; Docker pgvector/pgvector:pg17; client on same host via TCP; MS MARCO passages pid 0..999999 + dev.small first 1000 queries, bge-base-en-v1.5 (sentence-transformers 6.3.0, CLS, normalized, fp32, no query prefix) embedded on 1x RTX 4090 via deploy/phase3_gpu_embed.sh; GPU vs CPU cosine >= 0.999999 on 144k overlapping passages and all queries"
echo "[phase3] done $(date -u +%FT%TZ)"

#!/usr/bin/env bash
# Phase 3 corpus embedding on 1x RTX 4090 (vast.ai, pytorch/pytorch:2.5.1-cuda12.4 image; run
# `pip uninstall torchvision torchaudio` first). fp32, ~1,210 passages/s. corpus.npy/queries.npy are then
# copied into the phase3_run.sh directory, which skips its own CPU embedding step when they exist.
set -euo pipefail
cd /root
pip install -q sentence-transformers numpy
[ -f collection.tsv ] || curl -sSL https://msmarco.z22.web.core.windows.net/msmarcoranking/collectionandqueries.tar.gz | tar xz
[ -f passages.txt ] || head -n 1000000 collection.tsv | cut -f2 > passages.txt
[ -f queries.txt ] || head -n 1000 queries.dev.small.tsv | cut -f2 > queries.txt
[ "$(grep -c . passages.txt)" -eq 1000000 ] && [ "$(grep -c . queries.txt)" -eq 1000 ]
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
python embed_chunked.py --device cuda --passages queries.txt --limit 1000 --chunk 1000 --outdir qchunks --cache queries.npy
python embed_chunked.py --device cuda --batch 256 --chunk 10000 --passages passages.txt --limit 1000000 --outdir chunks --cache corpus.npy
sha256sum corpus.npy queries.npy > sha256.txt
echo "[gpu] done $(date -u +%FT%TZ)"

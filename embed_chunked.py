#!/usr/bin/env python3
"""
Resumable, chunked embedding of a passages file with bge-base-en-v1.5.
Writes one .npy per chunk into --outdir and merges them into --cache at the end, so an
out-of-memory failure or an interruption costs one chunk, not the run. Clears the accelerator
cache between chunks (the MPS allocator otherwise grows across a long run on an 8 GB machine).

    python3 embed_chunked.py --passages passages.txt --limit 100000 --outdir chunks/ --cache embeddings_100k.npy
"""
from __future__ import annotations

import argparse
import gc
import os
import time

import numpy as np

MODEL = "BAAI/bge-base-en-v1.5"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", required=True)
    ap.add_argument("--limit", type=int, default=100_000)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--cache", required=True)
    ap.add_argument("--chunk", type=int, default=2000)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=384)
    ap.add_argument("--device", default=None, help="mps | cpu (default: mps if available)")
    ap.add_argument("--offset", type=int, default=0, help="this worker's chunk offset (distributed runs)")
    ap.add_argument("--stride", type=int, default=1, help="number of workers sharing the chunks")
    ap.add_argument("--merge-only", action="store_true", help="only merge existing chunk files")
    args = ap.parse_args()

    texts = []
    with open(args.passages, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                texts.append(s)
            if len(texts) >= args.limit:
                break
    os.makedirs(args.outdir, exist_ok=True)

    n_chunks = (len(texts) + args.chunk - 1) // args.chunk
    if not args.merge_only:
        import torch
        from sentence_transformers import SentenceTransformer
        device = args.device or ("mps" if torch.backends.mps.is_available() else "cpu")
        model = SentenceTransformer(MODEL, device=device)
        model.max_seq_length = args.max_seq
        print(f"[embed] {len(texts)} passages, device={device}, chunk={args.chunk}, batch={args.batch}, "
              f"worker {args.offset}/{args.stride}")
    t_start = time.time()
    done = 0
    for ci in range(n_chunks):
        path = os.path.join(args.outdir, f"chunk_{ci:05d}.npy")
        if args.merge_only or os.path.exists(path) or ci % args.stride != args.offset:
            continue
        lo, hi = ci * args.chunk, min((ci + 1) * args.chunk, len(texts))
        t0 = time.time()
        vecs = model.encode(texts[lo:hi], batch_size=args.batch, normalize_embeddings=True,
                            convert_to_numpy=True, show_progress_bar=False).astype("float32")
        np.save(path + ".tmp.npy", vecs)
        os.replace(path + ".tmp.npy", path)
        if device == "mps":
            torch.mps.empty_cache()
        gc.collect()
        done += hi - lo
        print(f"[embed] chunk {ci+1}/{n_chunks} ({hi - lo} in {time.time()-t0:.0f}s) "
              f"this worker {done} done, {done/(time.time()-t_start):.1f}/s", flush=True)

    missing = [ci for ci in range(n_chunks) if not os.path.exists(os.path.join(args.outdir, f"chunk_{ci:05d}.npy"))]
    if missing:
        print(f"[embed] worker finished; {len(missing)} chunks still missing (other workers): {missing[:10]}...")
        return
    parts = [np.load(os.path.join(args.outdir, f"chunk_{ci:05d}.npy")) for ci in range(n_chunks)]
    all_vecs = np.concatenate(parts, axis=0)
    assert all_vecs.shape == (len(texts), 768), all_vecs.shape
    np.save(args.cache, all_vecs)
    print(f"[done] {all_vecs.shape} -> {args.cache}")


if __name__ == "__main__":
    main()

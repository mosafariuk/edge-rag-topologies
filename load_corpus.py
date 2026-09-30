#!/usr/bin/env python3
"""
Load a text corpus with bge-base-en-v1.5 embeddings into the `documents` table (schema.sql)
using pgvector's BINARY COPY wire format, then build the HNSW index.

Why binary: the July 2026 benchmark showed text COPY bottlenecked on client-side float
formatting (28.5k rows/min). Binary COPY sends each vector as int16 dim, int16 unused, then
dim x big-endian float4, straight from a NumPy buffer.

Input: a UTF-8 text file with one passage per line (--passages), or --generate N synthetic
passages (for smoke tests only; real text is preferred for the end-to-end probes).

    pip install numpy "psycopg[binary]" sentence-transformers
    DATABASE_URL=postgres://... python3 load_corpus.py --passages passages.txt --limit 100000
    DATABASE_URL=postgres://... python3 load_corpus.py --passages passages.txt --limit 100000 \
        --target replica_url   # to load a second database identically (same seed/order)

The same file loaded in the same order into the primary and every replica candidate gives
identical ids, so retrieval results are comparable across topologies.
"""
from __future__ import annotations

import argparse
import os
import struct
import time

import numpy as np
import psycopg

MODEL = "BAAI/bge-base-en-v1.5"
DIM = 768


def passages_from_file(path: str, limit: int) -> list[str]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                out.append(s)
            if len(out) >= limit:
                break
    return out


def synthetic_passages(n: int, seed: int = 42) -> list[str]:
    rng = np.random.default_rng(seed)
    topics = ["fibre latency", "vector indexes", "edge computing", "connection pooling",
              "retrieval-augmented generation", "PostgreSQL tuning", "serverless isolates",
              "replication lag", "time to first token", "HNSW graphs"]
    verbs = ["reduces", "explains", "measures", "bounds", "compares", "dominates", "amortizes"]
    objs = ["query latency", "recall at ten", "cold-start time", "index build memory",
            "round-trip cost", "page packing", "tail latency", "throughput"]
    return [f"Passage {i}: {rng.choice(topics)} {rng.choice(verbs)} {rng.choice(objs)} "
            f"in configuration {rng.integers(1, 10_000)}." for i in range(n)]


def embed_all(model, texts: list[str], batch: int = 32) -> np.ndarray:
    vecs = model.encode(texts, batch_size=batch, normalize_embeddings=True,
                        show_progress_bar=True, convert_to_numpy=True)
    return np.ascontiguousarray(vecs.astype("float32"))


def binary_copy(conn: psycopg.Connection, texts: list[str], vecs: np.ndarray) -> None:
    """pgvector binary format per value: int16 dim, int16 unused(0), then dim x float4 BE."""
    hdr = struct.pack(">hh", DIM, 0)
    with conn.cursor() as cur:
        with cur.copy("COPY documents (content, embedding) FROM STDIN WITH (FORMAT BINARY)") as cp:
            cp.set_types(["text", "vector"])
            be = vecs.astype(">f4")
            for i, text in enumerate(texts):
                cp.write_row([text, hdr + be[i].tobytes()])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", help="text file, one passage per line")
    ap.add_argument("--generate", type=int, default=0, help="synthetic passages (smoke test only)")
    ap.add_argument("--limit", type=int, default=100_000)
    ap.add_argument("--target", default=os.environ.get("DATABASE_URL"), help="Postgres URL")
    ap.add_argument("--m", type=int, default=16)
    ap.add_argument("--ef-construction", type=int, default=200)
    ap.add_argument("--skip-index", action="store_true")
    ap.add_argument("--cache", help=".npy file to save/reuse embeddings so replicas load identical vectors")
    ap.add_argument("--embed-only", action="store_true", help="compute --cache and exit (no database)")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--max-seq", type=int, default=384, help="token limit per passage (model max 512)")
    args = ap.parse_args()
    if not args.target and not args.embed_only:
        raise SystemExit("set DATABASE_URL or --target")

    texts = passages_from_file(args.passages, args.limit) if args.passages else synthetic_passages(args.generate or args.limit)
    print(f"[corpus] {len(texts)} passages ({'file' if args.passages else 'SYNTHETIC'})")

    if args.cache and os.path.exists(args.cache):
        vecs = np.load(args.cache)
        assert vecs.shape == (len(texts), DIM), f"cache shape {vecs.shape} != {(len(texts), DIM)}"
        print(f"[embed] reused {args.cache} {vecs.shape}")
    else:
        from sentence_transformers import SentenceTransformer
        import torch
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        model = SentenceTransformer(MODEL, device=device)
        model.max_seq_length = args.max_seq  # MS MARCO passages are short; 384 tokens covers them
        print(f"[embed] device={device} batch={args.batch} max_seq_length={args.max_seq}")
        t0 = time.time()
        vecs = embed_all(model, texts, batch=args.batch)
        print(f"[embed] {vecs.shape} in {time.time()-t0:.0f}s")
        if args.cache:
            np.save(args.cache, vecs)
    if args.embed_only:
        print(f"[done] embeddings -> {args.cache}")
        return

    with psycopg.connect(args.target) as conn:
        with conn.cursor() as cur:
            cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
            cur.execute("""CREATE TABLE IF NOT EXISTS documents (
                id bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
                content text NOT NULL, source text, created_at timestamptz NOT NULL DEFAULT now(),
                embedding vector(768) NOT NULL)""")
            cur.execute("TRUNCATE documents RESTART IDENTITY")
        conn.commit()
        t0 = time.time()
        binary_copy(conn, texts, vecs)
        conn.commit()
        print(f"[copy] {len(texts)} rows in {time.time()-t0:.1f}s ({len(texts)/(time.time()-t0)*60:,.0f} rows/min)")
        if not args.skip_index:
            with conn.cursor() as cur:
                cur.execute("SET maintenance_work_mem = '1GB'")
                cur.execute("DROP INDEX IF EXISTS documents_embedding_hnsw")
                t0 = time.time()
                cur.execute(f"CREATE INDEX documents_embedding_hnsw ON documents USING hnsw "
                            f"(embedding vector_cosine_ops) WITH (m = {args.m}, ef_construction = {args.ef_construction})")
                cur.execute("ANALYZE documents")
            conn.commit()
            print(f"[index] hnsw m={args.m} ef_c={args.ef_construction} in {time.time()-t0:.1f}s")
        with conn.cursor() as cur:
            cur.execute("SELECT count(*), pg_size_pretty(pg_total_relation_size('documents')) FROM documents")
            print("[done]", cur.fetchone())


if __name__ == "__main__":
    main()

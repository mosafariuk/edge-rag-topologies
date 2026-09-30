#!/usr/bin/env python3
"""
Owned pgvector ANN benchmark: HNSW vs IVFFlat, recall vs latency vs QPS.
=======================================================================
Produces OWNED measurements (not blog numbers) to back paper §V-E. It:
  1. Generates N clustered, L2-normalized d-dim vectors (mixture-of-Gaussians to mimic the
     cluster structure of real embeddings) into a disk-backed float32 memmap.
  2. Computes EXACT top-k ground truth for a held-out query set via chunked brute force.
  3. Loads the vectors into Postgres/pgvector.
  4. Builds a grid of HNSW and IVFFlat indexes, and for each sweeps its query-time recall dial
     (hnsw.ef_search / ivfflat.probes), measuring recall@k, latency p50/p95/p99, and QPS.
  5. Writes results/results.csv (checkpointed per row) + results/summary.json.

Methodology notes:
  * Data is SYNTHETIC clustered vectors, not a real corpus: recall/latency characterise the
    index mechanics, not any specific embedding model's geometry. Flagged in output.
  * Each timed sweep is preceded by an untimed warm-up (--warmup) and repeated --repeats
    times; p50/p95/p99 are the median across runs and the per-run p50s are recorded, so
    cache-warming is not confounded with the recall dial.
  * QPS is single-connection (serial); concurrent QPS would be higher. Reported as such.
  * The exact invocation, host note, server version and pgvector version are written to
    results/summary.json by the harness itself.

Usage:
  python bench.py --n 1000000 --dim 768 --queries 1000 --k 10
Env: DSN defaults to the docker-compose DB on host port 55432.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import time
from pathlib import Path

import numpy as np
import psycopg

DSN = os.environ.get("BENCH_DSN", "postgresql://bench:bench@127.0.0.1:55432/bench")
OUT = Path(__file__).parent / "results"
DATA = Path(__file__).parent / "data"


# --------------------------------------------------------------------------- data
def generate_vectors(n: int, dim: int, n_clusters: int, seed: int, path: Path) -> np.memmap:
    """Mixture-of-Gaussians, then L2-normalize. Disk-backed to stay within RAM."""
    if path.exists() and path.stat().st_size == n * dim * 4:
        print(f"[data] reuse existing memmap {path} ({n}x{dim})")
        return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))
    print(f"[data] generating {n}x{dim} clustered vectors -> {path}")
    rng = np.random.default_rng(seed)
    centers = rng.standard_normal((n_clusters, dim)).astype("float32")
    mm = np.memmap(path, dtype="float32", mode="w+", shape=(n, dim))
    batch = 50_000
    for start in range(0, n, batch):
        end = min(start + batch, n)
        b = end - start
        cid = rng.integers(0, n_clusters, size=b)
        vecs = centers[cid] + 0.35 * rng.standard_normal((b, dim)).astype("float32")
        norms = np.linalg.norm(vecs, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mm[start:end] = (vecs / norms).astype("float32")
        if start % 250_000 == 0:
            print(f"[data]   {end}/{n}")
    mm.flush()
    return np.memmap(path, dtype="float32", mode="r", shape=(n, dim))


def make_queries(dim: int, q: int, n_clusters: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed + 1)
    centers = np.random.default_rng(seed).standard_normal((n_clusters, dim)).astype("float32")
    cid = rng.integers(0, n_clusters, size=q)
    vecs = centers[cid] + 0.35 * rng.standard_normal((q, dim)).astype("float32")
    vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
    return vecs.astype("float32")


def exact_topk(vectors: np.memmap, queries: np.ndarray, k: int) -> np.ndarray:
    """Chunked brute-force cosine (dot product on normalized vectors). Returns (Q,k) indices."""
    n, _ = vectors.shape
    q = queries.shape[0]
    best_sim = np.full((q, k), -np.inf, dtype="float32")
    best_idx = np.zeros((q, k), dtype="int64")
    chunk = 50_000
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = np.asarray(vectors[start:end])          # (c, dim)
        sims = queries @ block.T                          # (q, c)
        # merge current top-k with this block
        cand_sim = np.concatenate([best_sim, sims], axis=1)
        cand_idx = np.concatenate(
            [best_idx, np.broadcast_to(np.arange(start, end), (q, end - start))], axis=1
        )
        part = np.argpartition(-cand_sim, k - 1, axis=1)[:, :k]
        rows = np.arange(q)[:, None]
        best_sim = cand_sim[rows, part]
        best_idx = cand_idx[rows, part]
    print(f"[gt] exact top-{k} for {q} queries in {time.time()-t0:.1f}s")
    return best_idx


# --------------------------------------------------------------------------- db
def load_table(conn: psycopg.Connection, vectors: np.memmap, dim: int) -> None:
    n = vectors.shape[0]
    with conn.cursor() as cur:
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
        cur.execute("DROP TABLE IF EXISTS documents")
        cur.execute(f"CREATE TABLE documents (id bigint, embedding vector({dim}))")
    conn.commit()

    print(f"[load] COPY {n} rows ...")
    t0 = time.time()
    batch = 20_000
    with conn.cursor() as cur:
        with cur.copy("COPY documents (id, embedding) FROM STDIN") as copy:
            for start in range(0, n, batch):
                end = min(start + batch, n)
                block = np.asarray(vectors[start:end])
                # C-fast vectorized float formatting, then wrap each row as a pgvector literal.
                strs = np.char.mod("%.6g", block)                 # (b, dim) str array
                for i in range(end - start):
                    copy.write(f"{start+i}\t[{','.join(strs[i])}]\n")
                if start % 200_000 == 0:
                    print(f"[load]   {end}/{n}  ({time.time()-t0:.0f}s)")
    conn.commit()
    with conn.cursor() as cur:
        cur.execute("ANALYZE documents")
    conn.commit()
    print(f"[load] done in {time.time()-t0:.0f}s")


def drop_indexes(conn: psycopg.Connection) -> None:
    with conn.cursor() as cur:
        cur.execute("DROP INDEX IF EXISTS idx_ann")
    conn.commit()


def build_index(conn: psycopg.Connection, kind: str, params: dict) -> tuple[float, str]:
    drop_indexes(conn)
    with conn.cursor() as cur:
        # Serial build: uses backend-local maintenance_work_mem, NOT /dev/shm dynamic shared
        # memory, so it avoids the parallel-worker shm segment that overflows a small container
        # /dev/shm. Slower but robust on memory-constrained hosts.
        cur.execute("SET max_parallel_maintenance_workers = 0")
        cur.execute("SET maintenance_work_mem = '2GB'")
    conn.commit()
    if kind == "hnsw":
        opts = f"m = {params['m']}, ef_construction = {params['ef_construction']}"
        ddl = f"CREATE INDEX idx_ann ON documents USING hnsw (embedding vector_cosine_ops) WITH ({opts})"
    elif kind == "ivfflat":
        ddl = (f"CREATE INDEX idx_ann ON documents USING ivfflat (embedding vector_cosine_ops) "
               f"WITH (lists = {params['lists']})")
    else:
        raise ValueError(kind)
    t0 = time.time()
    with conn.cursor() as cur:
        cur.execute(ddl)
    conn.commit()
    build_s = time.time() - t0
    with conn.cursor() as cur:
        cur.execute("SELECT pg_size_pretty(pg_relation_size('idx_ann'))")
        size = cur.fetchone()[0]
    print(f"[build] {kind} {params} -> {build_s:.1f}s, index {size}")
    return build_s, size


def verify_index_used(conn: psycopg.Connection, queries: np.ndarray, k: int, setup: list[str]) -> str:
    """EXPLAIN one query to confirm the ANN index is actually used (not a seqscan fallback)."""
    vec = "[" + ",".join(f"{x:.6g}" for x in queries[0]) + "]"
    with conn.cursor() as cur:
        for s in setup:
            cur.execute(s)
        cur.execute(
            "EXPLAIN (FORMAT TEXT) SELECT id FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
            (vec, k),
        )
        plan = " ".join(r[0] for r in cur.fetchall())
    used = "idx_ann" in plan or "Index Scan" in plan
    if not used:
        print(f"[warn] index NOT used in plan: {plan[:160]}")
    return "index" if used else "seqscan"


def _sweep(conn: psycopg.Connection, queries: np.ndarray, gt: np.ndarray | None, k: int,
           setup: list[str]) -> tuple[np.ndarray, np.ndarray]:
    q = queries.shape[0]
    lat_ms = np.empty(q, dtype="float64")
    recalls = np.empty(q, dtype="float64")
    with conn.cursor() as cur:
        for i in range(q):
            vec = "[" + ",".join(f"{x:.6g}" for x in queries[i]) + "]"
            for s in setup:  # SET LOCAL enable_seqscan/ef_search/probes (per-txn)
                cur.execute(s)
            t0 = time.perf_counter()
            cur.execute(
                "SELECT id FROM documents ORDER BY embedding <=> %s::vector LIMIT %s",
                (vec, k),
            )
            got = [r[0] for r in cur.fetchall()]
            lat_ms[i] = (time.perf_counter() - t0) * 1000.0
            if gt is not None:
                recalls[i] = len(set(gt[i].tolist()).intersection(got)) / k
    return lat_ms, recalls


def measure(conn: psycopg.Connection, queries: np.ndarray, gt: np.ndarray, k: int,
            dial_sql: str, warmup: int = 50, repeats: int = 3) -> dict:
    """Warm-up (untimed) + `repeats` timed sweeps; percentiles are medians across runs."""
    setup = [s.strip() for s in dial_sql.split(";") if s.strip()]
    if warmup > 0:
        _sweep(conn, queries[:warmup], None, k, setup)
    runs = [_sweep(conn, queries, gt, k, setup) for _ in range(max(1, repeats))]
    p50s = [float(np.percentile(l, 50)) for l, _ in runs]
    p95s = [float(np.percentile(l, 95)) for l, _ in runs]
    p99s = [float(np.percentile(l, 99)) for l, _ in runs]
    means = [float(l.mean()) for l, _ in runs]
    return {
        "recall_at_k": float(np.mean([r.mean() for _, r in runs])),
        "p50_ms": float(np.median(p50s)),
        "p95_ms": float(np.median(p95s)),
        "p99_ms": float(np.median(p99s)),
        "mean_ms": float(np.median(means)),
        "qps_serial": float(1000.0 / np.median(means)),
        "n_queries": int(queries.shape[0]),
        "warmup": warmup,
        "repeats": len(runs),
        "p50_runs": json.dumps([round(x, 3) for x in p50s]),
    }


# --------------------------------------------------------------------------- grid
HNSW_BUILDS = [
    {"m": 16, "ef_construction": 64},    # pgvector defaults
    {"m": 16, "ef_construction": 200},
    {"m": 32, "ef_construction": 200},
]
HNSW_EF_SEARCH = [40, 100, 200, 400]
IVF_BUILDS = [{"lists": 1000}, {"lists": 4000}]
IVF_PROBES = [1, 10, 40, 100]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1_000_000)
    ap.add_argument("--dim", type=int, default=768)
    ap.add_argument("--queries", type=int, default=1000)
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--clusters", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--seqscan-queries", type=int, default=30)
    ap.add_argument("--resume", action="store_true",
                    help="reuse an already-loaded table + existing memmap; skip DB load and the "
                         "seqscan baseline; append to results.csv (for restarting after a build failure)")
    ap.add_argument("--warmup", type=int, default=50, help="untimed queries before each timed sweep")
    ap.add_argument("--repeats", type=int, default=3, help="timed sweeps per configuration")
    ap.add_argument("--note", default="host not described; pass --note with hardware/VM details")
    ap.add_argument("--summarize-only", action="store_true",
                    help="rebuild results/summary.json from an existing results.csv without a DB")
    ap.add_argument("--provenance-note", default="",
                    help="free-text provenance appended to summary.json (e.g. for runs made "
                         "before the harness recorded its own configuration)")
    args = ap.parse_args()

    OUT.mkdir(exist_ok=True)
    if args.summarize_only:
        write_summary(OUT / "results.csv", args, server=None, provenance=args.provenance_note)
        return

    DATA.mkdir(exist_ok=True)

    vectors = generate_vectors(args.n, args.dim, args.clusters, args.seed, DATA / "vectors.f32")
    queries = make_queries(args.dim, args.queries, args.clusters, args.seed)
    gt = exact_topk(vectors, queries, args.k)

    csv_path = OUT / "results.csv"
    fields = ["index", "build_params", "build_s", "index_size", "dial", "recall_at_k",
              "p50_ms", "p95_ms", "p99_ms", "mean_ms", "qps_serial", "n", "dim", "k",
              "queries", "warmup", "repeats", "p50_runs", "note"]
    resume = args.resume and csv_path.exists()
    if resume:
        with open(csv_path) as f:
            existing = next(csv.reader(f))
        if existing != fields:
            raise SystemExit(f"--resume: {csv_path} has an older column layout {existing}; "
                             f"move it aside or start a fresh run")
    fcsv = open(csv_path, "a" if resume else "w", newline="")
    writer = csv.DictWriter(fcsv, fieldnames=fields)
    if not resume:
        writer.writeheader()

    def record(index, bp, build_s, size, dial, m):
        m = dict(m)
        row = {"index": index, "build_params": json.dumps(bp), "build_s": round(build_s, 1),
               "index_size": size, "dial": dial, "n": args.n, "dim": args.dim, "k": args.k,
               "queries": m.pop("n_queries"), "note": args.note}
        row.update({kk: round(vv, 4) if isinstance(vv, float) else vv for kk, vv in m.items()})
        writer.writerow(row)
        fcsv.flush()
        print(f"[result] {index} {bp} {dial}: recall={m['recall_at_k']:.3f} "
              f"p50={m['p50_ms']:.2f}ms p99={m['p99_ms']:.2f}ms qps={m['qps_serial']:.0f}")

    with psycopg.connect(DSN, autocommit=False) as conn:
        with conn.cursor() as cur:
            cur.execute("SELECT version()")
            pg_version = cur.fetchone()[0]
            cur.execute("SELECT extversion FROM pg_extension WHERE extname = 'vector'")
            r = cur.fetchone()
            pgvector_version = r[0] if r else "not installed yet"
        server = {"postgres": pg_version, "pgvector": pgvector_version}
        if resume:
            with conn.cursor() as cur:
                cur.execute("SELECT count(*) FROM documents")
                have = cur.fetchone()[0]
            print(f"[resume] reusing loaded table ({have} rows); skipping load + seqscan baseline")
        else:
            load_table(conn, vectors, args.dim)

            # Exact seq-scan baseline (recall=1.0 by definition); fewer queries since it's slow.
            with conn.cursor() as cur:
                cur.execute("SET LOCAL enable_indexscan = off")
            drop_indexes(conn)
            base_q = queries[: args.seqscan_queries]
            base_gt = gt[: args.seqscan_queries]
            m = measure(conn, base_q, base_gt, args.k, "SET LOCAL enable_indexscan = off",
                        warmup=min(args.warmup, 5), repeats=args.repeats)
            record("seqscan_exact", {}, 0.0, "n/a", "exact", m)

        # HNSW grid. Force the index path (enable_seqscan=off) so we measure the index, not a
        # planner fallback to seqscan on any config.
        for bp in HNSW_BUILDS:
            build_s, size = build_index(conn, "hnsw", bp)
            print("[plan]", verify_index_used(conn, queries, args.k,
                  ["SET LOCAL enable_seqscan=off", "SET LOCAL hnsw.ef_search = 100"]))
            for ef in HNSW_EF_SEARCH:
                m = measure(conn, queries, gt, args.k,
                            f"SET LOCAL enable_seqscan=off; SET LOCAL hnsw.ef_search = {ef}",
                            warmup=args.warmup, repeats=args.repeats)
                record("hnsw", bp, build_s, size, f"ef_search={ef}", m)

        # IVFFlat grid
        for bp in IVF_BUILDS:
            build_s, size = build_index(conn, "ivfflat", bp)
            print("[plan]", verify_index_used(conn, queries, args.k,
                  ["SET LOCAL enable_seqscan=off", "SET LOCAL ivfflat.probes = 10"]))
            for pr in IVF_PROBES:
                m = measure(conn, queries, gt, args.k,
                            f"SET LOCAL enable_seqscan=off; SET LOCAL ivfflat.probes = {pr}",
                            warmup=args.warmup, repeats=args.repeats)
                record("ivfflat", bp, build_s, size, f"probes={pr}", m)

    fcsv.close()
    write_summary(csv_path, args, server, provenance=args.provenance_note)


def write_summary(csv_path: Path, args: argparse.Namespace, server: dict | None,
                  provenance: str = "") -> None:
    """summary.json is written ONLY by the harness so that config and data stay in sync."""
    import sys
    from datetime import datetime, timezone
    rows = list(csv.DictReader(open(csv_path)))
    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "argv": sys.argv,
        "config": vars(args),
        "server": server,
        "data": "synthetic clustered mixture-of-Gaussians, L2-normalized (NOT a real corpus)",
        "qps_note": "serial single-connection QPS; concurrent throughput would be higher",
        "provenance": provenance,
        "rows": rows,
    }
    json.dump(summary, open(OUT / "summary.json", "w"), indent=2)
    print(f"\n[done] {len(rows)} result rows -> {csv_path}; summary -> {OUT / 'summary.json'}")


if __name__ == "__main__":
    main()

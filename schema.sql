-- Edge-RAG retrieval tier: PostgreSQL + pgvector
-- ================================================
-- Target: dense retrieval over millions of 768-dim embeddings (bge-base-en-v1.5),
-- optimized for low-latency cosine ANN search at the database layer.
--
-- Requires PostgreSQL 13+ and pgvector >= 0.5.0 (for HNSW). Tested syntax against pgvector 0.7+.
-- Parameter guidance below follows the pgvector documentation; the recall/latency effect of
-- each dial on a 300k x 768 corpus is measured in the paper (Tables I-II, results/results.csv).

-- 1. Extension -------------------------------------------------------------
CREATE EXTENSION IF NOT EXISTS vector;

-- 2. Schema ----------------------------------------------------------------
-- Store L2-NORMALIZED embeddings at write time so cosine distance reduces to a dot product.
-- vector(768) = 768 * 4 bytes = 3,072 bytes of raw vector per row (dominant memory term at scale).
CREATE TABLE IF NOT EXISTS documents (
    id          bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    content     text        NOT NULL,
    -- optional metadata for hybrid / filtered retrieval
    source      text,
    created_at  timestamptz NOT NULL DEFAULT now(),
    embedding   vector(768) NOT NULL
);

-- 3. HNSW index (recommended default: highest query-time recall/throughput) ---
-- Operator class `vector_cosine_ops` powers the `<=>` cosine-distance operator.
--   m               : max edges per node. Default 16. Raise to 24/32 ONLY if recall is insufficient.
--                     Larger m -> better recall, more memory, slower build.
--   ef_construction : build-time candidate list. Default 64; useful range 64-200.
--                     Higher -> better graph quality, slower build.
-- Build tips for millions of rows:
--   * Raise maintenance_work_mem so the graph builds in memory (see session GUCs below).
--   * Use parallel workers (pgvector supports parallel HNSW builds).
--   * Consider building the index AFTER bulk load for speed.
SET maintenance_work_mem = '8GB';          -- size to available RAM; larger = faster in-memory build
SET max_parallel_maintenance_workers = 7;  -- parallelize the build across cores

CREATE INDEX IF NOT EXISTS documents_embedding_hnsw
    ON documents
    USING hnsw (embedding vector_cosine_ops)
    WITH (m = 16, ef_construction = 64);

-- 4. IVFFlat alternative (lower memory, faster build, generally lower peak recall) -----
-- Choose ONE index family; do not build both on the same column in production.
--   lists  ~= rows / 1000 for up to ~1M rows; ~= sqrt(rows) beyond that (rule of thumb).
-- IMPORTANT: build IVFFlat only AFTER the table has representative data (it clusters existing rows).
--
-- CREATE INDEX documents_embedding_ivf
--     ON documents
--     USING ivfflat (embedding vector_cosine_ops)
--     WITH (lists = 1000);

-- 5. Query-time recall/latency dial ---------------------------------------
-- hnsw.ef_search (default 40): larger -> higher recall, higher latency (measured: recall@10
-- 0.90 -> 0.97 from ef_search 40 -> 400 at m=16/ef_construction=64; Table I).
-- SET LOCAL only takes effect INSIDE a transaction block; outside one PostgreSQL warns and
-- ignores it. Under a transaction-mode pooler (Hyperdrive, PgBouncer) a session-level SET may
-- land on a different server connection than the query. Always pair SET LOCAL with the query
-- in one transaction, as worker.ts does:
--   BEGIN; SET LOCAL hnsw.ef_search = 100; SELECT ...; COMMIT;
--
-- For IVFFlat, the analogous dial is:
--   SET ivfflat.probes = 10;                  -- more probes -> higher recall, higher latency

-- 6. Example cosine ANN query ---------------------------------------------
-- `<=>` returns cosine distance in [0, 2]; similarity = 1 - distance for normalized vectors.
-- ORDER BY ... <=> query ASC + LIMIT k lets the planner use the ANN index.
--
--   SET LOCAL hnsw.ef_search = 100;
--   SELECT id, content, (embedding <=> $1::vector) AS distance
--   FROM documents
--   ORDER BY embedding <=> $1::vector
--   LIMIT 5;

-- 7. Operational notes ----------------------------------------------------
-- * Keep the working set (raw vectors + graph) in RAM. Once it spills to disk, HNSW's
--   random-access traversal degrades sharply (paper Sec. V-C). For 1M x 768-dim, raw vectors
--   alone are ~2.86 GiB. Index size is quantised by page packing: with 8 KiB pages exactly two
--   (element + neighbour) tuple pairs fit per page for dim=768 at any m up to ~48, so the
--   on-disk index is ~N/2 pages regardless of m (measured: 1172 MB for both m=16 and m=32).
-- * ANALYZE documents; after bulk load so the planner has fresh statistics.
-- * For filtered retrieval (WHERE source = ...), test recall carefully: pre-filtering can
--   interact poorly with ANN indexes. Consider partial indexes per high-cardinality filter.

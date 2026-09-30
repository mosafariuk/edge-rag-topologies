# Edge-RAG: Latency Optimization in RAG via Edge Computing

An IEEE-format paper, a reference implementation, an analytical latency model and an owned
pgvector benchmark for the question: *when does moving the retrieval prefix of RAG (query
embedding + vector search) to edge compute reduce time-to-first-token?*

**Short answer from the model:** an edge Worker with a pooled connection to a central vector tier
matches a centralized deployment; an unpooled edge Worker is far slower; the distance-dependent
gain (about one user-to-region round trip) appears only when the vector tier is replicated near
the point of presence. Generation remains the largest term everywhere.

## Contents

| File | What it is |
|------|-----------|
| `paper.md` | The paper in Markdown (abstract to references). |
| `ieee_manuscript.tex` / `.pdf` | The same paper in IEEE conference format (8 pages; `tectonic ieee_manuscript.tex`). |
| `SOURCES.md` | Every source with a quality tag, the figure taken from it, and the list of claims removed in the 2026-09-30 revision. |
| `worker.ts`, `wrangler.toml`, `package.json`, `tsconfig.json` | Cloudflare Worker: embed at the edge (Workers AI, 768-d) → pooled pgvector cosine search inside one transaction → `Server-Timing` per stage. `npm run typecheck`. |
| `schema.sql` | pgvector schema, HNSW / IVFFlat DDL and parameter notes. |
| `hyperdrive.md` | Edge→Postgres pooling: Hyperdrive and PgBouncer, what pooling does and does not fix. |
| `latency_sim.py` | Analytical calculator for four topologies (no I/O, every constant tagged) plus the HNSW cost and index-size models. |
| `bench.py`, `docker-compose.yml` | Owned pgvector benchmark harness (HNSW vs IVFFlat recall@10 / latency / QPS grid). |
| `results/` | Raw run outputs (`results.csv`, `run.log`), harness-written `summary.json`, and `ERRATA.md`. |

## Run the model

```bash
python3 latency_sim.py                          # four topologies × five distances
python3 latency_sim.py --rtt sydney=205         # replace the modelled RTT with a measured one
python3 latency_sim.py --hnsw-sweep             # HNSW cost + page-packing index-size model
python3 latency_sim.py --validate results/results.csv   # cost-model floor vs measured p50s
python3 latency_sim.py --json
```

## Reproduce the retrieval benchmark

```bash
python3 -m venv .venv && ./.venv/bin/pip install numpy "psycopg[binary]"
docker compose up -d
BENCH_DSN="postgresql://bench:bench@127.0.0.1:55432/bench" \
  ./.venv/bin/python -u bench.py --n 300000 --dim 768 --queries 1000 --k 10 --clusters 1000 \
  --warmup 50 --repeats 3 --note "<describe the host and VM>"
```

The harness writes `results/summary.json` itself (invocation, versions, config). Use `--resume`
to restart from the index-build step after a failure. The July 2026 run in `results/` predates
the warm-up/repeats options; `results/ERRATA.md` lists its known artifacts.

## Deploy the Worker (sketch)

```bash
npm install && npm run typecheck
npx wrangler hyperdrive create edge-rag-pg --connection-string="postgres://USER:PWD@HOST:5432/DB"
# paste the id into wrangler.toml, then:
npx wrangler deploy
```

Apply `schema.sql`, load L2-normalized 768-d embeddings into `documents`, build the HNSW index and
`ANALYZE`. Responses carry `Server-Timing: embed;dur=…, search;dur=…, total;dur=…`.

## Status and next step

The four topologies are modelled, not yet measured end to end. The next phase deploys the Worker
in the centralized, edge-unpooled, edge-pooled and edge-replica configurations and probes them
from at least five geographies (paper Section VII-D).

## Revision note (2026-09-30)

An audit against the cited sources and the raw data found misattributed cold-start figures, a
mis-specified simulator baseline, an HNSW cost model contradicted by the measurements, and a
Worker bug that left `ef_search` at its default. All were corrected in this revision; the removed
claims are listed at the end of `SOURCES.md`.

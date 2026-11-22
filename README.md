# Edge-RAG: Latency Optimization in RAG via Edge Computing

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.24447712.svg)](https://doi.org/10.5281/zenodo.24447712)

An IEEE-format measurement study, with all code and raw data, of the question: *when does moving
the retrieval prefix of RAG (query embedding + vector search) to edge compute make it faster?*

**Measured answer (30,000 requests, five regions, five topologies, 1 Oct 2026):**

| Probe | Centralized | Edge, unpooled | Edge, pooled | Edge + near-user DB copy (Sydney) |
|---|---|---|---|---|
| Virginia | 92 | 141 | **76** | 316 |
| London | 167 | 923 | **150** | 397 |
| Sao Paulo | **205** | 1,360 | 257 | 495 |
| Mumbai | **275** | 2,595 | 353 | 411 |
| Sydney | 295 | 2,358 | 427 | **191** |

Median wall time in ms on warm connections. A pooled edge Worker is roughly level near the
database's region, loses for distant users unless the database is near them too, and an unpooled
one pays 9-13 connection-setup round trips. On cold connections the pooled edge Worker beats the
centralized server everywhere outside the database's region because TLS terminates at the PoP;
an edge TLS proxy in front of the unchanged central server does better still from São Paulo,
Mumbai and Sydney.

## Contents

| File | What it is |
|------|-----------|
| `paper.md` | The paper in Markdown (abstract to references). |
| `ieee_manuscript.tex` / `.pdf` | The same paper in IEEE conference format (9 pages; `tectonic ieee_manuscript.tex`). |
| `SOURCES.md` | Every source with a quality tag, the figure taken from it, and the list of claims removed in the 2026-09-30 revision. |
| `worker.ts`, `wrangler.toml`, `package.json`, `tsconfig.json` | Cloudflare Worker: embed at the edge (Workers AI, 768-d) → pooled pgvector cosine search inside one transaction → `Server-Timing` per stage. `npm run typecheck`. |
| `schema.sql` | pgvector schema, HNSW / IVFFlat DDL and parameter notes. |
| `hyperdrive.md` | Edge→Postgres pooling: Hyperdrive and PgBouncer, what pooling does and does not fix. |
| `latency_sim.py` | Analytical calculator for four topologies (no I/O, every constant tagged) plus the HNSW cost and index-size models. |
| `bench.py`, `docker-compose.yml` | Owned pgvector benchmark harness (HNSW vs IVFFlat recall@10 / latency / QPS grid). |
| `results_phase3/` | Retrieval-tier benchmark behind Tables IV–V: 1M MS MARCO bge-base embeddings, raw CSV, logs, harness-written `summary.json`, embedding checksums. |
| `results/` | Superseded July 2026 retrieval run on 300k synthetic vectors (kept with `ERRATA.md`). |
| `baseline/`, `deploy/`, `load_corpus.py`, `embed_chunked.py` | Centralized baseline server, TLS pgvector deployment, binary-COPY corpus loader, resumable embedder. |
| `probe/` | Probe client, matrix runner, analyzer; `probe/results/` holds the 30,000-row end-to-end dataset and its summary. `PHASE2.md` is the runbook. |

## Run the model

```bash
python3 latency_sim.py                          # four topologies × five distances
python3 latency_sim.py --rtt sydney=205         # replace the modelled RTT with a measured one
python3 latency_sim.py --hnsw-sweep             # HNSW cost + page-packing index-size model
python3 latency_sim.py --validate results_phase3/results.csv   # cost-model floor vs measured p50s
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

The harness writes `summary.json` itself (invocation, versions, config). Use `--resume`
to restart from the index-build step after a failure. The July 2026 run in `results/` predates
the warm-up/repeats options; `results/ERRATA.md` lists its known artifacts.

The paper's Tables IV–V come from the real-embedding run: `deploy/phase3_gpu_embed.sh` embeds
1M MS MARCO passages and 1,000 dev queries on one GPU (about 15 min on an RTX 4090), and
`deploy/phase3_run.sh` runs on one large Linux VM (we used GCP `c3-standard-22`): it installs
Docker and pgvector, embeds on CPU if `corpus.npy`/`queries.npy` are absent (about 4.5 h), and runs
`bench.py --vectors corpus.npy --query-vectors queries.npy --out results_phase3 --repeats 5`.

## Deploy the Worker (sketch)

```bash
npm install && npm run typecheck
npx wrangler hyperdrive create edge-rag-pg --connection-string="postgres://USER:PWD@HOST:5432/DB"
# paste the id into wrangler.toml, then:
npx wrangler deploy
```

Apply `schema.sql`, load L2-normalized 768-d embeddings into `documents`, build the HNSW index and
`ANALYZE`. Responses carry `Server-Timing: embed;dur=…, search;dur=…, total;dur=…`.

## Regenerate the paper's tables

```bash
python3 probe/analyze.py probe/results/*.csv          # Table I / II inputs (medians, CIs, stages)
python3 latency_sim.py --compare probe/results/summary.json   # Table III (model vs measurement)
python3 latency_sim.py --validate results_phase3/results.csv  # HNSW cost bound vs Table IV
```

## Status

End-to-end measurements are from a single session (01:20-03:38 UTC). A repeat session was lost
when the cloud instances were terminated before collection; the paper states this limitation.
Open items: repeat sessions, a larger retrieval benchmark on server-class hardware, a comparison
with Workers AI + Vectorize, and peer review.

## Revision note (2026-09-30)

An audit against the cited sources and the raw data found misattributed cold-start figures, a
mis-specified simulator baseline, an HNSW cost model contradicted by the measurements, and a
Worker bug that left `ef_search` at its default. All were corrected in this revision; the removed
claims are listed at the end of `SOURCES.md`.

## License

Paper and data: CC BY 4.0. Code: MIT. See `LICENSE`.

# Phase 2 runbook: measure the four topologies end to end

Goal: replace the assumptions in paper Sec. V-D with measurements. One corpus, one embedding
model, one query set, four (optionally six) deployed topologies, probed from five geographies.

## 0. Inputs needed

| Item | Used for |
|---|---|
| Cloudflare account with Workers AI + Hyperdrive (`npx wrangler login`) | the three edge Workers |
| Postgres + pgvector **primary** in the central region (URL) | all topologies |
| Postgres + pgvector **read replica(s)** in probe regions (URL per region) | edge-replica |
| One VM in the central region (≥4 GB RAM) with a public hostname + TLS | centralized baseline |
| One small VM per probe region: Sydney, London, São Paulo, Mumbai, Virginia | probes |

## 1. Corpus (once, same file into every database)

```bash
pip install numpy "psycopg[binary]" sentence-transformers
DATABASE_URL=<primary> python3 load_corpus.py --passages passages.txt --limit 100000
DATABASE_URL=<replica> python3 load_corpus.py --passages passages.txt --limit 100000   # only if the
# replica is a separate database; a streaming replica inherits the primary's data automatically
```
`passages.txt` = one real passage per line (e.g. Wikipedia or MS MARCO passages). 100k × 768-d fits
free/entry tiers (~300 MB table + ~400 MB index). Binary COPY is used (paper Sec. IV-D).

## 2. Centralized baseline (VM in the DB's region)

```bash
pip install -r baseline/requirements.txt
DATABASE_URL=<primary> POOL=1 EF_SEARCH=100 PORT=8080 python3 baseline/server.py
# TLS in front (same handshake cost as the Workers):  caddy reverse-proxy --from central.<domain> --to :8080
# optional in-region cold-connection variant on another port:  POOL=0 PORT=8081
```

## 3. Edge Workers

```bash
npm install
npx wrangler hyperdrive create edge-rag-primary --connection-string="<primary>"
npx wrangler hyperdrive create edge-rag-replica --connection-string="<replica>"
# paste both ids into wrangler.toml, then:
npx wrangler deploy --env pooled
npx wrangler deploy --env replica
npx wrangler secret put DATABASE_URL --env unpooled     # paste <primary> with sslmode=require
npx wrangler deploy --env unpooled
```
`EF_SEARCH=100` everywhere (same operating point as the baseline). Smart Placement is off.

Sanity check from your laptop (expect `timingMs` and a `colo`):
```bash
curl -s -X POST https://edge-rag-pooled.<acct>.workers.dev -d '{"query":"what does ef_search control"}' -D -
```

## 4. Probes (each region VM)

```bash
scp probe/probe.py probe/run_matrix.sh <vm>:~/
ssh <vm> 'REGION=sydney N=300 CENTRAL_URL=https://central.<domain> \
  EDGE_POOLED_URL=https://edge-rag-pooled.<acct>.workers.dev \
  EDGE_REPLICA_URL=https://edge-rag-replica.<acct>.workers.dev \
  EDGE_UNPOOLED_URL=https://edge-rag-unpooled.<acct>.workers.dev ./run_matrix.sh'
scp <vm>:~/results/sydney.csv probe/results/
```
Design: 300 requests × {cold, warm} × topology × 2 interleaved passes = 1,200 requests per
topology per region. Cold = new TCP+TLS per request; warm = keep-alive. ~15 min per region.
Workers AI free tier covers ~10k neurons/day; 5 regions × 3 Workers × 1,200 ≈ 18k embeddings,
so run over two days or enable pay-as-you-go (cents).

## 5. Analysis

```bash
python3 probe/analyze.py probe/results/*.csv --md probe/results/summary.md --json probe/results/summary.json
python3 latency_sim.py --rtt sydney=<measured> --rtt london=<measured> ...   # model vs measured
```
What goes into the paper: per region × topology, warm and cold wall-time medians with bootstrap
CIs, the server-side stage split (embed / search), the client network component
(wall − server total), the PoP that served each region, and the fitted route factor per path.

## 6. Optional: vendor-default comparison

A fourth Worker using Workers AI + Vectorize with the same corpus and query set
(`VECTORIZE_URL`). Not required for the core result; useful for Sec. II's positioning.

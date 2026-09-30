# Edge-to-Postgres connection pooling

A serverless edge invocation must not open a fresh Postgres connection per request. From a
distant edge PoP, connection *setup* ("upwards of seven round-trips (TCP handshake; TLS
negotiation; then auth)", each "50-200 milliseconds" over long paths **[vendor, Cloudflare 2023]**;
"a bare minimum of 5" in Cloudflare's 2025 description) can dominate the request. In the paper's
model (`latency_sim.py`), an unpooled edge Worker querying a central database from Sydney pays
about 1.4 s of setup per request and is far slower than a centralized deployment; pooling removes
that and brings the edge Worker back to parity. Two options follow.

## Option A — Cloudflare Hyperdrive (managed, recommended for Workers)

Hyperdrive "maintains a pool of connections to your database... optimally placed to minimize the
latency for your applications" **[vendor]**, and the Worker connects to a local pooled endpoint
instead of dialing the origin per request.

```bash
# 1. Create the Hyperdrive config against your existing Postgres origin.
npx wrangler hyperdrive create edge-rag-pg \
  --connection-string="postgres://USER:PASSWORD@HOST:5432/DBNAME"

# 2. Copy the returned id into wrangler.toml under [[hyperdrive]].
# 3. In the Worker, connect using env.HYPERDRIVE.connectionString (see worker.ts).
```

**When the win is large:** Worker and DB are far apart, and connections are reused across many
requests. **When it is small:** Worker and DB are same-region. Pooling makes an edge Worker viable;
it does not by itself make it faster than a centralized deployment (paper Sec. V-D).

## Option B — Self-managed PgBouncer (transaction pooling)

The classic equivalent when not on Cloudflare, or in front of the origin regardless. Use
**transaction** pooling for short serverless queries.

```ini
# pgbouncer.ini
[databases]
appdb = host=127.0.0.1 port=5432 dbname=appdb

[pgbouncer]
listen_addr = 0.0.0.0
listen_port = 6432
auth_type = scram-sha-256
auth_file = /etc/pgbouncer/userlist.txt

# transaction pooling: a server connection is assigned to a client only for the duration
# of a transaction — ideal for many short edge queries sharing a small server pool.
pool_mode = transaction

default_pool_size = 20      # server connections per (user, db) pair
max_client_conn = 5000      # many edge clients multiplexed onto few server connections
server_idle_timeout = 300
```

> Note: under transaction pooling, session-level features (e.g. named prepared statements) can
> misbehave, and a session-level `SET` may land on a different server connection than the query.
> The reference `worker.ts` sets `prepare: false` and issues `SET LOCAL hnsw.ef_search` together
> with the `SELECT` inside one transaction, which pins a single server connection for both.

## What pooling does and does not fix

- **Fixes:** the per-request TCP/TLS/auth setup tax; connection storms from bursty edge traffic.
- **Does not fix:** the *propagation* distance between the pool and the origin. If the origin is
  cross-continent, each query still crosses that distance once. Reducing that requires placing a
  read replica of the vector tier nearer the edge (introduces replication lag; paper Sec. VII).

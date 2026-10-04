# Latency Optimization in Retrieval-Augmented Generation via Edge Computing: A Measurement Study of Edge-RAG Deployment Topologies

Mohammad Safari, Independent Research (mo@aioapex.com)

---

## Abstract

Retrieval-Augmented Generation (RAG) is the prevailing pattern for grounding large language models in external knowledge. Its canonical deployment routes every request to one cloud region that performs query embedding, vector search and generation, so a distant user pays cross-continent round trips before the first token can stream. This paper measures when relocating the retrieval prefix of RAG, query embedding plus vector search, to globally distributed edge compute reduces that delay. We deploy five topologies over one corpus of 100,000 MS MARCO passages in PostgreSQL with pgvector: a centralized server in the database's region, the same server behind an edge TLS proxy, and a Cloudflare Worker that embeds at the edge and reaches the database through an unpooled connection, a pooled connection, or a pooled connection to a copy of the database near the user. Probes in five regions issued 30,000 requests with per-stage server timing. Four findings follow. First, on warm connections an edge Worker with a pooled connection to a central database is 16 ms faster than the centralized baseline for nearby users and 52 to 133 ms slower for distant ones, because the pooled query path costs 1.15 to 1.57 round trips and because managed edge inference embeds a query in a median of about 40 ms at one point of presence and about 165 ms at another. Second, a database copy near the user makes the edge Worker the fastest option there, 104 ms (35 percent) below the baseline, about half of what a propagation-only model predicts. Third, an unpooled edge Worker pays 9 to 13 connection-setup round trips per request, 0.9 to 2.6 s. Fourth, on cold connections the centralized baseline pays two additional round trips for TCP and TLS, so the pooled edge Worker beats it in every region outside the database's own, by 151 to 298 ms, and the near-user copy beats it by 533 ms. An additive latency model predicts the centralized and near-replica topologies within 4 percent and underestimates the pooled and unpooled paths by 7 to 42 percent; we report the calibrated parameters. Separate measurements of the retrieval tier on 300,000 vectors quantify the recall and latency of HNSW and IVFFlat indexes and show that the on-disk HNSW index size is fixed by page packing rather than graph degree. Code, harnesses and raw data are released.

**Index Terms:** retrieval-augmented generation, edge computing, serverless, V8 isolates, pgvector, HNSW, approximate nearest neighbor search, connection pooling, latency measurement, time-to-first-token.

---

## I. Introduction

### A. Retrieval-augmented generation and the primacy of latency

Retrieval-Augmented Generation was formalized by Lewis et al. [1] as the combination of a parametric memory, a pre-trained sequence-to-sequence generator, with a non-parametric memory, "a dense vector index of Wikipedia, accessed with a pre-trained neural retriever". The motivation is that parametric-only models' "ability to access and precisely manipulate knowledge is still limited, and hence on knowledge-intensive tasks, their performance lags behind task-specific architectures" [1]. The non-parametric component introduces a retrieval round trip into the request path and is therefore the component most exposed to where the computation is placed.

For interactive systems that stream tokens, the user-visible latency metric is time-to-first-token (TTFT). Every stage that precedes generation, reaching the service, computing the query embedding and searching the vector index, is charged in full against TTFT. We call these stages the retrieval prefix. This work measures under which conditions moving the retrieval prefix closer to the user shortens it, and by how much.

### B. Why centralization is costly

The default cloud posture concentrates inference and data in a small number of regions. Three effects compound for a distant user:

1. **Propagation floor.** Light in silica fibre travels at roughly two thirds of its vacuum speed, about $2.0\times10^{5}$ km/s, or approximately 5 µs per kilometre one way [5].
2. **Path inflation.** Deployed paths are longer than great-circle paths. Bozkurt et al. [5] measured the ratio of deployed-path latency to the straight-line fibre ideal across all pairs of the 120 largest U.S. population centres and found a median of 2.1, reducible to about 1.3. That result is scoped to the U.S. terrestrial backbone. For the four intercontinental paths in this study we measure the ratio directly and obtain 1.30 to 1.50 (Section VI-B).
3. **Connection setup.** Establishing a new PostgreSQL connection over TLS requires, in Cloudflare's description, "upwards of seven round-trips (TCP handshake; TLS negotiation; then auth)" [14]; a later Cloudflare post gives "a bare minimum of 5" [15]. We measure 9 to 13 (Section VI-D).

### C. Contributions

1. An end-to-end measurement of five deployment topologies for the retrieval prefix of RAG from five regions: 30,000 requests with per-stage server timing, on cold and warm client connections (Section VI).
2. Four empirical findings on when edge placement helps: parity or a small gain near well-provisioned points of presence, a loss for distant users unless the database is also near them, a large penalty without pooling, and a consistent gain on cold connections.
3. An additive latency model, compared against the measurements topology by topology, with the calibrated pooled-path and connection-setup parameters that the comparison yields (Sections V and VI-E).
4. Measurements of the pgvector retrieval tier on 300,000 vectors, a corrected HNSW search-cost bound checked against them, and a page-packing model that explains why index size is invariant to the graph degree parameter at this dimensionality (Sections V-C and VII).
5. Released artifacts: the Worker with per-stage timing, the centralized baseline, the loader, the probe and analysis tools, the simulator, and all raw result files.

---

## II. Related Work

**Retrieval-augmented generation and its latency.** Lewis et al. [1] introduced RAG. Subsequent systems work treats retrieval as a first-order latency component of serving: RAGCache [10] caches the intermediate states of retrieved knowledge across GPU and host memory; PipeRAG [11] overlaps retrieval with generation through pipeline parallelism; TeleRAG [12] prefetches index data from CPU to GPU in parallel with generation. These works optimize retrieval inside a single serving cluster. VectorLiteRAG [7] shows that, for a 128-million-vector database, CPU-side retrieval raises TTFT from 197 ms to 606 ms, and the near-memory accelerator of Quinn et al. [8] reports 13.4 to 27.9 times faster exact search over a 512 GB database than a server CPU, translating into 1.7 to 26.3 times lower end-to-end inference time. Those results concern exact search over corpora far larger than ours; we cite them to establish that retrieval can dominate TTFT at scale, not to transfer their multipliers.

**Approximate nearest-neighbour search.** HNSW [2] builds a multi-layer proximity graph with links "separated by their characteristic distance scales"; the authors report empirically logarithmic scaling of search cost with corpus size. Product quantization [3] and its inverted-file variant established the memory-for-scale trade underlying IVF-style indexes. ANN-Benchmarks [4] finds that "very different approaches to k-NN search yield comparable quality-performance trade-offs", so index choice is an operating-point decision. pgvector [24] exposes both an HNSW and an IVFFlat index inside PostgreSQL.

**On-device RAG.** EdgeRAG [6] targets resource-constrained end-user devices and is evaluated on an NVIDIA Jetson Orin Nano. Its "edge" is the device, not a provider's point of presence; we cite it for two phenomena relevant here, the memory footprint of dense indexes (an 18.5 GB index for 5.23 million FEVER records) and embedding-generation tail latency (most clusters under 500 ms, some over 2 s; an IVF baseline whose 95th-percentile retrieval latency exceeds the median by over 64 times on Natural Questions).

**Serverless execution and cold starts.** Cloudflare states that its V8-isolate runtime loads a Worker in about 5 ms [13]; Python Workers, which carry an interpreter, start in about 1 s with memory snapshots [19]. Cloudflare's container product documents cold starts "often in the 1–3 second range" [18]; a third-party benchmark of 700 boots reports medians of 1.7 to 10.5 s, 95th percentiles of 15.6 to 24.4 s and a maximum of 58.6 s across four workloads [20]. Dirigent [9] shows that sandbox initialization of "10–100s of milliseconds" is dwarfed by cluster scheduling latency in real FaaS deployments. All cold-start figures used here are either vendor-reported or single-benchmark; we treat them as order-of-magnitude evidence only.

**Edge RAG on managed platforms.** Cloudflare's own tutorial [21] builds RAG from Workers AI (the `bge-base-en-v1.5` embedding model) and Vectorize, which the vendor describes as "a globally distributed vector database" [22]. That stack is the vendor default for the problem studied here. This paper instead pairs the edge Worker with a self-managed PostgreSQL and pgvector tier through a connection pool, because it allows the vector tier to be shared with an existing relational schema and because pgvector's index internals are open and measurable. A direct comparison against Vectorize is left to future work (Section IX).

**Fibre-infrastructure latency.** Bozkurt et al. [5] provide the route-inflation measurement discussed in Section V-B.

---

## III. System Architecture

### A. Topologies

We consider a user at great-circle distance $D$ from the cloud region that hosts the vector database, and a nearby point of presence (PoP).

```
 centralized      User ──────────────────────────────► Region: server (embed, CPU) ── pgvector
 central-tunnel   User ──► PoP (TLS proxy) ──────────► Region: server (embed, CPU) ── pgvector
 edge-unpooled    User ──► PoP: Worker (embed) ══ new TCP+TLS+auth per request ════► pgvector
 edge-pooled      User ──► PoP: Worker (embed) ── connection pool ─────────────────► pgvector
 edge-replica     User ──► PoP: Worker (embed) ── connection pool ──► copy of the database near the PoP
```

In the **centralized** baseline one server in the database's region embeds the query on CPU and queries the database over a pooled in-region connection. **Central-tunnel** places a Cloudflare edge proxy in front of the same server, so the user's TLS session terminates at a nearby PoP while all computation stays central. In the three **edge** topologies a Cloudflare Worker at the PoP embeds the query through Workers AI and reaches the database with a fresh connection per request (unpooled), through Cloudflare Hyperdrive (pooled), or through Hyperdrive to a copy of the database in the user's region (replica). Generation is outside the scope of the measurements.

### B. Isolates versus containers for per-request edge execution

Per-request execution in hundreds of locations without a warm pool everywhere is the workload where the start-up cost of the execution unit dominates. Containers and microVMs construct an image, a runtime and an application before serving; the vendor's own figure for its container product is 1 to 3 s [18] and third-party measurements show a long tail [20]. V8 isolates run many tenants' code as sandboxed contexts inside one already-running process; the vendor states about 5 ms to load a JavaScript Worker [13], and about 1 s for Python Workers, which carry an interpreter [19]. These are vendor figures. Our probes include an untimed warm-up and therefore do not measure isolate cold starts.

### C. Edge embedding

The reference implementation uses Workers AI, whose `@cf/baai/bge-base-en-v1.5` model produces 768-dimensional vectors with a 512-token input limit [17]. One implementation detail affects retrieval quality: the service defaults to mean pooling for this model, whereas the model's reference pooling, used when the corpus is embedded with standard tooling, takes the first token. With the default, the Worker and the baseline returned different passages for the same query; with first-token pooling selected explicitly they return the same passages with cosine similarities equal to four decimals.

### D. Retrieval tier and pooling

The retrieval tier is PostgreSQL with pgvector [24]: vectors in a `vector(768)` column, cosine distance through the `<=>` operator, and an HNSW index [2]. Cloudflare Hyperdrive "maintains a pool of connections to your database. These are optimally placed to minimize the latency for your applications" [16]; PgBouncer in transaction mode is the self-managed equivalent. Pooling removes connection setup from the request path. It does not remove the distance between the PoP and the database.

---

## IV. Implementation

The repository contains `worker.ts`, `wrangler.toml`, `baseline/server.py`, `load_corpus.py`, `embed_chunked.py`, `schema.sql`, `deploy/`, `probe/`, `latency_sim.py`, `bench.py`, and the raw results.

### A. Worker

The Worker validates the request, embeds the query, runs the search, and returns the top-$k$ passages with a `Server-Timing` header carrying the measured duration of the embedding stage, the search stage and the invocation. Three details proved necessary.

- **`SET LOCAL` needs a transaction.** `hnsw.ef_search` is set per query with `SET LOCAL`, which PostgreSQL ignores outside a transaction block; under a transaction-mode pooler a session-level `SET` may also reach a different server connection than the query.
- **The transaction must be one round trip.** Sent as four statements, `BEGIN`, `SET LOCAL`, `SELECT`, `COMMIT` each crossed the PoP-to-database distance through the pooler: about 460 ms of search time in pre-test requests served by the London PoP. Sent as a single multi-statement query string, the same work took 125 to 130 ms in the same pre-test and 98 ms at the median of the measurement run. Only validated numbers are inlined into that string.
- **Direct connections need a publicly trusted certificate.** The Workers runtime rejects a self-signed database certificate, and the driver then retries until the invocation's subrequest limit is reached. The database therefore presents a Let's Encrypt certificate; Hyperdrive accepts either.

### B. Centralized baseline

`baseline/server.py` implements the same request contract and the same single-round-trip transaction in Python. It embeds with the same model through `sentence-transformers` on the server's two vCPUs and keeps a connection pool to the database in the same region. Caddy terminates TLS in front of it with a Let's Encrypt certificate.

### C. Corpus loading

The corpus is the first 100,000 passages of the MS MARCO passage collection [25], embedded once; the same vectors are loaded into both databases, so identifiers and results are comparable across topologies. Loading uses pgvector's binary `COPY` format (`int16` dimension, `int16` unused, big-endian `float4` values). Binary `COPY` loaded 100,000 rows in 5.1 s, 1.18 million rows per minute, on the central server; the text `COPY` used for the retrieval-tier benchmark of Section VII, which formats every float as a string on the client, reached 28,500 rows per minute on a laptop. The two were measured on different hosts, so the ratio is indicative only. The HNSW index ($m = 16$, `ef_construction = 200`) built in 124 s.

### D. Observations from building the retrieval benchmark

- **Parallel index builds need shared memory.** pgvector's parallel HNSW build allocates a dynamic shared-memory segment; on a container with a 1 GB `/dev/shm`, a 300,000-vector build requested 1.25 GB and aborted. Building serially uses backend-local `maintenance_work_mem` instead.
- **IVFFlat build memory scales with `lists`.** A `lists = 4000` build required 3,738 MB of `maintenance_work_mem` against a 2,048 MB budget and failed.

---

## V. Latency Model

### A. Additive model

We model the latency to first useful output as

$$
T_{total} = T_{network} + T_{embed} + T_{search} + T_{generation}.
$$

$T_{network}$ collects propagation and protocol round trips on the request path; $T_{embed}$ is embedding compute plus any hop to the embedder; $T_{search}$ is index traversal plus, if the connection is not pooled, connection setup. The measurements of Section VI cover the first three terms. Generation is not measured and is unaffected by the topologies studied.

### B. Network terms

Let $c_f \approx 2.0\times10^{5}$ km/s. For straight-line distance $D$ and route-inflation factor $\rho$, $\text{RTT}(D) = 2\rho D / c_f$. With $R$ the round-trip time between the user's region and the database's region, the structural assumptions for warm client connections are

$$
\begin{aligned}
T_{centralized} &= R + E_c + S, \\
T_{pooled} &= \delta + E_e + k_{pool}\,R + S, \\
T_{unpooled} &= \delta + E_e + (n_{rt} + 1)\,R + S, \\
T_{replica} &= \delta + E_e + S_{local},
\end{aligned}
$$

where $E_c$ and $E_e$ are the central and edge embedding times, $S$ the search time at the database, $\delta$ the client-to-PoP overhead, $k_{pool}$ the number of round trips the pooled query path costs, and $n_{rt}$ the number of connection-setup round trips. A propagation-only reading sets $k_{pool} = 1$ and takes $n_{rt} = 7$ from [14]. On a cold client connection the centralized topology adds the client's own TCP and TLS handshakes over $R$; the edge topologies add them over the short client-to-PoP path.

### C. HNSW search cost and index size

**Search cost.** The candidate-list-bounded beam search of HNSW runs only at layer 0, where each element has up to $2m$ links; the upper layers are a greedy descent with a candidate list of one over about $\log N$ levels. The number of distance evaluations per query is therefore additive in the two terms,

$$
N_{dist} \approx c\,\text{ef}\,(2m) + m\log_2 N,
$$

with $c < 1$ accounting for neighbours already visited. Each evaluation over $d$-dimensional single-precision vectors reads $4d$ bytes; for $d = 768$ that is 3,072 bytes. Because traversal is random access, bytes moved per query, $4d\,N_{dist}$, divided by an effective memory bandwidth gives a lower bound on search compute. With $c = 0.5$ and an assumed 8 GB/s the bound ranges from 0.36 to 5.14 ms over the twelve HNSW configurations of Section VII and lies below every measured median, at 1 to 59 percent of it. A multiplicative form, $c\,\text{ef}\,\bar m \log N$, which appeared in an earlier draft, predicts a floor of 44.7 ms for a configuration measured at 6.7 ms and is withdrawn.

**Index size.** pgvector stores each element's vector and its neighbour list in 8 KiB pages, and tuples are not split across pages. With $d = 768$ an element tuple is about 3,104 bytes and a neighbour tuple about $6(2m + m/(m-1)) + 8$ bytes, so exactly two element-plus-neighbour pairs fit in a page for every $m$ up to about 48. The index therefore occupies $\lceil N/2 \rceil$ pages regardless of $m$: 150,000 pages, 1,171.9 MiB, for $N = 300{,}000$, which is the size PostgreSQL reported for both the $m = 16$ and the $m = 32$ index (Table IV). At this dimensionality raising $m$ costs build time and query bandwidth, not storage.

---

## VI. End-to-End Measurements

### A. Setup

**Servers.** Five AWS `m7i-flex.large` instances (2 vCPU, 8 GB, Ubuntu 26.04): `us-east-1` hosting the central database and the baseline server, and one each in `eu-west-2` (London), `sa-east-1` (São Paulo), `ap-south-1` (Mumbai) and `ap-southeast-2` (Sydney). The Sydney instance also hosts the near-user copy of the database. Both databases run PostgreSQL 17.11 with pgvector 0.8.6 in Docker with TLS, the corpus of Section IV-C, and `hnsw.ef_search = 100`.

**Topologies.** As in Section III-A. The three Workers are one code base deployed with different bindings on the Cloudflare free plan, with Smart Placement off. Edge-replica points its pool at the Sydney database from every PoP, so it is a near-user copy only for the Sydney probe; from the other regions it measures a pool to a distant database. The copy is an independently loaded, identical database rather than a streaming replica, so replication lag is not measured.

**Probes.** A standard-library Python client on each instance issues `POST` requests with one of ten fixed queries and $k = 5$. In warm mode all requests reuse one keep-alive connection; in cold mode each request opens a new TCP and TLS connection. For each topology and mode a probe sends 10 untimed warm-up requests and 300 timed requests at 0.5 s spacing, in two passes over all topologies, with the topology order rotated per region. Each cell therefore holds 600 requests; the whole run holds 30,000, on 1 October 2026 between 01:20 and 03:38 UTC. All 30,000 returned HTTP 200 with complete server timing. The Virginia probe runs on the central server itself, so its centralized requests cross no network. We report medians with bootstrap 95 percent confidence intervals of the median (2,000 resamples).

**What is measured.** Client wall time from sending the request to receiving the full response, and the server-reported embedding, search and total times. Wall time minus server total is the client-side network component. For Workers, the search time includes the pooled or direct path from the PoP to the database.

### B. Network baseline

The median TCP connect time from each probe to the central server is 76.9 ms (London), 115.1 ms (São Paulo), 184.9 ms (Mumbai) and 218.1 ms (Sydney). Against great-circle distances to Ashburn of 5,917, 7,664, 12,855 and 15,674 km, these correspond to route factors $\rho$ of 1.30, 1.50, 1.44 and 1.39. The U.S. terrestrial median of 2.1 [5] would overestimate all four.

### C. Results

**Table I. Median wall time of the retrieval prefix (ms), 600 requests per cell. Brackets: 95 percent confidence interval of the median. The fastest topology per row is in bold.**

| Probe | Mode | Centralized | Central-tunnel | Edge-unpooled | Edge-pooled | Edge-replica (Sydney DB) |
|---|---|---|---|---|---|---|
| Virginia | warm | 92 [91, 93] | 114 [111, 116] | 141 [140, 142] | **76 [75, 77]** | 316 [315, 318] |
| London | warm | 167 [166, 168] | 187 [184, 189] | 923 [918, 925] | **150 [149, 152]** | 397 [395, 399] |
| São Paulo | warm | **205 [204, 206]** | 236 [235, 237] | 1,360 [1,341, 1,370] | 257 [253, 260] | 495 [492, 498] |
| Mumbai | warm | **275 [274, 276]** | 303 [301, 306] | 2,595 [2,584, 2,615] | 353 [351, 355] | 411 [374, 543] |
| Sydney | warm | 294 [294, 295] | 324 [323, 326] | 2,358 [2,343, 2,373] | 427 [425, 430] | **191 [189, 192]** |
| Virginia | cold | **96 [95, 97]** | 134 [132, 136] | 166 [164, 169] | 106 [104, 109] | 348 [346, 350] |
| London | cold | 327 [325, 330] | 205 [204, 207] | 941 [938, 943] | **170 [168, 171]** | 417 [415, 419] |
| São Paulo | cold | 436 [435, 437] | **251 [249, 253]** | 1,431 [1,418, 1,439] | 284 [281, 291] | 508 [503, 514] |
| Mumbai | cold | 655 [650, 656] | **319 [317, 320]** | 2,654 [2,646, 2,665] | 373 [371, 376] | 351 [348, 355] |
| Sydney | cold | 738 [736, 741] | 338 [336, 341] | 2,363 [2,355, 2,372] | 441 [437, 444] | **205 [204, 207]** |

**Table II. Server-side stage medians (ms) and derived path parameters, warm connections.**

| Probe | PoP | $R$ | Embed, central CPU | Embed, edge | Search, central | Search, pooled | $k_{pool}$ | Search, unpooled | $n_{rt}$ |
|---|---|---|---|---|---|---|---|---|---|
| Virginia | IAD | 1.1 | 85 | 47 | 4 | 14 | n/a | 71 | n/a |
| London | LHR | 76.9 | 84 | 39 | 4 | 98 | 1.22 | 870 | 10.3 |
| São Paulo | GRU | 115.1 | 85 | 91 | 4 | 152 | 1.29 | 1,239 | 9.7 |
| Mumbai | BOM | 184.9 | 84 | 53 | 4 | 294 | 1.57 | 2,538 | 12.7 |
| Sydney | SYD | 218.1 | 83 | 165 | 4 | 254 | 1.15 | 2,174 | 9.0 |

$k_{pool} = (\text{search}_{pooled} - \text{search}_{central}) / R$; $n_{rt} = (\text{search}_{unpooled} - \text{search}_{central}) / R - 1$. For the Sydney probe the edge-replica search time is 12.5 ms.

### D. Findings

**1. Warm connections: a pooled edge Worker helps near a well-provisioned PoP and hurts far from the database.** At Virginia and London the pooled Worker is 16 ms faster than the centralized baseline. At São Paulo, Mumbai and Sydney it is 52, 78 and 133 ms slower. Two measured effects explain this. The pooled query path costs more than one round trip: $k_{pool}$ is 1.15 to 1.57, and the minimum search time in every region exceeds $R$, so no response was served from a cache. And edge embedding time depends on the PoP: the median is 39 to 53 ms at London, Virginia and Mumbai, below the baseline's 84 ms on two CPU cores, but 91 ms at São Paulo and 165 ms at Sydney. The same model is evidently not served with the same capacity at every location.

**2. A database copy near the user makes the edge fastest there, by about half the modelled amount.** For the Sydney probe, edge-replica takes 191 ms against 294 ms for the baseline, a 104 ms or 35 percent reduction, and 236 ms below the pooled Worker that queries Virginia. Removing one 218 ms round trip does not yield 218 ms because embedding at the Sydney PoP takes 165 ms against 83 ms centrally. From the other four regions the same Sydney database is distant, and edge-replica is the slowest pooled option: placing data away from the user is as costly as placing it near is beneficial.

**3. Without pooling, an edge Worker pays 9 to 13 setup round trips on every request.** The unpooled search time is 870 ms from London and over 2.1 s from Mumbai and Sydney. Even next to the database (Virginia) connection setup costs 71 ms against 14 ms pooled. The vendor's figure of seven round trips [14] underestimates what we observe with TLS and SCRAM authentication through the Workers socket API.

**4. Cold connections favour the pooled edge Worker everywhere outside the database's region.** A new connection to the centralized server costs the client two additional round trips, TCP and TLS 1.3, over the long path: the median connect time is 156 ms from London and 438 ms from Sydney, twice $R$. Edge topologies terminate TLS at the PoP in 8 to 15 ms. On cold connections the pooled Worker is therefore 157 ms faster than the baseline at London, 151 ms at São Paulo, 282 ms at Mumbai and 298 ms at Sydney, and the near-user copy is 533 ms faster at Sydney. The tunnel row isolates this effect: with all computation still central, edge TLS termination alone brings Sydney from 738 to 338 ms cold, while costing 20 to 31 ms on warm connections.

**5. Embedding tails differ by an order of magnitude.** Across 3,600 edge requests per region, the 99th-percentile embedding time is 83 ms at London, 150 ms at Virginia, 174 ms at Mumbai, 300 ms at São Paulo and 385 ms at Sydney, with maxima of 2 to 28 s. The CPU baseline has a 99th percentile of 202 to 233 ms and a maximum of 310 ms.

### E. Model versus measurement

**Table III. Predicted and measured median wall time (ms), warm connections. Predictions use the equations of Section V-B with measured $R$, embedding and search medians, $k_{pool} = 1$ and $n_{rt} = 7$.**

| Probe | Centralized | | Edge-pooled | | Edge-unpooled | | Edge-replica | |
|---|---|---|---|---|---|---|---|---|
| | pred. | meas. | pred. | meas. | pred. | meas. | pred. | meas. |
| Virginia | 90 | 92 | 67 | 76 | 81 | 141 | | |
| London | 165 | 167 | 132 | 150 | 680 | 923 | | |
| São Paulo | 204 | 205 | 223 | 257 | 1,022 | 1,360 | | |
| Mumbai | 273 | 275 | 251 | 353 | 1,542 | 2,595 | | |
| Sydney | 305 | 294 | 396 | 427 | 1,931 | 2,358 | 191 | 191 |

The additive structure predicts the centralized topology within 4 percent and the near-user copy within 1 percent. With $k_{pool} = 1$ it underestimates the pooled Worker by 7 to 29 percent, and with $n_{rt} = 7$ the unpooled Worker by 18 to 42 percent. The data imply $k_{pool}$ between 1.15 and 1.57 (median 1.26) and $n_{rt}$ between 9.0 and 12.7 (median 10.0). With these calibrated values the model can be used for regions not probed; `latency_sim.py --compare` regenerates the table from the raw data.

---

## VII. Retrieval-Tier Measurements

This section characterizes the index that all topologies share, on a separate, larger synthetic corpus.

**Setup.** PostgreSQL 17.10 with pgvector 0.8.5 in a Docker VM limited to 4.8 GiB of memory on an 8-core, 8 GB laptop-class host. The corpus is 300,000 L2-normalized 768-dimensional vectors drawn from a mixture of 1,000 Gaussians. Ground truth is the exact top-10 by brute-force cosine. Recall@10 is measured for 1,000 held-out queries drawn from the same mixture; the exact-scan baseline used the first 30. Queries are issued serially over one connection. The index path is forced with `enable_seqscan = off` and confirmed with `EXPLAIN`. This run had no warm-up pass and a single timed sweep per configuration, with `ef_search` swept in ascending order after each build; `results/ERRATA.md` lists its artifacts.

**Table IV. HNSW on 300,000 × 768 vectors: Recall@10, latency and serial throughput versus `ef_search`. Index size 1172 MB for every build.**

| build (`m`, `ef_construction`) | build time (s) | `ef_search` | Recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | QPS (serial) |
|---|---|---|---|---|---|---|---|
| 16, 64  | 293.9 | 40  | 0.903 | 12.50 | 42.45 | 72.41 | 60.1 |
| 16, 64  | 293.9 | 100 | 0.933 | 8.07 | 35.63 | 110.16 | 77.6 |
| 16, 64  | 293.9 | 200 | 0.952 | 3.92 | 7.42 | 11.26 | 228.3 |
| 16, 64  | 293.9 | 400 | 0.973 | 8.03 | 25.74 | 51.12 | 96.0 |
| 16, 200 | 669.4 | 40  | 0.972 | 25.71 | 114.27 | 217.55 | 27.4 |
| 16, 200 | 669.4 | 100 | 0.998 | 12.28 | 30.40 | 55.85 | 70.3 |
| 16, 200 | 669.4 | 200 | 0.9999 | 9.43 | 40.84 | 83.95 | 70.3 |
| 16, 200 | 669.4 | 400 | 0.9999 | 6.70 | 18.72 | 35.70 | 116.7 |
| 32, 200 | n/a† | 40  | 0.996 | 21.93 | 79.94 | 162.82 | 33.9 |
| 32, 200 | n/a† | 100 | 0.999 | 7.98 | 24.63 | 57.70 | 97.5 |
| 32, 200 | n/a† | 200 | 0.999 | 6.29 | 12.26 | 23.97 | 146.5 |
| 32, 200 | n/a† | 400 | 0.9998 | 8.71 | 24.94 | 43.32 | 90.8 |

† The host suspended overnight during this build; the recorded wall-clock includes system sleep and is not reported. The query rows for this build ran after wake and are valid.

**Table V. IVFFlat (`lists = 1000`, build 166.7 s, index 1178 MB): Recall@10 and latency versus `probes`.**

| `probes` | Recall@10 | p50 (ms) | p95 (ms) | p99 (ms) | QPS (serial) |
|---|---|---|---|---|---|
| 1   | 0.873 | 9.55 | 29.20 | 55.82 | 77.6 |
| 10  | 0.9999 | 18.08 | 39.95 | 94.43 | 45.3 |
| 40  | 0.9999 | 41.63 | 75.24 | 97.17 | 22.7 |
| 100 | 0.9999 | 75.33 | 164.22 | 208.94 | 11.7 |

A second IVFFlat build with `lists = 4000` failed for lack of `maintenance_work_mem` (Section IV-D). An exact sequential scan over the corpus, the brute-force baseline, measured Recall@10 of 1.000 at a median of 1,061.6 ms over 30 queries.

**Findings.**

1. Recall rises with `ef_search` within every build and with `ef_construction` and `m` across builds. At `ef_search = 40` the recall floor is 0.903, 0.972 and 0.996 for the three builds; the single exception to monotonicity is at `ef_search = 400`, where the `m = 32` build scores 0.9998 against 0.9999 for `m = 16`.
2. At Recall@10 of 0.9999, HNSW serves at a median of 6.70, 8.71 or 9.43 ms depending on the build, against 18.08 ms for IVFFlat with `probes = 10`: 1.9 to 2.7 times lower latency. We report the range because the median varies by up to 3.2 times between sweeps of the same build.
3. At Recall@10 of at least 0.99, HNSW medians range from 6.3 to 21.9 ms, 48 to 169 times faster than the exact scan.
4. IVFFlat recall saturates by `probes = 10`; raising `probes` to 40 and 100 multiplies the median latency by 2.3 and 4.2 with no recall gain.
5. IVFFlat built in 166.7 s, HNSW in 293.9 s at `ef_construction = 64` and 669.4 s at 200.
6. Index size is invariant to `m` at this dimensionality, as the page-packing model predicts.

On the server-class instances of Section VI, with real text embeddings, 100,000 vectors and `ef_search = 100`, the server-side search median was 4 ms, consistent with the lower end of these laptop measurements.

---

## VIII. Discussion

The measurements support a conditional answer to the question in the title. Moving the retrieval prefix to the edge is not a uniform improvement. Its effect is the sum of three terms that differ in sign across regions: client connection setup, which the edge always shortens; embedding time, which the edge shortens or lengthens depending on the inference capacity of the serving PoP; and the path to the database, which pooling makes tolerable and which only a nearby copy of the data makes short.

For a deployment, this yields three practical rules. A Worker without a connection pool should not be deployed against a distant database under any workload. A pooled Worker with a central database is the better choice when sessions are short and connections are mostly cold, and the worse choice for long-lived warm connections from distant regions. And a regional copy of the vector tier is worthwhile only together with a check of the embedding latency at the PoPs that will serve it, since slow edge inference can consume half of the gain.

The centralized baseline was also improved by an edge component: terminating TLS at the edge and proxying to the central server captured most of the cold-connection benefit with no change to where embedding or search run. A fair evaluation of an edge architecture should include this cheaper alternative.

The single-round-trip transaction was the largest implementation effect we encountered, a factor of about four in pooled search time. Statement count per request matters more at the edge than in a region, because every statement is a wide-area round trip.

---

## IX. Threats to Validity and Future Work

### A. Internal validity

- **Single session.** All end-to-end data come from one session of about two hours and twenty minutes. A second session at a different time of day was started and lost when the instances were terminated before collection. Diurnal variation in PoP load and routing is therefore not captured.
- **Shared baseline.** The five probes ran concurrently against one two-core baseline server. Its embedding median was 83 to 91 ms in every cell during the run, against about 70 ms when idle, so concurrency added roughly 15 ms to the centralized measurements, near-uniformly across regions.
- **Co-located Virginia probe.** The Virginia probe ran on the central server; its centralized requests use the loopback interface and its probe process shares the server's CPUs.
- **Retrieval-tier latency noise.** The Table IV run had no warm-up, so cache warming is confounded with the `ef_search` dial; recall is unaffected.

### B. External validity

- **Probes are data-centre machines.** Their network paths to the PoPs and to the central region are better than residential or mobile paths. Last-mile latency would add to every topology and would increase the relative value of local TLS termination.
- **One provider pair, one embedding model, one plan.** Cloudflare Workers, Workers AI and Hyperdrive on the free plan, AWS for origins. PoP inference capacity and pool placement are operational properties that may change.
- **Small query set and corpus.** Ten distinct queries and 100,000 passages. The search compute term is 4 ms here and grows with corpus size and recall target (Section VII).
- **Baseline hardware.** The baseline embeds on two CPU cores. A GPU-backed central embedder would lower $E_c$ and move every warm comparison in the baseline's favour.
- **The near-user copy is not a replica.** It is an independently loaded identical database; replication lag and its effect on freshness are not measured.
- **Synthetic vectors in Section VII.** Gaussian mixtures do not reproduce the geometry of text embeddings.

### C. Construct validity and source quality

- The measured quantity is the retrieval prefix, not TTFT; generation is excluded.
- $k_{pool}$ and $n_{rt}$ are computed with $R$ measured between the probe instance and the central server, while the Worker's path starts at the PoP; the two differ by the instance-to-PoP distance, a few milliseconds.
- Isolate and container start-up figures in Section III-B are vendor-reported or from one third-party benchmark and are not measured here.
- Practitioner benchmarks of pgvector [23] report, for example, 253 queries per second at 5.51 ms 99th-percentile for HNSW against 8 queries per second at 150 ms for IVFFlat on a one-million-vector corpus; these are context, not evidence.

### D. Future work

1. Repeat the probe matrix over several days and times, with more regions and a larger query set.
2. Compare against the vendor-default stack, Workers AI with Vectorize [21], [22], and against a GPU-backed central embedder.
3. Replace the independent copy with streaming replicas and measure staleness.
4. Rerun the retrieval benchmark with warm-up and repeats on server-class memory, at one million or more real text embeddings.
5. Add generation to measure TTFT end to end.

---

## X. Conclusion

We deployed five topologies for the retrieval prefix of retrieval-augmented generation and measured them with 30,000 requests from five regions. An edge Worker with a pooled connection to a central vector database is slightly faster than a centralized server for users near a well-provisioned point of presence and 52 to 133 ms slower for distant users on warm connections, because the pooled path costs more than one round trip and edge embedding time varies fourfold between points of presence. A copy of the database near the user makes the edge the fastest option there, by 104 ms at Sydney. An unpooled edge Worker pays 9 to 13 setup round trips per request. On cold connections the pooled edge Worker beats the centralized server in every region outside the database's own, by 151 to 298 ms, because TLS terminates nearby. An additive model reproduces the centralized and near-copy topologies within 4 percent and, once calibrated, the pooled and unpooled ones. The retrieval tier itself contributes 4 ms at this corpus size; its recall and latency behaviour and the page-packing invariance of HNSW index size are characterized separately. All code and data are released.

---

## References

[1] P. Lewis, E. Perez, A. Piktus, F. Petroni, V. Karpukhin, N. Goyal, H. Küttler, M. Lewis, W.-t. Yih, T. Rocktäschel, S. Riedel, and D. Kiela, "Retrieval-augmented generation for knowledge-intensive NLP tasks," in *Proc. NeurIPS*, 2020. arXiv:2005.11401.

[2] Yu. A. Malkov and D. A. Yashunin, "Efficient and robust approximate nearest neighbor search using hierarchical navigable small world graphs," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 42, no. 4, pp. 824–836, Apr. 2020. arXiv:1603.09320.

[3] H. Jégou, M. Douze, and C. Schmid, "Product quantization for nearest neighbor search," *IEEE Trans. Pattern Anal. Mach. Intell.*, vol. 33, no. 1, pp. 117–128, Jan. 2011.

[4] M. Aumüller, E. Bernhardsson, and A. Faithfull, "ANN-Benchmarks: A benchmarking tool for approximate nearest neighbor algorithms," *Information Systems*, vol. 87, 2020. arXiv:1807.05614.

[5] I. N. Bozkurt, W. Aqeel, D. Bhattacherjee, B. Chandrasekaran, P. B. Godfrey, G. Laughlin, B. M. Maggs, and A. Singla, "Dissecting latency in the Internet's fiber infrastructure," in *Proc. ACM IMC*, 2018. arXiv:1811.10737.

[6] K. Seemakhupt, S. Liu, and S. Khan, "EdgeRAG: Online-indexed RAG for edge devices," arXiv:2412.21023, Dec. 2024.

[7] J. Kim and D. Mahajan, "VectorLiteRAG: Latency-aware and fine-grained resource partitioning for efficient RAG," arXiv:2504.08930, Apr. 2025 (to appear in *Proc. IEEE HPCA*, 2026).

[8] D. Quinn, M. Nouri, N. Patel, J. Salihu, A. Salemi, S. Lee, H. Zamani, and M. Alian, "Accelerating retrieval-augmented generation," in *Proc. ACM ASPLOS*, 2025. arXiv:2412.15246.

[9] L. Cvetković, F. Costa, M. Djokic, M. Friedman, and A. Klimovic, "Dirigent: Lightweight serverless orchestration," in *Proc. ACM SOSP*, 2024. arXiv:2404.16393.

[10] C. Jin, Z. Zhang, X. Jiang, F. Liu, X. Liu, X. Liu, and X. Jin, "RAGCache: Efficient knowledge caching for retrieval-augmented generation," arXiv:2404.12457, 2024.

[11] W. Jiang, S. Zhang, B. Han, J. Wang, B. Wang, and T. Kraska, "PipeRAG: Fast retrieval-augmented generation via algorithm-system co-design," arXiv:2403.05676, 2024.

[12] C.-Y. Lin, K. Kamahori, Y. Liu, X. Shi, M. Kashyap, Y. Gu, R. Shao, Z. Ye, K. Zhu, R. Kadekodi, S. Wang, A. Krishnamurthy, L. Ceze, and B. Kasikci, "TeleRAG: Efficient retrieval-augmented generation inference with lookahead retrieval," arXiv:2502.20969, 2025.

[13] Cloudflare, "Eliminating cold starts with Cloudflare Workers," Cloudflare Blog, Jul. 30, 2020. [Online]. Available: https://blog.cloudflare.com/eliminating-cold-starts-with-cloudflare-workers/ (accessed Sep. 30, 2026).

[14] M. Silverlock and A. Robinson, "Hyperdrive: making databases feel like they're global," Cloudflare Blog, Sep. 28, 2023. [Online]. Available: https://blog.cloudflare.com/hyperdrive-making-regional-databases-feel-distributed/ (accessed Sep. 30, 2026).

[15] Cloudflare, "Pools across the sea: how Hyperdrive speeds up access to databases and why we're making it free," Cloudflare Blog, Apr. 8, 2025. [Online]. Available: https://blog.cloudflare.com/how-hyperdrive-speeds-up-database-access/ (accessed Sep. 30, 2026).

[16] Cloudflare, "Hyperdrive: Connection pooling," Cloudflare Developer Docs. [Online]. Available: https://developers.cloudflare.com/hyperdrive/concepts/connection-pooling/ (accessed Sep. 30, 2026).

[17] Cloudflare, "Workers AI model: bge-base-en-v1.5," Cloudflare Developer Docs. [Online]. Available: https://developers.cloudflare.com/workers-ai/models/bge-base-en-v1.5/ (accessed Sep. 30, 2026).

[18] Cloudflare, "Containers: Lifecycle of a Container," Cloudflare Developer Docs. [Online]. Available: https://developers.cloudflare.com/containers/platform-details/architecture/ (accessed Sep. 30, 2026).

[19] Cloudflare, "Python Workers redux: fast cold starts, packages, and a uv-first workflow," Cloudflare Blog, Dec. 8, 2025. [Online]. Available: https://blog.cloudflare.com/python-workers-advancements/ (accessed Sep. 30, 2026).

[20] Alchemy, "Benchmarking Cloudflare Containers vs AWS MicroVMs," Jul. 1, 2026. [Online]. Available: https://alchemy.run/blog/2026-07-01-microvm-cold-starts/ (accessed Sep. 30, 2026).

[21] Cloudflare, "Build a Retrieval Augmented Generation (RAG) AI," Workers AI tutorial, Cloudflare Developer Docs. [Online]. Available: https://developers.cloudflare.com/workers-ai/guides/tutorials/build-a-retrieval-augmented-generation-ai/ (accessed Sep. 30, 2026).

[22] Cloudflare, "Vectorize," Cloudflare Developer Docs. [Online]. Available: https://developers.cloudflare.com/vectorize/ (accessed Sep. 30, 2026).

[23] J. Katz, "The 150x pgvector speedup: a year-in-review," Apr. 30, 2024. [Online]. Available: https://jkatz05.com/post/postgres/pgvector-performance-150x-speedup/ (accessed Sep. 30, 2026).

[24] pgvector contributors, "pgvector: Open-source vector similarity search for Postgres," GitHub. [Online]. Available: https://github.com/pgvector/pgvector (accessed Sep. 30, 2026).

[25] P. Bajaj, D. Campos, N. Craswell, L. Deng, J. Gao, X. Liu, R. Majumder, A. McNamara, B. Mitra, T. Nguyen, M. Rosenberg, X. Song, A. Stoica, S. Tiwary, and T. Wang, "MS MARCO: A human generated machine reading comprehension dataset," arXiv:1611.09268, 2016.

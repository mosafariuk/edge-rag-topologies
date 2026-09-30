# Sources

Quality tags: **[primary]** peer-reviewed / archival · **[vendor]** first-party docs or blog ·
**[blog]** third-party practitioner material · **[owned]** measured in this project.
Numbered to match the reference list in `paper.md` / `ieee_manuscript.tex`. Every entry was
re-fetched on 2026-09-30 and the figure the paper takes from it is noted.

## Primary
- [1] Lewis et al., RAG, NeurIPS 2020 — https://arxiv.org/abs/2005.11401 **[primary]** — two abstract quotes.
- [2] Malkov & Yashunin, HNSW, TPAMI 42(4) 2020 — https://arxiv.org/abs/1603.09320 **[primary]** — "links separated by their characteristic distance scales"; empirical log scaling.
- [3] Jégou, Douze, Schmid, PQ, TPAMI 33(1) 2011 — https://inria.hal.science/inria-00514462/document **[primary]**.
- [4] Aumüller, Bernhardsson, Faithfull, ANN-Benchmarks, Inf. Syst. 87, 2020 — https://arxiv.org/abs/1807.05614 **[primary]** — "very different approaches ... comparable quality-performance trade-offs" (verbatim).
- [5] Bozkurt et al., IMC 2018 — https://arxiv.org/abs/1811.10737 **[primary]** — median conversion factor 2.1 → 1.3, 120 largest U.S. population centres. **Scope: U.S. terrestrial only.**
- [6] Seemakhupt, Liu, Khan, EdgeRAG, arXiv:2412.21023 — https://arxiv.org/html/2412.21023v2 **[primary]** — evaluated on Jetson Orin Nano only (phones appear in a memory-spec table); 18.5 GB / 5.23 M FEVER; GTE-base 768-d; IVF p95 > 64× median on NQ; <500 ms typical, >2 s tail; 24k-char threshold.
- [7] Kim & Mahajan, VectorLiteRAG, arXiv:2504.08930 (HPCA 2026) **[primary]** — 128 M vectors, TTFT 197 → 606 ms.
- [8] Quinn et al., Accelerating RAG (IKS), ASPLOS 2025, arXiv:2412.15246 **[primary]** — 13.4–27.9× exact search over 512 GB; 1.7–26.3× end-to-end.
- [9] Cvetković et al., Dirigent, SOSP 2024, arXiv:2404.16393 **[primary]** — sandbox init "10–100s of milliseconds".
- [10] Jin et al., RAGCache, arXiv:2404.12457 **[primary]** — related work only.
- [11] Jiang et al., PipeRAG, arXiv:2403.05676 **[primary]** — related work only.
- [12] Lin et al., TeleRAG, arXiv:2502.20969 **[primary]** — related work only.

## Vendor (Cloudflare)
- [13] "Eliminating cold starts with Cloudflare Workers," Jul. 30, 2020 — https://blog.cloudflare.com/eliminating-cold-starts-with-cloudflare-workers/ **[vendor]** — "5 milliseconds to load a Worker"; "single-digit milliseconds". Note: 2020 material.
- [14] Silverlock & Robinson, "Hyperdrive: making databases feel like they're global," Sep. 28, 2023 — https://blog.cloudflare.com/hyperdrive-making-regional-databases-feel-distributed/ **[vendor]** — "upwards of seven round-trips (TCP handshake; TLS negotiation; then auth)"; "50-200 milliseconds" per round trip.
- [15] "Pools across the sea ...," Apr. 8, 2025 — https://blog.cloudflare.com/how-hyperdrive-speeds-up-database-access/ **[vendor]** — "a bare minimum of 5 round trips"; ~125 ms Chicago→London. (Does NOT contain the 7-round-trip or any Sydney figure.)
- [16] Hyperdrive connection pooling docs — https://developers.cloudflare.com/hyperdrive/concepts/connection-pooling/ **[vendor]** — "optimally placed to minimize the latency" quote.
- [17] Workers AI bge-base-en-v1.5 — https://developers.cloudflare.com/workers-ai/models/bge-base-en-v1.5/ **[vendor]** — 768 dims, 512 max input tokens.
- [18] Containers: Lifecycle of a Container — https://developers.cloudflare.com/containers/platform-details/architecture/ **[vendor]** — "often in the 1-3 second range". (Does NOT contain any p50/p95/max percentiles.)
- [19] "Python Workers redux," Dec. 8, 2025 — https://blog.cloudflare.com/python-workers-advancements/ **[vendor]** — mean cold start 1.027 s with snapshots; ~10 s without.
- [21] Workers AI tutorial "Build a RAG AI" — https://developers.cloudflare.com/workers-ai/guides/tutorials/build-a-retrieval-augmented-generation-ai/ **[vendor]** — vendor-default edge RAG stack (Workers AI + Vectorize + D1).
- [22] Vectorize overview — https://developers.cloudflare.com/vectorize/ **[vendor]** — "a globally distributed vector database".

## Third-party
- [20] Alchemy, "Benchmarking Cloudflare Containers vs AWS MicroVMs," Jul. 1, 2026 — https://alchemy.run/blog/2026-07-01-microvm-cold-starts/ **[blog]** — 700 boots, 16 variants; Cloudflare Containers p50 1.7–10.5 s, p95 15.6–24.4 s, max 58.6 s; the quote "boot time is a property of the workload, plus a tail you can't control" is the **author's**, not Cloudflare's; also reports Firecracker snapshot boots of 2–4.5 s. (Does NOT contain a Workers-JS-vs-Lambda cold-start comparison.)
- [23] J. Katz, "The 150x pgvector speedup: a year-in-review," Apr. 30, 2024 — https://jkatz05.com/post/postgres/pgvector-performance-150x-speedup/ **[blog]** — dbpedia-openai-1000k-angular @ 99 % recall: HNSW (0.7.0) 253 QPS / 5.51 ms p99; IVFFlat (0.4.1) 8 QPS / 150.16 ms p99. Context only; not used as evidence.
- [24] pgvector README — https://github.com/pgvector/pgvector **[vendor/open source]** — parameter semantics.

## Owned
- `results/results.csv`, `results/run.log`, `results/summary.json`, `results/ERRATA.md` **[owned]** — the July 2026 pgvector 0.8.5 / PG 17.10 run (300k × 768). See ERRATA for the known artifacts.
- `latency_sim.py` **[owned, analytical]** — no measurements; every constant tagged.

## Removed in the 2026-09-30 revision (attribution or support failed on re-check)
1. Container boot percentiles attributed to Cloudflare documentation → they are from [20]; Cloudflare's own figure is "1–3 s".
2. "Cloudflare's container platform documents ... 'a tail you can't control'" → the quote is Alchemy's.
3. "Independent 2026 benchmarks: <5 ms JS Worker vs 1.2–2.8 s p95 Lambda" → not present on the cited page; no replacement source found. Dropped.
4. "Sydney↔us-east-1 150–200 ms per round trip [Hyperdrive]" → neither Hyperdrive post mentions Sydney; [14] says 50–200 ms generically. Dropped.
5. "Python Workers ~1 s [2020 cold-start post]" → the 2020 post predates Python Workers; re-cited to [19].
6. "EdgeRAG evaluated on Jetson, iPhone 16 Pro, Galaxy S24" → evaluated on Jetson only.
7. Katz figures "243 QPS / 5.74 ms / 16× build" → page shows 253 / 5.51 ms / 30×. Corrected and demoted to context.
8. "Firecracker 2–4.5 s: refuted (unanimous)" → no counter-evidence existed; the paper now simply makes no Firecracker claim.
9. Ingestion "~48k rows/min, backend <10 % CPU" → run.log shows 28.5k rows/min; no CPU record. Corrected / dropped.
10. "800 mixture clusters" → harness default 1000; no artifact for 800. Corrected.
11. Applying ρ = 2.1 to Sydney↔Ashburn (334 ms, claimed "consistent with 150–200 ms") → scope violation; intercontinental paths now use an explicit 1.3 assumption or a measured RTT.
12. Simulator baseline that charged the centralized topology 7 cross-continent round trips of DB setup → the "2.3 s setup / 2.34 s saving" figures were an artifact; withdrawn.
13. HNSW cost model N_dist ≈ c·ef·log N·m̄ → predicted a 44.7 ms lower bound for a 6.7 ms measurement; replaced by the additive form and validated (`latency_sim.py --validate`).
14. Memory formula with L ≈ log N layers per element and 4-byte links → replaced by the page-packing model (2 pairs per 8 KiB page → 1172 MB, matching both measured indexes).

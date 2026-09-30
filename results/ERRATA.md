# Errata and provenance for the 2026-07 benchmark run

`results.csv` and `run.log` are the raw outputs of the July 2026 run and are left unmodified.
The following corrections apply when reading them:

1. **`queries` column, `seqscan_exact` row.** The harness at the time wrote the value of
   `--queries` (1000) into every row. The exact-scan baseline actually used the first 30 queries
   (`--seqscan-queries` default). The HNSW and IVFFlat rows did use all 1,000 queries. The
   harness now records the true per-row count.
2. **`--clusters`.** The run log does not record this flag and the reproduction command in the
   README did not pass it, so the value was the harness default, **1000** mixture components.
   An earlier draft of the paper stated 800; that figure had no artifact behind it.
3. **`hnsw m=32` `build_s = 55777.9`.** Invalid: the host suspended overnight mid-build. The
   wall-clock includes system sleep. Query rows for that build ran after wake and are valid.
4. **`ivfflat lists=4000`.** Build failed: `memory required is 3738 MB, maintenance_work_mem is
   2048 MB`. No rows exist for it (see `run.log`).
5. **No warm-up, single sweep.** This run had no untimed warm-up and one timed sweep per
   configuration, with `ef_search` swept in ascending order after each build. Cache warming is
   therefore confounded with the dial for the first sweeps of each build (e.g. p50 12.50 →
   8.07 → 3.92 → 8.03 ms within the `m=16, ef_construction=64` build). The harness now warms
   up and repeats; a rerun on server-class hardware is planned (paper Sec. VII).
6. **`summary.json`.** The original file was hand-edited and did not match the harness schema.
   It has been regenerated from `results.csv` with `bench.py --summarize-only` and carries a
   provenance note pointing here.
7. **Ingestion rate.** `run.log` shows 300,000 rows loaded in 631 s (≈28.5k rows/min). A
   draft figure of ~48k rows/min and a "<10% backend CPU" observation had no logged basis
   and were removed from the paper.

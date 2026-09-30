#!/usr/bin/env python3
"""
Edge-RAG latency model: a transparent calculator, not a measurement
====================================================================
Instantiates the additive model from the paper (Sec. V)

    T_total = T_network + T_embed + T_search + T_generation

for FOUR deployment topologies and prints a stage-by-stage TTFT comparison across
user<->region distances. It performs NO network I/O and fabricates NO measurements.
Every input is a named constant with a source tag; every one can be overridden on the
command line so each figure in the paper can be regenerated and stress-tested.

Topologies (paper Sec. III-A / V-B):
  centralized      user -> central region; embedder is a separate in-region service;
                   DB is IN-REGION, so a cold DB connection costs 7 x in-region RTT.
  edge-unpooled    user -> nearby PoP; embed in PoP; Worker opens a FRESH connection to the
                   central DB on every request (7 x PoP<->DB RTT of setup, then the query).
  edge-pooled      as above but the PoP<->DB connection is pooled and warm (Hyperdrive /
                   PgBouncer): one PoP<->DB round trip per query.
  edge-replica     as edge-pooled, plus a read replica of the vector tier near the PoP.

The pooling comparison that matters is therefore edge-unpooled vs edge-pooled. An earlier
version of this script charged the *centralized* baseline seven cross-continent round
trips of DB setup; that mis-specified the baseline (a centralized app's DB is in-region)
and is corrected here.

Source tags:
  [primary] peer-reviewed / archival   [vendor] first-party docs/blog   [blog] practitioner
  [assume]  a modeling assumption (a knob, not a measured fact)

Run:
    python3 latency_sim.py                       # default scenario table
    python3 latency_sim.py --rtt sydney=205      # replace the model with a MEASURED RTT (ms)
    python3 latency_sim.py --hnsw-sweep          # HNSW search-cost sensitivity
    python3 latency_sim.py --validate results/results.csv   # model vs owned measurements
    python3 latency_sim.py --json                # machine-readable output
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass, asdict, field

# ---------------------------------------------------------------------------
# Physical + protocol constants
# ---------------------------------------------------------------------------
C_FIBER_KM_S = 2.0e5      # [primary] light in silica fibre ~ 2/3 c ~ 200,000 km/s (~5 us/km)

# Route-inflation factor rho = deployed path latency / straight-line fibre latency.
RHO_US_MEDIAN = 2.1       # [primary] Bozkurt et al., IMC 2018: MEDIAN over pairs of the 120
                          #   largest U.S. population centres (reducible to ~1.3). Its scope
                          #   is the U.S. terrestrial backbone. It is NOT applied to
                          #   intercontinental paths here.
RHO_INTL_ASSUMED = 1.3    # [assume] submarine cables follow near-great-circle routes; this is
                          #   a placeholder until Phase-2 measurements replace it. Prefer
                          #   --rtt <scenario>=<measured ms> whenever a measurement exists.

N_SETUP_ROUND_TRIPS = 7   # [vendor] Cloudflare (Hyperdrive launch post, Sep 2023): "upwards of
                          #   seven round-trips (TCP handshake; TLS negotiation; then auth)".
                          #   A 2025 Cloudflare post revises this to "a bare minimum of 5".
IN_REGION_RTT_MS = 1.0    # [assume] app server <-> DB inside one cloud region
EDGE_POP_KM = 25.0        # [assume] user <-> nearest PoP straight-line distance


@dataclass(frozen=True)
class Scenario:
    key: str
    label: str
    distance_km: float      # user <-> central region (us-east-1 / Ashburn), great-circle
    intercontinental: bool  # True -> RHO_INTL_ASSUMED applies, not the U.S. median


SCENARIOS = [
    Scenario("metro",   "same-metro (edge-ideal)",                 25,    False),
    Scenario("nyc",     "same-continent (NYC->Ashburn)",            330,   False),
    Scenario("oregon",  "US coast-to-coast (Oregon->Ashburn)",      3500,  False),
    Scenario("london",  "transatlantic (London->Ashburn)",          5900,  True),
    Scenario("sydney",  "transpacific (Sydney->Ashburn)",           15900, True),
]


# ---------------------------------------------------------------------------
# Compute-stage assumptions (labeled; NOT measurements of the whole system)
# ---------------------------------------------------------------------------
@dataclass
class ComputeModel:
    embed_ms_edge: float = 25.0           # [assume] small encoder on edge GPU/WASM, warm
    embed_ms_central: float = 25.0        # [assume] same model in the central region
    central_embed_hop_ms: float = 3.0     # [assume] in-region service-to-service RTT when the
                                          #   embedder is a separate service (common pattern)
    search_compute_ms: float = 6.7        # [owned] pgvector HNSW p50 at recall 0.9999,
                                          #   300k x 768, results/results.csv (Sec. V-E)
    generation_prefill_ms: float = 400.0  # [blog] typical LLM prefill-to-first-token; held
                                          #   fixed across topologies by construction


# ---------------------------------------------------------------------------
# Network model
# ---------------------------------------------------------------------------
def rtt_model_ms(distance_km: float, rho: float) -> float:
    """Round-trip fibre propagation with route inflation, in milliseconds."""
    return 2.0 * rho * (distance_km / C_FIBER_KM_S) * 1000.0


@dataclass
class Net:
    """Resolved round-trip times for one scenario (ms)."""
    user_region: float   # user <-> central region
    user_pop: float      # user <-> nearest edge PoP
    pop_db: float        # PoP <-> central DB (assumed same distance class as user<->region)
    in_region: float
    rho_used: float
    rtt_source: str      # "model" or "measured"


def resolve_net(sc: Scenario, rho_us: float, rho_intl: float,
                measured_rtt: dict[str, float]) -> Net:
    rho = rho_intl if sc.intercontinental else rho_us
    if sc.key in measured_rtt:
        user_region = measured_rtt[sc.key]
        src = "measured"
    else:
        user_region = rtt_model_ms(sc.distance_km, rho)
        src = "model"
    user_pop = rtt_model_ms(EDGE_POP_KM, rho_us)
    return Net(user_region, user_pop, user_region, IN_REGION_RTT_MS, rho, src)


# ---------------------------------------------------------------------------
# Stage decomposition per topology
# ---------------------------------------------------------------------------
@dataclass
class Breakdown:
    scenario: str
    topology: str
    T_network_ms: float
    T_embed_ms: float
    T_search_ms: float
    T_generation_ms: float
    notes: list[str] = field(default_factory=list)

    @property
    def ttft_ms(self) -> float:
        return self.T_network_ms + self.T_embed_ms + self.T_search_ms + self.T_generation_ms

    @property
    def prefix_ms(self) -> float:
        """The retrieval prefix: everything edge placement can act on."""
        return self.T_network_ms + self.T_embed_ms + self.T_search_ms


def centralized(sc: Scenario, n: Net, cm: ComputeModel, cold_conn: bool) -> Breakdown:
    net = n.user_region                          # request in + first byte out
    notes = []
    if cold_conn:
        net += N_SETUP_ROUND_TRIPS * n.in_region  # DB is in-region: setup is cheap
        notes.append("cold DB conn = 7 x in-region RTT")
    embed = cm.embed_ms_central + cm.central_embed_hop_ms
    return Breakdown(sc.label, "centralized", net, embed, cm.search_compute_ms,
                     cm.generation_prefill_ms, notes)


def edge_unpooled(sc: Scenario, n: Net, cm: ComputeModel) -> Breakdown:
    net = n.user_pop + N_SETUP_ROUND_TRIPS * n.pop_db + n.pop_db
    return Breakdown(sc.label, "edge-unpooled", net, cm.embed_ms_edge, cm.search_compute_ms,
                     cm.generation_prefill_ms, ["fresh DB conn per request: 7 x PoP<->DB RTT"])


def edge_pooled(sc: Scenario, n: Net, cm: ComputeModel) -> Breakdown:
    net = n.user_pop + n.pop_db
    return Breakdown(sc.label, "edge-pooled", net, cm.embed_ms_edge, cm.search_compute_ms,
                     cm.generation_prefill_ms, ["warm pooled conn: 1 x PoP<->DB RTT"])


def edge_replica(sc: Scenario, n: Net, cm: ComputeModel) -> Breakdown:
    net = n.user_pop + n.user_pop            # replica assumed as near as the PoP
    return Breakdown(sc.label, "edge-replica", net, cm.embed_ms_edge, cm.search_compute_ms,
                     cm.generation_prefill_ms, ["read replica near PoP; staleness not modeled"])


TOPOLOGIES = ("centralized", "edge-unpooled", "edge-pooled", "edge-replica")


def all_topologies(sc: Scenario, n: Net, cm: ComputeModel, cold_conn: bool) -> list[Breakdown]:
    return [centralized(sc, n, cm, cold_conn), edge_unpooled(sc, n, cm),
            edge_pooled(sc, n, cm), edge_replica(sc, n, cm)]


# ---------------------------------------------------------------------------
# HNSW search-cost model (paper Sec. V-C), calibrated against owned data
# ---------------------------------------------------------------------------
def hnsw_cost(n_vectors: int, dim: int, ef_search: int, m: int = 16,
              c: float = 0.5, eff_bw_gb_s: float = 8.0) -> dict:
    """
    Distance evaluations per query. The ef-bounded beam search runs only at layer 0, where
    each element has up to 2m links; the upper layers are a greedy descent with ef=1 over
    ~log(N) levels. So the cost is ADDITIVE, not multiplicative, in ef and log N:

        N_dist ~= c * ef * 2m  +  m * log2(N)

    c (default 0.5, [assume]) accounts for neighbours already in the visited set. The
    bytes-moved floor is N_dist * 4 * dim; the bandwidth-bound time floor divides by an
    EFFECTIVE random-access bandwidth (8 GB/s, [assume]). This is a lower bound on search
    compute; --validate checks that it lies below the measured p50s in results/results.csv.
    """
    layer0 = c * ef_search * 2 * m
    upper = m * math.log2(max(n_vectors, 2))
    n_dist = layer0 + upper
    bytes_moved = n_dist * 4 * dim
    t_floor_ms = (bytes_moved / (eff_bw_gb_s * 1e9)) * 1000.0
    return {
        "n_vectors": n_vectors, "dim": dim, "ef_search": ef_search, "m": m,
        "est_distance_evals": round(n_dist),
        "MB_moved_per_query": round(bytes_moved / 1e6, 3),
        "flops_per_query": round(n_dist * 2 * dim),
        "bandwidth_bound_floor_ms": round(t_floor_ms, 3),
        "raw_vector_memory_GiB": round(n_vectors * 4 * dim / (1024 ** 3), 3),
    }


def pgvector_hnsw_index_bytes(n_vectors: int, dim: int, m: int) -> dict:
    """
    Page-packing estimate of a pgvector HNSW index (8 KiB heap pages).
      element tuple  ~ 4*dim + 8 (varlena + dim/unused) + 24 (tuple + element header)
      neighbour tuple~ 6 B per ItemPointer x (2m links at layer 0 + m per upper layer) + 8
      expected upper layers per element = sum_k m^-k = 1/(m-1)   (level ~ Geometric(1/m))
    Tuples are not split across pages, so the index size is quantised by how many
    (element + neighbour) pairs fit in one page. For dim=768 that is 2 per page for every
    m up to ~48, which is why the paper's m=16 and m=32 indexes are both exactly 1172 MB.
    """
    elem = 4 * dim + 8 + 24
    upper_layers = 1.0 / (m - 1)
    neigh = 6 * (2 * m + m * upper_layers) + 8
    per_page = max(1, int(8168 // (elem + neigh)))
    pages = math.ceil(n_vectors / per_page)
    return {"element_bytes": elem, "neighbour_bytes": round(neigh), "pairs_per_page": per_page,
            "pages": pages, "index_MiB": round(pages * 8192 / 2 ** 20, 1)}


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def fmt(x: float) -> str:
    return f"{x:9.1f}"


def print_table(cm: ComputeModel, rho_us: float, rho_intl: float, cold_conn: bool,
                measured_rtt: dict[str, float]) -> list[dict]:
    rows: list[dict] = []
    header = (f"{'scenario':<38}{'topology':<15}{'T_net':>9}{'T_embed':>9}"
              f"{'T_search':>10}{'T_gen':>9}{'TTFT':>10}{'prefix':>9}")
    print(header)
    print("-" * len(header))
    for sc in SCENARIOS:
        n = resolve_net(sc, rho_us, rho_intl, measured_rtt)
        bds = all_topologies(sc, n, cm, cold_conn)
        base = bds[0]
        tag = f"[RTT user<->region {n.user_region:.1f} ms, {n.rtt_source}, rho={n.rho_used}]"
        print(f"{sc.label:<38}{tag}")
        for b in bds:
            print(f"{'':<38}{b.topology:<15}{fmt(b.T_network_ms)}{fmt(b.T_embed_ms)}"
                  f"{fmt(b.T_search_ms)}{fmt(b.T_generation_ms)}{fmt(b.ttft_ms)}{fmt(b.prefix_ms)}"
                  + (f"   ({base.ttft_ms - b.ttft_ms:+.1f} ms vs centralized)"
                     if b is not base else ""))
            rows.append({"scenario": sc.key, "topology": b.topology, **asdict(b),
                         "ttft_ms": round(b.ttft_ms, 1), "prefix_ms": round(b.prefix_ms, 1),
                         "rtt_user_region_ms": round(n.user_region, 1),
                         "rtt_source": n.rtt_source, "rho_used": n.rho_used})
        print()
    return rows


def print_hnsw_sweep() -> list[dict]:
    print("\nHNSW search-cost sensitivity (paper Sec. V-C). dim=768, m=16, c=0.5, BW=8 GB/s [assume].")
    header = f"{'N_vectors':>12}{'ef_search':>11}{'dist_evals':>12}{'MB/query':>10}{'floor ms':>10}{'raw GiB':>10}"
    print(header)
    print("-" * len(header))
    out = []
    for n in (300_000, 1_000_000, 10_000_000, 100_000_000):
        for ef in (40, 100, 400, 800):
            r = hnsw_cost(n, 768, ef)
            print(f"{n:>12,}{ef:>11}{r['est_distance_evals']:>12,}{r['MB_moved_per_query']:>10}"
                  f"{r['bandwidth_bound_floor_ms']:>10}{r['raw_vector_memory_GiB']:>10}")
            out.append(r)
    print("\nIndex size (page-packing model) for N=300k, dim=768:")
    for m in (16, 24, 32, 48, 64):
        print(f"  m={m:<3} -> {pgvector_hnsw_index_bytes(300_000, 768, m)}")
    print("\nThe floor is a LOWER bound from bytes moved; real latency adds index-page overhead,\n"
          "cache misses and planner/executor cost. ef_search enters linearly; N only through log N.")
    return out


def validate(csv_path: str, quiet: bool = False) -> list[dict]:
    """Compare the cost-model floor with measured HNSW p50s from bench.py output."""
    say = (lambda *a, **k: None) if quiet else print
    say(f"\nModel vs measurement ({csv_path}): floor must be BELOW every measured p50.")
    header = f"{'build':<26}{'ef':>5}{'meas p50 ms':>13}{'model floor ms':>16}{'floor/meas':>12}{'ok':>5}"
    say(header)
    say("-" * len(header))
    out = []
    with open(csv_path) as f:
        for row in csv.DictReader(f):
            if row["index"] != "hnsw":
                continue
            bp = json.loads(row["build_params"])
            ef = int(row["dial"].split("=")[1])
            r = hnsw_cost(int(row["n"]), int(row["dim"]), ef, m=int(bp["m"]))
            meas = float(row["p50_ms"])
            floor = r["bandwidth_bound_floor_ms"]
            ok = floor <= meas
            say(f"{json.dumps(bp):<26}{ef:>5}{meas:>13.2f}{floor:>16.2f}{floor / meas:>12.2f}{'yes' if ok else 'NO':>5}")
            out.append({"build": bp, "ef_search": ef, "measured_p50_ms": meas,
                        "model_floor_ms": floor, "consistent": ok})
    return out


def parse_rtt(items: list[str]) -> dict[str, float]:
    out = {}
    for it in items:
        k, v = it.split("=", 1)
        out[k.strip()] = float(v)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Edge-RAG analytical latency model")
    ap.add_argument("--rho", type=float, default=RHO_US_MEDIAN,
                    help="U.S. route-inflation factor (default 2.1, Bozkurt et al. median)")
    ap.add_argument("--rho-intl", type=float, default=RHO_INTL_ASSUMED,
                    help="intercontinental route-inflation factor (default 1.3, an ASSUMPTION)")
    ap.add_argument("--rtt", action="append", default=[], metavar="SCENARIO=MS",
                    help="measured user<->region RTT in ms for a scenario key "
                         "(metro|nyc|oregon|london|sydney); overrides the model")
    ap.add_argument("--cold-conn", action="store_true",
                    help="charge the centralized path a cold in-region DB connection (7 x ~1 ms)")
    ap.add_argument("--generation-ms", type=float, default=None, help="override T_generation (ms)")
    ap.add_argument("--search-ms", type=float, default=None, help="override T_search compute (ms)")
    ap.add_argument("--hnsw-sweep", action="store_true", help="print HNSW search-cost sensitivity")
    ap.add_argument("--validate", metavar="CSV", help="check the HNSW floor against bench.py results")
    ap.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    args = ap.parse_args()

    cm = ComputeModel()
    if args.generation_ms is not None:
        cm.generation_prefill_ms = args.generation_ms
    if args.search_ms is not None:
        cm.search_compute_ms = args.search_ms
    measured = parse_rtt(args.rtt)

    if args.json:
        payload = {
            "constants": {"c_fiber_km_s": C_FIBER_KM_S, "rho_us": args.rho,
                          "rho_intl_assumed": args.rho_intl,
                          "n_setup_round_trips": N_SETUP_ROUND_TRIPS,
                          "in_region_rtt_ms": IN_REGION_RTT_MS, "edge_pop_km": EDGE_POP_KM,
                          "measured_rtt_ms": measured},
            "compute_model": asdict(cm),
            "scenarios": [],
            "hnsw_sweep": [hnsw_cost(n, 768, ef) for n in (300_000, 1_000_000, 10_000_000)
                           for ef in (40, 100, 400, 800)],
            "hnsw_index_size_model": {m: pgvector_hnsw_index_bytes(300_000, 768, m) for m in (16, 32)},
        }
        for sc in SCENARIOS:
            n = resolve_net(sc, args.rho, args.rho_intl, measured)
            payload["scenarios"].append({
                "key": sc.key, "label": sc.label, "distance_km": sc.distance_km,
                "net": asdict(n),
                "topologies": {b.topology: {**asdict(b), "ttft_ms": round(b.ttft_ms, 1),
                                            "prefix_ms": round(b.prefix_ms, 1)}
                               for b in all_topologies(sc, n, cm, args.cold_conn)},
            })
        if args.validate:
            payload["validation"] = validate(args.validate, quiet=True)
        print(json.dumps(payload, indent=2))
        return

    print("Edge-RAG latency model: analytical, no measurements.")
    print(f"rho_us={args.rho}  rho_intl={args.rho_intl} [assume]  cold_conn(centralized)={args.cold_conn}"
          f"  T_search={cm.search_compute_ms} ms [owned]  T_gen={cm.generation_prefill_ms} ms [blog]\n")
    print_table(cm, args.rho, args.rho_intl, args.cold_conn, measured)
    if args.hnsw_sweep:
        print_hnsw_sweep()
    if args.validate:
        validate(args.validate)
    print("\nBoundary: T_generation is held fixed across topologies. Edge placement acts on the\n"
          "retrieval prefix only. Against a correctly specified centralized baseline, an edge Worker\n"
          "that still queries a central DB gains nothing on the network term; the gain appears only\n"
          "when the vector tier is replicated near the PoP. Pooling is what makes an edge Worker\n"
          "viable at all (compare edge-unpooled vs edge-pooled), not what beats the central baseline.")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Aggregate probe CSVs into the paper's measured tables (stdlib only).

    python3 analyze.py results/*.csv --md results/summary.md --json results/summary.json

Per (region, topology, mode): n, median / p95 wall time with a bootstrap 95% CI on the median,
median Server-Timing stages, and the client-side network component (wall - st_total).
Also prints `--rtt` arguments for latency_sim.py derived from the centralized/warm rows.
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import statistics
from collections import defaultdict


def pct(xs: list[float], p: float) -> float:
    if not xs:
        return float("nan")
    s = sorted(xs)
    k = (len(s) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def boot_ci(xs: list[float], reps: int = 2000, seed: int = 1) -> tuple[float, float]:
    if len(xs) < 5:
        return (float("nan"), float("nan"))
    rng = random.Random(seed)
    meds = sorted(statistics.median(rng.choices(xs, k=len(xs))) for _ in range(reps))
    return (meds[int(0.025 * reps)], meds[int(0.975 * reps)])


def fnum(x) -> float | None:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("csvs", nargs="+")
    ap.add_argument("--md")
    ap.add_argument("--json")
    args = ap.parse_args()

    groups: dict[tuple, list[dict]] = defaultdict(list)
    errors: dict[tuple, int] = defaultdict(int)
    for path in args.csvs:
        for r in csv.DictReader(open(path)):
            key = (r["region"], r["topology"], r["mode"])
            if r["status"] != "200":
                errors[key] += 1
                continue
            groups[key].append(r)

    rows = []
    for key in sorted(groups):
        g = groups[key]
        wall = [fnum(r["wall_ms"]) for r in g if fnum(r["wall_ms"]) is not None]
        st_total = [fnum(r["st_total"]) for r in g if fnum(r["st_total"]) is not None]
        embed = [fnum(r["st_embed"]) for r in g if fnum(r["st_embed"]) is not None]
        search = [fnum(r["st_search"]) for r in g if fnum(r["st_search"]) is not None]
        conn = [fnum(r["connect_ms"]) for r in g if fnum(r["connect_ms"]) is not None]
        net = [fnum(r["wall_ms"]) - fnum(r["st_total"]) for r in g
               if fnum(r["wall_ms"]) is not None and fnum(r["st_total"]) is not None]
        lo, hi = boot_ci(wall)
        colos = sorted({r["colo"] for r in g if r["colo"]})
        rows.append({
            "region": key[0], "topology": key[1], "mode": key[2], "n": len(wall),
            "errors": errors.get(key, 0),
            "wall_p50": round(statistics.median(wall), 1) if wall else None,
            "wall_p50_ci95": [round(lo, 1), round(hi, 1)],
            "wall_p95": round(pct(wall, 0.95), 1) if wall else None,
            "connect_p50": round(statistics.median(conn), 1) if conn else None,
            "net_p50": round(statistics.median(net), 1) if net else None,
            "embed_p50": round(statistics.median(embed), 1) if embed else None,
            "search_p50": round(statistics.median(search), 1) if search else None,
            "server_total_p50": round(statistics.median(st_total), 1) if st_total else None,
            "tcp_rtt_ms": fnum(g[0].get("tcp_rtt_ms")),
            "colos": colos,
        })

    hdr = ("| region | topology | mode | n | wall p50 (ms) [95% CI] | wall p95 | connect p50 | "
           "net p50 (wall−server) | embed p50 | search p50 | colo |")
    sep = "|" + "---|" * 11
    lines = [hdr, sep]
    for r in rows:
        ci = f"[{r['wall_p50_ci95'][0]}, {r['wall_p50_ci95'][1]}]"
        lines.append(f"| {r['region']} | {r['topology']} | {r['mode']} | {r['n']} | {r['wall_p50']} {ci} | "
                     f"{r['wall_p95']} | {r['connect_p50']} | {r['net_p50']} | {r['embed_p50']} | "
                     f"{r['search_p50']} | {','.join(r['colos']) or '-'} |")
    md = "\n".join(lines)
    print(md)

    rtts = [f"--rtt {r['region']}={r['tcp_rtt_ms']:.0f}" for r in rows
            if r["topology"] == "centralized" and r["mode"] == "warm" and r["tcp_rtt_ms"]]
    if rtts:
        print("\nlatency_sim.py inputs (TCP connect ≈ RTT to the central server):\n  python3 latency_sim.py "
              + " ".join(sorted(set(rtts))))
    if args.md:
        open(args.md, "w").write(md + "\n")
    if args.json:
        json.dump(rows, open(args.json, "w"), indent=2)


if __name__ == "__main__":
    main()

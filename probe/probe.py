#!/usr/bin/env python3
"""
Edge-RAG end-to-end probe (stdlib only; runs on any VM with python3 >= 3.8).
==========================================================================
Sends N POST /query requests to one deployed topology and records, per request:
  wall_ms       client-observed request time (what the user experiences before generation)
  connect_ms    TCP+TLS connection setup (cold mode only; 0 in warm mode)
  st_embed / st_search / st_total   server-side Server-Timing stages (ms)
  colo          Cloudflare PoP that served the request (Workers only)
so that  wall_ms - st_total  is the client<->server network + transfer component.

Modes:
  warm  one keep-alive connection, reused for every request (steady-state user)
  cold  a new TCP+TLS connection per request (first request of a session)

    python3 probe.py --target https://edge-rag-pooled.<acct>.workers.dev --label edge-pooled \
        --region sydney --n 300 --mode both --out results/sydney.csv
"""
from __future__ import annotations

import argparse
import csv
import http.client
import json
import os
import random
import socket
import ssl
import time
from datetime import datetime, timezone
from urllib.parse import urlparse

# Cloudflare's bot filter returns 403 to Python's default user agent; identify the probe instead.
USER_AGENT = "edge-rag-probe/0.1 (+research; contact in repo README)"

DEFAULT_QUERIES = [
    "how does connection pooling reduce database latency",
    "what limits the recall of an HNSW index",
    "why is fibre propagation slower than light in vacuum",
    "when does edge computing reduce time to first token",
    "how much memory does a 768-dimensional vector index need",
    "what is the cold start time of a V8 isolate",
    "how do read replicas affect data freshness",
    "what does ef_search control in pgvector",
    "why are container cold starts slow",
    "how is recall at ten measured",
]


def parse_server_timing(h: str | None) -> dict[str, float]:
    out: dict[str, float] = {}
    if not h:
        return out
    for part in h.split(","):
        name, *attrs = [p.strip() for p in part.split(";")]
        for a in attrs:
            if a.startswith("dur="):
                try:
                    out[name] = float(a[4:])
                except ValueError:
                    pass
    return out


def make_conn(u, timeout: float) -> http.client.HTTPConnection:
    if u.scheme == "https":
        return http.client.HTTPSConnection(u.hostname, u.port or 443, timeout=timeout,
                                           context=ssl.create_default_context())
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout)


def tcp_rtt_ms(host: str, port: int, samples: int = 5) -> float:
    """Median TCP connect time: a cheap proxy for the network RTT to the target."""
    vals = []
    for _ in range(samples):
        t = time.perf_counter()
        try:
            s = socket.create_connection((host, port), timeout=10)
            vals.append((time.perf_counter() - t) * 1000)
            s.close()
        except OSError:
            pass
    vals.sort()
    return vals[len(vals) // 2] if vals else float("nan")


def one_request(conn, path: str, payload: bytes, fresh: bool) -> dict:
    row: dict = {}
    t0 = time.perf_counter()
    if fresh:
        conn.connect()
        row["connect_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    else:
        row["connect_ms"] = 0.0
    conn.request("POST", path, body=payload, headers={"content-type": "application/json",
                                                       "content-length": str(len(payload)),
                                                       "user-agent": USER_AGENT})
    resp = conn.getresponse()
    body = resp.read()
    row["wall_ms"] = round((time.perf_counter() - t0) * 1000, 2)
    row["status"] = resp.status
    st = parse_server_timing(resp.getheader("server-timing"))
    row["st_embed"] = st.get("embed", "")
    row["st_search"] = st.get("search", "")
    row["st_total"] = st.get("total", "")
    try:
        j = json.loads(body)
        row["colo"] = j.get("colo", "")
        row["server_topology"] = j.get("topology", "")
        row["n_results"] = len(j.get("results", []))
    except Exception:
        row["colo"], row["server_topology"], row["n_results"] = "", "", ""
    return row


def run(args) -> None:
    u = urlparse(args.target)
    path = u.path or "/"
    if args.label.startswith("central") and path == "/":
        path = "/query"
    queries = [q.strip() for q in open(args.queries)] if args.queries else DEFAULT_QUERIES
    rng = random.Random(args.seed)
    rtt = tcp_rtt_ms(u.hostname, u.port or (443 if u.scheme == "https" else 80))
    print(f"[probe] {args.region} -> {args.label} ({u.hostname}) tcp-connect median {rtt:.1f} ms")

    fields = ["ts", "region", "topology", "mode", "i", "status", "wall_ms", "connect_ms",
              "st_embed", "st_search", "st_total", "colo", "server_topology", "n_results",
              "tcp_rtt_ms", "query"]
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    new = not os.path.exists(args.out)
    with open(args.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        if new:
            w.writeheader()
        modes = ["cold", "warm"] if args.mode == "both" else [args.mode]
        for mode in modes:
            fresh = mode == "cold"
            conn = None if fresh else make_conn(u, args.timeout)
            # untimed warm-up so the Worker/model/pool are warm before the timed run
            for _ in range(args.warmup):
                c = make_conn(u, args.timeout) if fresh else conn
                try:
                    one_request(c, path, json.dumps({"query": rng.choice(queries), "topK": 5}).encode(), fresh)
                except Exception:
                    pass
                if fresh:
                    c.close()
            for i in range(args.n):
                q = rng.choice(queries)
                payload = json.dumps({"query": q, "topK": 5}).encode()
                c = make_conn(u, args.timeout) if fresh else conn
                try:
                    row = one_request(c, path, payload, fresh)
                except Exception as e:  # noqa: BLE001
                    row = {"status": f"ERR:{type(e).__name__}", "wall_ms": "", "connect_ms": "",
                           "st_embed": "", "st_search": "", "st_total": "", "colo": "",
                           "server_topology": "", "n_results": ""}
                    if not fresh:
                        conn.close()
                        conn = make_conn(u, args.timeout)
                if fresh:
                    c.close()
                row.update({"ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                            "region": args.region, "topology": args.label, "mode": mode, "i": i,
                            "tcp_rtt_ms": round(rtt, 2), "query": q})
                w.writerow(row)
                f.flush()
                if args.interval:
                    time.sleep(args.interval)
                if (i + 1) % 50 == 0:
                    print(f"[probe]   {mode} {i+1}/{args.n} last wall={row['wall_ms']} ms st_total={row['st_total']}")
            if conn is not None:
                conn.close()
    print(f"[probe] done -> {args.out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True, help="base URL of the deployed topology")
    ap.add_argument("--label", required=True,
                    help="centralized | centralized-unpooled | edge-unpooled | edge-pooled | edge-replica | vectorize")
    ap.add_argument("--region", required=True, help="probe location label, e.g. sydney")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--mode", choices=["cold", "warm", "both"], default="both")
    ap.add_argument("--warmup", type=int, default=10)
    ap.add_argument("--interval", type=float, default=0.2, help="seconds between requests")
    ap.add_argument("--timeout", type=float, default=30.0)
    ap.add_argument("--queries", help="file with one query per line (default: built-in 10)")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--out", required=True, help="CSV to append to")
    run(ap.parse_args())

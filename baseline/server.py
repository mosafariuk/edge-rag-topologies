#!/usr/bin/env python3
"""
Centralized RAG baseline: the same retrieval prefix as worker.ts, run IN THE DATABASE'S REGION.
=================================================================================================
POST /query {"query": str, "topK"?: int}
  1. embed the query with bge-base-en-v1.5 (the same model Workers AI serves), locally on this VM
  2. SET LOCAL hnsw.ef_search + cosine ANN query on pgvector, inside one transaction
  3. return top-k + a Server-Timing header (embed / search / total, ms)

This is the "centralized" topology of the paper (Sec. III-A): the user pays one cross-region
round trip to reach this server; the server reaches the database in-region, over a pooled
connection by default (POOL=1) or a fresh connection per request (POOL=0) for the in-region
cold-connection variant.

Run on a VM in the database's region:
    pip install "psycopg[binary,pool]" sentence-transformers torch --index-url https://download.pytorch.org/whl/cpu
    DATABASE_URL=postgres://... POOL=1 EF_SEARCH=100 PORT=8080 python3 server.py
Put it behind TLS (Caddy: `caddy reverse-proxy --from probe-central.example.com --to :8080`)
so the probe's TLS cost is the same as for the Workers.
"""
from __future__ import annotations

import json
import math
import os
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import psycopg
from psycopg_pool import ConnectionPool
from sentence_transformers import SentenceTransformer

DATABASE_URL = os.environ["DATABASE_URL"]
POOL = os.environ.get("POOL", "1") == "1"
EF_SEARCH = int(os.environ.get("EF_SEARCH", "100"))
PORT = int(os.environ.get("PORT", "8080"))
MODEL_NAME = os.environ.get("MODEL", "BAAI/bge-base-en-v1.5")
DEFAULT_TOP_K, MAX_TOP_K = 5, 50

model = SentenceTransformer(MODEL_NAME, device="cpu")
model.encode(["warm-up"], normalize_embeddings=True)  # load weights before serving
pool = ConnectionPool(DATABASE_URL, min_size=2, max_size=8, open=True, kwargs={"autocommit": True}) if POOL else None

def retrieve(vec: list[float], k: int) -> list[dict]:
    # Same single-round-trip shape as worker.ts: BEGIN; SET LOCAL; SELECT; COMMIT in one query
    # string (simple protocol). Only validated numbers are inlined.
    assert all(math.isfinite(x) for x in vec)
    literal = "[" + ",".join(f"{x:.7g}" for x in vec) + "]"
    q = (f"BEGIN; SET LOCAL hnsw.ef_search = {int(EF_SEARCH)}; "
         f"SELECT id, content, (embedding <=> '{literal}'::vector) AS distance "
         f"FROM documents ORDER BY embedding <=> '{literal}'::vector LIMIT {int(k)}; COMMIT")

    def run(conn: psycopg.Connection) -> list[dict]:
        with conn.cursor() as cur:
            cur.execute(q)
            while cur.description is None and cur.nextset():
                pass
            rows = cur.fetchall()
            while cur.nextset():  # drain COMMIT
                pass
            return [{"id": r[0], "content": r[1], "similarity": 1 - float(r[2])} for r in rows]

    if pool is not None:
        with pool.connection() as conn:
            return run(conn)
    with psycopg.connect(DATABASE_URL, autocommit=True) as conn:  # fresh TCP+TLS+auth per request
        return run(conn)


class Handler(BaseHTTPRequestHandler):
    server_version = "edge-rag-central/0.1"

    def log_message(self, *_):  # quiet
        pass

    def _json(self, obj: dict, status: int = 200, timing: dict | None = None) -> None:
        body = json.dumps(obj).encode()
        self.send_response(status)
        self.send_header("content-type", "application/json")
        self.send_header("content-length", str(len(body)))
        if timing:
            self.send_header("server-timing", ", ".join(f"{k};dur={v}" for k, v in timing.items()))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:
        t0 = time.perf_counter()
        if self.path != "/query":
            return self._json({"error": "POST /query"}, 404)
        try:
            n = int(self.headers.get("content-length", "0"))
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json({"error": "Invalid JSON body"}, 400)
        query = (body.get("query") or "").strip() if isinstance(body.get("query"), str) else ""
        if not query:
            return self._json({"error": "`query` must be a non-empty string"}, 400)
        try:
            k = max(1, min(int(body.get("topK", DEFAULT_TOP_K)), MAX_TOP_K))
        except (TypeError, ValueError):
            k = DEFAULT_TOP_K
        timing: dict[str, int] = {}
        try:
            t = time.perf_counter()
            vec = model.encode([query], normalize_embeddings=True)[0].tolist()
            timing["embed"] = round((time.perf_counter() - t) * 1000)
            t = time.perf_counter()
            rows = retrieve(vec, k)
            timing["search"] = round((time.perf_counter() - t) * 1000)
            timing["total"] = round((time.perf_counter() - t0) * 1000)
            self._json({"query": query, "model": MODEL_NAME,
                        "topology": "centralized" if POOL else "centralized-unpooled",
                        "topK": k, "efSearch": EF_SEARCH, "timingMs": timing, "results": rows},
                       200, timing)
        except Exception as e:  # noqa: BLE001
            timing["total"] = round((time.perf_counter() - t0) * 1000)
            self._json({"error": "Retrieval failed", "detail": str(e), "timingMs": timing}, 502, timing)

    def do_GET(self) -> None:
        self._json({"ok": True, "topology": "centralized" if POOL else "centralized-unpooled"})


if __name__ == "__main__":
    print(f"[central] {MODEL_NAME} loaded; pool={'on' if POOL else 'off'}; ef_search={EF_SEARCH}; :{PORT}")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()

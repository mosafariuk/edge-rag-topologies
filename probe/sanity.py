#!/usr/bin/env python3
"""Query every deployment a few times and print timing, PoP and the top result."""
import json, sys, time, urllib.request, urllib.error
UA = {"User-Agent": "edge-rag-probe/0.1 (+research)", "content-type": "application/json"}
targets = [("centralized", "https://34-207-205-29.sslip.io/query"),
           ("central-tunnel", "https://cyber-zealand-partition-bush.trycloudflare.com/query"),
           ("edge-pooled", "https://edge-rag-pooled.aipco.workers.dev/"),
           ("edge-replica", "https://edge-rag-replica.aipco.workers.dev/"),
           ("edge-unpooled", "https://edge-rag-unpooled.aipco.workers.dev/")]
body = json.dumps({"query": "what was the Manhattan Project", "topK": 3}).encode()
n = int(sys.argv[1]) if len(sys.argv) > 1 else 3
for label, url in targets:
    for i in range(n):
        t = time.perf_counter()
        try:
            r = urllib.request.urlopen(urllib.request.Request(url, data=body, headers=UA), timeout=60)
            wall = (time.perf_counter() - t) * 1000
            st = r.getheader("server-timing") or ""
            j = json.loads(r.read())
            res = j.get("results") or []
            top = (res[0]["content"][:50] + "… sim=%.3f" % res[0]["similarity"]) if res else j.get("error")
            print(f"{label:<15} {i+1} wall={wall:6.0f}ms  {st:<42} colo={j.get('colo')} | {top}")
        except urllib.error.HTTPError as e:
            print(f"{label:<15} {i+1} HTTP {e.code}: {e.read()[:220]}")
        except Exception as e:  # noqa: BLE001
            print(f"{label:<15} {i+1} ERR {type(e).__name__}: {e}")

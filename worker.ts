/**
 * Edge-RAG reference Worker
 * -------------------------
 * Cloudflare Workers (module syntax). On each request it:
 *   1. Validates the query.
 *   2. Generates the query embedding at the edge via Workers AI (bge-base-en-v1.5, 768-dim).
 *   3. Runs a cosine ANN search in Postgres/pgvector over a Hyperdrive-pooled connection.
 *   4. Returns the top-k retrieved context plus a Server-Timing header with the measured
 *      duration of every stage, so that end-to-end probes (paper, Phase 2 measurement plan)
 *      can attribute latency to embed / search / total without instrumenting the client.
 *
 * Generation is intentionally out of scope: it is the compute-bound term the edge does not
 * eliminate (paper Sec. V-A).
 *
 * Correctness note on ef_search: `SET LOCAL` only takes effect inside a transaction block.
 * Outside one, PostgreSQL emits "WARNING: SET LOCAL can only be used in transaction blocks"
 * and the statement is a no-op, so the query would silently run at the default ef_search=40.
 * With a pooled backend the SET and the SELECT could also land on different server
 * connections. Both problems are avoided by issuing the SET LOCAL and the SELECT inside one
 * `sql.begin(...)` transaction, which pins a single connection for its duration.
 *
 * Runtime deps: see package.json ("postgres" driver, @cloudflare/workers-types for types).
 * Wrangler bindings: see wrangler.toml (AI, HYPERDRIVE).
 */

import postgres from "postgres";

export interface Env {
  /** Workers AI binding (edge inference). */
  AI: Ai;
  /**
   * Hyperdrive binding: a pooled Postgres connection string. Which database it points at
   * (central primary or a near-PoP replica) is decided per wrangler environment, so the
   * "edge-pooled" and "edge-replica" topologies share this code and differ only in config.
   */
  HYPERDRIVE?: Hyperdrive;
  /**
   * Direct connection string (a Wrangler secret) for the "edge-unpooled" topology: the Worker
   * dials the origin itself on every request and pays the full TCP+TLS+auth setup cost.
   */
  DATABASE_URL?: string;
  /** "hyperdrive" (default) or "direct". Set per wrangler environment. */
  TOPOLOGY?: string;
  /** Label echoed in responses so probes can tag rows (e.g. "edge-pooled"). */
  TOPOLOGY_LABEL?: string;
  /** Optional: cap results; defaults applied below. */
  MAX_TOP_K?: string;
  /** Optional: HNSW ef_search operating point (default 100). */
  EF_SEARCH?: string;
}

/** Embedding model: BAAI bge-base-en-v1.5 -> 768-dim vectors (must match schema.sql vector(768)). */
const EMBEDDING_MODEL = "@cf/baai/bge-base-en-v1.5" as const;
const EMBEDDING_DIM = 768;
const DEFAULT_TOP_K = 5;
const HARD_MAX_TOP_K = 50;
const DEFAULT_EF_SEARCH = 100;
const MAX_EF_SEARCH = 1000;

interface RetrievedChunk {
  id: number;
  content: string;
  /** Cosine distance in [0, 2]; cosine similarity = 1 - distance. */
  distance: number;
}

export default {
  async fetch(request: Request, env: Env, ctx: ExecutionContext): Promise<Response> {
    const tStart = Date.now();

    // GET /health?q=... : embed-only path (no database). Measures Workers AI embedding latency
    // at the serving PoP, independently of the retrieval tier.
    if (request.method === "GET") {
      const url = new URL(request.url);
      if (url.pathname !== "/health") {
        return json({ error: "Use POST / with { query: string, topK?: number }, or GET /health" }, 404);
      }
      const q = (url.searchParams.get("q") ?? "health check query").trim() || "health check query";
      const timing: Record<string, number> = {};
      try {
        const tEmbed = Date.now();
        const vec = await embedQuery(env, q);
        timing.embed = Date.now() - tEmbed;
        timing.total = Date.now() - tStart;
        return json(
          { ok: true, topology: env.TOPOLOGY_LABEL ?? null,
            colo: (request.cf as { colo?: string } | undefined)?.colo ?? null,
            dims: vec.length, timingMs: timing },
          200, { "server-timing": serverTiming(timing) });
      } catch (err) {
        timing.total = Date.now() - tStart;
        return json({ ok: false, detail: String(err), timingMs: timing }, 502,
                    { "server-timing": serverTiming(timing) });
      }
    }
    if (request.method !== "POST") {
      return json({ error: "Use POST with { query: string, topK?: number }" }, 405);
    }

    // 1. Validate.
    let body: { query?: unknown; topK?: unknown };
    try {
      body = await request.json();
    } catch {
      return json({ error: "Invalid JSON body" }, 400);
    }
    const query = typeof body.query === "string" ? body.query.trim() : "";
    if (!query) {
      return json({ error: "`query` must be a non-empty string" }, 400);
    }
    const topK = clampInt(body.topK, DEFAULT_TOP_K, 1, clampInt(env.MAX_TOP_K, HARD_MAX_TOP_K, 1, HARD_MAX_TOP_K));
    const efSearch = clampInt(env.EF_SEARCH, DEFAULT_EF_SEARCH, topK, MAX_EF_SEARCH);

    // Resolve the topology. A driver instance per invocation is the documented Hyperdrive
    // pattern: with Hyperdrive the TCP/TLS/auth work is done once by the pool, not by this
    // short-lived isolate; with a direct URL the isolate pays it on every request.
    const direct = (env.TOPOLOGY ?? "hyperdrive").toLowerCase() === "direct";
    const connectionString = direct ? env.DATABASE_URL : env.HYPERDRIVE?.connectionString;
    if (!connectionString) {
      return json({ error: `No connection string for TOPOLOGY=${env.TOPOLOGY ?? "hyperdrive"}` }, 500);
    }
    const sql = postgres(connectionString, {
      max: 1,             // one connection per invocation
      fetch_types: false, // avoid a type-lookup round trip on first use
      prepare: false,     // named prepared statements are unsafe under transaction pooling
      ssl: direct ? "require" : undefined, // origin TLS when dialing directly
    });

    const timing: Record<string, number> = {};
    try {
      // 2. Embed at the edge.
      const tEmbed = Date.now();
      const embedding = await embedQuery(env, query);
      timing.embed = Date.now() - tEmbed;

      // 3. Cosine ANN search. The SET LOCAL and the SELECT run inside ONE transaction so the
      //    ef_search operating point actually applies to this query (see header note).
      const literal = toVectorLiteral(embedding);
      const tSearch = Date.now();
      const rows = await sql.begin(async (tx) => {
        await tx.unsafe(`SET LOCAL hnsw.ef_search = ${efSearch}`);
        return tx<RetrievedChunk[]>`
          SELECT id, content, (embedding <=> ${literal}::vector) AS distance
          FROM documents
          ORDER BY embedding <=> ${literal}::vector
          LIMIT ${topK}
        `;
      });
      timing.search = Date.now() - tSearch;
      timing.total = Date.now() - tStart;

      // 4. Return retrieved context (generation happens downstream).
      return json(
        {
          query,
          model: EMBEDDING_MODEL,
          topology: env.TOPOLOGY_LABEL ?? (direct ? "edge-unpooled" : "edge-pooled"),
          colo: (request.cf as { colo?: string } | undefined)?.colo ?? null,
          topK,
          efSearch,
          timingMs: timing,
          results: rows.map((r) => ({
            id: r.id,
            content: r.content,
            similarity: 1 - Number(r.distance),
          })),
        },
        200,
        { "server-timing": serverTiming(timing) },
      );
    } catch (err) {
      timing.total = Date.now() - tStart;
      return json({ error: "Retrieval failed", detail: String(err), timingMs: timing }, 502, {
        "server-timing": serverTiming(timing),
      });
    } finally {
      // Release the driver handle after the response is sent; Hyperdrive keeps its pool warm.
      ctx.waitUntil(sql.end({ timeout: 5 }));
    }
  },
} satisfies ExportedHandler<Env>;

/** Generate a 768-dim embedding at the edge; validate dimensionality defensively. */
async function embedQuery(env: Env, text: string): Promise<number[]> {
  const out = (await env.AI.run(EMBEDDING_MODEL, { text: [text] })) as { data?: number[][] };
  const vec = out?.data?.[0];
  if (!Array.isArray(vec) || vec.length !== EMBEDDING_DIM) {
    throw new Error(`Embedding model returned ${vec?.length ?? "no"} dims; expected ${EMBEDDING_DIM}`);
  }
  return vec;
}

/** pgvector accepts a bracketed, comma-separated literal, e.g. "[0.1,0.2,...]". */
function toVectorLiteral(vec: number[]): string {
  return `[${vec.join(",")}]`;
}

/** Parse an integer-like value, falling back to `dflt`, and clamp to [lo, hi]. */
function clampInt(raw: unknown, dflt: number, lo: number, hi: number): number {
  const n = Number(raw);
  const v = Number.isFinite(n) && n > 0 ? Math.floor(n) : dflt;
  return Math.max(lo, Math.min(v, hi));
}

/** Format a Server-Timing header: `embed;dur=23, search;dur=8, total;dur=35`. */
function serverTiming(t: Record<string, number>): string {
  return Object.entries(t)
    .map(([k, v]) => `${k};dur=${v}`)
    .join(", ");
}

function json(obj: unknown, status = 200, extraHeaders: Record<string, string> = {}): Response {
  return new Response(JSON.stringify(obj), {
    status,
    headers: { "content-type": "application/json", ...extraHeaders },
  });
}

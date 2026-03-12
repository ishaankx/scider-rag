# Scider RAG Pipeline

Multi-agent retrieval and reasoning pipeline over scientific data — built for Problem B of the Scider backend engineering challenge.

---

## Quick Start

```bash
git clone <repo>
cd scider-rag
cp .env.example .env
# Edit .env and set OPENAI_API_KEY=sk-...
docker compose up --build
```

The API will be available at `http://localhost:8001`. OpenAPI docs at `http://localhost:8001/docs`.

**Seed sample data and run a smoke test:**
```bash
python -m scripts.seed_data
```

**Run the full feature demo (7 tests covering every level):**
```bash
python -m scripts.test_pipeline
```

This exercises the entire pipeline end-to-end: query API, caching + invalidation, 8-way concurrent queries, graph traversal, hallucination detection, batch evaluation, and sandbox security. Each test prints what it does and why.

---

## Requirements

- Docker + Docker Compose
- An OpenAI API key (models: `text-embedding-3-small`, `gpt-4o-mini`)
- No other local dependencies — everything runs in containers

---

## Feature Matrix

| Requirement | Level | Implementation |
|---|---|---|
| Ingest ≥ 2 source types | L1 Core | PDF, CSV, JSON, TXT via handler chain |
| Retrieval agent with search strategy | L1 Core | LLM-planned vector / keyword / hybrid search |
| Reasoning agent with ≥ 1 non-retrieval tool | L1 Core | ReAct loop with Calculator, GraphTraversal, CodeExecutor |
| Query API: answer + sources + latency | L1 Core | `/api/v1/query` with `LatencyBreakdown` |
| Concurrent query isolation | L2 Scale | Per-request `AgentContext`, global `asyncio.Semaphore` |
| Intelligent cache with staleness control | L2 Scale | Redis + version-based invalidation on ingestion |
| Entity graph traversal | L2 Scale | Recursive CTE in PostgreSQL, no separate graph DB |
| Hallucination detection | L3 Robust | LLM-as-judge grounding check, opt-in via `check_grounding` |
| Systematic evaluation framework | L3 Robust | `/api/v1/eval` with LLM-as-judge correctness + p95 latency |
| Sandboxed code execution | L3 Robust | Subprocess isolation, `resource.setrlimit`, import whitelist |

---

## Architecture

```
POST /query
     │
     ▼
AgentOrchestrator ── asyncio.wait_for(total_timeout)
  ├── QueryCache (Redis) ──── cache hit? return early
  │     ├── Version-based invalidation (INCR on ingestion)
  │     └── Stampede lock (SETNX → double-check → compute)
  ├── RetrievalAgent
  │     ├── LLM plans strategy (vector / keyword / hybrid)
  │     ├── VectorSearchTool  → Qdrant (cosine, 1536-dim)
  │     └── KeywordSearchTool → PostgreSQL (plainto_tsquery)
  └── ReasoningAgent (ReAct loop, max 5 iterations)
        ├── CalculatorTool     (safe AST eval)
        ├── GraphTraversalTool (recursive CTE, multi-hop)
        └── CodeExecutorTool   (sandboxed subprocess)

POST /ingest
     │
     ▼
IngestionPipeline
  ├── FileHandler (PDF/CSV/JSON/TXT) ── magic byte validation
  ├── Chunker (recursive split, 512 tokens, 50 overlap)
  ├── EmbeddingService (OpenAI, batched, circuit breaker)
  ├── VectorStore.upsert → Qdrant
  ├── EntityExtraction (LLM) → GraphStore (PostgreSQL)
  └── QueryCache.invalidate_all()
```

Each query gets a fresh `AgentContext` — no mutable state is shared between requests.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/api/v1/health` | Service health check (Postgres, Redis, Qdrant) |
| `POST` | `/api/v1/ingest` | Upload a file (PDF, CSV, JSON, TXT) |
| `GET` | `/api/v1/documents` | List ingested documents |
| `POST` | `/api/v1/query` | Ask a research question |
| `POST` | `/api/v1/eval` | Batch evaluation with LLM-as-judge scoring |

### Query request body

```json
{
  "question": "How does CRISPR-Cas9 achieve site-specific cleavage?",
  "max_sources": 5,
  "check_grounding": false
}
```

Setting `check_grounding: true` runs hallucination detection and returns a `grounding` object with per-claim support status. It adds one extra LLM call (~1–2 s).

### Query response

```json
{
  "answer": "...",
  "sources": [{"document_title": "...", "chunk_content": "...", "relevance_score": 0.87, ...}],
  "latency": {"retrieval_ms": 1240, "reasoning_ms": 4100, "total_ms": 5360},
  "confidence": 0.72,
  "request_id": "...",
  "grounding": null
}
```

---

## Non-Obvious Design Decisions

### Why no LangChain / LlamaIndex?

The brief asked to avoid black-box wrappers. Every abstraction here is written from scratch so the behaviour is fully transparent:
- ReAct loop is ~40 lines in `src/agents/reasoning.py`
- Retrieval strategy planning is a structured JSON prompt in `src/agents/retrieval.py`
- OpenAI function-calling drives tool dispatch — no framework routing

### Why PostgreSQL for the entity graph instead of Neo4j?

Neo4j is a separate operational dependency. The relationship queries needed here (multi-hop traversal, cycle prevention) are expressible as a single recursive CTE. This keeps the stack to 4 services and the graph stays transactionally consistent with the rest of the document data.

```sql
WITH RECURSIVE graph_walk AS (
    -- base case: direct neighbours
    -- recursive step: follow edges, guard cycles with path array
)
SELECT DISTINCT entity_id, depth, ...
```

### Version-based cache invalidation

Rather than scanning and deleting keys when new data arrives, the cache key includes a version number fetched from Redis. On every ingestion, the version is incremented (`INCR cache:version`). Old cache entries become unreachable immediately (they use the previous version in their key) and expire naturally via TTL.

### Cache stampede prevention

When a cache miss occurs for a popular query, concurrent requests would all trigger the expensive LLM pipeline simultaneously. A Redis distributed lock (`SETNX`) ensures only one request computes the result; others see the lock, re-check the cache, and return the freshly-written value.

### Sandboxed code execution

The reasoning agent can execute Python to help answer quantitative questions. The sandbox uses:
- A subprocess with a clean minimal environment (no inherited secrets)
- `resource.setrlimit` to cap CPU time and memory
- An explicit import allowlist (`math`, `statistics`, `json`, `re`, `itertools`, `collections`, `datetime`, `decimal`, `fractions`, `functools`, `string`, `textwrap`)
- Static analysis blocking dangerous patterns (`exec`, `eval`, `__import__`, `open`, `subprocess`, `os.system`)

The process is killed after `SANDBOX_TIMEOUT_SECONDS` regardless.

### Circuit breaker for external APIs

The OpenAI embedding service uses a circuit breaker (CLOSED → OPEN → HALF_OPEN state machine) layered on top of tenacity retries. After 5 consecutive failures, the circuit opens and calls are rejected immediately for 30 seconds — preventing cascading failures and wasted API credits during outages. A probe request after the timeout transitions to HALF_OPEN; if it succeeds, the circuit closes.

### Hallucination detection is opt-in

Running a grounding check on every query adds ~1–2 s and doubles the LLM cost per request. It is gated behind `check_grounding: true` so production workloads pay only when they need claim-level attribution. The evaluation framework (`/eval`) always runs it for batch quality measurement.

### Confidence score

The `confidence` field is a heuristic: the mean cosine similarity of the top-5 retrieved chunks. It is not the LLM's self-reported confidence (which is unreliable). When `check_grounding: true`, callers can also inspect `grounding.confidence`, which is the fraction of claims judged "supported" by a separate LLM pass.

---

## Security Hardening

| Layer | Mechanism |
|---|---|
| Input sanitization | HTML/XSS stripping (bleach), SQL injection pattern detection, filename sanitization |
| File upload validation | Extension allowlist + magic byte verification (rejects spoofed `.pdf`/`.json` files) |
| Rate limiting | Redis-backed sliding window per IP (`RateLimitMiddleware`) |
| Request isolation | UUID `X-Request-ID` header, per-request `AgentContext`, no shared mutable state |
| Code sandbox | Subprocess isolation, import allowlist, `resource.setrlimit`, clean env (no secrets) |
| Security headers | `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, body size limit |
| API resilience | Circuit breaker on OpenAI calls, total pipeline timeout (`asyncio.wait_for`) |

---

## Project Structure

```
src/
  api/          FastAPI routes, middleware, Pydantic schemas
  agents/       Orchestrator, RetrievalAgent, ReasoningAgent, tools
  ingestion/    Pipeline, chunker, embeddings, file handlers
  storage/      VectorStore (Qdrant), DocumentStore, GraphStore, QueryCache
  evaluation/   HallucinationDetector, PipelineEvaluator, LLM-as-judge metrics
  security/     Input sanitizer, circuit breaker, sandbox policy
  config.py     Pydantic Settings with env var loading
  dependencies.py  Shared connection pools, FastAPI dependency providers

tests/
  unit/         54 tests — chunker, sanitizer, calculator, cache, circuit breaker, magic bytes
  integration/  API validation tests (mocked services)
  fixtures/     sample_paper.txt, sample_data.csv, eval_questions.json

scripts/
  test_pipeline.py  Comprehensive 7-test demo (query, cache, concurrency, graph, grounding, eval, sandbox)
  seed_data.py      Ingest sample files + run smoke query
  run_eval.py       CLI evaluation runner
  init_db.py        Standalone schema creation
```

---

## Running Tests

```bash
# Unit tests (no external services needed)
python -m pytest tests/unit/ -v

# All tests inside Docker (includes magic byte tests requiring FastAPI)
docker compose exec app python -m pytest tests/unit/ -v

# Full pipeline demo (requires running stack + ingested data)
python -m scripts.test_pipeline

# Full evaluation against live stack
python -m scripts.run_eval
```

---

## Infrastructure

| Service | Image | Purpose |
|---|---|---|
| app | Python 3.12-slim | FastAPI + uvicorn |
| postgres | postgres:16-alpine | Documents, chunks, entity graph |
| redis | redis:7-alpine | Query cache, distributed locks, rate limiting |
| qdrant | qdrant/qdrant:v1.12.4 | Vector embeddings (1536-dim, cosine) |

All services run as non-root users. The app container uses a dedicated `appuser`.

---

## Known Limitations and What I Would Do Differently With More Time

**Latency** — End-to-end query latency is 10–20 s at current settings because the ReAct loop makes 2–4 sequential OpenAI calls. Mitigation options: streaming responses to the client, caching embedding lookups, or switching to a local embedding model to eliminate that round trip.

**Embedding dimension mismatch on schema change** — If `EMBEDDING_DIMENSIONS` is changed after the Qdrant collection is created, ingestion will fail silently. A migration path (recreate collection, re-embed all chunks) should be automated.

**Entity extraction quality** — Entities are extracted with a single LLM call using a JSON prompt. On complex scientific documents this can produce noisy or redundant entities. A dedicated NER model (e.g. SciSpacy) would improve graph quality substantially.

**No streaming** — The query API returns the full answer as a single JSON blob. Adding Server-Sent Events would allow the UI to show the answer token by token, significantly improving perceived latency.

**Evaluation dataset** — The fixture questions cover CRISPR and basic chemistry. A real evaluation set should have 50–200 questions with ground-truth answers derived from the ingested corpus.

**Rate limiting is per-IP** — A more robust production system would rate-limit per authenticated user/API key, not by IP, to prevent trivial bypass via proxies.

**Structured logging** — The application uses Python's standard `logging` module with a text format. For production observability, switching to JSON-structured logs (e.g. `structlog` or a custom `logging.Formatter`) would make log aggregation and alerting significantly easier.

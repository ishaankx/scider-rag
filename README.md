# Scider RAG Pipeline

Multi-agent retrieval and reasoning pipeline over scientific data — built for **Problem B** of the Scider backend engineering challenge.

Ingests scientific papers (PDF, CSV, JSON, TXT), builds a knowledge graph, and answers research questions using a ReAct-style agent loop with tool use, hallucination detection, and real-time SSE streaming with a live reasoning trace.

---

## Quick Start

```bash
git clone <repo>
cd scider-rag
cp .env.example .env          # Set OPENAI_API_KEY=sk-...
docker compose up --build -d
```

Once healthy (~15 s), seed data and run the full test suite:

```bash
python -m scripts.seed_data        # Ingest 7 scientific papers
python -m scripts.test_pipeline    # 8 end-to-end tests (query, cache, concurrency, graph, grounding, eval, sandbox, streaming)
```

| URL | What |
|---|---|
| `http://localhost:8001` | **Chat UI** — ask questions with live reasoning trace |
| `http://localhost:8001/docs` | OpenAPI / Swagger UI |
| `http://localhost:8001/redoc` | ReDoc API documentation |

### Try the streaming chat

Open `http://localhost:8001` in your browser. Click any sample question and watch the pipeline reason in real time — cache check, retrieval, tool calls, and reasoning iterations stream live, then sources and answer render progressively. Toggle **"Hallucination check"** for claim-level grounding.

Or from the terminal:

```bash
curl -N -X POST http://localhost:8001/api/v1/query/stream \
  -H "Content-Type: application/json" \
  -d '{"question": "How does self-attention work in Transformers?"}'
```

---

## Requirements

- Docker + Docker Compose
- An OpenAI API key (models: `text-embedding-3-small`, `gpt-4o-mini`)
- No other local dependencies — everything runs in containers

---

## Feature Matrix

| Requirement | Level | Implementation |
|---|---|---|
| Ingest ≥ 2 source types | **L1 Core** | PDF, CSV, JSON, TXT via handler chain with magic byte validation |
| Retrieval agent with search strategy | **L1 Core** | LLM-planned vector / keyword / hybrid search |
| Reasoning agent with ≥ 1 non-retrieval tool | **L1 Core** | ReAct loop with Calculator, GraphTraversal, CodeExecutor |
| Query API: answer + sources + latency | **L1 Core** | `POST /query` with latency breakdown |
| Concurrent query isolation | **L2 Scale** | Per-request `AgentContext`, global `asyncio.Semaphore` |
| Intelligent cache with staleness control | **L2 Scale** | Redis + version-based invalidation on ingestion |
| Entity graph traversal | **L2 Scale** | Recursive CTE in PostgreSQL — no separate graph DB |
| Hallucination detection | **L3 Robust** | LLM-as-judge grounding check, opt-in via `check_grounding` |
| Systematic evaluation framework | **L3 Robust** | `POST /eval` with LLM-as-judge correctness + p95 latency |
| Sandboxed code execution | **L3 Robust** | Subprocess isolation, `resource.setrlimit`, import allowlist |

### Beyond the brief

| Feature | Why |
|---|---|
| **SSE streaming with reasoning trace** | Real-time visibility into every pipeline step (cache → retrieval → tool calls → reasoning → answer). Designed for IDE panel integration — the exact use case Scider serves. |
| **Chat UI** | Single-file HTML/CSS/JS served at `/`, consumes the streaming endpoint. Dark IDE theme, source cards with relevance bars, collapsible thinking trace, grounding toggle. Zero build tools, zero extra dependencies. |
| **Security hardening** | Rate limiting, request ID tracking, magic byte file validation, input sanitization (XSS/SQL injection/path traversal), security headers, circuit breaker on external APIs, pipeline timeout. |

---

## Architecture

```
                            ┌──────────────────────────────────┐
                            │  Chat UI (http://localhost:8001) │
                            └───────────────┬──────────────────┘
                                            │ SSE / JSON
                                            ▼
POST /query ─────────────────────► AgentOrchestrator ◄──── POST /query/stream
                                     │
                        ┌────────────┼────────────────┐
                        ▼            ▼                ▼
                   QueryCache   RetrievalAgent   ReasoningAgent
                    (Redis)     │                 (ReAct loop)
                    │           │                 │
                    │     ┌─────┴─────┐     ┌─────┴──────────┐
                    │     ▼           ▼     ▼     ▼          ▼
                    │  VectorSearch  Keyword Calculator  GraphTraversal
                    │  (Qdrant)     (PG FTS) (AST eval)  (recursive CTE)
                    │                                          ▼
                    │                                    CodeExecutor
                    │                                    (sandboxed subprocess)
                    │
                    └──── Version-based invalidation on ingestion

POST /ingest ───► IngestionPipeline
                    ├── FileHandler (PDF/CSV/JSON/TXT) ── magic byte check
                    ├── Chunker (recursive split, 512 chars, 50 overlap)
                    ├── EmbeddingService (OpenAI, batched, circuit breaker)
                    ├── VectorStore.upsert → Qdrant
                    ├── EntityExtraction (LLM) → GraphStore (PostgreSQL)
                    └── QueryCache.invalidate_all()
```

Each query gets a fresh `AgentContext` — no mutable state is shared between requests. The semaphore caps concurrent LLM calls at 10, and a total pipeline timeout (`3 × LLM_TIMEOUT_SECONDS`) prevents runaway requests.

---

## API Endpoints

| Method | Path | Description |
|---|---|---|
| `GET` | `/` | Chat UI (live streaming reasoning trace) |
| `GET` | `/api/v1/health` | Service health check (Postgres, Redis, Qdrant) |
| `POST` | `/api/v1/ingest` | Upload a file (PDF, CSV, JSON, TXT) |
| `GET` | `/api/v1/documents` | List ingested documents |
| `POST` | `/api/v1/query` | Ask a research question |
| `POST` | `/api/v1/query/stream` | Streaming variant (Server-Sent Events) |
| `POST` | `/api/v1/eval` | Batch evaluation with LLM-as-judge scoring |

### Query request

```json
{
  "question": "How does scaled dot-product attention work in the Transformer?",
  "max_sources": 5,
  "check_grounding": false
}
```

### Query response

```json
{
  "answer": "...",
  "sources": [{"document_title": "...", "chunk_content": "...", "relevance_score": 0.87}],
  "latency": {"retrieval_ms": 1240, "reasoning_ms": 4100, "total_ms": 5360},
  "confidence": 0.72,
  "request_id": "a1b2c3d4",
  "grounding": null
}
```

Setting `check_grounding: true` runs hallucination detection and returns a `grounding` object with per-claim support status. Adds ~1–2 s (one extra LLM call).

### Streaming events (SSE)

`POST /query/stream` returns a `text/event-stream` with this event sequence:

| Event | Payload | When |
|---|---|---|
| `status` | `{step, message}` | Each pipeline step (cache, retrieval, tool calls, reasoning iterations) |
| `sources` | `[{document_title, relevance_score, …}]` | After retrieval — rendered early for progressive UI |
| `answer` | `{text}` | After reasoning completes |
| `done` | Full result (same shape as `/query`) | Pipeline finished |
| `error` | `{message, request_id}` | On failure |

---

## Key Architectural Decisions

### No LangChain / LlamaIndex

The brief asked to avoid black-box wrappers. Every abstraction is written from scratch so behaviour is fully transparent — the ReAct loop is ~40 lines in `src/agents/reasoning.py`, retrieval strategy planning is a structured JSON prompt in `src/agents/retrieval.py`, and OpenAI function-calling drives tool dispatch directly. This also avoids the dependency bloat and version churn typical of framework-heavy RAG stacks.

### PostgreSQL for the entity graph (no Neo4j)

Neo4j would add a fifth operational dependency. The relationship queries needed here — multi-hop traversal with cycle prevention — are expressible as a single recursive CTE in PostgreSQL. This keeps the stack to four services and the graph stays transactionally consistent with document and chunk data.

```sql
WITH RECURSIVE graph_walk AS (
    -- base case: direct neighbours
    -- recursive step: follow edges, guard cycles with path array
)
SELECT DISTINCT entity_id, depth, ...
```

### Version-based cache invalidation

Rather than scanning and deleting keys when new data arrives, the cache key includes a version number fetched from Redis. On every ingestion, the version is incremented (`INCR cache:version`). Old cache entries become unreachable immediately — they use the previous version in their key — and expire naturally via TTL. This is O(1) on write and eliminates key-scan latency spikes.

### Cache stampede prevention

When a cache miss occurs for a popular query, concurrent requests would all trigger the expensive LLM pipeline simultaneously. A Redis distributed lock (`SETNX`) ensures only one request computes the result; others see the lock, re-check the cache, and return the freshly-written value. This prevents cascading load under traffic bursts.

### Streaming as an additive layer

The SSE streaming endpoint (`POST /query/stream`) is implemented alongside the existing `POST /query` — not as a replacement. `run_stream()` and `execute_stream()` are parallel async generators added to `AgentOrchestrator` and `ReasoningAgent` respectively. The original `run()` and `execute()` are completely untouched. An internal `_result` event type lets generators pass structured data back to the orchestrator without forwarding it to clients. This makes the streaming feature fully backward-compatible and independently testable.

### Hallucination detection is opt-in

Running a grounding check on every query doubles LLM cost per request. It is gated behind `check_grounding: true` so production workloads pay only when they need claim-level attribution. Grounding is never cached — even on cache hits, a fresh grounding check runs when requested, because the detection model may have changed or the user may want fresh verification.

### Confidence score is retrieval-based, not self-reported

The `confidence` field is the mean cosine similarity of the top-5 retrieved chunks — a factual measurement from the retrieval step. It is intentionally not the LLM's self-reported confidence (which is unreliable). When `check_grounding: true`, callers also get `grounding.confidence`, which is the fraction of claims judged "supported" by a separate LLM pass.

### Sandboxed code execution

The reasoning agent can run Python to answer quantitative questions. The sandbox uses subprocess isolation with `resource.setrlimit` to cap CPU time (10 s) and memory (256 MB), a static analysis pass blocking dangerous patterns (`exec`, `eval`, `__import__`, `open`, `subprocess`, `os.system`), an import allowlist (`math`, `statistics`, `json`, `re`, `itertools`, `collections`, `datetime`, `decimal`, `fractions`, `functools`, `string`, `textwrap`), and a clean minimal environment (no inherited secrets). The process is force-killed after the timeout regardless of state.

### Circuit breaker on OpenAI API

The embedding service uses a circuit breaker (CLOSED → OPEN → HALF_OPEN state machine) layered on top of tenacity retries. After 5 consecutive failures, the circuit opens and calls are rejected immediately for 30 seconds — preventing cascading failures and wasted API credits during outages. A probe request after the timeout transitions to HALF_OPEN; if it succeeds, the circuit closes.

---

## Security Hardening

| Layer | Mechanism |
|---|---|
| Input sanitization | HTML/XSS stripping (bleach), SQL injection pattern detection, filename sanitization, path traversal prevention |
| File upload validation | Extension allowlist + magic byte signature verification (rejects spoofed files) |
| Rate limiting | Redis-backed sliding window per IP (`RateLimitMiddleware`, 60 req/min) |
| Request isolation | UUID `X-Request-ID` header, per-request `AgentContext`, no shared mutable state |
| Code sandbox | Subprocess isolation, import allowlist, `resource.setrlimit` (CPU + memory), clean env |
| Security headers | `X-Content-Type-Options`, `X-Frame-Options`, `X-XSS-Protection`, body size limit |
| API resilience | Circuit breaker on OpenAI calls, total pipeline timeout (`asyncio.wait_for`) |
| Body size limits | Configurable max upload size (50 MB default), enforced at middleware layer |

---

## Project Structure

```
src/
  main.py               App factory, middleware stack, route registration, chat UI mount
  config.py              Pydantic Settings — all config from env vars
  dependencies.py        Shared connection pools and FastAPI dependency injection

  api/
    v1/
      query.py           POST /query — full pipeline, JSON response
      stream.py          POST /query/stream — SSE with reasoning trace
      ingest.py          POST /ingest, GET /documents
      eval.py            POST /eval — batch evaluation
      health.py          GET /health — dependency checks
    middleware/
      request_id.py      X-Request-ID tracking (contextvars)
      rate_limit.py      Redis sliding window rate limiter
      security.py        Security headers + body size enforcement
    schemas/             Pydantic request/response models

  agents/
    orchestrator.py      AgentOrchestrator — run() and run_stream() pipelines
    retrieval.py         RetrievalAgent — LLM-planned search strategy
    reasoning.py         ReasoningAgent — ReAct loop, execute() and execute_stream()
    base.py              AgentContext, AgentResult, BaseAgent protocol
    tools/
      calculator.py      Safe AST-based math evaluation
      code_executor.py   Sandboxed subprocess Python execution
      graph_traversal.py Multi-hop entity traversal via recursive CTE
      search.py          VectorSearchTool, KeywordSearchTool

  ingestion/
    pipeline.py          Full ingestion flow: parse → chunk → embed → store → extract entities
    chunker.py           Recursive text splitting with configurable size and overlap
    embeddings.py        OpenAI embeddings with batching, retries, circuit breaker
    handlers/            PDF (PyMuPDF), CSV, JSON, TXT file parsers

  storage/
    models.py            SQLAlchemy ORM — documents, chunks, entities, relationships
    document_store.py    PostgreSQL CRUD + full-text search
    vector_store.py      Qdrant operations (upsert, search, delete)
    graph_store.py       Entity/relationship storage, recursive CTE traversal
    cache.py             Redis query cache with version-based invalidation + stampede locks
    init_db.py           Schema creation, Qdrant collection setup

  evaluation/
    evaluator.py         Batch pipeline evaluation (PipelineEvaluator)
    hallucination.py     LLM-based claim-level grounding check
    metrics.py           LLM-as-judge correctness scoring

  security/
    sanitizer.py         XSS, SQL injection, path traversal prevention
    circuit_breaker.py   Async circuit breaker (CLOSED → OPEN → HALF_OPEN)
    sandbox.py           Code execution security policy re-exports

  static/
    index.html           Chat UI — single-file dark-themed IDE-style interface

tests/
  unit/                  68 tests — chunker, cache, calculator, sanitizer, circuit breaker,
                         magic bytes, SSE formatting, event contracts
  integration/           API tests with mocked services

scripts/
  test_pipeline.py       8 end-to-end tests (query, cache, concurrency, graph, grounding,
                         eval, sandbox, streaming) across AI/ML, quantum, and biology domains
  seed_data.py           Ingest sample papers + smoke test
  ingest_data.py         Bulk ingestion
  run_eval.py            CLI evaluation runner
```

---

## Running Tests

```bash
# Unit tests (68 tests, no external services needed — run inside Docker)
docker compose exec app python -m pytest tests/unit/ -v

# Full pipeline demo (requires running stack + ingested data)
python -m scripts.test_pipeline

# Evaluation against live stack
python -m scripts.run_eval
```

### Test coverage by domain

The end-to-end test suite (`scripts/test_pipeline.py`) exercises all three scientific domains present in the ingested corpus:

- **AI/ML**: Transformer self-attention, BERT masked language modelling, GPT-3 few-shot learning
- **Quantum computing**: Grover's algorithm, quantum speedup complexity, superposition
- **Biology**: CRISPR-Cas adaptive immunity, GDSC drug sensitivity biomarkers, cancer genomics

This ensures the pipeline generalises across domains rather than being tested against a single paper.

---

## Infrastructure

| Service | Image | Purpose | Port |
|---|---|---|---|
| app | Python 3.12-slim | FastAPI + uvicorn | 8001 |
| postgres | postgres:16-alpine | Documents, chunks, entity graph | 5433 |
| redis | redis:7-alpine | Query cache, distributed locks, rate limits | 6380 |
| qdrant | qdrant/qdrant:v1.12.4 | Vector embeddings (1536-dim, cosine) | 6334 |

All services run as non-root users. The app container uses a dedicated `appuser`. Health checks ensure `app` only starts after all dependencies are ready.

---

## Environment Variables

Copy `.env.example` to `.env` and configure:

| Variable | Default | Description |
|---|---|---|
| `OPENAI_API_KEY` | *(required)* | OpenAI API key |
| `LLM_MODEL` | `gpt-4o-mini` | Chat completion model |
| `EMBEDDING_MODEL` | `text-embedding-3-small` | Embedding model |
| `EMBEDDING_DIMENSIONS` | `1536` | Embedding vector dimensions |
| `LLM_TEMPERATURE` | `0.1` | LLM sampling temperature |
| `LLM_MAX_TOKENS` | `2048` | Max tokens per LLM response |
| `LLM_TIMEOUT_SECONDS` | `30` | Per-call LLM timeout |
| `DATABASE_URL` | `postgresql+asyncpg://...` | PostgreSQL connection |
| `REDIS_URL` | `redis://redis:6379/0` | Redis connection |
| `QDRANT_HOST` / `QDRANT_PORT` | `qdrant` / `6333` | Qdrant connection |
| `CHUNK_SIZE` | `512` | Chunk size (chars) |
| `CHUNK_OVERLAP` | `50` | Chunk overlap (chars) |
| `RATE_LIMIT_PER_MINUTE` | `60` | Max requests per IP per minute |
| `AGENT_MAX_ITERATIONS` | `5` | ReAct loop iteration cap |
| `RETRIEVAL_TOP_K` | `10` | Max chunks retrieved per query |
| `SANDBOX_TIMEOUT_SECONDS` | `10` | Code execution timeout |
| `SANDBOX_MAX_MEMORY_MB` | `256` | Code execution memory limit |
| `MAX_FILE_SIZE_MB` | `50` | Upload size limit |

---

## Known Limitations

**Latency** — End-to-end query latency is 5–15 s on cache miss because the ReAct loop makes 2–4 sequential OpenAI API calls. The streaming endpoint mitigates perceived latency (the user sees progress live), and cache hits return in <50 ms.

**Embedding dimension lock-in** — If `EMBEDDING_DIMENSIONS` is changed after the Qdrant collection is created, ingestion will fail. A migration path (recreate collection, re-embed all chunks) should be automated.

**Entity extraction quality** — Entities are extracted with a single LLM call using a JSON prompt. On complex scientific documents this produces noisy or redundant entities. A dedicated NER model (e.g. SciSpacy) would improve graph quality.

**Rate limiting is per-IP** — A production system should rate-limit per authenticated user/API key to prevent trivial bypass via proxies.

**Evaluation dataset** — The fixture questions cover three domains but only 3–8 questions each. A real evaluation set should have 50–200 questions with ground-truth answers derived from the corpus.

---

## What I Would Do Differently With More Time

**Token-level streaming** — The current SSE streaming shows pipeline steps in real time, but the answer text arrives as a complete block. Integrating OpenAI's streaming API (`stream=True`) would let the answer render token-by-token, matching the ChatGPT/Claude experience and further reducing perceived latency.

**Hybrid reranking** — The retrieval agent uses cosine similarity from the embedding model. Adding a cross-encoder reranker (e.g. `bge-reranker-v2-m3`) as a second pass would significantly improve retrieval precision, especially for domain-specific scientific queries.

**Structured logging** — The application uses Python's standard `logging` module. For production observability, JSON-structured logs (via a custom formatter or `python-json-logger`) would make log aggregation, alerting, and trace correlation across services far more effective.

**Authentication and RBAC** — The system is currently open. Adding JWT-based auth with scoped API keys would enable per-user rate limiting, audit trails, and multi-tenant isolation.

**Incremental ingestion** — Currently, re-ingesting a modified file creates a new document record. A proper incremental pipeline would diff against existing chunks, update only what changed, and selectively re-embed — saving significant API cost on large corpora.

**Observability** — Adding OpenTelemetry traces (spans for each pipeline step) and Prometheus metrics (query latency histograms, cache hit rates, LLM call counts) would give production-grade visibility into system behaviour under load.

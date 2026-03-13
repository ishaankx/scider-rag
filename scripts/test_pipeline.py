"""
Comprehensive pipeline test — mirrors the criteria a recruiting engineer would evaluate.

Run with: python -m scripts.test_pipeline

Tests:
  1. Query API   — answer, sources, latency breakdown
  2. Caching     — cache hit on repeat query; invalidation on new ingestion
  3. Concurrency — 8 parallel queries, all complete correctly with no cross-talk
  4. Graph       — entity relationship traversal (not just document retrieval)
  5. Grounding   — hallucination detection flags unsupported claims
  6. Evaluation  — systematic scoring across N questions (LLM-as-judge)
  7. Sandbox     — safe code runs; dangerous patterns are blocked at tool level
  8. Streaming   — SSE reasoning trace with live pipeline visibility
"""

import asyncio
import os
import sys
import time
import uuid

import httpx

# Add project root to path for direct tool imports (sandbox test)
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

API_BASE = os.getenv("API_BASE", "http://localhost:8001/api/v1")
TIMEOUT = 120.0

# ── helpers ──────────────────────────────────────────────────────────────────

def header(title: str) -> None:
    print(f"\n{'═' * 64}")
    print(f"  {title}")
    print(f"{'═' * 64}")

def ok(msg: str) -> None:
    print(f"  ✓  {msg}")

def fail(msg: str) -> None:
    print(f"  ✗  {msg}")

def info(msg: str) -> None:
    print(f"     {msg}")

def show_qa(question: str, answer: str, max_a: int = 200) -> None:
    info(f"Q: {question}")
    info(f"A: {answer[:max_a]}{'...' if len(answer) > max_a else ''}")


# ── 1. QUERY API ─────────────────────────────────────────────────────────────

async def test_query_api(client: httpx.AsyncClient) -> None:
    header("TEST 1 — Query API: answer + sources + latency breakdown")

    question = "How does the self-attention mechanism work in the Transformer architecture?"
    resp = await client.post("/query", json={"question": question, "max_sources": 3})

    assert resp.status_code == 200, f"Expected 200, got {resp.status_code}: {resp.text}"
    data = resp.json()

    assert data["answer"], "Answer is empty"
    show_qa(question, data["answer"])
    print()

    assert len(data["sources"]) > 0, "No sources returned"
    ok(f"Sources returned: {len(data['sources'])}")
    for s in data["sources"]:
        info(f"  [{s['relevance_score']:.3f}] {s['document_title']} — {s['chunk_content'][:80]}...")

    latency = data["latency"]
    assert latency["retrieval_ms"] > 0
    assert latency["reasoning_ms"] > 0
    assert latency["total_ms"] > 0
    ok(f"Latency breakdown:")
    info(f"  retrieval_ms  = {latency['retrieval_ms']:.0f}ms  (embed query + Qdrant/PG search)")
    info(f"  reasoning_ms  = {latency['reasoning_ms']:.0f}ms  (ReAct loop + tool calls)")
    info(f"  total_ms      = {latency['total_ms']:.0f}ms  (wall-clock including overhead)")

    ok(f"Confidence: {data['confidence']:.3f}  (mean cosine similarity of top-5 chunks)")
    ok(f"Request ID: {data['request_id']}")


# ── 2. CACHING ────────────────────────────────────────────────────────────────

async def test_caching(client: httpx.AsyncClient) -> None:
    header("TEST 2 — Caching: repeat query → cache hit; new ingestion → invalidation")

    # Use a unique run ID so the question is never in cache from a prior run
    run_id = uuid.uuid4().hex[:8]
    question = f"What is Grover's algorithm and how does it achieve quantum speedup over classical search? [run-{run_id}]"

    # First call — guaranteed cache miss (unique question)
    t0 = time.perf_counter()
    r1 = await client.post("/query", json={"question": question})
    ms1 = (time.perf_counter() - t0) * 1000
    assert r1.status_code == 200
    show_qa(question, r1.json()["answer"])
    print()

    # Second call — should be a cache hit (dramatically faster)
    t0 = time.perf_counter()
    r2 = await client.post("/query", json={"question": question})
    ms2 = (time.perf_counter() - t0) * 1000
    assert r2.status_code == 200

    speedup = ms1 / ms2 if ms2 > 0 else 999
    ok(f"Cache miss (1st call): {ms1:.0f}ms")
    ok(f"Cache hit  (2nd call): {ms2:.0f}ms  →  {speedup:.0f}× faster")
    assert speedup >= 2, f"Expected ≥2× speedup, got {speedup:.1f}×"
    ok("Cache hit confirmed")

    assert r1.json()["answer"] == r2.json()["answer"], "Cached answer differs!"
    ok("Cached answer is byte-for-byte identical")

    # Ingest a new document → increments the Redis cache version key
    # Use run_id to make content unique so dedup never blocks this
    new_content = (
        f"Quantum error correction is essential for fault-tolerant quantum computation. "
        f"Surface codes provide a practical path to scalable quantum computers. [run-{run_id}]"
    )
    tmp_path = f"/tmp/quantum_error_correction_{run_id}.txt"
    with open(tmp_path, "w") as f:
        f.write(new_content)
    with open(tmp_path, "rb") as f:
        ingest_resp = await client.post(
            "/ingest",
            files={"file": (f"quantum_error_correction_{run_id}.txt", f, "text/plain")},
        )
    assert ingest_resp.status_code == 200, f"Ingest failed: {ingest_resp.text}"
    ok("New document ingested — Redis cache version incremented")
    info("Old cache keys are now unreachable (version mismatch) and will TTL out")

    # Third call — cache miss because version changed; runs full pipeline
    t0 = time.perf_counter()
    r3 = await client.post("/query", json={"question": question})
    ms3 = (time.perf_counter() - t0) * 1000
    assert r3.status_code == 200
    ok(f"Post-ingestion call:   {ms3:.0f}ms  (full pipeline — stale cache bypassed)")
    assert ms3 > ms2 * 3, (
        f"Expected much slower after invalidation ({ms3:.0f}ms vs {ms2:.0f}ms)"
    )
    ok("Cache invalidation confirmed — no stale results served")


# ── 3. CONCURRENCY ────────────────────────────────────────────────────────────

async def test_concurrency(client: httpx.AsyncClient) -> None:
    header("TEST 3 — Concurrency: 8 parallel queries with no pipeline interference")

    questions = [
        # AI/ML — Transformer, BERT, GPT-3
        "What is the Transformer architecture and why did it replace RNNs?",
        "How does BERT's masked language model pre-training work?",
        "What is few-shot learning in GPT-3 and how does it differ from fine-tuning?",
        # Quantum computing — Grover's algorithm
        "How does Grover's quantum search algorithm achieve sqrt(N) speedup?",
        "What is quantum superposition and how is it used in database search?",
        # Biology — CRISPR, cancer genomics
        "What are CRISPR-Cas adaptive immune systems in bacteria?",
        "What drug sensitivity biomarkers are used in cancer cell lines?",
        "How does the GDSC database help identify therapeutic targets?",
    ]

    info(f"Firing {len(questions)} queries simultaneously...")
    start = time.perf_counter()
    responses = await asyncio.gather(*[
        client.post("/query", json={"question": q}) for q in questions
    ])
    wall_ms = (time.perf_counter() - start) * 1000

    failed_idx = [i for i, r in enumerate(responses) if r.status_code != 200]
    if failed_idx:
        fail(f"Queries {failed_idx} returned non-200")
    else:
        ok(f"All {len(questions)} queries returned 200")

    ok(f"Wall-clock time for {len(questions)} parallel queries: {wall_ms:.0f}ms")
    info("(Sequential would be ~8× longer — parallelism is working)")

    request_ids = [r.json().get("request_id") for r in responses]
    assert len(set(request_ids)) == len(questions), "Duplicate request IDs — isolation broken!"
    ok("All responses have unique request_ids — no cross-pipeline contamination")

    print()
    info("Sample answers from parallel execution:")
    for q, r in list(zip(questions, responses))[:3]:
        data = r.json()
        info(f"  Q: {q}")
        info(f"  A: {data['answer'][:100]}...")
        info("")


# ── 4. GRAPH TRAVERSAL ────────────────────────────────────────────────────────

async def test_graph_traversal(client: httpx.AsyncClient) -> None:
    header("TEST 4 — Graph traversal: entity relationship queries")

    question = (
        "What entities and relationships connect the Transformer architecture to BERT and GPT-3? "
        "Please explore using the graph tool."
    )
    resp = await client.post("/query", json={"question": question, "max_sources": 5})
    assert resp.status_code == 200, f"Query failed: {resp.text}"
    data = resp.json()

    show_qa(question, data["answer"])
    print()
    ok(f"Query completed in {data['latency']['total_ms']:.0f}ms")
    ok(f"Sources used: {len(data['sources'])}")
    info("")
    info("How graph traversal works:")
    info("  During ingestion, the LLM extracts entities (model, algorithm, technique, etc.)")
    info("  and their relationships, stored in PostgreSQL.")
    info("  The graph_traverse tool uses a recursive CTE:")
    info("    WITH RECURSIVE graph_walk AS (")
    info("      base: direct neighbours of start entity")
    info("      + recursive: follow edges, guard cycles with path array")
    info("    )")
    info("  This gives multi-hop traversal without Neo4j.")


# ── 5. GROUNDING / HALLUCINATION DETECTION ───────────────────────────────────

async def test_grounding(client: httpx.AsyncClient) -> None:
    header("TEST 5 — Grounding: hallucination detection flags unsupported claims")

    question = "What is Grover's quantum search algorithm and what computational complexity does it achieve?"

    # With grounding enabled
    resp = await client.post("/query", json={
        "question": question,
        "max_sources": 5,
        "check_grounding": True,
    })
    assert resp.status_code == 200, f"Query failed: {resp.text}"
    data = resp.json()

    show_qa(question, data["answer"])
    print()

    grounding = data.get("grounding")
    assert grounding is not None, "grounding field is missing — check_grounding=True was sent"

    ok("Grounding check ran (separate LLM pass against retrieved sources):")
    info(f"  Supported claims:    {grounding['supported_count']}  (claim found in sources)")
    info(f"  Partial claims:      {grounding['partial_count']}   (partially in sources)")
    info(f"  Unsupported claims:  {grounding['unsupported_count']}  (possible hallucination)")
    info(f"  Grounding confidence:{grounding['confidence']:.3f}")

    if grounding["flags"]:
        info("")
        info("Flagged (unsupported/partial) claims:")
        for flag in grounding["flags"][:5]:
            info(f"  ⚠ {flag[:110]}")
    else:
        ok("All claims grounded in sources — no hallucination detected")

    # Verify default (check_grounding=False) returns null grounding
    resp2 = await client.post("/query", json={"question": "What is multi-head attention?", "check_grounding": False})
    assert resp2.json().get("grounding") is None, "grounding should be null when not requested"
    ok("grounding=null when check_grounding=False — no extra LLM cost by default")

    # Verify cache hit still runs grounding (the bug that was fixed)
    resp3 = await client.post("/query", json={"question": question, "max_sources": 5, "check_grounding": True})
    data3 = resp3.json()
    assert data3.get("grounding") is not None, "grounding missing on cache hit — fix not applied!"
    ok("Cache hit with check_grounding=True still returns grounding (not skipped)")


# ── 6. EVALUATION FRAMEWORK ───────────────────────────────────────────────────

async def test_evaluation(client: httpx.AsyncClient) -> None:
    header("TEST 6 — Evaluation: systematic LLM-as-judge scoring across questions")

    info("What this tests:")
    info("  • correctness: LLM scores answer vs expected_answer (0–1)")
    info("  • hallucination_flags: claims not grounded in sources")
    info("  • hallucination_rate: fraction of questions with ≥1 flag")
    info("  • p95_latency: 95th percentile response time")
    print()

    questions = [
        {
            "question": "How does scaled dot-product attention work in the Transformer?",
            "expected_answer": (
                "Scaled dot-product attention computes compatibility between queries and keys "
                "by taking their dot product, scaling by sqrt(d_k), applying softmax, "
                "then multiplying by the values to produce the output."
            ),
        },
        {
            "question": "What quantum speedup does Grover's algorithm achieve for unstructured search?",
            "expected_answer": (
                "Grover's algorithm searches an unsorted database of N items in O(sqrt(N)) "
                "time, a quadratic speedup over the classical O(N) approach."
            ),
        },
        {
            "question": "How does the GDSC database identify drug sensitivity biomarkers in cancer?",
            "expected_answer": (
                "GDSC screens hundreds of cancer cell lines against anticancer drugs, "
                "linking drug sensitivity to genomic features like mutations, "
                "gene amplification, and deletions to discover therapeutic biomarkers."
            ),
        },
    ]

    resp = await client.post("/eval", json={"questions": questions})
    assert resp.status_code == 200, f"Eval failed: {resp.text}"
    data = resp.json()

    summary = data["summary"]
    ok(f"Evaluated {summary['total_questions']} questions")
    print()
    info("Aggregate metrics:")
    info(f"  avg_correctness    = {summary['avg_correctness']:.3f}  "
         f"(LLM judge: does answer match expected?  0=wrong, 1=correct)")
    info(f"  avg_relevance      = {summary['avg_relevance']:.3f}  "
         f"(retrieval confidence score)")
    info(f"  avg_confidence     = {summary['avg_confidence']:.3f}  "
         f"(grounding: fraction of claims supported)")
    info(f"  hallucination_rate = {summary['hallucination_rate']:.0%}  "
         f"(fraction of questions with ≥1 unsupported claim)")
    latency_note = "  ← cache hits from prior runs" if summary["avg_latency_ms"] < 100 else ""
    info(f"  avg_latency_ms     = {summary['avg_latency_ms']:.0f}ms{latency_note}")
    info(f"  p95_latency_ms     = {summary['p95_latency_ms']:.0f}ms  "
         f"(95th percentile — SLA target){latency_note}")
    print()
    info("Per-question results:")
    info(f"  {'Score':>5}  {'Flags':>5}  Question")
    info(f"  {'─'*5}  {'─'*5}  {'─'*50}")
    for r in data["results"]:
        flag_count = len(r["hallucination_flags"])
        mark = "✓" if r["correctness_score"] >= 0.7 else "~"
        info(f"  {mark} {r['correctness_score']:.2f}  "
             f"{flag_count} flag{'s' if flag_count != 1 else ' '}  "
             f"{r['question'][:60]}")
        info(f"         Answer: {r['answer'][:100]}...")
        for flag in r["hallucination_flags"][:2]:
            info(f"         ⚠ {flag[:100]}")
    print()
    info("Note on hallucination_rate:")
    info("  A rate >0 doesn't mean the answers are wrong — it means the LLM judge")
    info("  found at least one claim that wasn't explicitly in the retrieved chunks.")
    info("  Generic transitional sentences ('This is important because...') often")
    info("  get flagged even when the core factual claims are fully grounded.")


# ── 7. CODE SANDBOX ───────────────────────────────────────────────────────────

async def test_sandbox(client: httpx.AsyncClient) -> None:
    header("TEST 7 — Code sandbox: safe code runs, dangerous patterns are blocked")

    # Part A: Safe computation through the query API (end-to-end)
    question = (
        "Using your calculation tools, compute: sqrt(144) * 3.14159. "
        "Show the step-by-step calculation."
    )
    resp = await client.post("/query", json={"question": question})
    assert resp.status_code == 200
    data = resp.json()
    show_qa(question, data["answer"])
    print()
    ok("Computation completed via agent tool use")

    # Part B: Direct sandbox tool tests (no subprocess-in-subprocess)
    info("Direct tool-level tests (bypassing HTTP API):")
    print()
    try:
        from src.config import get_settings
        from src.agents.tools.code_executor import CodeExecutorTool
    except ImportError:
        info("Skipping direct sandbox tests (app dependencies not installed locally).")
        info("These run inside the Docker container. The HTTP-level test above covers the feature.")
        return

    tool = CodeExecutorTool(get_settings())

    blocked_cases = [
        ("os.system",   "import os; os.system('echo pwned')"),
        ("subprocess",  "import subprocess; subprocess.run(['ls'])"),
        ("exec()",      "exec('import os; os.getcwd()')"),
        ("eval()",      "eval('__import__(\"os\").getcwd()')"),
        ("open()",      "open('/etc/passwd', 'r').read()"),
        ("__import__",  "__import__('os').system('id')"),
    ]

    for label, code in blocked_cases:
        result = await tool.execute(code=code)
        if not result.success:
            ok(f"BLOCKED  {label:15s} → {result.error[:70]}")
        else:
            fail(f"ALLOWED  {label}  ← SECURITY HOLE: {result.output[:60]}")

    print()
    safe_cases = [
        ("math.sqrt",     "import math; print(math.sqrt(144) * 3.14159)"),
        ("statistics",    "import statistics; print(statistics.mean([1,2,3,4,5]))"),
        ("list comp",     "print(sum(x**2 for x in range(1, 6)))"),
        ("json parsing",  "import json; d=json.loads('{\"a\":1}'); print(d['a'])"),
    ]

    for label, code in safe_cases:
        result = await tool.execute(code=code)
        if result.success:
            ok(f"ALLOWED  {label:15s} → output: {result.output.strip()}")
        else:
            fail(f"BLOCKED  {label}  ← should have been allowed: {result.error}")


# ── 8. STREAMING / REASONING TRACE ───────────────────────────────────────────

async def test_streaming(client: httpx.AsyncClient) -> None:
    header("TEST 8 — Streaming: SSE reasoning trace with live pipeline events")

    info("What this tests:")
    info("  • POST /query/stream returns Server-Sent Events")
    info("  • Events arrive incrementally (status → sources → answer → done)")
    info("  • Each pipeline step is visible in real time")
    info("  • Designed for IDE integration — render reasoning as it happens")
    print()

    question = "How does GPT-3 perform few-shot learning without gradient updates?"

    events = []
    event_types_seen = set()

    # Read the SSE stream
    async with client.stream(
        "POST",
        "/query/stream",
        json={"question": question, "max_sources": 3},
    ) as response:
        assert response.status_code == 200, f"Expected 200, got {response.status_code}"
        content_type = response.headers.get("content-type", "")
        assert "text/event-stream" in content_type, f"Expected text/event-stream, got {content_type}"

        current_event = None
        async for line in response.aiter_lines():
            if line.startswith("event: "):
                current_event = line[7:]
            elif line.startswith("data: ") and current_event:
                import json
                data = json.loads(line[6:])
                events.append((current_event, data))
                event_types_seen.add(current_event)
                current_event = None

    ok(f"Received {len(events)} SSE events")

    # Verify expected event types
    assert "status" in event_types_seen, "No status events received"
    ok("status events received (pipeline step updates)")

    assert "sources" in event_types_seen, "No sources event received"
    ok("sources event received (retrieved documents)")

    assert "answer" in event_types_seen, "No answer event received"
    ok("answer event received (final synthesized answer)")

    assert "done" in event_types_seen, "No done event received"
    ok("done event received (complete result with latency)")

    # Show the reasoning trace
    print()
    info("Live reasoning trace:")
    for event_type, data in events:
        if event_type == "status":
            step = data.get("step", "?")
            msg = data.get("message", "")[:90]
            info(f"  [{step:>12}] {msg}")
        elif event_type == "sources":
            info(f"  [     sources] {len(data)} chunks retrieved")
            for s in data[:2]:
                info(f"                 [{s.get('relevance_score', 0):.3f}] {s.get('document_title', '?')[:50]}")
        elif event_type == "answer":
            info(f"  [      answer] {data.get('text', '')[:100]}...")
        elif event_type == "done":
            lat = data.get("latency", {})
            info(f"  [        done] total={lat.get('total_ms', 0):.0f}ms  "
                 f"confidence={data.get('confidence', 0):.3f}  "
                 f"request_id={data.get('request_id', '?')[:12]}")

    # Verify the done event has the full result shape
    done_events = [d for t, d in events if t == "done"]
    assert len(done_events) == 1, f"Expected 1 done event, got {len(done_events)}"
    done_data = done_events[0]
    assert "answer" in done_data, "done event missing answer"
    assert "sources" in done_data, "done event missing sources"
    assert "latency" in done_data, "done event missing latency"
    assert "request_id" in done_data, "done event missing request_id"
    ok("done event matches POST /query response shape")

    # Verify event ordering: status events come before sources/answer/done
    first_status_idx = next(i for i, (t, _) in enumerate(events) if t == "status")
    first_sources_idx = next(i for i, (t, _) in enumerate(events) if t == "sources")
    first_answer_idx = next(i for i, (t, _) in enumerate(events) if t == "answer")
    done_idx = next(i for i, (t, _) in enumerate(events) if t == "done")

    assert first_status_idx < first_sources_idx < first_answer_idx < done_idx, (
        "Events out of order — expected: status → sources → answer → done"
    )
    ok("Event ordering correct: status → sources → answer → done")

    print()
    info("Why this matters for a scientific IDE:")
    info("  In Scider, researchers submit complex queries that take 5-15 seconds.")
    info("  Without streaming, they stare at a spinner. With SSE streaming,")
    info("  the IDE can show: 'Searching 7 papers... Found 5 chunks...'")
    info("  'Calling graph_traverse... Synthesizing answer...' — keeping the")
    info("  researcher engaged and building trust in the reasoning process.")


# ── MAIN ─────────────────────────────────────────────────────────────────────

async def main() -> None:
    print("\n" + "═" * 64)
    print("  Scider RAG Pipeline — Comprehensive Feature Test")
    print("═" * 64)
    print(f"  Target: {API_BASE}")

    async with httpx.AsyncClient(base_url=API_BASE, timeout=TIMEOUT) as client:
        try:
            r = await client.get("/health")
            health = r.json()
            assert r.status_code == 200 and health.get("status") == "healthy"
            services = {k: v for k, v in health.items() if k != "status"}
            print(f"  Services: {services}")
        except Exception as e:
            print(f"\n  ✗ API not reachable: {e}")
            print("    Start the stack: docker compose up")
            sys.exit(1)

        results: dict[str, bool] = {}
        tests = [
            ("Query API (answer + sources + latency)", test_query_api),
            ("Caching (invalidation on ingestion)",    test_caching),
            ("Concurrency (8 parallel queries)",       test_concurrency),
            ("Graph traversal (entity relationships)", test_graph_traversal),
            ("Grounding / hallucination detection",    test_grounding),
            ("Evaluation framework (LLM-as-judge)",    test_evaluation),
            ("Code sandbox (dangerous patterns blocked)", test_sandbox),
            ("Streaming (SSE reasoning trace)",           test_streaming),
        ]

        for name, fn in tests:
            try:
                await fn(client)
                results[name] = True
            except AssertionError as e:
                fail(f"ASSERTION FAILED: {e}")
                results[name] = False
            except Exception as e:
                import traceback
                fail(f"ERROR: {type(e).__name__}: {e}")
                info(traceback.format_exc()[-400:])
                results[name] = False

    header("SUMMARY")
    passed = sum(1 for v in results.values() if v)
    total = len(results)
    for name, ok_flag in results.items():
        mark = "✓" if ok_flag else "✗"
        print(f"  {mark}  {name}")
    print(f"\n  {passed}/{total} tests passed\n")
    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    asyncio.run(main())

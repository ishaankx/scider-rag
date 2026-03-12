"""
Run the evaluation suite against the pipeline.
Usage: python -m scripts.run_eval [--file path/to/questions.json]
"""

import argparse
import asyncio
import json
import os
import sys

import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8000/api/v1")
DEFAULT_EVAL_FILE = "tests/fixtures/eval_questions.json"


async def main(eval_file: str):
    # Load questions
    if not os.path.exists(eval_file):
        print(f"Evaluation file not found: {eval_file}")
        sys.exit(1)

    with open(eval_file) as f:
        questions = json.load(f)

    print(f"Running evaluation with {len(questions)} questions...")
    print(f"API: {API_BASE}")
    print()

    async with httpx.AsyncClient(timeout=120.0) as client:
        resp = await client.post(
            f"{API_BASE}/eval",
            json={"questions": questions},
        )

        if resp.status_code != 200:
            print(f"Evaluation failed ({resp.status_code}): {resp.text}")
            sys.exit(1)

        result = resp.json()

    # Print summary
    summary = result["summary"]
    print("=" * 60)
    print("EVALUATION SUMMARY")
    print("=" * 60)
    print(f"  Total questions:    {summary['total_questions']}")
    print(f"  Avg relevance:      {summary['avg_relevance']:.3f}")
    print(f"  Avg correctness:    {summary['avg_correctness']:.3f}")
    print(f"  Avg confidence:     {summary['avg_confidence']:.3f}")
    print(f"  Avg latency:        {summary['avg_latency_ms']:.0f}ms")
    print(f"  P95 latency:        {summary['p95_latency_ms']:.0f}ms")
    print(f"  Hallucination rate: {summary['hallucination_rate']:.1%}")
    print()

    # Print per-question results
    for i, r in enumerate(result["results"], 1):
        print(f"--- Question {i} ---")
        print(f"  Q: {r['question'][:80]}")
        print(f"  A: {r['answer'][:120]}...")
        print(f"  Relevance:  {r['relevance_score']:.3f}")
        print(f"  Correctness: {r['correctness_score']:.3f}")
        print(f"  Confidence: {r['confidence']:.3f}")
        print(f"  Latency:    {r['latency_ms']:.0f}ms")
        print(f"  Sources:    {r['sources_found']}")
        if r["hallucination_flags"]:
            print(f"  Flags:      {r['hallucination_flags']}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run RAG pipeline evaluation")
    parser.add_argument(
        "--file", default=DEFAULT_EVAL_FILE, help="Path to evaluation questions JSON"
    )
    args = parser.parse_args()
    asyncio.run(main(args.file))

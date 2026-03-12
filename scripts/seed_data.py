"""
Seed script to load sample data into the pipeline.
Run with: python -m scripts.seed_data
"""

import asyncio
import os
import sys

import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8001/api/v1")

SAMPLE_FILES = [
    ("tests/fixtures/sample_paper.txt", "text/plain"),
    ("tests/fixtures/sample_data.csv", "text/csv"),
]


async def main():
    async with httpx.AsyncClient(timeout=60.0) as client:
        # Check health
        print("Checking API health...")
        try:
            resp = await client.get(f"{API_BASE}/health")
            health = resp.json()
            print(f"  Status: {health.get('status')}")
        except Exception as exc:
            print(f"  API not reachable: {exc}")
            sys.exit(1)

        # Ingest sample files
        for file_path, content_type in SAMPLE_FILES:
            if not os.path.exists(file_path):
                print(f"  Skipping {file_path} (not found)")
                continue

            file_name = os.path.basename(file_path)
            print(f"\nIngesting {file_name}...")

            with open(file_path, "rb") as f:
                resp = await client.post(
                    f"{API_BASE}/ingest",
                    files={"file": (file_name, f, content_type)},
                )

            if resp.status_code == 200:
                data = resp.json()
                print(f"  Document ID: {data['document_id']}")
                print(f"  Chunks: {data['chunks_created']}")
                print(f"  Entities: {data['entities_extracted']}")
            else:
                print(f"  Error ({resp.status_code}): {resp.text}")

        # List documents
        print("\n--- Ingested Documents ---")
        resp = await client.get(f"{API_BASE}/documents")
        if resp.status_code == 200:
            docs = resp.json()
            for doc in docs.get("documents", []):
                print(f"  [{doc['source_type']}] {doc['title']} ({doc['chunk_count']} chunks)")
            print(f"  Total: {docs.get('total', 0)} documents")

        # Run a sample query
        print("\n--- Sample Query ---")
        resp = await client.post(
            f"{API_BASE}/query",
            json={"question": "How does CRISPR-Cas9 work?", "max_sources": 3},
        )
        if resp.status_code == 200:
            result = resp.json()
            print(f"  Answer: {result['answer'][:200]}...")
            print(f"  Sources: {len(result['sources'])}")
            print(f"  Confidence: {result['confidence']}")
            print(f"  Latency: {result['latency']['total_ms']:.0f}ms")
        else:
            print(f"  Error ({resp.status_code}): {resp.text}")

    print("\nSeed complete.")


if __name__ == "__main__":
    asyncio.run(main())

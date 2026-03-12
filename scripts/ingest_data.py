"""
Batch-ingest all files from the data/ directory.

Usage:
    python -m scripts.ingest_data               # ingest data/papers/ + data/datasets/
    python -m scripts.ingest_data path/to/file  # ingest a single file

Supported formats: .pdf .txt .csv .json
"""

import asyncio
import os
import sys
from pathlib import Path

import httpx

API_BASE = os.getenv("API_BASE", "http://localhost:8001/api/v1")
TIMEOUT = 120.0
SUPPORTED = {".pdf", ".txt", ".csv", ".json"}

MIME = {
    ".pdf": "application/pdf",
    ".txt": "text/plain",
    ".csv": "text/csv",
    ".json": "application/json",
}

ROOT = Path(__file__).parent.parent


def collect_files(paths: list[Path]) -> list[Path]:
    files = []
    for p in paths:
        if p.is_file() and p.suffix.lower() in SUPPORTED:
            files.append(p)
        elif p.is_dir():
            for f in sorted(p.rglob("*")):
                if f.is_file() and f.suffix.lower() in SUPPORTED:
                    files.append(f)
    return files


async def ingest_file(client: httpx.AsyncClient, path: Path) -> bool:
    mime = MIME.get(path.suffix.lower(), "application/octet-stream")
    with open(path, "rb") as f:
        resp = await client.post(
            "/ingest",
            files={"file": (path.name, f, mime)},
            timeout=TIMEOUT,
        )

    if resp.status_code == 200:
        data = resp.json()
        print(f"  ✓  {path.name}  →  {data.get('chunk_count', '?')} chunks  (id: {data.get('document_id', '?')[:8]}...)")
        return True
    elif "already ingested" in resp.text:
        print(f"  -  {path.name}  (already ingested — skipped)")
        return True
    else:
        print(f"  ✗  {path.name}  FAILED: {resp.text[:120]}")
        return False


async def main(targets: list[str]) -> None:
    print(f"\n  Scider RAG — Batch Ingest")
    print(f"  Target API: {API_BASE}\n")

    if targets:
        paths = [Path(t) for t in targets]
    else:
        paths = [ROOT / "data" / "papers", ROOT / "data" / "datasets"]

    files = collect_files(paths)

    if not files:
        print("  No files found. Put .pdf / .txt / .csv / .json files in:")
        print("    data/papers/    ← research papers")
        print("    data/datasets/  ← structured data (CSV, JSON)")
        print("\n  Or pass a path directly:")
        print("    python -m scripts.ingest_data path/to/paper.pdf")
        return

    print(f"  Found {len(files)} file(s) to ingest:\n")

    async with httpx.AsyncClient(base_url=API_BASE) as client:
        # Health check
        try:
            h = await client.get("/health", timeout=5)
            h.raise_for_status()
        except Exception as e:
            print(f"  ✗  API not reachable at {API_BASE}: {e}")
            print("     Run: docker compose up -d")
            sys.exit(1)

        ok = 0
        for f in files:
            result = await ingest_file(client, f)
            if result:
                ok += 1

    print(f"\n  {ok}/{len(files)} files ingested successfully.")
    if ok > 0:
        print("  Redis cache version incremented — old cached answers are now stale.")
    print()


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1:]))

#!/usr/bin/env bash
set -euo pipefail

echo "=== Scider RAG Pipeline ==="

# Check if .env exists
if [ ! -f .env ]; then
    echo "No .env file found. Copying from .env.example..."
    cp .env.example .env
    echo "Please edit .env and set your OPENAI_API_KEY, then re-run this script."
    exit 1
fi

# Check if OPENAI_API_KEY is set
if grep -q "sk-your-openai-api-key-here" .env; then
    echo "ERROR: Please set your OPENAI_API_KEY in .env before running."
    exit 1
fi

echo "Starting services with Docker Compose..."
docker compose up --build -d

echo ""
echo "Waiting for services to be healthy..."
docker compose ps

echo ""
echo "Initializing database schema..."
docker compose exec app python -m scripts.init_db

echo ""
echo "=== Services Running ==="
echo "  API:     http://localhost:8000"
echo "  Docs:    http://localhost:8000/docs"
echo "  Health:  http://localhost:8000/api/v1/health"
echo ""
echo "To seed sample data:  docker compose exec app python -m scripts.seed_data"
echo "To view logs:         docker compose logs -f app"
echo "To stop:              docker compose down"

#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."
export STORAGE_MODE="${STORAGE_MODE:-local}"
export QUEUE_MODE="${QUEUE_MODE:-memory}"
exec uvicorn app.main:app --reload --host 0.0.0.0 --port 8000

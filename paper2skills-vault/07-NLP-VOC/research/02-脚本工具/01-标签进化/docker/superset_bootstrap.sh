#!/usr/bin/env bash
# Phase 7 D2 — Superset bootstrap (run on container start)
# - Ensures psycopg2-binary installed for postgres connection
# - Then chains to original Superset entrypoint (run-server.sh)
#
# Image apache/superset:4.1.1 ships without psycopg2; we need it to talk to voc_bi.
# Re-installing on every start is idempotent and only takes ~3s after first cache.

set -euo pipefail

if ! python3 -c "import psycopg2" >/dev/null 2>&1; then
  echo "[bootstrap] installing psycopg2-binary..."
  pip install --quiet --no-warn-script-location psycopg2-binary
fi

echo "[bootstrap] handing off to /usr/bin/run-server.sh"
exec /usr/bin/run-server.sh

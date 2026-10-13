#!/bin/sh
# Coral Verifier entrypoint. Options come from /data/options.json (read by the service itself).
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
cd /app || exit 1
echo "coral_verifier: starting ($(date -u +%FT%TZ))"
exec python3 /app/server.py

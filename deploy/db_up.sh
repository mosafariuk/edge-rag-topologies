#!/usr/bin/env bash
# Bring up the pgvector container with TLS on this host. Usage: PGPASSWORD=... ./db_up.sh
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p certs
if [ ! -f certs/server.key ]; then
  openssl req -x509 -newkey rsa:2048 -nodes -days 90 -subj "/CN=edge-rag-pg" \
    -keyout certs/server.key -out certs/server.crt >/dev/null 2>&1
  sudo chown 999:999 certs/server.key certs/server.crt   # postgres uid inside the image
  sudo chmod 600 certs/server.key
fi
export PGPASSWORD
docker compose -f docker-compose.db.yml up -d
for i in $(seq 1 40); do
  docker exec edge_rag_pg pg_isready -U edgerag -d edgerag >/dev/null 2>&1 && break; sleep 2
done
docker exec edge_rag_pg psql -U edgerag -d edgerag -c "CREATE EXTENSION IF NOT EXISTS vector" -c "SHOW ssl" -c "SELECT extversion FROM pg_extension WHERE extname='vector'" -c "SELECT version()"

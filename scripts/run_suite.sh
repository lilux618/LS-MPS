#!/usr/bin/env bash
set -euo pipefail
for c in bulk_uniform wall_film rain_injection narrow_gap; do
  echo "=== $c ==="
  ./bin/lsmps-bench "config/$c.cfg" results
  echo
done

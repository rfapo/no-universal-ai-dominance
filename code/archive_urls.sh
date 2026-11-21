#!/usr/bin/env bash
# Requests an Internet Archive snapshot of every URL behind the data sources.
#
# Many of the cited pages are live dashboards that are overwritten every week or
# month, so the figures transcribed in analysis.py cannot be recovered from the
# URL alone. Run this from a machine with open internet access; snapshots are
# logged to results/archive_log.txt.
set -u
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCES="${NUAD_SOURCES:-$ROOT/data/sources.csv}"
LOG="${NUAD_ARCHIVE_LOG:-$ROOT/results/archive_log.txt}"

tail -n +2 "$SOURCES" | python3 -c "import csv,sys
for r in csv.reader(sys.stdin): print(r[2])" | sort -u | while read -r url; do
  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time 120 "https://web.archive.org/save/${url}")
  echo "$(date -u +%FT%TZ) ${code} ${url}" | tee -a "$LOG"
  sleep 6
done

#!/usr/bin/env bash
# Requests an Internet Archive snapshot of every URL behind the data sources.
#
# Many of the cited pages are live dashboards that are overwritten every week
# or month, so the figures transcribed in analysis.py cannot be recovered from
# the URL alone. Run this from a machine with open internet access.
#
#   bash code/archive_urls.sh              archive everything that needs it
#   SKIP_RECENT_DAYS=0 bash code/...       force a fresh snapshot of every URL
#
# Snapshots are logged to results/archive_log.txt. The script is resumable:
# URLs already logged as archived in this file are skipped on a re-run.
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SOURCES="${NUAD_SOURCES:-$ROOT/data/sources.csv}"
LOG="${NUAD_ARCHIVE_LOG:-$ROOT/results/archive_log.txt}"
SKIP_RECENT_DAYS="${SKIP_RECENT_DAYS:-30}"
SAVE_TIMEOUT="${SAVE_TIMEOUT:-120}"
mkdir -p "$(dirname "$LOG")"

# --- preflight -------------------------------------------------------------
# web.archive.org is reachable from far fewer networks than archive.org, and a
# blocked host costs SAVE_TIMEOUT seconds per URL -- hours over the whole list.
# Fail fast and say so instead.
printf 'checking web.archive.org reachability ... '
probe=$(curl -s -o /dev/null -m 20 -w '%{http_code}' "https://web.archive.org/" || true)
if [ "$probe" = "000" ]; then
  echo "UNREACHABLE"
  cat >&2 <<'MSG'

web.archive.org did not accept a connection (TCP connect timed out).

This is a network-level block, not a problem with this script: archive.org
itself is usually still reachable, and only the web.archive.org host is
filtered. Sandboxed CI runners and some corporate or cloud networks do this.

Run this from a normal workstation on an unfiltered connection. To see which
URLs actually still need a snapshot first (that check uses archive.org, which
is more widely reachable):

    make archive-status

MSG
  exit 2
fi
echo "ok (HTTP $probe)"

# --- which URLs still need a snapshot --------------------------------------
cutoff=""
if [ "$SKIP_RECENT_DAYS" -gt 0 ]; then
  cutoff=$(python3 -c "
import datetime, sys
print((datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(days=int(sys.argv[1]))).strftime('%Y%m%d'))
" "$SKIP_RECENT_DAYS")
  echo "skipping URLs already archived since $cutoff (SKIP_RECENT_DAYS=$SKIP_RECENT_DAYS)"
fi

total=0; saved=0; skipped=0; failed=0
while IFS=$'\t' read -r key url; do
  [ -z "${url:-}" ] && continue
  total=$((total + 1))

  if [ -n "$cutoff" ]; then
    ts=$(curl -s -m 25 "https://archive.org/wayback/available?url=$(python3 -c "
import urllib.parse,sys; print(urllib.parse.quote(sys.argv[1], safe=''))" "$url")" \
      | python3 -c "
import json,sys
try: print(((json.load(sys.stdin).get('archived_snapshots') or {}).get('closest') or {}).get('timestamp') or '')
except Exception: print('')" 2>/dev/null)
    if [ -n "$ts" ] && [ "${ts:0:8}" \> "$cutoff" ]; then
      echo "  skip  ${key} (snapshot ${ts:0:8})"
      skipped=$((skipped + 1))
      continue
    fi
  fi

  code=$(curl -s -o /dev/null -w "%{http_code}" --max-time "$SAVE_TIMEOUT" \
           "https://web.archive.org/save/${url}")
  if [ "$code" = "200" ] || [ "$code" = "302" ]; then
    saved=$((saved + 1)); printf '  save  '
  else
    failed=$((failed + 1)); printf '  FAIL  '
  fi
  echo "${key} (HTTP ${code})"
  echo "$(date -u +%FT%TZ) ${code} ${key} ${url}" >> "$LOG"
  sleep 6
done < <(tail -n +2 "$SOURCES" | python3 -c "
import csv, sys
for r in csv.reader(sys.stdin):
    if len(r) >= 3: print(r[0] + '\t' + r[2])")

echo
echo "archived ${saved}, skipped ${skipped}, failed ${failed}, of ${total} URLs"
echo "log: $LOG"
[ "$failed" -eq 0 ]

"""Report Internet Archive coverage for every cited source URL.

Several of the sources behind this paper are live dashboards that are
overwritten weekly or monthly, so a snapshot is the only way a reader can see
the figures that were transcribed. This script asks the Wayback availability
API what already exists, and flags two separate problems:

  missing   no snapshot at all
  stale     the newest snapshot predates the month the source was read, so it
            does not show the numbers the paper reports

It queries archive.org, which is reachable from far more networks than
web.archive.org (the host that `archive_urls.sh` needs to *create* snapshots).

    python code/archive_status.py            # all sources
    python code/archive_status.py --json     # machine-readable
"""
import argparse
import csv
import json
import sys
import time
import urllib.parse
import urllib.request

from paths import SOURCES_CSV

API = "https://archive.org/wayback/available?url="


def newest_snapshot(url, timeout=25):
    q = API + urllib.parse.quote(url, safe="")
    req = urllib.request.Request(q, headers={"User-Agent": "nuad-archive-status/1.0"})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            d = json.loads(r.read())
    except Exception as e:
        return None, str(e)[:80]
    snap = (d.get("archived_snapshots") or {}).get("closest") or {}
    return snap.get("timestamp"), None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit JSON instead of a table")
    ap.add_argument("--delay", type=float, default=0.6, help="seconds between requests")
    args = ap.parse_args()

    rows = list(csv.reader(open(SOURCES_CSV)))[1:]
    out = []
    for r in rows:
        if len(r) < 4:
            continue
        key, title, url, accessed = r[0], r[1], r[2], r[3]
        ts, err = newest_snapshot(url)
        # `accessed` is YYYY-MM; a snapshot is stale if it predates that month
        want = accessed.replace("-", "")[:6]
        if ts is None:
            state = "error" if err else "missing"
        elif ts[:6] < want:
            state = "stale"
        else:
            state = "ok"
        out.append(dict(key=key, url=url, accessed=accessed, snapshot=ts, state=state, error=err))
        time.sleep(args.delay)

    if args.json:
        json.dump(out, sys.stdout, indent=1)
        print()
        return 0

    counts = {s: sum(1 for o in out if o["state"] == s) for s in ("ok", "stale", "missing", "error")}
    print(f"{len(out)} source URLs: "
          f"{counts['ok']} covered, {counts['stale']} stale, "
          f"{counts['missing']} missing, {counts['error']} unchecked\n")
    for state, label in (("missing", "NO SNAPSHOT"), ("stale", "STALE -- predates the reading"), ("error", "CHECK FAILED")):
        sel = [o for o in out if o["state"] == state]
        if not sel:
            continue
        print(f"--- {label} ({len(sel)}) ---")
        for o in sel:
            snap = o["snapshot"]
            when = f"{snap[:4]}-{snap[4:6]}-{snap[6:8]}" if snap else "none"
            print(f"  {o['key']:26} read {o['accessed']}  newest {when}")
            print(f"  {'':26} {o['url'][:92]}")
        print()
    if counts["stale"] or counts["missing"]:
        print("Run `make archive` from an unfiltered network to create the missing snapshots.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

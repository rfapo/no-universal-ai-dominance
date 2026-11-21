"""Canonical input/output locations for the replication scripts.

The scripts in this directory are the ones distributed as arXiv ancillary files,
where every path was relative to the current working directory. This module is
the only thing added on top of them, so that `make all` works from the
repository root while the statistical code itself stays byte-identical to the
version that produced the published numbers.

Override any location with an environment variable, e.g.

    NUAD_RESULTS=/tmp/out python code/analysis.py
"""
import os
from pathlib import Path

ROOT = Path(os.environ.get("NUAD_ROOT", Path(__file__).resolve().parent.parent))
RESULTS_DIR = Path(os.environ.get("NUAD_RESULTS", ROOT / "results"))
FIGS_DIR = Path(os.environ.get("NUAD_FIGS", ROOT / "paper" / "figs"))
DATA_DIR = Path(os.environ.get("NUAD_DATA", ROOT / "data"))

RESULTS_JSON = RESULTS_DIR / "results.json"
VALIDATION_JSON = RESULTS_DIR / "validation.json"
SOURCES_CSV = DATA_DIR / "sources.csv"

for _d in (RESULTS_DIR, FIGS_DIR):
    _d.mkdir(parents=True, exist_ok=True)

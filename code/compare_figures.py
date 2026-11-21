"""Compare two directories of rendered figures.

Used by `make verify`. What is reproducible here needs stating precisely,
because the two artefacts have different guarantees.

The *statistics* are platform-independent: every simulation is seeded and
NumPy's PCG64 stream is stable across releases, so `results/results.json` and
`results/validation.json` must match byte for byte anywhere, and `make verify`
fails if they do not.

A *rendered figure* carries no such guarantee. Its bytes depend on the
Matplotlib, Pillow and FreeType versions and on which fonts the host has
installed, and a different font stack moves glyph outlines by fractions of a
pixel across every label in the figure. Byte equality is therefore the wrong
invariant for a cross-platform check, and so is any pixel threshold loose
enough to absorb re-rasterised text, because by then it is also loose enough to
absorb a genuinely changed bar.

So this script separates the two questions:

  default      every figure was produced, and has the dimensions of the
               reference. Pixel differences are measured and reported, but do
               not fail the check: on a different font stack they are expected.
  --strict     every figure matches byte for byte. This is the right check
               between two runs that share a font stack -- the same machine,
               or a container pinned to the same Matplotlib and FreeType.

Usage:
    python code/compare_figures.py REFERENCE_DIR CANDIDATE_DIR [--strict]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

# A pixel counts as visibly different when any channel moves by more than this
# fraction of full scale. Used for reporting only.
VISIBLE = 0.1


def load(path):
    with Image.open(path) as im:
        return np.asarray(im.convert("RGB"), dtype=np.float64) / 255.0


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("reference", type=Path, help="directory of committed figures")
    ap.add_argument("candidate", type=Path, help="directory of freshly rendered figures")
    ap.add_argument(
        "--strict",
        action="store_true",
        help="require byte-for-byte equality (same font stack only)",
    )
    args = ap.parse_args()

    refs = sorted(args.reference.glob("*.png"))
    if not refs:
        print(f"no reference figures in {args.reference}", file=sys.stderr)
        return 1

    failures, identical, differing = [], [], []
    worst_mad = worst_visible = 0.0

    for ref in refs:
        cand = args.candidate / ref.name
        if not cand.exists():
            failures.append(f"{ref.name}: not produced")
            continue
        if cand.stat().st_size < 1024:
            failures.append(f"{ref.name}: produced but only {cand.stat().st_size} bytes")
            continue

        if ref.read_bytes() == cand.read_bytes():
            identical.append(ref.name)
            continue

        a, b = load(ref), load(cand)
        if a.shape != b.shape:
            failures.append(
                f"{ref.name}: {b.shape[1]}x{b.shape[0]} against a reference "
                f"of {a.shape[1]}x{a.shape[0]}"
            )
            continue

        diff = np.abs(a - b)
        mad = float(diff.mean())
        visible = float((diff.max(axis=2) > VISIBLE).mean())
        worst_mad = max(worst_mad, mad)
        worst_visible = max(worst_visible, visible)
        differing.append((ref.name, mad, visible))
        if args.strict:
            failures.append(
                f"{ref.name}: bytes differ (mean abs. difference {mad:.5f}, "
                f"{visible:.2%} of pixels visibly different)"
            )

    for line in failures:
        print(f"  {line}", file=sys.stderr)
    if failures:
        print(f"{len(failures)} of {len(refs)} figures failed the check", file=sys.stderr)
        return 1

    if not differing:
        print(f"  OK all {len(refs)} figures (byte for byte)")
        return 0

    print(
        f"  OK all {len(refs)} figures produced at the reference dimensions "
        f"({len(identical)} byte-identical, {len(differing)} re-rasterised)"
    )
    print(
        f"     worst mean abs. difference {worst_mad:.5f}, "
        f"worst visibly-different pixels {worst_visible:.2%}"
    )
    for name, mad, visible in sorted(differing, key=lambda r: -r[1])[:5]:
        print(f"     {name}: {mad:.5f} mean, {visible:.2%} visible")
    print(
        "     pixel differences are expected on a different Matplotlib, Pillow,\n"
        "     FreeType or font stack; run `make verify-strict` on the machine\n"
        "     that produced the committed figures to require identical bytes."
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

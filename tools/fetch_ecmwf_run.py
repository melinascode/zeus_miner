"""Fetch one ECMWF open-data run (AIFS single + AIFS-ENS) for Zeus serving.

Downloads, for a given init date/time:
  - aifs-single oper fc : 2t, 100u, 100v, ssrd   (steps 0..360 by 6)
  - aifs-ens   enfo em  : 2t                     (ensemble mean, steps 6..360 by 6)
  - aifs-ens   enfo cf+pf: 100u, 100v            (51 members, steps 0..360 by 6)

Usage:
  python tools/fetch_ecmwf_run.py --date 2026-08-18 --time 12 [--skip-ens]
"""

import argparse
import os
import time

from ecmwf.opendata import Client

STEPS = list(range(0, 361, 6))
OUT_ROOT = "/Zeus/data/evaluation/ecmwf_live"


def fetch(client, out_dir, name, **kwargs):
    """Download to a temp file and rename, so partial files are never kept."""
    target = os.path.join(out_dir, name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        print(f"[skip] {name} already present ({os.path.getsize(target)/1e6:.0f} MB)")
        return
    tmp = target + ".tmp"
    t0 = time.time()
    client.retrieve(target=tmp, **kwargs)
    os.replace(tmp, target)
    print(f"[done] {name}: {os.path.getsize(target)/1e6:.0f} MB in {time.time()-t0:.0f}s", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", type=int, required=True)
    ap.add_argument("--skip-ens", action="store_true")
    args = ap.parse_args()

    stamp = f"{args.date.replace('-', '')}T{args.time:02d}0000Z"
    out_dir = os.path.join(OUT_ROOT, stamp)
    os.makedirs(out_dir, exist_ok=True)

    single = Client(source="azure", model="aifs-single")
    ens = Client(source="azure", model="aifs-ens")

    fetch(single, out_dir, "aifs_single_all.grib2",
          date=args.date, time=args.time, stream="oper", type="fc",
          param=["2t", "100u", "100v", "ssrd"], step=STEPS)

    # em has no step-0 field and only carries 2t among our params
    fetch(ens, out_dir, "aifs_ens_em_2t.grib2",
          date=args.date, time=args.time, stream="enfo", type="em",
          param="2t", step=STEPS[1:])

    if not args.skip_ens:
        fetch(ens, out_dir, "aifs_ens_cf_winds.grib2",
              date=args.date, time=args.time, stream="enfo", type="cf",
              param=["100u", "100v"], step=STEPS)
        # Chunk perturbed members so each request stays inside the Azure SAS
        # token lifetime (~45 min); the full 50-member file exceeds it.
        for lo in range(1, 51, 10):
            members = list(range(lo, min(lo + 10, 51)))
            fetch(ens, out_dir, f"aifs_ens_pf_winds_{lo:02d}_{members[-1]:02d}.grib2",
                  date=args.date, time=args.time, stream="enfo", type="pf",
                  param=["100u", "100v"], number=members, step=STEPS)

    print(f"Bundle complete: {out_dir}")


if __name__ == "__main__":
    main()

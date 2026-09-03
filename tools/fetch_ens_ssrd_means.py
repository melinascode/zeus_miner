"""Fetch ensemble-member SSRD from the Azure archive and build ensemble means.

Both ensembles carry per-member accumulated ssrd (J/m2 since init) but no
ensemble-mean product for it:
  aifs-ens enfo cf+pf  51 members
  ifs      enfo pf     50 members

Output per model: float32 .npy of shape (61, 721, 1440) on the Zeus grid
(lat -90..90), the member-mean ACCUMULATED ssrd at steps 0..360 by 6, saved
to /Zeus/data/evaluation/{aifs_ens_ssrd_mean,ifs_ens_ssrd_mean}/<cycle>.npy.

Usage:
  python tools/fetch_ens_ssrd_means.py --cycle 20260801T000000Z --model aifs-ens
  python tools/fetch_ens_ssrd_means.py --cycle 20260801T000000Z --model ifs
"""

from __future__ import annotations

import argparse
import concurrent.futures
import os
import time
from datetime import datetime, timezone
from pathlib import Path

import eccodes
import numpy as np

STEPS = list(range(0, 361, 6))
SHAPE = (721, 1440)
MEMBER_CHUNKS = [list(range(lo, min(lo + 10, 51))) for lo in range(1, 51, 10)]
OUT_ROOTS = {
    "aifs-ens": Path("/Zeus/data/evaluation/aifs_ens_ssrd"),
    "ifs": Path("/Zeus/data/evaluation/ifs_ens_ssrd"),
}
MEAN_ROOTS = {
    "aifs-ens": Path("/Zeus/data/evaluation/aifs_ens_ssrd_mean"),
    "ifs": Path("/Zeus/data/evaluation/ifs_ens_ssrd_mean"),
}


def fetch_one(model: str, run_dir: Path, name: str, request: dict) -> str:
    import random

    from ecmwf.opendata import Client

    target = run_dir / name
    if target.is_file() and target.stat().st_size > 0:
        return f"[skip] {name}"
    tmp = str(target) + ".tmp"
    started = time.time()
    last_exc: Exception | None = None
    for attempt in range(5):
        if attempt:
            time.sleep(min(300, 15 * 2**attempt) + random.uniform(0, 10))
        try:
            client = Client(source="azure", model=model)
            try:
                client.retrieve(target=tmp, **request)
            except ValueError:
                if request.get("step", [None])[0] == 0:
                    retry = dict(request)
                    retry["step"] = [s for s in request["step"] if s != 0]
                    client.retrieve(target=tmp, **retry)
                else:
                    raise
            break
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            print(f"[retry {attempt + 1}] {name}: {exc}", flush=True)
    else:
        raise RuntimeError(f"{name} failed after retries") from last_exc
    os.replace(tmp, target)
    return f"[done] {name}: {target.stat().st_size/1e6:.0f} MB in {time.time()-started:.0f}s"


def accumulate_file(path: Path, sums: np.ndarray, counts: np.ndarray) -> None:
    step_index = {s: i for i, s in enumerate(STEPS)}
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            short = eccodes.codes_get(gid, "shortName")
            step = int(eccodes.codes_get(gid, "endStep"))
            if short == "ssrd" and step in step_index:
                values = eccodes.codes_get_values(gid).reshape(SHAPE)
                sums[step_index[step]] += values[::-1, :]
                counts[step_index[step]] += 1
            eccodes.codes_release(gid)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--model", required=True, choices=("aifs-ens", "ifs"))
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()

    cycle_time = datetime.strptime(args.cycle, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    date = cycle_time.strftime("%Y-%m-%d")
    hour = cycle_time.hour
    run_dir = OUT_ROOTS[args.model] / args.cycle
    run_dir.mkdir(parents=True, exist_ok=True)
    mean_root = MEAN_ROOTS[args.model]
    mean_root.mkdir(parents=True, exist_ok=True)
    mean_path = mean_root / f"{args.cycle}.npy"
    if mean_path.is_file():
        print(f"mean already exists: {mean_path}")
        return 0

    jobs = []
    if args.model == "aifs-ens":
        jobs.append(
            ("ssrd_cf.grib2",
             dict(date=date, time=hour, stream="enfo", type="cf",
                  param="ssrd", step=STEPS))
        )
    for members in MEMBER_CHUNKS:
        jobs.append(
            (f"ssrd_pf_{members[0]:02d}_{members[-1]:02d}.grib2",
             dict(date=date, time=hour, stream="enfo", type="pf",
                  param="ssrd", number=members, step=STEPS))
        )

    print(f"{args.model} {args.cycle}: {len(jobs)} files, {args.workers} workers",
          flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(fetch_one, args.model, run_dir, name, request): name
            for name, request in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(futures[future])
                print(f"[fail] {futures[future]}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"failed: {failures}")

    print("building member mean ...", flush=True)
    sums = np.zeros((len(STEPS), *SHAPE), np.float64)
    counts = np.zeros(len(STEPS), np.int64)
    for grib in sorted(run_dir.glob("ssrd_*.grib2")):
        accumulate_file(grib, sums, counts)
        print(f"  {grib.name} decoded", flush=True)
    mean = np.zeros((len(STEPS), *SHAPE), np.float32)
    for i in range(len(STEPS)):
        if counts[i] > 0:
            mean[i] = (sums[i] / counts[i]).astype(np.float32)
    # Accumulated-since-init: step 0 is exactly zero even if not published.
    np.save(mean_path, np.clip(mean, 0.0, None))
    print(
        f"wrote {mean_path} ({mean_path.stat().st_size/1e9:.2f} GB), "
        f"members min={int(counts[1:].min())} max={int(counts.max())}",
        flush=True,
    )
    for grib in run_dir.glob("ssrd_*.grib2"):
        grib.unlink()
    print("removed GRIBs", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

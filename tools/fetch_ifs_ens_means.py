"""Fetch IFS-ENS (enfo) members from the Azure archive and build ensemble means.

The Azure ECMWF open-data mirror has no cf/em entries for 2t/100u/100v, so the
mean is over the 50 perturbed members. Output matches the aifs_ens_mean layout:
one float16 .npy of shape (61, 3, 721, 1440) on the Zeus grid (lat -90..90),
channels (t2m, u100, v100), steps 0..360 by 6.

Also fetches the deterministic IFS oper run (same params) for reference.

Usage:
  python tools/fetch_ifs_ens_means.py --cycle 20260727T000000Z [--workers 3]
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

OUT_ROOT = Path("/Zeus/data/evaluation/ifs_ens")
MEAN_ROOT = Path("/Zeus/data/evaluation/ifs_ens_mean")
PARAMS = ["2t", "100u", "100v"]
SHORT_TO_CHANNEL = {"2t": 0, "100u": 1, "100v": 2}
STEPS = list(range(0, 361, 6))
SHAPE = (721, 1440)
MEMBER_CHUNKS = [list(range(lo, min(lo + 10, 51))) for lo in range(1, 51, 10)]


def to_zeus_grid(field: np.ndarray) -> np.ndarray:
    return np.ascontiguousarray(field[::-1, :])


def fetch_one(run_dir: Path, name: str, request: dict) -> str:
    """Download one GRIB with its own client (own SAS token). Resumable.

    Retries with backoff: the Planetary Computer SAS token endpoint returns
    429 when several clients request tokens at the same instant.
    """
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
            client = Client(source="azure", model="ifs")
            try:
                client.retrieve(target=tmp, **request)
            except Exception:
                # Some cycles miss step 0 in enfo; retry without it.
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
    size = target.stat().st_size / 1e6
    return f"[done] {name}: {size:.0f} MB in {time.time() - started:.0f}s"


def read_member_sums(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Sum fields per (channel, step) over all members in one chunk file."""
    sums = np.zeros((len(STEPS), 3, *SHAPE), np.float64)
    counts = np.zeros((len(STEPS), 3), np.int32)
    step_index = {s: i for i, s in enumerate(STEPS)}
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            short = eccodes.codes_get(gid, "shortName")
            step = int(eccodes.codes_get(gid, "endStep"))
            if short in SHORT_TO_CHANNEL and step in step_index:
                channel = SHORT_TO_CHANNEL[short]
                values = eccodes.codes_get_values(gid).reshape(SHAPE)
                sums[step_index[step], channel] += to_zeus_grid(values)
                counts[step_index[step], channel] += 1
            eccodes.codes_release(gid)
    return sums, counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", required=True, help="e.g. 20260727T000000Z")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--keep-gribs", action="store_true")
    args = parser.parse_args()

    cycle_time = datetime.strptime(args.cycle, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    date = cycle_time.strftime("%Y-%m-%d")
    hour = cycle_time.hour
    run_dir = OUT_ROOT / args.cycle
    run_dir.mkdir(parents=True, exist_ok=True)
    MEAN_ROOT.mkdir(parents=True, exist_ok=True)
    mean_path = MEAN_ROOT / f"{args.cycle}.npy"
    if mean_path.is_file():
        print(f"mean already exists: {mean_path}")
        return 0

    jobs = [
        (
            "ifs_oper_all.grib2",
            dict(date=date, time=hour, stream="oper", type="fc",
                 param=PARAMS, step=STEPS),
        )
    ] + [
        (
            f"ifs_enfo_pf_{members[0]:02d}_{members[-1]:02d}.grib2",
            dict(date=date, time=hour, stream="enfo", type="pf",
                 param=PARAMS, number=members, step=STEPS),
        )
        for members in MEMBER_CHUNKS
    ]

    print(f"{args.cycle}: {len(jobs)} files, {args.workers} workers", flush=True)
    failures = []
    with concurrent.futures.ThreadPoolExecutor(args.workers) as pool:
        futures = {
            pool.submit(fetch_one, run_dir, name, request): name
            for name, request in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            try:
                print(future.result(), flush=True)
            except Exception as exc:  # noqa: BLE001
                failures.append(futures[future])
                print(f"[fail] {futures[future]}: {exc}", flush=True)
    if failures:
        raise SystemExit(f"{len(failures)} downloads failed: {failures}")

    print("building 50-member mean ...", flush=True)
    total = np.zeros((len(STEPS), 3, *SHAPE), np.float64)
    counts = np.zeros((len(STEPS), 3), np.int32)
    for members in MEMBER_CHUNKS:
        chunk = run_dir / f"ifs_enfo_pf_{members[0]:02d}_{members[-1]:02d}.grib2"
        sums, chunk_counts = read_member_sums(chunk)
        total += sums
        counts += chunk_counts
        print(f"  {chunk.name} decoded", flush=True)

    mean = np.zeros((len(STEPS), 3, *SHAPE), np.float32)
    member_count = int(counts[1:].min()) if counts[1:].min() > 0 else 0
    for i, step in enumerate(STEPS):
        for channel in range(3):
            n = counts[i, channel]
            if n > 0:
                mean[i, channel] = (total[i, channel] / n).astype(np.float32)
    # Fill a missing step 0 from the deterministic run (analysis-like state).
    if counts[0].min() == 0:
        print("  enfo has no step 0; filling from oper", flush=True)
        oper_sums, oper_counts = read_member_sums(run_dir / "ifs_oper_all.grib2")
        if oper_counts[0].min() == 0:
            raise RuntimeError("oper run also missing step 0")
        mean[0] = (oper_sums[0] / np.maximum(oper_counts[0], 1)[:, None, None]).astype(
            np.float32
        )

    np.save(mean_path, mean.astype(np.float16))
    print(
        f"wrote {mean_path} ({mean_path.stat().st_size/1e9:.2f} GB), "
        f"members per step >= {member_count}",
        flush=True,
    )
    if not args.keep_gribs:
        for grib in run_dir.glob("ifs_enfo_pf_*.grib2"):
            grib.unlink()
        print("removed pf GRIBs (kept oper)", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

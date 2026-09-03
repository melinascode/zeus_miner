#!/usr/bin/env python3
"""Parallel daily AIFS-ENS mean fetcher, Europe crop only.

Downloads 00z AIFS-ENS cycles (2t/100u/100v, steps 0-360h, control + 50
perturbed members), accumulates the 51-member mean, and stores ONLY the
Europe crop (208 x 368, 28.00-79.75N / 40.00W-51.75E) as fp16 of shape
(61, 3, 208, 368) — ~28 MB per cycle. The global GRIBs are decoded
streaming and never kept.

Self-contained on purpose: needs only numpy + eccodes + ecmwf-opendata,
so it can run on a bare RunPod CPU pod without the Zeus repo.

Parallelism: a worker pool over days. Each worker owns its own Azure SAS
token pool (tokens die after ~45 min; we refresh at 12 min) and downloads
member chunks of 5 (~0.9 GB each). Azure throttles per connection at
~2-4 MB/s, so N workers gives ~N-fold speedup until the token API rate
limit (HTTP 429) pushes back — 6 workers is a safe default.

Resumable: finished crops are skipped; rerun the same command after any
stop. A day that fails is logged and retried on the next run.

Usage (RunPod CPU pod, ~2-3 days wall time with 6 workers):
  python fetch_aifs_ens_crops_parallel.py \
      --start 2025-09-03 --end 2026-08-26 \
      --out-root /workspace/europe_crops/ens --workers 6
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from multiprocessing import Pool, current_process

import eccodes
import numpy as np
from ecmwf.opendata import Client

PARAMS = ("2t", "100u", "100v")
PARAM_INDEX = {name: i for i, name in enumerate(PARAMS)}
STEPS = list(range(0, 361, 6))
N_STEPS = len(STEPS)
STEP_INDEX = {step: i for i, step in enumerate(STEPS)}
N_MEMBERS = 51  # control + 50 perturbed
MEMBER_CHUNK = 5
FULL_HEIGHT, FULL_WIDTH = 721, 1440
# Europe crop on the Zeus grid (lat -90..90 ascending, lon -180..179.75):
# 28.00-79.75N, 40.00W-51.75E -> 208 x 368. Matches
# tools/preprocess_europe_crops.py exactly.
LAT_SLICE = slice(472, 680)
LON_SLICE = slice(560, 928)
CROP_HEIGHT = LAT_SLICE.stop - LAT_SLICE.start
CROP_WIDTH = LON_SLICE.stop - LON_SLICE.start
SAS_MAX_AGE_SEC = 12 * 60
MIN_CROP_BYTES = 25_000_000  # a full fp16 crop is ~28 MB


class AzureClientPool:
    """One per worker: reuse a SAS token only while young enough to finish."""

    def __init__(self, source: str, max_age_sec: int = SAS_MAX_AGE_SEC) -> None:
        self.source = source
        self.max_age_sec = max_age_sec
        self.client: Client | None = None
        self.born = 0.0

    def get(self, *, force: bool = False) -> Client:
        aged = self.client is not None and (time.time() - self.born) >= self.max_age_sec
        if force or self.client is None or aged:
            self.client = Client(source=self.source, model="aifs-ens")
            self.born = time.time()
        return self.client


def is_token_error(exc: Exception) -> bool:
    text = str(exc)
    return "AuthenticationFailed" in text or "403" in text


def count_grib_messages(path: str) -> int:
    n = 0
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle, headers_only=True)
            if gid is None:
                break
            n += 1
            eccodes.codes_release(gid)
    return n


def accumulate_grib_cropped(
    path: str, sums: np.ndarray, counts: np.ndarray
) -> None:
    """Decode each global message, flip latitude, keep only the Europe crop."""
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            try:
                short = eccodes.codes_get(gid, "shortName")
                step = int(eccodes.codes_get(gid, "endStep"))
                if short in PARAM_INDEX and step in STEP_INDEX:
                    values = eccodes.codes_get_values(gid).reshape(
                        FULL_HEIGHT, FULL_WIDTH
                    )
                    # GRIB rows run 90N -> 90S; Zeus grid is ascending.
                    crop = values[::-1][LAT_SLICE, LON_SLICE]
                    si, vi = STEP_INDEX[step], PARAM_INDEX[short]
                    sums[si, vi] += crop
                    counts[si, vi] += 1
            finally:
                eccodes.codes_release(gid)


def retrieve_chunk(
    pool: AzureClientPool,
    target: str,
    retries: int,
    *,
    n_members: int,
    **request,
) -> None:
    expected = n_members * len(PARAMS) * N_STEPS
    last = ""
    force_new = False
    for attempt in range(1, retries + 1):
        if os.path.exists(target):
            os.remove(target)
        try:
            client = pool.get(force=force_new)
            client.retrieve(target=target, **request)
            got = count_grib_messages(target)
            if got == expected:
                return
            last = f"truncated: {got}/{expected} messages"
            force_new = True
        except Exception as exc:  # noqa: BLE001 - classify and retry
            last = f"{type(exc).__name__}: {str(exc)[:160]}"
            force_new = is_token_error(exc)
        text = last
        if "429" in text:
            time.sleep(90 * attempt)
        elif force_new:
            time.sleep(3)
        else:
            time.sleep(min(30 * attempt, 120))
    raise RuntimeError(f"retrieve failed after {retries} tries: {last}")


# --- worker process state -------------------------------------------------

_POOL: AzureClientPool | None = None
_OUT_ROOT = ""
_RETRIES = 8


def init_worker(source: str, out_root: str, retries: int, stagger_sec: float) -> None:
    global _POOL, _OUT_ROOT, _RETRIES
    _POOL = AzureClientPool(source)
    _OUT_ROOT = out_root
    _RETRIES = retries
    # Stagger token minting so N workers do not hit the token API at once.
    identity = current_process()._identity
    rank = identity[0] if identity else 1
    time.sleep((rank - 1) * stagger_sec)


def fetch_day(stamp: str) -> str:
    target = os.path.join(_OUT_ROOT, f"{stamp}.npy")
    if os.path.exists(target) and os.path.getsize(target) >= MIN_CROP_BYTES:
        return f"SKIP {stamp}"
    day = datetime.strptime(stamp, "%Y%m%dT%H%M%SZ")
    t0 = time.time()
    sums = np.zeros((N_STEPS, len(PARAMS), CROP_HEIGHT, CROP_WIDTH), np.float32)
    counts = np.zeros((N_STEPS, len(PARAMS)), np.int32)
    tmp_grib = os.path.join(_OUT_ROOT, f".{stamp}.{os.getpid()}.chunk.grib2")
    common = dict(
        date=day.strftime("%Y-%m-%d"),
        time=0,
        stream="enfo",
        param=list(PARAMS),
        step=STEPS,
    )
    try:
        retrieve_chunk(_POOL, tmp_grib, _RETRIES, n_members=1, type="cf", **common)
        accumulate_grib_cropped(tmp_grib, sums, counts)
        for lo in range(1, 51, MEMBER_CHUNK):
            members = list(range(lo, min(lo + MEMBER_CHUNK, 51)))
            retrieve_chunk(
                _POOL,
                tmp_grib,
                _RETRIES,
                n_members=len(members),
                type="pf",
                number=members,
                **common,
            )
            accumulate_grib_cropped(tmp_grib, sums, counts)
    finally:
        if os.path.exists(tmp_grib):
            os.remove(tmp_grib)
    if not (counts == N_MEMBERS).all():
        bad = int((counts != N_MEMBERS).sum())
        return f"FAIL {stamp} incomplete: {bad} (step,var) cells != {N_MEMBERS}"
    mean = (sums / float(N_MEMBERS)).astype(np.float16)
    tmp_npy = target + f".{os.getpid()}.tmp.npy"
    np.save(tmp_npy, mean)
    os.replace(tmp_npy, target)
    return f"DONE {stamp} {os.path.getsize(target)/1e6:.0f}MB in {time.time()-t0:.0f}s"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-09-03")
    parser.add_argument("--end", default="2026-08-26")
    parser.add_argument("--every", type=int, default=1, help="days between cycles")
    parser.add_argument("--out-root", default="/workspace/europe_crops/ens")
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--source", default="azure")
    parser.add_argument("--retries", type=int, default=8)
    parser.add_argument("--stagger", type=float, default=20.0,
                        help="seconds between worker token-pool startups")
    parser.add_argument("--dry-run", action="store_true",
                        help="list pending days and exit")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    stamps = []
    cur = start
    while cur <= end:
        stamps.append(cur.strftime("%Y%m%dT000000Z"))
        cur += timedelta(days=args.every)
    os.makedirs(args.out_root, exist_ok=True)

    def done(stamp: str) -> bool:
        path = os.path.join(args.out_root, f"{stamp}.npy")
        return os.path.exists(path) and os.path.getsize(path) >= MIN_CROP_BYTES

    pending = [s for s in stamps if not done(s)]
    print(
        f"{len(stamps)} cycles {args.start}..{args.end}, "
        f"{len(stamps) - len(pending)} already done, {len(pending)} pending, "
        f"{args.workers} workers -> {args.out_root}",
        flush=True,
    )
    if args.dry_run:
        for s in pending[:5]:
            print(f"  pending {s}")
        if len(pending) > 5:
            print(f"  ... and {len(pending) - 5} more")
        return 0
    if not pending:
        print("Nothing to do.")
        return 0

    log_path = os.path.join(args.out_root, "fetch_parallel.log")
    ok = fail = 0
    t_all = time.time()
    with Pool(
        processes=args.workers,
        initializer=init_worker,
        initargs=(args.source, args.out_root, args.retries, args.stagger),
    ) as pool, open(log_path, "a", encoding="utf-8") as log:
        for i, line in enumerate(pool.imap_unordered(fetch_day, pending), start=1):
            if line.startswith("DONE") or line.startswith("SKIP"):
                ok += 1
            else:
                fail += 1
            elapsed_h = (time.time() - t_all) / 3600
            rate = i / elapsed_h if elapsed_h > 0 else 0.0
            remaining_h = (len(pending) - i) / rate if rate > 0 else float("inf")
            msg = (
                f"[{i}/{len(pending)} {ok}ok {fail}fail "
                f"{rate:.1f}/h eta {remaining_h:.1f}h] {line}"
            )
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    print(
        f"Finished: {ok} ok, {fail} failed in {(time.time()-t_all)/3600:.2f}h",
        flush=True,
    )
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

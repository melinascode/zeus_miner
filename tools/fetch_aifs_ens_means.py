"""Download AIFS-ENS cycles and keep only the 51-member ensemble mean.

For each 00z cycle: fetch the control run plus the 50 perturbed members
(2t/100u/100v, steps 0-360h) in member chunks that finish well inside one
Azure SAS token lifetime (~45 min). Stream-decode each chunk into a running
sum, delete the raw GRIB immediately, and store the mean as one fp16 array
of shape (61, 3, 721, 1440) on the Zeus grid (~380 MB per cycle).

Azure SAS tokens expire in ~45 minutes. A 10-member / 61-step retrieve is
~1.9 GB and takes ~13 minutes, so reusing one token across several of those
walks into AuthenticationFailed mid-file. This script therefore:
  - mints a new Client (new SAS) when the current one is older than 12 min
  - treats AuthenticationFailed / 403 as an immediate token refresh
  - downloads 5 members at a time (~0.9 GB, ~6 min)
  - counts GRIB messages before accumulating, so a truncated file is retried

Resumable: rerun the same command after a stop; finished cycles are skipped.

Usage:
  python tools/fetch_aifs_ens_means.py --start 2025-09-03 --end 2026-07-22 --every 7
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

import eccodes
import numpy as np
from ecmwf.opendata import Client

PARAMS = ("2t", "100u", "100v")
STEPS = list(range(0, 361, 6))
N_STEPS = len(STEPS)
SHAPE = (721, 1440)
N_MEMBERS = 51  # control + 50 perturbed
OUT_ROOT = "/Zeus/data/evaluation/aifs_ens_mean"
MEMBER_CHUNK = 5
SAS_MAX_AGE_SEC = 12 * 60
# 5 members × 3 vars × 61 steps is ~900 MB; a truncated file is far smaller.
MIN_BYTES_PER_MEMBER = 80_000_000


class AzureClientPool:
    """Reuse a SAS token only while it is still young enough to finish a chunk."""

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


def count_grib_messages(path: str) -> int:
    n = 0
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            n += 1
            eccodes.codes_release(gid)
    return n


def accumulate_grib(
    path: str,
    sums: np.ndarray,
    counts: np.ndarray,
) -> None:
    """Add every (param, step) field in the file to the running sums."""

    index = {name: i for i, name in enumerate(PARAMS)}
    with open(path, "rb") as handle:
        while True:
            gid = eccodes.codes_grib_new_from_file(handle)
            if gid is None:
                break
            short = eccodes.codes_get(gid, "shortName")
            step = int(eccodes.codes_get(gid, "endStep"))
            if short in index and step % 6 == 0 and step <= 360:
                s = step // 6
                v = index[short]
                sums[s, v] += eccodes.codes_get_values(gid).reshape(SHAPE)
                counts[s, v] += 1
            eccodes.codes_release(gid)


def is_token_error(exc: BaseException) -> bool:
    text = str(exc)
    return any(
        marker in text
        for marker in (
            "403",
            "AuthenticationFailed",
            "Signature not valid",
            "token",
        )
    )


def retrieve_chunk(
    pool: AzureClientPool,
    target: str,
    retries: int,
    *,
    n_members: int,
    **request,
) -> None:
    """Download one GRIB chunk; retry on SAS expiry or a truncated file."""

    expected_msgs = n_members * len(PARAMS) * N_STEPS
    min_bytes = n_members * MIN_BYTES_PER_MEMBER
    last: BaseException | None = None
    force = True  # always start a chunk with a fresh-enough token
    for attempt in range(1, retries + 1):
        try:
            if os.path.exists(target):
                os.remove(target)
            pool.get(force=force).retrieve(target=target, **request)
            size = os.path.getsize(target) if os.path.exists(target) else 0
            if size < min_bytes:
                last = RuntimeError(f"too small ({size} bytes, need {min_bytes})")
                force = True
                time.sleep(5)
                continue
            n_msgs = count_grib_messages(target)
            if n_msgs != expected_msgs:
                last = RuntimeError(
                    f"got {n_msgs} GRIB messages, expected {expected_msgs}"
                )
                force = True
                time.sleep(5)
                continue
            return
        except Exception as exc:  # noqa: BLE001 - retry on any transport error
            last = exc
            force = True
            text = str(exc)
            if "429" in text:
                time.sleep(90 * attempt)
            elif is_token_error(exc):
                time.sleep(3)
            else:
                time.sleep(min(30 * attempt, 120))
    raise RuntimeError(f"retrieve failed after {retries} tries: {last}")


def fetch_cycle_mean(
    pool: AzureClientPool,
    day: datetime,
    out_root: str,
    retries: int,
) -> str:
    stamp = day.strftime("%Y%m%dT000000Z")
    target = os.path.join(out_root, f"{stamp}.npy")
    if os.path.exists(target) and os.path.getsize(target) > 300_000_000:
        return f"SKIP {stamp}"
    t0 = time.time()
    sums = np.zeros((N_STEPS, len(PARAMS), *SHAPE), np.float32)
    counts = np.zeros((N_STEPS, len(PARAMS)), np.int32)
    tmp_grib = os.path.join(out_root, f".{stamp}.chunk.grib2")
    common = dict(
        date=day.strftime("%Y-%m-%d"),
        time=0,
        stream="enfo",
        param=list(PARAMS),
        step=STEPS,
    )
    try:
        retrieve_chunk(pool, tmp_grib, retries, n_members=1, type="cf", **common)
        accumulate_grib(tmp_grib, sums, counts)
        print(f"  {stamp} cf done", flush=True)
        for lo in range(1, 51, MEMBER_CHUNK):
            members = list(range(lo, min(lo + MEMBER_CHUNK, 51)))
            retrieve_chunk(
                pool,
                tmp_grib,
                retries,
                n_members=len(members),
                type="pf",
                number=members,
                **common,
            )
            accumulate_grib(tmp_grib, sums, counts)
            print(f"  {stamp} pf {members[0]:02d}-{members[-1]:02d} done", flush=True)
    finally:
        if os.path.exists(tmp_grib):
            os.remove(tmp_grib)
    if not (counts == N_MEMBERS).all():
        bad = int((counts != N_MEMBERS).sum())
        return f"FAIL {stamp} incomplete: {bad} (step,var) cells != {N_MEMBERS} members"
    mean = sums / float(N_MEMBERS)
    # ECMWF open data already starts at -180 longitude; flip latitude only.
    mean = mean[:, :, ::-1, :]
    tmp_npy = target + ".tmp.npy"
    np.save(tmp_npy, mean.astype(np.float16))
    os.replace(tmp_npy, target)
    meta = {
        "cycle": stamp,
        "members": N_MEMBERS,
        "params": PARAMS,
        "steps": STEPS,
        "grid": "zeus lat -90..90, lon -180..179.75",
    }
    with open(os.path.join(out_root, f"{stamp}.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f)
    return f"DONE {stamp} {os.path.getsize(target)/1e6:.0f}MB in {time.time()-t0:.0f}s"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-09-03")
    parser.add_argument("--end", default="2026-07-22")
    parser.add_argument("--every", type=int, default=7, help="days between cycles")
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--source", default="azure")
    parser.add_argument("--retries", type=int, default=6)
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=args.every)
    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(args.out_root, "fetch.log")
    pool = AzureClientPool(args.source)
    print(
        f"Fetching ensemble means for {len(days)} cycles "
        f"{args.start}..{args.end} every {args.every}d -> {args.out_root}",
        flush=True,
    )
    done = skip = fail = 0
    t_all = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        for i, day in enumerate(days, start=1):
            try:
                line = fetch_cycle_mean(pool, day, args.out_root, args.retries)
            except Exception as exc:  # noqa: BLE001 - keep the batch going
                line = f"FAIL {day.date()} {type(exc).__name__}: {str(exc)[:160]}"
            if line.startswith("DONE"):
                done += 1
            elif line.startswith("SKIP"):
                skip += 1
            else:
                fail += 1
            msg = f"[{i}/{len(days)} {done}ok {skip}skip {fail}fail] {line}"
            print(msg, flush=True)
            log.write(msg + "\n")
            log.flush()
    print(
        f"Finished: {done} done, {skip} skipped, {fail} failed "
        f"in {(time.time()-t_all)/3600:.2f}h",
        flush=True,
    )
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

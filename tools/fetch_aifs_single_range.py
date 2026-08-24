"""Download AIFS Single 00z runs (2t, 100u, 100v) over a date range.

Azure first; skip-existing; interleaved so all seasons appear early.
Resumable: rerun the same command after a stop.

Usage:
  python tools/fetch_aifs_single_range.py --start 2025-05-01 --end 2026-04-30
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone

from ecmwf.opendata import Client

STEPS = list(range(0, 361, 6))
PARAMS = ["2t", "100u", "100v"]
OUT_ROOT = "/Zeus/data/evaluation/aifs_single"
MIN_BYTES = 50_000_000  # a full 61-step 3-var file is ~183 MB


def daterange(start: datetime, end: datetime) -> list[datetime]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=1)
    return days


def interleaved(days: list[datetime], stride: int = 4) -> list[datetime]:
    ordered = []
    for offset in range(stride):
        ordered.extend(days[offset::stride])
    return ordered


def aifs_version(day: datetime) -> str:
    if day >= datetime(2026, 5, 12, tzinfo=timezone.utc):
        return "v2"
    if day >= datetime(2025, 8, 27, tzinfo=timezone.utc):
        return "v1.1"
    return "v1.0"


class AzureClientPool:
    """Reuse one SAS token across several runs; refresh before it expires.

    A new Client every file hammers Planetary Computer's token API (HTTP 429).
    Tokens themselves die after ~45 min (HTTP 403). Refresh every `max_uses`
    successful retrieves, or immediately after 403/429.
    """

    def __init__(self, source: str, max_uses: int = 15) -> None:
        self.source = source
        self.max_uses = max_uses
        self.client: Client | None = None
        self.uses = 0

    def get(self, *, force: bool = False) -> Client:
        if force or self.client is None or self.uses >= self.max_uses:
            self.client = Client(source=self.source, model="aifs-single")
            self.uses = 0
        self.uses += 1
        return self.client


def fetch_one(pool: AzureClientPool, day: datetime, out_dir: str, retries: int) -> str:
    stamp = day.strftime("%Y%m%dT000000Z")
    target = os.path.join(out_dir, f"{stamp}.grib2")
    if os.path.exists(target) and os.path.getsize(target) >= MIN_BYTES:
        return f"SKIP {stamp} {os.path.getsize(target)/1e6:.0f}MB"
    tmp = target + ".tmp"
    last_err = ""
    force_new = False
    for attempt in range(1, retries + 1):
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
            t0 = time.time()
            client = pool.get(force=force_new)
            force_new = False
            client.retrieve(
                target=tmp,
                date=day.strftime("%Y-%m-%d"),
                time=0,
                stream="oper",
                type="fc",
                param=PARAMS,
                step=STEPS,
            )
            size = os.path.getsize(tmp)
            if size < MIN_BYTES:
                last_err = f"too small ({size} bytes)"
                os.remove(tmp)
                time.sleep(15)
                continue
            os.replace(tmp, target)
            return (
                f"DONE {stamp} {size/1e6:.0f}MB "
                f"{time.time()-t0:.0f}s {aifs_version(day)} try={attempt}"
            )
        except Exception as exc:
            last_err = f"{type(exc).__name__}: {exc}"
            if os.path.exists(tmp):
                try:
                    os.remove(tmp)
                except OSError:
                    pass
            text = str(exc)
            force_new = True
            if "429" in text:
                time.sleep(90 * attempt)
            elif "403" in text:
                time.sleep(5)
            else:
                time.sleep(min(30 * attempt, 120))
    return f"FAIL {stamp} {last_err[:180]}"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2025-05-01")
    parser.add_argument("--end", default="2026-04-30")
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--source", default="azure")
    args = parser.parse_args()

    start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    days = interleaved(daterange(start, end), stride=args.stride)
    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(args.out_root, "fetch.log")
    pool = AzureClientPool(args.source)

    print(
        f"Fetching {len(days)} 00z runs {args.start}..{args.end} "
        f"source={args.source} interleaved stride={args.stride}",
        flush=True,
    )
    done = skip = fail = 0
    t_all = time.time()
    with open(log_path, "a", encoding="utf-8") as log:
        for i, day in enumerate(days, start=1):
            line = fetch_one(pool, day, args.out_root, args.retries)
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
        f"Finished: {done} downloaded, {skip} skipped, {fail} failed "
        f"in {time.time()-t_all:.0f}s  log={log_path}",
        flush=True,
    )
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())

"""Download AIFS Single 00z SSRD (accumulated J/m², steps 0..360 by 6).

Azure first; skip-existing. Default is the ENS-mean archive dates so the
AIFS-downscaler train/validation split can be reused without extra mapping.

Usage:
  python tools/fetch_aifs_ssrd_range.py
  python tools/fetch_aifs_ssrd_range.py --start 2025-09-03 --end 2026-07-22 --every 7
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

from ecmwf.opendata import Client

STEPS = list(range(0, 361, 6))
PARAMS = ["ssrd"]
OUT_ROOT = "/Zeus/data/evaluation/aifs_ssrd"
ENS_ROOT = "/Zeus/data/evaluation/aifs_ens_mean"
# A full 61-step ssrd file is ~64 MB; anything much smaller is truncated.
MIN_BYTES = 20_000_000


def aifs_version(day: datetime) -> str:
    if day >= datetime(2026, 5, 12, tzinfo=timezone.utc):
        return "v2"
    if day >= datetime(2025, 8, 27, tzinfo=timezone.utc):
        return "v1.1"
    return "v1.0"


def dates_from_ens_root(root: str) -> list[datetime]:
    days = []
    for path in sorted(Path(root).glob("*.npy")):
        days.append(
            datetime.strptime(path.stem, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)
        )
    if not days:
        raise SystemExit(f"no .npy cycles in {root}")
    return days


def dates_every(start: datetime, end: datetime, every: int) -> list[datetime]:
    days = []
    cur = start
    while cur <= end:
        days.append(cur)
        cur += timedelta(days=every)
    return days


class AzureClientPool:
    """Reuse one SAS token across several runs; refresh before it expires."""

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
    parser.add_argument(
        "--from-ens-root",
        default=ENS_ROOT,
        help="Use cycle dates from this ENS-mean .npy directory (default). "
        "Pass empty string to use --start/--end/--every instead.",
    )
    parser.add_argument("--start", default="2025-09-03")
    parser.add_argument("--end", default="2026-07-22")
    parser.add_argument("--every", type=int, default=7)
    parser.add_argument("--out-root", default=OUT_ROOT)
    parser.add_argument("--retries", type=int, default=6)
    parser.add_argument("--source", default="azure")
    args = parser.parse_args()

    if args.from_ens_root:
        days = dates_from_ens_root(args.from_ens_root)
        origin = f"ens archive {args.from_ens_root} ({len(days)} cycles)"
    else:
        start = datetime.strptime(args.start, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        end = datetime.strptime(args.end, "%Y-%m-%d").replace(tzinfo=timezone.utc)
        days = dates_every(start, end, args.every)
        origin = f"{args.start}..{args.end} every {args.every}d"

    os.makedirs(args.out_root, exist_ok=True)
    log_path = os.path.join(args.out_root, "fetch.log")
    pool = AzureClientPool(args.source)

    print(
        f"Fetching {len(days)} AIFS Single ssrd 00z runs from {origin} "
        f"source={args.source} -> {args.out_root}",
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

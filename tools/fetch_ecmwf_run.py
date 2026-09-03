"""Fetch one ECMWF open-data run (AIFS single + AIFS-ENS + IFS-ENS members).

Downloads, for a given init date/time:
  - aifs-single oper fc : 2t, 100u, 100v, ssrd   (steps 0..360 by 6)
  - aifs-ens   enfo em  : 2t                     (ensemble mean, steps 6..360 by 6)
  - aifs-ens   enfo cf+pf: 100u, 100v, ssrd      (51 members, steps 0..360 by 6)
  - ifs        enfo pf  : 100u, 100v, ssrd       (50 members, best effort)

The IFS winds feed the lead-ramped wind blend (alpha 0 at <=72h rising to
0.65 at 360h); member ssrd feeds the ensemble-mean SSRD mix (-16..-18%
360h SSRD vs the deterministic single run). IFS is optional — if it is late
or missing the run is served AIFS-only. 06/18z IFS ensembles only reach
144h, so those requests fall back to the shorter step list.

Files are fetched by a thread pool (default 4 workers). Azure throttles to
~2.5 MB/s per connection, so parallelism cuts wall time roughly linearly.
Each job creates its own Client so every download gets a fresh SAS token.

Usage:
  python tools/fetch_ecmwf_run.py --date 2026-08-18 --time 12 [--skip-ens]
         [--skip-ifs] [--workers 4]
"""

import argparse
import concurrent.futures
import glob
import os
import time

STEPS = list(range(0, 361, 6))
OUT_ROOT = "/Zeus/data/evaluation/ecmwf_live"


def fetch(model, out_dir, name, step_candidates=None, **kwargs):
    """Download to a temp file and rename, so partial files are never kept.

    Retries with backoff: the Planetary Computer SAS token endpoint returns
    429 when several clients request tokens at the same instant.

    `step_candidates` is an ordered list of step lists; when the index has no
    entries for one (e.g. 06/18z IFS ensembles stop at 144h, some products
    lack step 0) the next candidate is tried without burning a retry.
    """
    import random

    from ecmwf.opendata import Client

    target = os.path.join(out_dir, name)
    if os.path.exists(target) and os.path.getsize(target) > 0:
        return f"[skip] {name} already present ({os.path.getsize(target)/1e6:.0f} MB)"
    tmp = target + ".tmp"
    t0 = time.time()
    candidates = list(step_candidates) if step_candidates else [kwargs.pop("step")]
    candidate = 0
    last_exc = None
    for attempt in range(5):
        if attempt:
            time.sleep(min(300, 15 * 2**attempt) + random.uniform(0, 10))
        try:
            client = Client(source="azure", model=model)
            client.retrieve(target=tmp, step=candidates[candidate], **kwargs)
            break
        except ValueError as exc:
            # "Cannot find index entries" -> product doesn't have these steps.
            last_exc = exc
            if candidate + 1 < len(candidates):
                candidate += 1
                print(f"[fallback] {name}: trying shorter step list", flush=True)
            else:
                raise
        except Exception as exc:
            last_exc = exc
            # A 404 on an index blob means a step in this candidate list is
            # not published for this run (e.g. >144h for 06/18z IFS): move to
            # the next candidate instead of burning retries.
            if "404" in str(exc) and candidate + 1 < len(candidates):
                candidate += 1
                print(f"[fallback] {name}: 404, trying shorter step list", flush=True)
                continue
            print(f"[retry {attempt + 1}] {name}: {exc}", flush=True)
    else:
        raise RuntimeError(f"{name} failed after retries") from last_exc
    os.replace(tmp, target)
    return f"[done] {name}: {os.path.getsize(target)/1e6:.0f} MB in {time.time()-t0:.0f}s"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--date", required=True)
    ap.add_argument("--time", type=int, required=True)
    ap.add_argument("--skip-ens", action="store_true")
    ap.add_argument("--skip-ifs", action="store_true")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()

    stamp = f"{args.date.replace('-', '')}T{args.time:02d}0000Z"
    out_dir = os.path.join(OUT_ROOT, stamp)
    os.makedirs(out_dir, exist_ok=True)

    # (model, filename, request, required)
    jobs = [
        ("aifs-single", "aifs_single_all.grib2",
         dict(date=args.date, time=args.time, stream="oper", type="fc",
              param=["2t", "100u", "100v", "ssrd"], step=STEPS), True),
        # em has no step-0 field and only carries 2t among our params
        ("aifs-ens", "aifs_ens_em_2t.grib2",
         dict(date=args.date, time=args.time, stream="enfo", type="em",
              param="2t", step=STEPS[1:]), True),
    ]
    if not args.skip_ens:
        jobs.append(
            ("aifs-ens", "aifs_ens_cf_winds.grib2",
             dict(date=args.date, time=args.time, stream="enfo", type="cf",
                  param=["100u", "100v", "ssrd"], step=STEPS), True)
        )
        # Chunk perturbed members so each request stays inside the Azure SAS
        # token lifetime (~45 min); the full 50-member file exceeds it.
        for lo in range(1, 51, 10):
            members = list(range(lo, min(lo + 10, 51)))
            jobs.append(
                ("aifs-ens", f"aifs_ens_pf_winds_{lo:02d}_{members[-1]:02d}.grib2",
                 dict(date=args.date, time=args.time, stream="enfo", type="pf",
                      param=["100u", "100v", "ssrd"], number=members, step=STEPS), True)
            )
    if not args.skip_ifs:
        # 06/18z IFS ensembles stop at 144h; requesting past that 404s, so
        # start those runs directly on the short list.
        if args.time in (6, 18):
            ifs_step_candidates = [
                [s for s in STEPS if s <= 144],
                [s for s in STEPS if 0 < s <= 144],
            ]
        else:
            ifs_step_candidates = [
                STEPS,
                STEPS[1:],
                [s for s in STEPS if s <= 144],
                [s for s in STEPS if 0 < s <= 144],
            ]
        for lo in range(1, 51, 10):
            members = list(range(lo, min(lo + 10, 51)))
            jobs.append(
                ("ifs", f"ifs_ens_pf_winds_{lo:02d}_{members[-1]:02d}.grib2",
                 dict(date=args.date, time=args.time, stream="enfo", type="pf",
                      param=["100u", "100v", "ssrd"], number=members,
                      step_candidates=ifs_step_candidates), False)
            )
        # IFS-ENS ssrd at native 3h resolution for the 49h challenge window:
        # halving the zenith-redistribution interval scored -8.7/-9.1% SSRD
        # vs the 6h mix on 20260826/20260801. Optional, like the winds.
        ifs3h_steps = list(range(0, 61, 3))
        for lo in range(1, 51, 10):
            members = list(range(lo, min(lo + 10, 51)))
            jobs.append(
                ("ifs", f"ifs3h_ssrd_pf_{lo:02d}_{members[-1]:02d}.grib2",
                 dict(date=args.date, time=args.time, stream="enfo", type="pf",
                      param="ssrd", number=members,
                      step_candidates=[ifs3h_steps, ifs3h_steps[1:]]), False)
            )

    required_failures, optional_failures = [], []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(fetch, model, out_dir, name, **request): (name, required)
            for model, name, request, required in jobs
        }
        for future in concurrent.futures.as_completed(futures):
            name, required = futures[future]
            try:
                print(future.result(), flush=True)
            except Exception as exc:
                (required_failures if required else optional_failures).append(name)
                print(f"[fail] {name}: {exc}", flush=True)

    if optional_failures:
        # Remove partial IFS coverage per family: each blend needs all of its
        # member chunks, but a failed 3h-ssrd chunk must not discard winds.
        print(f"[warn] optional IFS downloads failed: {optional_failures}",
              flush=True)
        for family in ("ifs_ens_pf_winds", "ifs3h_ssrd_pf"):
            if any(name.startswith(family) for name in optional_failures):
                print(f"[warn] dropping partial {family} chunks", flush=True)
                for path in glob.glob(os.path.join(out_dir, f"{family}_*.grib2")):
                    os.unlink(path)
    if required_failures:
        raise SystemExit(f"{len(required_failures)} downloads failed: {required_failures}")
    print(f"Bundle complete: {out_dir}")


if __name__ == "__main__":
    main()

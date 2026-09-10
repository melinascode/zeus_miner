"""Live ECMWF -> Zeus ForecastStore bundle builder.

Feeds the prepared-bundle miner (ZEUS_PREPARED_BUNDLE_ONLY=1). Two loops in
one daemon:

  Pipeline (worker thread, ~2.5h per run, 4 runs/day):
    1. Probe ECMWF open data for the newest fully-published AIFS run.
    2. tools/fetch_ecmwf_run.py       Single + ENS em/cf/pf GRIBs (~11 GB)
    3. tools/build_ecmwf_bundle.py    6-hourly ENS-mean bundle.npz
    4. In-process: hourly cube 0..360h
         t2m/u100/v100  interpolated ENS-mean + AIFS downscaler v2
         ssrd           zenith-weighted redistribution of 6h interval energy
       Written as float16 .npy next to the run; GRIBs and bundle.npz deleted.

  Publisher (main loop, every minute):
    For challenge cycle T (00/06/12/18 UTC) inside its publish window
    [T-45min, T+15min], slice the newest cube at offset delta = T - run_init
    into 8 artifacts ({var} x {0_48, 0_360}), copy the 49h recipe into hours
    0..48 of each 361h artifact, blosc2-compress, hash with the miner hotkey
    and publish atomically into the ForecastStore. The miner wakes at T:30,
    loads this bundle and commits the hashes by T:45.

Hours past the run's +360h horizon are padded by wrapping the final 24
forecast hours, preserving the diurnal cycle.

Run under the zeus-fourvar env with PYTHONPATH=/Zeus (see
start_bundle_builder.sh).
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
from datetime import datetime, timedelta, timezone
from logging.handlers import RotatingFileHandler
from pathlib import Path

import numpy as np

REPO = Path("/Zeus")
RUN_ROOT = REPO / "data" / "evaluation" / "ecmwf_live"
DEFAULT_STORE = REPO / "data" / "forecast_store_v2"
CUBE_TUV = "hourly_tuv_f16.npy"
CUBE_SSRD = "hourly_ssrd_f16.npy"
CUBE_SSRD3H = "hourly_ssrd3h_f16.npy"  # 0..60h from IFS-ENS 3h intervals
CUBE_META = "cube.meta.json"

PUBLISH_WINDOW_BEFORE = timedelta(minutes=45)
PUBLISH_WINDOW_AFTER = timedelta(minutes=15)
MAX_STALENESS_HOURS = 72
MIN_FREE_GB_FOR_FETCH = 25.0  # ~11 GB AIFS + ~7 GB IFS winds + bundle/cube
KEEP_RUN_DIRS = 3
PROBE_INTERVAL_SECONDS = 15 * 60
# IFS-ENS full 360h coverage publishes ~2h after AIFS (e.g. 00z run: AIFS
# ~06:10 UTC, IFS enfo to 360h ~08:50 UTC). Bounded wait for the wind-blend
# inputs; the publisher keeps using the newest older cube meanwhile.
IFS_WAIT_MAX_SECONDS = 150 * 60

VARIABLE_CHANNEL = {
    "2m_temperature": 0,
    "100m_u_component_of_wind": 1,
    "100m_v_component_of_wind": 2,
}
SSRD_VARIABLE = "surface_solar_radiation_downwards"

# Freshness split for the 49h challenge. The full run used for the 361h
# artifacts is 12h stale at publish time because AIFS-ENS publishes ~6.3h
# after init, just past the commit window. The AIFS-single of the run 6h
# before the cycle, however, publishes ~5.7h after init, ~15min before the
# window closes. When it is up we build a 49h cube from it and use that for
# the short artifacts (long artifacts keep the stale ENS+CNN cube).
FRESH_SHORT_ENABLED = os.environ.get("ZEUS_FRESH_SHORT", "1") == "1"
# raw = linear interp of the fresh single; cnn = fresh single through the v2
# CNN; blend_50 = per-variable weighted mean of the fresh CNN field and the
# stale serving field.
FRESH_SHORT_MODE = os.environ.get("ZEUS_FRESH_SHORT_MODE", "blend_50")
# Fresh-member weight per channel. 0.05 grid on 20260801T00 + 20260826
# 00/06/12 (V3 cubes, official scalars). Naive argmin goes to 0.75 from
# 12z alone; the Pareto step that improves every 0826 cycle without
# hurting 0801 is 0.45/0.65/0.60. v stays 0.60: 0.65 helps 12z but
# slightly hurts 00z/06z.
FRESH_ALPHA_TUV = np.array([0.45, 0.65, 0.60], dtype=np.float32)[:, None, None]
FRESH_ALPHA_SSRD = 0.5
# Give up waiting for the fresh single this long after cycle time and
# publish stale-only. Publishing takes ~12 min; the miner reads at T+30.
FRESH_FALLBACK_DEADLINE = timedelta(minutes=2)
FRESH_STEPS = list(range(0, 61, 6))  # 6h steps to 60h: leads 6..54 need them
FRESH_TUV = "fresh_short_tuv_f16.npy"
FRESH_SSRD = "fresh_short_ssrd_f16.npy"
FRESH_META = "fresh_short.meta.json"
SHORT_HOURS = 49

# Step-1 MOS bias corrector: per-cell (optionally hour-of-day) mean error of
# our own published short stack vs ERA5, subtracted from the 49h artifacts at
# publish time. Empty ZEUS_BIAS_CORRECTOR (default) disables it entirely.
BIAS_CORRECTOR_PATH = os.environ.get("ZEUS_BIAS_CORRECTOR", "").strip()
BIAS_MODE = os.environ.get("ZEUS_BIAS_MODE", "flat")  # flat | hod
BIAS_SHRINK = float(os.environ.get("ZEUS_BIAS_SHRINK", "0.5"))
BIAS_VARIABLES = tuple(
    v
    for v in os.environ.get(
        "ZEUS_BIAS_VARS",
        "100m_u_component_of_wind,100m_v_component_of_wind",
    ).split(",")
    if v
)

# Long-window upgrades, validated on a 20260801T12z live-geometry replay:
#   - ZEUS_BIAS_LONG_VARS: HOD MOS extended to hours 49..360 (2t full-window
#     C -1.2%, improves every lead segment incl. day 7-15). Only applied to
#     vars also present in ZEUS_BIAS_VARS.
#   - TAIL_COMPOSITE_DAYS: wrapped tail hours (349..360 at delta=12) served
#     as a multi-day diurnal composite instead of repeating the last day.
#     SSRD 3-day: tail C -32%, full long C -2.2%. 2t 2-day: small gain.
#     Winds keep the plain wrap (composites scored worse).
BIAS_LONG_VARIABLES = tuple(
    v
    for v in os.environ.get("ZEUS_BIAS_LONG_VARS", "2m_temperature").split(",")
    if v
)
TAIL_COMPOSITE_DAYS = {
    "surface_solar_radiation_downwards": 3,
    "2m_temperature": 2,
}

logger = logging.getLogger("zeus.bundle_builder")


def configure_logging() -> None:
    # bittensor's import hook sets pre-existing loggers to CRITICAL; import it
    # first so our level survives (later imports elsewhere are then no-ops).
    import bittensor  # noqa: F401

    logger.setLevel(logging.INFO)
    if logger.handlers:
        return
    formatter = logging.Formatter(
        "%(asctime)s | %(levelname)-7s | %(threadName)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    console = logging.StreamHandler(sys.stdout)
    console.setFormatter(formatter)
    logger.addHandler(console)
    log_dir = REPO / "logs"
    log_dir.mkdir(exist_ok=True)
    file_handler = RotatingFileHandler(
        log_dir / "bundle_builder.log",
        maxBytes=20 * 1024 * 1024,
        backupCount=3,
        encoding="utf-8",
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)


def run_stamp(when: datetime) -> str:
    return when.strftime("%Y%m%dT%H%M%SZ")


def parse_stamp(stamp: str) -> datetime:
    return datetime.strptime(stamp, "%Y%m%dT%H%M%SZ").replace(tzinfo=timezone.utc)


def load_hotkey_ss58(wallet_name: str, wallet_hotkey: str) -> str:
    """Read the hotkey ss58 address from the (unencrypted) keyfile."""
    path = (
        Path.home()
        / ".bittensor"
        / "wallets"
        / wallet_name
        / "hotkeys"
        / wallet_hotkey
    )
    payload = json.loads(path.read_text(encoding="utf-8"))
    address = payload.get("ss58Address")
    if not isinstance(address, str) or not address:
        raise ValueError(f"No ss58Address in {path}")
    return address


def free_gb(path: Path) -> float:
    usage = shutil.disk_usage(path)
    return usage.free / 1e9


# --------------------------------------------------------------------------
# Pipeline: fetch -> 6h bundle -> hourly cube
# --------------------------------------------------------------------------

def probe_latest_complete_run() -> datetime | None:
    """Newest run for which every product we need is fully published."""
    from ecmwf.opendata import Client

    single = Client(source="azure", model="aifs-single")
    ens = Client(source="azure", model="aifs-ens")
    try:
        candidates = [
            single.latest(stream="oper", type="fc", param="2t", step=360),
            ens.latest(stream="enfo", type="em", param="2t", step=360),
            ens.latest(stream="enfo", type="pf", param="100u", number=50, step=360),
        ]
    except Exception as exc:
        logger.warning("Availability probe failed: %s", exc)
        return None
    latest = min(
        c.replace(tzinfo=timezone.utc) if c.tzinfo is None else c
        for c in candidates
    )
    return latest


def cube_exists(run_dir: Path) -> bool:
    return (
        (run_dir / CUBE_TUV).is_file()
        and (run_dir / CUBE_SSRD).is_file()
        and (run_dir / CUBE_META).is_file()
    )


def ifs_winds_available(run_time: datetime) -> bool:
    """Cheap index probe: is the same-cycle IFS-ENS wind data published?

    00/12z IFS ensembles reach 360h, 06/18z stop at 144h. Probing the final
    step matters: 360h fields publish ~1h after the 144h ones, and fetching
    in between would truncate the blend to 144h for a 00/12z run.
    """
    from ecmwf.opendata import Client

    final_step = 360 if run_time.hour in (0, 12) else 144
    try:
        latest = Client(source="azure", model="ifs").latest(
            stream="enfo", type="pf", param="100u", number=50, step=final_step
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("IFS availability probe failed: %s", exc)
        return False
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return latest >= run_time


def run_fetch_script(run_time: datetime, skip_ifs: bool) -> bool:
    command = [
        sys.executable,
        str(REPO / "tools" / "fetch_ecmwf_run.py"),
        "--date",
        run_time.strftime("%Y-%m-%d"),
        "--time",
        str(run_time.hour),
    ]
    if skip_ifs:
        command.append("--skip-ifs")
    result = subprocess.run(
        command, cwd=REPO, env={**os.environ, "PYTHONPATH": str(REPO)}
    )
    return result.returncode == 0


def fetch_and_bundle(run_time: datetime) -> Path | None:
    """Drive the existing fetch + bundle tools via subprocess. Resumable."""
    stamp = run_stamp(run_time)
    run_dir = RUN_ROOT / stamp
    run_dir.mkdir(parents=True, exist_ok=True)

    if not (run_dir / "bundle.npz").is_file():
        if free_gb(RUN_ROOT) < MIN_FREE_GB_FOR_FETCH:
            logger.error(
                "Only %.1f GB free; refusing to fetch %s", free_gb(RUN_ROOT), stamp
            )
            return None
        # IFS-ENS publishes ~1h after AIFS. Fetch AIFS immediately (skipping
        # IFS if its index isn't there yet), then wait a bounded time for the
        # IFS winds; if they never appear the run is served AIFS-only.
        ifs_ready = ifs_winds_available(run_time)
        logger.info("Fetching run %s (IFS winds available: %s) ...", stamp, ifs_ready)
        if not run_fetch_script(run_time, skip_ifs=not ifs_ready):
            logger.error("fetch_ecmwf_run.py failed for %s", stamp)
            return None
        if not any(run_dir.glob("ifs_ens_pf_winds_*.grib2")):
            deadline = time.monotonic() + IFS_WAIT_MAX_SECONDS
            while time.monotonic() < deadline:
                if ifs_winds_available(run_time):
                    logger.info("IFS winds now published; fetching for %s", stamp)
                    run_fetch_script(run_time, skip_ifs=False)
                    break
                time.sleep(180)
            if not any(run_dir.glob("ifs_ens_pf_winds_*.grib2")):
                logger.warning(
                    "No IFS winds for %s after waiting; serving AIFS-only", stamp
                )

        logger.info("Building 6-hourly bundle for %s ...", stamp)
        result = subprocess.run(
            [
                sys.executable,
                str(REPO / "tools" / "build_ecmwf_bundle.py"),
                "--run",
                stamp,
            ],
            cwd=REPO,
            env={**os.environ, "PYTHONPATH": str(REPO)},
        )
        if result.returncode != 0:
            logger.error("build_ecmwf_bundle.py failed for %s", stamp)
            return None

    # GRIBs are no longer needed once bundle.npz exists.
    for grib in run_dir.glob("*.grib2"):
        grib.unlink()
    return run_dir


_COMPOSER = None
_COMPOSER_LOCK = threading.Lock()

_BIAS = None


def get_bias_corrector():
    """Load the bias maps once. Returns None when disabled."""
    global _BIAS
    if _BIAS is None:
        if not BIAS_CORRECTOR_PATH:
            _BIAS = False
        else:
            data = np.load(BIAS_CORRECTOR_PATH)
            _BIAS = {
                "flat": data["flat"].astype(np.float32),   # (3, 721, 1440)
                "hod": data["hod"].astype(np.float32),     # (24, 3, 721, 1440)
            }
            logger.info(
                "bias corrector loaded %s (mode=%s shrink=%.2f vars=%s)",
                BIAS_CORRECTOR_PATH,
                BIAS_MODE,
                BIAS_SHRINK,
                ",".join(BIAS_VARIABLES),
            )
    return _BIAS or None


def apply_bias_correction(
    array: np.ndarray, variable: str, cycle_time: datetime
) -> np.ndarray:
    """Subtract shrink x fitted mean error from a short (49h) tuv artifact."""
    bias = get_bias_corrector()
    if bias is None or variable not in BIAS_VARIABLES:
        return array
    channel = VARIABLE_CHANNEL[variable]
    if BIAS_MODE == "hod":
        idx = (cycle_time.hour + np.arange(array.shape[0])) % 24
        correction = bias["hod"][idx, channel]
    else:
        correction = bias["flat"][channel][None]
    return (
        array.astype(np.float32) - np.float32(BIAS_SHRINK) * correction
    ).astype(np.float16)


def get_composer():
    """Shared ForecastComposer; loads the global CNN + statics once."""
    global _COMPOSER
    with _COMPOSER_LOCK:
        if _COMPOSER is None:
            from zeus_ml.serve.compose_forecast import ForecastComposer

            _COMPOSER = ForecastComposer()
            logger.info(
                "composer loaded global_checkpoint=%s lagged=%s",
                _COMPOSER.config.global_checkpoint,
                _COMPOSER.g_lagged,
            )
        return _COMPOSER


# Lead-ramped IFS-ENS wind blend: no IFS weight through 72h (protects the
# 49h challenge), rising linearly to 0.65 at 360h where the IFS-ENS mean
# adds real skill (verified on 20260727 + 20260801: never worse anywhere,
# up to -3.6% wind iwRMSE on the 361h window).
WIND_RAMP_START = 72
WIND_RAMP_END = 360
WIND_RAMP_ALPHA_MAX = 0.65


def apply_ifs_wind_ramp(cube: np.ndarray, bundle, stamp: str) -> None:
    """Blend IFS-ENS mean winds into cube channels 1:3, in place."""
    ifs_steps = bundle["ifs_steps"].astype(np.int64)
    ifs_uv = np.stack([bundle["ifs_u100"], bundle["ifs_v100"]], axis=1).astype(
        np.float32
    )
    last_ifs = int(ifs_steps[-1])
    logger.info(
        "cube %s: blending IFS-ENS winds (ramp %d..%dh, alpha max %.2f, IFS to %dh)",
        stamp,
        WIND_RAMP_START,
        WIND_RAMP_END,
        WIND_RAMP_ALPHA_MAX,
        last_ifs,
    )
    span = float(WIND_RAMP_END - WIND_RAMP_START)
    for lead in range(WIND_RAMP_START + 1, min(360, last_ifs) + 1):
        alpha = WIND_RAMP_ALPHA_MAX * min(1.0, (lead - WIND_RAMP_START) / span)
        right = int(np.searchsorted(ifs_steps, lead, side="left"))
        if int(ifs_steps[right]) == lead:
            ifs_hour = ifs_uv[right]
        else:
            left = right - 1
            fraction = (lead - float(ifs_steps[left])) / float(
                ifs_steps[right] - ifs_steps[left]
            )
            ifs_hour = (1.0 - fraction) * ifs_uv[left] + fraction * ifs_uv[right]
        blended = (1.0 - alpha) * cube[lead, 1:].astype(np.float32) + alpha * ifs_hour
        cube[lead, 1:] = blended.astype(np.float16)


def fresh_single_available(run_time: datetime) -> bool:
    """Is the AIFS-single of ``run_time`` published through step 60?"""
    from ecmwf.opendata import Client

    try:
        latest = Client(source="azure", model="aifs-single").latest(
            stream="oper", type="fc", param="2t", step=FRESH_STEPS[-1]
        )
    except Exception as exc:  # noqa: BLE001
        logger.warning("Fresh-single probe failed: %s", exc)
        return False
    if latest.tzinfo is None:
        latest = latest.replace(tzinfo=timezone.utc)
    return latest >= run_time


def fresh_short_exists(run_dir: Path) -> bool:
    return (run_dir / FRESH_TUV).is_file() and (run_dir / FRESH_SSRD).is_file()


def build_fresh_short(
    fresh_run: datetime, stale_cube_dir: Path, cycle_time: datetime
) -> bool:
    """Fetch the fresh AIFS-single and build the 49h short cube.

    Writes FRESH_TUV (49, 3, 721, 1440) and FRESH_SSRD (49, 721, 1440) into
    the fresh run's directory, indexed by challenge hour (lead = 6 + hour).
    """
    from tools.build_ecmwf_bundle import read_grib, stack
    from tools.fetch_ecmwf_run import fetch
    from tools.score_aifs_ssrd import reconstruct_hourly
    from zeus_ml.serve.compose_forecast import _interpolate

    started = time.monotonic()
    fresh_dir = RUN_ROOT / run_stamp(fresh_run)
    fresh_dir.mkdir(parents=True, exist_ok=True)
    grib = fresh_dir / "fresh_single_short.grib2"
    if not grib.is_file():
        if free_gb(RUN_ROOT) < 5.0:
            logger.error("Fresh short: <5 GB free, skipping")
            return False
        logger.info("Fresh short: fetching %s AIFS-single to 60h", fresh_dir.name)
        try:
            fetch(
                "aifs-single",
                str(fresh_dir),
                grib.name,
                date=fresh_run.strftime("%Y-%m-%d"),
                time=fresh_run.hour,
                stream="oper",
                type="fc",
                param=["2t", "100u", "100v", "ssrd"],
                step=FRESH_STEPS,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("Fresh short: fetch failed: %s", exc)
            grib.unlink(missing_ok=True)
            return False

    composer = get_composer()
    fields = read_grib(str(grib), ["2t", "100u", "100v", "ssrd"])
    steps = np.asarray(FRESH_STEPS)
    single = np.stack(
        [stack(fields["2t"], steps), stack(fields["100u"], steps),
         stack(fields["100v"], steps)],
        axis=1,
    ).astype(np.float32)
    acc = stack(fields["ssrd"], steps).astype(np.float64)
    del fields

    # reconstruct_hourly expects the full 361h accumulated array (61 rows);
    # pad past step 60 with the final cumulative value (zero energy there).
    n_int = acc.shape[0] - 1
    accumulated = np.zeros((61, 721, 1440), dtype=np.float64)
    np.cumsum(
        np.clip(np.diff(acc, axis=0), 0.0, None),
        axis=0,
        out=accumulated[1 : n_int + 1],
    )
    accumulated[n_int + 1 :] = accumulated[n_int]
    hourly_ssrd = np.clip(
        reconstruct_hourly(
            accumulated.astype(np.float32),
            method="zenith",
            cycle_time=fresh_run,
            latitudes=composer.latitudes,
            longitudes=composer.longitudes,
            zenith_samples=4,
        ),
        0.0,
        None,
    ).astype(np.float32)
    del acc, accumulated

    delta_stale = int((cycle_time - parse_stamp(stale_cube_dir.name)).total_seconds() // 3600)
    stale_tuv = np.load(stale_cube_dir / CUBE_TUV, mmap_mode="r")
    stale_ssrd = np.load(stale_cube_dir / CUBE_SSRD, mmap_mode="r")

    tuv = np.empty((SHORT_HOURS, 3, 721, 1440), dtype=np.float16)
    ssrd = np.empty((SHORT_HOURS, 721, 1440), dtype=np.float16)
    for hour in range(SHORT_HOURS):
        lead = 6 + hour
        # The blend's fresh member is the CNN-corrected single: verified on
        # 20260826 that raw single 2t is far worse than its CNN correction.
        if FRESH_SHORT_MODE == "raw":
            fresh_field = _interpolate(single, lead)
        else:
            fresh_field = composer._apply_global(
                single, None, lead, fresh_run
            ).numpy()
        fresh_solar = hourly_ssrd[lead]
        if FRESH_SHORT_MODE == "blend_50":
            stale_field = stale_tuv[delta_stale + hour].astype(np.float32)
            tuv[hour] = (
                FRESH_ALPHA_TUV * fresh_field
                + (1.0 - FRESH_ALPHA_TUV) * stale_field
            ).astype(np.float16)
            ssrd[hour] = np.clip(
                FRESH_ALPHA_SSRD * fresh_solar
                + (1.0 - FRESH_ALPHA_SSRD)
                * stale_ssrd[delta_stale + hour].astype(np.float32),
                0.0,
                None,
            ).astype(np.float16)
        else:
            tuv[hour] = fresh_field.astype(np.float16)
            ssrd[hour] = np.clip(fresh_solar, 0.0, None).astype(np.float16)

    for name, array in ((FRESH_TUV, tuv), (FRESH_SSRD, ssrd)):
        tmp = fresh_dir / (name + ".tmp.npy")
        np.save(tmp, array)
        os.replace(tmp, fresh_dir / name)
    (fresh_dir / FRESH_META).write_text(
        json.dumps(
            {
                "fresh_run": run_stamp(fresh_run),
                "mode": FRESH_SHORT_MODE,
                "stale_run": stale_cube_dir.name,
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "elapsed_minutes": round((time.monotonic() - started) / 60.0, 1),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    grib.unlink(missing_ok=True)
    logger.info(
        "Fresh short cube for %s ready in %.1f min (mode %s)",
        fresh_dir.name,
        (time.monotonic() - started) / 60.0,
        FRESH_SHORT_MODE,
    )
    return True


def build_hourly_cube(run_dir: Path) -> None:
    """bundle.npz (6-hourly) -> hourly float16 cubes with v2 CNN + zenith SSRD."""
    from tools.score_aifs_ssrd import reconstruct_hourly

    stamp = run_dir.name
    cycle_time = parse_stamp(stamp)
    composer = get_composer()

    bundle = np.load(run_dir / "bundle.npz")
    required = {"t2m", "u100", "v100", "ssrd"}
    missing = required - set(bundle.files)
    if missing:
        raise ValueError(f"{run_dir.name}/bundle.npz is missing {sorted(missing)}")

    started = time.monotonic()
    wind_blend_applied = False
    if not (run_dir / CUBE_TUV).is_file():
        ens = np.stack(
            [bundle["t2m"], bundle["u100"], bundle["v100"]], axis=1
        ).astype(np.float32)
        if "t2m_single" in bundle:
            single = np.stack(
                [bundle["t2m_single"], bundle["u100_single"], bundle["v100_single"]],
                axis=1,
            ).astype(np.float32)
        else:
            single = None

        cube = np.empty((361, 3, 721, 1440), dtype=np.float16)
        for lead in range(361):
            corrected = composer._apply_global(ens, single, lead, cycle_time)
            cube[lead] = corrected.numpy().astype(np.float16)
            if lead % 48 == 0:
                logger.info(
                    "cube %s lead %d/360 (%.1f min elapsed)",
                    stamp,
                    lead,
                    (time.monotonic() - started) / 60.0,
                )
        del ens, single
        if "ifs_u100" in bundle.files:
            apply_ifs_wind_ramp(cube, bundle, stamp)
            wind_blend_applied = True
        tmp = run_dir / (CUBE_TUV + ".tmp.npy")
        np.save(tmp, cube)
        os.replace(tmp, run_dir / CUBE_TUV)
        del cube
    else:
        logger.info("cube %s: %s already present, skipping CNN pass", stamp, CUBE_TUV)

    logger.info("cube %s: reconstructing hourly SSRD (zenith) ...", stamp)
    interval_mean = bundle["ssrd"].astype(np.float64)  # (60, 721, 1440) W/m2
    accumulated = np.zeros((61, 721, 1440), dtype=np.float64)
    np.cumsum(interval_mean * (6 * 3600.0), axis=0, out=accumulated[1:])
    hourly = reconstruct_hourly(
        accumulated.astype(np.float32),
        method="zenith",
        cycle_time=cycle_time,
        latitudes=composer.latitudes,
        longitudes=composer.longitudes,
        zenith_samples=4,
    )
    tmp = run_dir / (CUBE_SSRD + ".tmp.npy")
    np.save(tmp, np.clip(hourly, 0.0, None).astype(np.float16))
    os.replace(tmp, run_dir / CUBE_SSRD)

    ssrd3h_built = False
    if "ifs_ssrd3h_acc" in bundle.files:
        logger.info(
            "cube %s: reconstructing hourly IFS 3h SSRD (zenith, 0..60h) ...",
            stamp,
        )
        hourly_3h = reconstruct_hourly(
            bundle["ifs_ssrd3h_acc"].astype(np.float32),
            method="zenith",
            cycle_time=cycle_time,
            latitudes=composer.latitudes,
            longitudes=composer.longitudes,
            zenith_samples=4,
            step_hours=3,
        )
        tmp = run_dir / (CUBE_SSRD3H + ".tmp.npy")
        np.save(tmp, np.clip(hourly_3h, 0.0, None).astype(np.float16))
        os.replace(tmp, run_dir / CUBE_SSRD3H)
        ssrd3h_built = True

    ckpt_stem = Path(composer.config.global_checkpoint).stem
    model_name = f"aifs_ens_mean+{ckpt_stem}+zenith_ssrd"
    if ssrd3h_built:
        model_name += "+ifs_ssrd3h60"
    if wind_blend_applied:
        model_name += "+ifs_wind_ramp65"
    try:
        bundle_meta = json.loads((run_dir / "bundle.meta.json").read_text())
        ssrd_source = bundle_meta.get("ssrd_source", "single")
        if ssrd_source != "single":
            model_name += f"+ssrd_{ssrd_source}"
    except (OSError, json.JSONDecodeError):
        pass
    meta = {
        "run": stamp,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": model_name,
        "global_checkpoint": composer.config.global_checkpoint,
        "elapsed_minutes": round((time.monotonic() - started) / 60.0, 1),
    }
    (run_dir / CUBE_META).write_text(json.dumps(meta, indent=2), encoding="utf-8")
    logger.info("cube %s complete in %.1f min", stamp, meta["elapsed_minutes"])

    # The 6-hourly bundle is no longer needed once the cube exists (~1.7 GB).
    (run_dir / "bundle.npz").unlink(missing_ok=True)


def garbage_collect_runs() -> None:
    run_dirs = sorted(
        (p for p in RUN_ROOT.iterdir() if p.is_dir()),
        key=lambda p: p.name,
        reverse=True,
    )
    kept = 0
    for path in run_dirs:
        if cube_exists(path):
            kept += 1
            if kept > KEEP_RUN_DIRS:
                logger.info("GC: removing old run %s", path.name)
                shutil.rmtree(path, ignore_errors=True)
        elif kept >= 1 and not any(path.glob("*.tmp*")):
            # Incomplete leftovers older than the newest good cube.
            newest_good = next((p.name for p in run_dirs if cube_exists(p)), "")
            if path.name < newest_good:
                logger.info("GC: removing incomplete run %s", path.name)
                shutil.rmtree(path, ignore_errors=True)


def pipeline_step() -> None:
    """One check: is there a newer complete run than our newest cube?"""
    latest = probe_latest_complete_run()
    if latest is None:
        return
    stamp = run_stamp(latest)
    run_dir = RUN_ROOT / stamp
    if cube_exists(run_dir):
        return
    logger.info("New run available: %s", stamp)
    run_dir = fetch_and_bundle(latest)
    if run_dir is None:
        return
    build_hourly_cube(run_dir)
    garbage_collect_runs()


# --------------------------------------------------------------------------
# Publisher: cube -> ForecastStore bundle for a challenge cycle
# --------------------------------------------------------------------------

def newest_cube_dir() -> Path | None:
    candidates = sorted(
        (p for p in RUN_ROOT.iterdir() if p.is_dir() and cube_exists(p)),
        key=lambda p: p.name,
        reverse=True,
    )
    return candidates[0] if candidates else None


def cycle_in_publish_window(now: datetime) -> datetime | None:
    """Challenge cycle T with T-45min <= now <= T+15min, if any."""
    floor = now.replace(minute=0, second=0, microsecond=0)
    floor = floor.replace(hour=(floor.hour // 6) * 6)
    for candidate in (floor, floor + timedelta(hours=6)):
        if candidate - PUBLISH_WINDOW_BEFORE <= now <= candidate + PUBLISH_WINDOW_AFTER:
            return candidate
    return None


def tail_wrapped_indices(delta_hours: int) -> np.ndarray:
    """Cube indices for artifact hours 0..360 at offset delta.

    Hours beyond the run's +360h horizon wrap into the final 24 forecast
    hours at the same hour-of-day, so the diurnal cycle stays aligned.
    """
    indices = delta_hours + np.arange(361)
    base = 360 - 24
    over = indices > 360
    indices[over] = base + ((indices[over] - base) % 24)
    return indices


def publish_bundle(
    store,
    cycle_time: datetime,
    hotkey: str,
    expected_state_keys: tuple[str, ...],
) -> bool:
    from zeus.utils.compression import compress_prediction
    from zeus.utils.hash import prediction_hash
    from zeus.validator.challenge_spec import make_state_key
    from zeus.validator.constants import LONG_CHALLENGE, SHORT_CHALLENGE

    if store.bundle_exists(cycle_time, expected_state_keys):
        return False

    cube_dir = newest_cube_dir()
    if cube_dir is None:
        logger.warning("No hourly cube available; cannot publish %s", cycle_time)
        return False

    run_time = parse_stamp(cube_dir.name)
    delta = int((cycle_time - run_time).total_seconds() // 3600)
    if delta < 0:
        logger.error("Cube %s is newer than cycle %s; skipping", cube_dir.name, cycle_time)
        return False
    if delta > MAX_STALENESS_HOURS:
        logger.error(
            "Cube %s is %dh stale for cycle %s; refusing to publish",
            cube_dir.name,
            delta,
            cycle_time,
        )
        return False

    # Freshness split: try to serve the 49h artifacts from the 6h-fresher
    # AIFS-single. Wait (return False -> retry next tick) until it publishes
    # or the fallback deadline passes.
    fresh_run = cycle_time - timedelta(hours=6)
    fresh_dir = RUN_ROOT / run_stamp(fresh_run)
    use_fresh = FRESH_SHORT_ENABLED and fresh_run > run_time
    if use_fresh and not fresh_short_exists(fresh_dir):
        now = datetime.now(timezone.utc)
        if now > cycle_time + FRESH_FALLBACK_DEADLINE:
            logger.warning(
                "Fresh single %s missed deadline; publishing stale short",
                run_stamp(fresh_run),
            )
            use_fresh = False
        elif fresh_single_available(fresh_run):
            if not build_fresh_short(fresh_run, cube_dir, cycle_time):
                use_fresh = False
        else:
            logger.info(
                "Fresh single %s not yet published; retrying (deadline %s)",
                run_stamp(fresh_run),
                (cycle_time + FRESH_FALLBACK_DEADLINE).strftime("%H:%M"),
            )
            return False
    fresh_tuv = fresh_ssrd = None
    if use_fresh and fresh_short_exists(fresh_dir):
        fresh_tuv = np.load(fresh_dir / FRESH_TUV)
        fresh_ssrd = np.load(fresh_dir / FRESH_SSRD)
        if not (
            np.isfinite(fresh_tuv).all() and np.isfinite(fresh_ssrd).all()
        ):
            logger.error("Fresh short cube has non-finite values; using stale")
            fresh_tuv = fresh_ssrd = None
    use_fresh = fresh_tuv is not None

    started = time.monotonic()
    indices = tail_wrapped_indices(delta)
    wrapped_hours = np.nonzero(delta + np.arange(361) > 360)[0]
    tuv = np.load(cube_dir / CUBE_TUV, mmap_mode="r")
    ssrd = np.load(cube_dir / CUBE_SSRD, mmap_mode="r")

    # Short-window SSRD: 0.4 * (6h AIFS/IFS mix) + 0.6 * (IFS-ENS 3h zenith).
    # 0.05 grid on 0801+0826 reconfirmed 0.4 as the 0826 holdout min
    # (0801 prefers 0.45 by 0.03 combined; not enough to move). Falls
    # back to the fresh blend / stale slice when the 3h cube is unavailable.
    short_ssrd = None
    ssrd3h_path = cube_dir / CUBE_SSRD3H
    if ssrd3h_path.is_file() and delta + SHORT_HOURS <= 61:
        ssrd3h = np.load(ssrd3h_path).astype(np.float32)
        stale_slice = np.asarray(
            ssrd[indices[:SHORT_HOURS]], dtype=np.float32
        )
        candidate = 0.4 * stale_slice + 0.6 * ssrd3h[delta : delta + SHORT_HOURS]
        if np.isfinite(candidate).all():
            short_ssrd = np.clip(candidate, 0.0, None).astype(np.float16)
        else:
            logger.error("IFS 3h SSRD blend has non-finite values; skipping")

    logger.info(
        "Publishing cycle %s from run %s (staleness %dh, padded tail %dh, "
        "short tuv %s, short ssrd %s)",
        store.cycle_key(cycle_time),
        cube_dir.name,
        delta,
        delta,
        f"fresh {run_stamp(fresh_run)} ({FRESH_SHORT_MODE})" if use_fresh else "stale",
        "ifs3h blend" if short_ssrd is not None
        else ("fresh blend" if use_fresh else "stale"),
    )

    with store.begin_bundle(cycle_time, expected_state_keys) as writer:
        for variable in sorted(list(VARIABLE_CHANNEL) + [SSRD_VARIABLE]):
            if variable == SSRD_VARIABLE:
                long_array = np.ascontiguousarray(ssrd[indices], dtype=np.float16)
            else:
                channel = VARIABLE_CHANNEL[variable]
                long_array = np.ascontiguousarray(
                    tuv[indices, channel], dtype=np.float16
                )
            comp_days = TAIL_COMPOSITE_DAYS.get(variable, 1)
            if comp_days > 1 and wrapped_hours.size:
                # Multi-day diurnal composite for the wrapped tail hours.
                for h in wrapped_hours:
                    lead = int(indices[h])
                    if variable == SSRD_VARIABLE:
                        stack = [
                            np.asarray(ssrd[lead - 24 * k], np.float32)
                            for k in range(comp_days)
                        ]
                    else:
                        stack = [
                            np.asarray(tuv[lead - 24 * k, channel], np.float32)
                            for k in range(comp_days)
                        ]
                    long_array[h] = np.mean(stack, axis=0).astype(np.float16)
            if not np.isfinite(long_array).all():
                raise ValueError(f"{variable}: non-finite values in cube slice")
            if variable == SSRD_VARIABLE:
                # Validators now penalty any SSRD cell < 0 (Orpheus-AI/Zeus#83 / #87).
                long_array = np.clip(long_array, 0.0, None)

            short_source_time = run_time
            short_array = None
            if variable == SSRD_VARIABLE and short_ssrd is not None:
                short_array = np.ascontiguousarray(short_ssrd[:SHORT_HOURS])
            elif use_fresh:
                if variable == SSRD_VARIABLE:
                    short_array = np.ascontiguousarray(
                        fresh_ssrd[:SHORT_HOURS], dtype=np.float16
                    )
                else:
                    short_array = np.ascontiguousarray(
                        fresh_tuv[:SHORT_HOURS, VARIABLE_CHANNEL[variable]],
                        dtype=np.float16,
                    )
                short_source_time = fresh_run
            if short_array is None:
                short_array = np.ascontiguousarray(long_array[:SHORT_HOURS])
            if variable != SSRD_VARIABLE:
                short_array = apply_bias_correction(
                    short_array, variable, cycle_time
                )
                if not np.isfinite(short_array).all():
                    raise ValueError(
                        f"{variable}: bias correction produced non-finite values"
                    )
            if variable == SSRD_VARIABLE:
                short_array = np.clip(short_array, 0.0, None).astype(
                    np.float16, copy=False
                )
                long_array = np.clip(long_array, 0.0, None)

            # Long-lead MOS: reuse the short-window HOD maps/shrink on
            # hours 0..360 (hours 0..48 are then overwritten by the short
            # prefix, which carries its own single application).
            if variable in BIAS_LONG_VARIABLES and variable != SSRD_VARIABLE:
                long_array = apply_bias_correction(
                    long_array, variable, cycle_time
                )
                if not np.isfinite(long_array).all():
                    raise ValueError(
                        f"{variable}: long bias correction non-finite"
                    )

            # Long challenge still scores hours 0..48. Those hours on the
            # stale cube are worse than the 49h artifact (fresh blend + MOS
            # / IFS 3h SSRD). Copy the short recipe into the long prefix so
            # both windows serve the same 0-48h field. Hours 49..360 stay
            # on the stale cube (+ IFS wind ramp after 72h).
            long_array = np.ascontiguousarray(long_array)
            long_array[:SHORT_HOURS] = np.ascontiguousarray(
                short_array[:SHORT_HOURS]
            )

            for hours, window, array, source_time in (
                (361, LONG_CHALLENGE, long_array, run_time),
                (49, SHORT_CHALLENGE, short_array, short_source_time),
            ):
                compressed = compress_prediction(array)
                writer.write_artifact(
                    make_state_key(variable, window[0], window[1]),
                    compressed,
                    shape=array.shape,
                    dtype="float16",
                    variable=variable,
                    requested_hours=hours,
                    commitment_hash=prediction_hash(compressed, hotkey),
                    source_valid_time_utc=source_time.isoformat(),
                )
            del long_array, short_array
        manifest = writer.finalize(
            metadata={
                "model": "aifs_ens_mean+downscaler_v2+zenith_ssrd",
                "hotkey": hotkey,
                "fallback": False,
                "source_run_utc": run_time.isoformat(),
                "staleness_hours": delta,
                "padded_tail_hours": delta,
                "short_source": (
                    {"run": run_stamp(fresh_run), "mode": FRESH_SHORT_MODE}
                    if use_fresh
                    else "stale"
                ),
                "short_ssrd_source": (
                    "ifs3h_blend_60" if short_ssrd is not None
                    else ("fresh_blend" if use_fresh else "stale")
                ),
                "long_prefix_from_short": True,
                "long_bias_vars": list(BIAS_LONG_VARIABLES),
                "long_tail_composite_days": TAIL_COMPOSITE_DAYS,
                "bias_corrector": (
                    {
                        "path": BIAS_CORRECTOR_PATH,
                        "mode": BIAS_MODE,
                        "shrink": BIAS_SHRINK,
                        "variables": list(BIAS_VARIABLES),
                    }
                    if get_bias_corrector() is not None
                    else None
                ),
                "source_variables": {},
            }
        )
    logger.info(
        "Published %s: %d artifacts in %.1f min",
        manifest["cycle_key"],
        len(manifest["artifacts"]),
        (time.monotonic() - started) / 60.0,
    )
    return True


# --------------------------------------------------------------------------
# Daemon
# --------------------------------------------------------------------------

def daemon(store, hotkey: str, expected_state_keys: tuple[str, ...]) -> None:
    worker: threading.Thread | None = None
    last_probe = 0.0
    last_prune = 0.0

    logger.info("Daemon started | store=%s | hotkey=%s", store.directory, hotkey)
    while True:
        now = datetime.now(timezone.utc)
        try:
            cycle = cycle_in_publish_window(now)
            if cycle is not None:
                publish_bundle(store, cycle, hotkey, expected_state_keys)
        except Exception:
            logger.exception("Publisher iteration failed")

        try:
            if (worker is None or not worker.is_alive()) and (
                time.monotonic() - last_probe > PROBE_INTERVAL_SECONDS
            ):
                last_probe = time.monotonic()
                worker = threading.Thread(
                    target=_safe_pipeline_step, name="pipeline", daemon=True
                )
                worker.start()
        except Exception:
            logger.exception("Pipeline scheduling failed")

        if time.monotonic() - last_prune > 3600:
            last_prune = time.monotonic()
            try:
                removed = store.prune()
                if removed:
                    logger.info("Store prune removed %s", removed)
            except Exception:
                logger.exception("Store prune failed")

        time.sleep(60)


def _safe_pipeline_step() -> None:
    try:
        pipeline_step()
    except Exception:
        logger.exception("Pipeline step failed")


def main() -> int:
    configure_logging()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--store", default=str(DEFAULT_STORE))
    parser.add_argument(
        "--retention-days",
        type=int,
        default=int(os.environ.get("ZEUS_FORECAST_RETENTION_DAYS", "24")),
    )
    parser.add_argument("--wallet-name", default=os.environ.get("WALLET_NAME", ""))
    parser.add_argument("--wallet-hotkey", default=os.environ.get("WALLET_HOTKEY", ""))
    parser.add_argument(
        "--hotkey-ss58",
        default=os.environ.get("ZEUS_HOTKEY_SS58", ""),
        help="Override the hotkey address (skips reading the wallet keyfile).",
    )
    parser.add_argument(
        "--build-cube",
        metavar="RUN_STAMP",
        help="Build the hourly cube for an already-fetched run, then exit.",
    )
    parser.add_argument(
        "--publish-cycle",
        metavar="CYCLE_STAMP",
        help="Publish one bundle for the given cycle (e.g. 20260829T180000Z), then exit.",
    )
    parser.add_argument(
        "--once", action="store_true", help="One pipeline step, then exit."
    )
    args = parser.parse_args()

    if args.build_cube:
        build_hourly_cube(RUN_ROOT / args.build_cube)
        return 0
    if args.once:
        pipeline_step()
        return 0

    if args.hotkey_ss58:
        hotkey = args.hotkey_ss58
    else:
        if not args.wallet_name or not args.wallet_hotkey:
            parser.error("Provide --hotkey-ss58 or --wallet-name/--wallet-hotkey.")
        hotkey = load_hotkey_ss58(args.wallet_name, args.wallet_hotkey)

    from forecast.forecast_store import ForecastStore
    from zeus.validator.constants import CHALLENGE_REGISTRY

    store = ForecastStore(directory=args.store, retention_days=args.retention_days)
    expected_state_keys = tuple(sorted(CHALLENGE_REGISTRY))

    if args.publish_cycle:
        publish_bundle(store, parse_stamp(args.publish_cycle), hotkey, expected_state_keys)
        return 0

    daemon(store, hotkey, expected_state_keys)
    return 0


if __name__ == "__main__":
    sys.exit(main())

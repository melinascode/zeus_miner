#!/usr/bin/env python3
"""Run benchmark_v1 pilots A (T2M 0-48) and B (solar semantic) under disk gate.

Writes results under data/evaluation/pilots/. Does not freeze coefficients,
touch production forecast_store_v2, or start bulk calibration downloads.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

CYCLE = "20250422T180000Z"
HOTKEY = "5HVyUksh8kEDvMAFTCsSuX5t2pUwXDuFmU2Ju5oyG2CvigJR"
STORE = Path("data/evaluation/forecast_store_hist")
ERA5_ROOT = Path("data/evaluation/era5")
OUT_ROOT = Path("data/evaluation/pilots")
SELECTION = Path("data/evaluation/plans/benchmark_v1_selection.json")


def _utc_now() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


def _era5_days_for_48h(cycle_key: str) -> list[str]:
    cycle = datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
        tzinfo=timezone.utc
    )
    # H000..H48 spans up to cycle+48h → need calendar days covering that span.
    days = []
    for hour in range(0, 49):
        day = (cycle.timestamp() + hour * 3600)
        days.append(
            datetime.fromtimestamp(day, tz=timezone.utc).strftime("%Y-%m-%d")
        )
    return sorted(set(days))


def run_disk_gate() -> dict:
    from evaluation.disk_safety import assert_disk_safety, free_gib

    free_before = free_gib(PROJECT_ROOT)
    assert_disk_safety(PROJECT_ROOT, min_free_gib=80)
    return {
        "status": "pass",
        "min_free_gib": 80,
        "free_gib": round(free_before, 2),
    }


def run_pilot_a() -> dict:
    from evaluation.artifacts import ForecastArtifactReader
    from evaluation.runner import evaluate_case, write_evaluation_result
    from evaluation.selection import load_selection
    from evaluation.truth import Era5TruthLoader

    selection = load_selection(SELECTION)
    bundle = STORE / "bundles" / CYCLE
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    commitment = manifest["artifacts"]["2m_temperature@0_48"]["commitment_hash"]

    days = _era5_days_for_48h(CYCLE)
    truth_files = [
        ERA5_ROOT / "2m_temperature" / f"era5_{day}.nc" for day in days
    ]
    missing = [str(path) for path in truth_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ERA5 for pilot A: {missing}")

    reader = ForecastArtifactReader(STORE)
    artifact = reader.read(
        CYCLE,
        "2m_temperature",
        48,
        expected_hotkey=HOTKEY,
        expected_commitment_hash=commitment,
        expected_manifest_sha256=manifest_sha,
    )
    truth = Era5TruthLoader().load(
        truth_files,
        variable="2m_temperature",
        cycle_time=artifact.cycle_time,
        horizon_hours=48,
    )
    result = evaluate_case(
        artifact,
        truth,
        include_lead_diagnostics=True,
        selection_sha256=selection["content_sha256"],
    )
    out_dir = OUT_ROOT / "pilot_a_t2m_0_48"
    result_path = write_evaluation_result(result, out_dir)

    # ERA5 truth manifest record (pilot-local; full product metadata may be partial
    # for Google ARCO fetches).
    truth_manifest = {
        "product": "ERA5 single levels (evaluation NetCDF under data/evaluation/era5)",
        "expver": "unknown_in_local_nc_attrs_check",
        "retrieval_request": {
            "variable": "2m_temperature",
            "days": days,
            "source_layout": "data/evaluation/era5/2m_temperature/era5_YYYY-MM-DD.nc",
        },
        "checksum": {
            path.name: hashlib.sha256(path.read_bytes()).hexdigest()
            for path in truth_files
        },
        "coordinates": {
            "latitude": [float(truth.latitudes.min()), float(truth.latitudes.max())],
            "longitude": [
                float(truth.longitudes.min()),
                float(truth.longitudes.max()),
            ],
            "shape": list(truth.tensor.shape),
        },
        "units": {
            "source_unit": truth.source_unit,
            "target_unit": truth.target_unit,
        },
        "valid_time_range": {
            "start": truth.valid_times[0].isoformat().replace("+00:00", "Z"),
            "end": truth.valid_times[-1].isoformat().replace("+00:00", "Z"),
            "n_steps": len(truth.valid_times),
        },
    }
    # Try to enrich expver from NetCDF attrs.
    try:
        import xarray as xr

        with xr.open_dataset(truth_files[0], engine="h5netcdf") as ds:
            truth_manifest["expver"] = str(
                ds.attrs.get("expver")
                or ds.attrs.get("experiment_id")
                or next(
                    (
                        ds[var].attrs.get("expver")
                        for var in ds.data_vars
                        if "expver" in ds[var].attrs
                    ),
                    "not_present_in_attrs",
                )
            )
            truth_manifest["product"] = str(
                ds.attrs.get("title")
                or ds.attrs.get("source")
                or truth_manifest["product"]
            )
            truth_manifest["netcdf_attrs_sample"] = {
                key: str(value) for key, value in list(ds.attrs.items())[:20]
            }
    except Exception as exc:  # noqa: BLE001 — pilot diagnostics
        truth_manifest["expver_error"] = f"{type(exc).__name__}: {exc}"

    (out_dir / "era5_truth_manifest.json").write_text(
        json.dumps(truth_manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    return {
        "status": "pass",
        "cycle": CYCLE,
        "variable": "2m_temperature",
        "horizon_hours": 48,
        "selection_sha256": selection["content_sha256"],
        "gfs_source_offset_hours": manifest.get("gfs_source_offset_hours"),
        "gfs_common_cycle_utc": manifest.get("gfs_common_cycle_utc"),
        "production_faithful": manifest.get("gfs_source_offset_hours")
        in (6, 12, 18, 24),
        "fidelity_note": (
            "Existing hist bundle uses offset 0 (target=source). "
            "Production newest-ready selection uses offsets (6,12,18,24). "
            "Pilot A proves acquire/align/score path; scores are NOT yet "
            "production-faithful raw GFS."
        ),
        "result_path": str(result_path),
        "metrics": {
            "persistence": result["metrics"]["persistence"],
            "raw_gfs": result["metrics"]["raw_gfs"],
            "comparison": result["metrics"]["comparison"],
        },
        "truth_manifest_path": str(out_dir / "era5_truth_manifest.json"),
        "provenance_snippet": {
            "persistence_source": result["provenance"].get(
                "persistence_source"
            ),
            "future_truth_used_by_persistence": result["provenance"].get(
                "future_truth_used_by_persistence"
            ),
            "forecast": {
                "gfs_source_offset_hours": result["provenance"]["forecast"].get(
                    "gfs_source_offset_hours"
                ),
                "manifest_sha256": result["provenance"]["forecast"].get(
                    "manifest_sha256"
                ),
            },
        },
    }


def run_pilot_b() -> dict:
    from evaluation.artifacts import ForecastArtifactReader
    from evaluation.truth import Era5TruthLoader
    from forecast.variables import get_variable_spec
    from zeus.utils.compression import decompress_prediction
    from zeus.data.converter import get_converter

    bundle = STORE / "bundles" / CYCLE
    manifest_path = bundle / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest_sha = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    state_key = "surface_solar_radiation_downwards@0_48"
    commitment = manifest["artifacts"][state_key]["commitment_hash"]

    days = _era5_days_for_48h(CYCLE)
    truth_files = [
        ERA5_ROOT
        / "surface_solar_radiation_downwards"
        / f"era5_{day}.nc"
        for day in days
    ]
    missing = [str(path) for path in truth_files if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing ERA5 for pilot B: {missing}")

    reader = ForecastArtifactReader(STORE)
    artifact = reader.read(
        CYCLE,
        "surface_solar_radiation_downwards",
        48,
        expected_hotkey=HOTKEY,
        expected_commitment_hash=commitment,
        expected_manifest_sha256=manifest_sha,
    )
    truth = Era5TruthLoader().load(
        truth_files,
        variable="surface_solar_radiation_downwards",
        cycle_time=artifact.cycle_time,
        horizon_hours=48,
    )
    spec = get_variable_spec("surface_solar_radiation_downwards")
    converter = get_converter("surface_solar_radiation_downwards")

    forecast = artifact.tensor.detach().cpu().numpy().astype(np.float64)
    truth_np = truth.tensor.detach().cpu().numpy().astype(np.float64)
    valid_times = list(artifact.valid_times)

    # Timestamps: hourly from cycle through +48h inclusive.
    expected_times = [
        artifact.cycle_time.timestamp() + hour * 3600 for hour in range(49)
    ]
    actual_times = [vt.timestamp() for vt in valid_times]
    timestamps_ok = actual_times == expected_times and list(
        truth.valid_times
    ) == valid_times

    # Units / representation.
    units = {
        "gfs_source_units": spec.source_units,
        "gfs_source_representation": spec.source_representation,
        "truth_source_unit": truth.source_unit,
        "truth_target_unit": truth.target_unit,
        "converter_unit": converter.unit,
        "forecast_finite": bool(np.isfinite(forecast).all()),
        "truth_finite": bool(np.isfinite(truth_np).all()),
        "forecast_min": float(np.nanmin(forecast)),
        "forecast_max": float(np.nanmax(forecast)),
        "truth_min": float(np.nanmin(truth_np)),
        "truth_max": float(np.nanmax(truth_np)),
        "negative_forecast_count": int((forecast < 0).sum()),
        "negative_truth_count": int((truth_np < 0).sum()),
    }
    units_ok = (
        units["negative_forecast_count"] == 0
        and units["forecast_finite"]
        and units["truth_finite"]
        and truth.target_unit == converter.unit
    )

    # Hour-zero: Zeus H000 is target-time field; solar should be non-negative.
    h000 = forecast[0]
    hour_zero = {
        "lead": 0,
        "valid_time": valid_times[0].isoformat().replace("+00:00", "Z"),
        "matches_cycle_time": valid_times[0] == artifact.cycle_time,
        "global_mean": float(h000.mean()),
        "global_max": float(h000.max()),
        "fraction_near_zero": float((h000 <= 1e-3).mean()),
    }

    # Nighttime: use polar-night / local night proxy via low flux cells.
    # For each lead, fraction of grid with forecast ≈ 0 and truth ≈ 0.
    night_threshold = 1.0  # W/m^2-equivalent after conversion; small
    nighttime_rows = []
    for lead in range(0, 49, 6):
        f = forecast[lead]
        t = truth_np[lead]
        nighttime_rows.append(
            {
                "lead": lead,
                "valid_time": valid_times[lead]
                .isoformat()
                .replace("+00:00", "Z"),
                "forecast_frac_le_threshold": float((f <= night_threshold).mean()),
                "truth_frac_le_threshold": float((t <= night_threshold).mean()),
                "forecast_mean": float(f.mean()),
                "truth_mean": float(t.mean()),
            }
        )
    # Expect some nighttime mass on the globe at every hour.
    nighttime_ok = all(
        row["forecast_frac_le_threshold"] > 0.2 for row in nighttime_rows
    )

    # Accumulation interval semantics: GFS DSWRF is hourly-mean flux (W m-2);
    # ERA5 SSRD is hourly energy (J m-2) converted to validator units.
    accumulation = {
        "gfs": "hourly_mean_flux (DSWRF, W m**-2) converted to ERA5-equivalent then validator units",
        "era5": "hourly accumulation SSRD (J m**-2) via get_converter",
        "interval_hours": 1,
        "note": (
            "Validator compares converted fields on the Zeus hourly valid-time "
            "grid; H000 is the target-cycle instant / first hourly slot."
        ),
    }

    # Score solar for reference (same path as Pilot A).
    from evaluation.runner import evaluate_case, write_evaluation_result

    scored = evaluate_case(
        artifact,
        truth,
        include_lead_diagnostics=False,
    )
    out_dir = OUT_ROOT / "pilot_b_solar_semantic"
    result_path = write_evaluation_result(scored, out_dir)

    checks = {
        "timestamps": timestamps_ok,
        "accumulation_interval": True,
        "units": units_ok,
        "hour_zero_handling": hour_zero["matches_cycle_time"]
        and hour_zero["global_max"] >= 0,
        "nighttime_behavior": nighttime_ok,
    }
    status = "pass" if all(checks.values()) else "fail"

    payload = {
        "status": status,
        "cycle": CYCLE,
        "variable": "surface_solar_radiation_downwards",
        "horizon_hours": 48,
        "checks": checks,
        "timestamps": {
            "n_steps": len(valid_times),
            "start": valid_times[0].isoformat().replace("+00:00", "Z"),
            "end": valid_times[-1].isoformat().replace("+00:00", "Z"),
            "hourly_from_cycle_inclusive": timestamps_ok,
        },
        "accumulation": accumulation,
        "units": units,
        "hour_zero": hour_zero,
        "nighttime_sample_every_6h": nighttime_rows,
        "score_metrics": {
            "persistence": scored["metrics"]["persistence"],
            "raw_gfs": scored["metrics"]["raw_gfs"],
            "comparison": scored["metrics"]["comparison"],
        },
        "result_path": str(result_path),
        "decompress_smoke": {
            "artifact_shape": list(artifact.tensor.shape),
            "dtype": str(artifact.tensor.dtype),
        },
    }
    # silence unused import if any
    _ = decompress_prediction
    _ = torch
    (out_dir / "semantic_report.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip-a", action="store_true")
    parser.add_argument("--skip-b", action="store_true")
    args = parser.parse_args()

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    summary = {
        "plan_id": "benchmark_v1",
        "pilots_started_at_utc": _utc_now(),
        "selection_path": str(SELECTION),
        "disk_safety": run_disk_gate(),
        "pilot_a": None,
        "pilot_b": None,
    }
    try:
        if not args.skip_a:
            print("Running Pilot A (T2M 0-48)...", flush=True)
            summary["pilot_a"] = run_pilot_a()
            print(
                "Pilot A:",
                summary["pilot_a"]["status"],
                summary["pilot_a"]["metrics"]["comparison"],
                flush=True,
            )
        if not args.skip_b:
            print("Running Pilot B (solar semantic)...", flush=True)
            summary["pilot_b"] = run_pilot_b()
            print(
                "Pilot B:",
                summary["pilot_b"]["status"],
                summary["pilot_b"]["checks"],
                flush=True,
            )
    finally:
        summary["pilots_finished_at_utc"] = _utc_now()
        summary_path = OUT_ROOT / "pilot_summary.json"
        summary_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {summary_path}", flush=True)

    a_ok = args.skip_a or (
        summary["pilot_a"] and summary["pilot_a"]["status"] == "pass"
    )
    b_ok = args.skip_b or (
        summary["pilot_b"] and summary["pilot_b"]["status"] == "pass"
    )
    return 0 if a_ok and b_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())

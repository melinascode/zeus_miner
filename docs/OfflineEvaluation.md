# Validator-faithful offline evaluation

The offline evaluator reads immutable ForecastStore v2 artifacts and existing
local ERA5 NetCDF files. It does not download data, generate forecasts, modify
the forecast store, or interact with the miner/validator processes.

## Scoring behavior

For each complete `(time, latitude, longitude)` tensor, the evaluator calls the
validator's `custom_rmse` and `custom_mae` functions directly. It reproduces:

- float16 forecast decompression followed by float32 scoring;
- pole-corrected latitude weights;
- Europe weighting and the timestamp-dependent Germany weighting;
- one joint mean over time, latitude, and longitude;
- `combined_error = (RMSE + MAE) / 2`;
- strict shape and finite-value penalties.

Per-lead metrics are labeled diagnostics and are not validator scores.

## Ground truth

Supply hourly files from the CDS `reanalysis-era5-single-levels` product.
Expected NetCDF variables are `t2m`, `u100`, `v100`, and `ssrd`. Coordinates
must exactly match the ascending Zeus grid:

- latitude: `-90 ... 90`, 721 points;
- longitude: `-180 ... 179.75`, 1440 points;
- resolution: 0.25 degrees.

The loader performs the same longitude normalization and sorting as the
validator, then fails if times or coordinates do not match exactly. Solar
radiation is converted from hourly ERA5 `J m-2` to `W m-2` by dividing by
3600. Native unit metadata is mandatory; already-converted or unlabeled files
are rejected.

## Run one paired evaluation

Use the existing scientific environment:

```bash
PYTHONPATH=/Zeus /root/miniconda3/envs/zeus-fourvar/bin/python \
  tools/evaluate_forecast_bundle.py \
  --store-dir data/forecast_store_v2 \
  --cycle 20260728T000000Z \
  --variable 2m_temperature \
  --horizon 48 \
  --expected-commitment-hash <trusted-on-chain-sha256> \
  --expected-manifest-sha256 <trusted-manifest-sha256> \
  --truth-file /path/to/era5_2026-07-28.nc \
  --truth-file /path/to/era5_2026-07-29.nc \
  --truth-file /path/to/era5_2026-07-30.nc
```

The commitment and manifest hashes must come from trusted sources independent
of the local manifest. The commitment normally comes from the accepted
on-chain value; the manifest digest attests the model/source metadata that the
on-chain payload hash does not cover. Fallback bundles and source-cycle
misalignment are rejected.

Outputs are written atomically beneath:

```text
data/evaluation/<cycle>/<state-key>/<evaluation-id>/evaluation.json
```

Persistence repeats the raw GFS H000 field. ERA5 truth is never used as a
baseline input, and both candidates are scored against the same truth tensor
and valid-time sequence.

## 30-cycle three-model benchmark

The locked scientific plan lives at
`data/evaluation/plans/benchmark_v1_selection.json` (content SHA-256 recorded
in-file). Calibration issues are `20250101T000000Z`–`20250331T180000Z`. The
30 independent test cycles start at `20250422T180000Z` and step by 366 hours to
`20260709T000000Z`. Mutable cycle status is recorded only in
`benchmark_v1_registry.json`; never edit the selection file after locking.

Calibrated GFS uses a frozen additive lead-hour bias stratified by synoptic
hour, estimated on the calibration period with latitude weights only:

```text
b[V,h,c] = mean over calib(c) of latitude-weighted (Y - F_raw)
F_cal = F_raw + b[V,h,c]
solar: F_cal = max(0, F_cal)
```

Historical GFS and ERA5 for this study must be written under
`data/evaluation/` only (`forecast_store_hist`, `era5`, `gfs_cache`). Do not
write into `data/forecast_store_v2` or the validator ERA5 cache.

```bash
# Dry-run ERA5 plan
PYTHONPATH=/Zeus /root/miniconda3/envs/zeus-fourvar/bin/python \
  tools/fetch_era5_evaluation.py \
  --start-date 2025-01-01 --end-date 2025-04-15 --dry-run

# Historical native GFS bundle (evaluation store only)
PYTHONPATH=/Zeus /root/miniconda3/envs/zeus-fourvar/bin/python \
  tools/build_historical_gfs_bundle.py \
  --target-cycle 20250422T180000Z \
  --hotkey <hotkey>

# Fit frozen coefficients (after calib GFS+ERA5 exist)
PYTHONPATH=/Zeus /root/miniconda3/envs/zeus-fourvar/bin/python \
  tools/run_calibration_fit.py \
  --selection data/evaluation/plans/benchmark_v1_selection.json \
  --commitment-map /path/to/calib-commitment-map.json

# Three-model backtest
PYTHONPATH=/Zeus /root/miniconda3/envs/zeus-fourvar/bin/python \
  tools/run_evaluation_backtest.py \
  --plan /path/to/test-plan.json \
  --selection data/evaluation/plans/benchmark_v1_selection.json \
  --coefficients data/evaluation/plans/benchmark_v1_calibrated_gfs_coefficients.json \
  --store-dir data/evaluation/forecast_store_hist \
  --minimum-cycles 30 \
  --require-full-matrix \
  --require-independent \
  --continue-on-error
```

W&B group for three-model runs: `validator-faithful-benchmark-v1`.

## Multi-cycle backtest

Create a JSON plan containing explicit, immutable cases:

```json
{
  "cases": [
    {
      "cycle": "20260728T000000Z",
      "variable": "2m_temperature",
      "horizon": 48,
      "commitment_hash": "<trusted-on-chain-sha256>",
      "manifest_sha256": "<trusted-manifest-sha256>",
      "truth_files": [
        "/path/to/era5_2026-07-28.nc",
        "/path/to/era5_2026-07-29.nc",
        "/path/to/era5_2026-07-30.nc"
      ]
    }
  ]
}
```

Run the plan with:

```bash
PYTHONPATH=/Zeus /root/miniconda3/envs/zeus-fourvar/bin/python \
  tools/run_evaluation_backtest.py \
  --plan /path/to/evaluation-plan.json \
  --minimum-cycles 30 \
  --require-full-matrix \
  --require-independent
```

`--require-full-matrix` requires all four variables and both horizons for every
cycle. `--require-independent` rejects overlapping truth windows. For the
360-hour horizon, starts must therefore be more than 360 hours apart; on the
six-hour Zeus schedule this means at least 366 hours.

## W&B

Add `--wandb` to the evaluation command. Numerical evaluation remains in the
scientific environment, then the CLI launches the reporter with:

```text
/root/miniconda3/envs/zeus-eval/bin/python
```

The reporter uses:

- entity: `castelcasey79-marzoni-s-brick-oven-brewing`
- project: `zeus-forecast-evaluation`
- group: `validator-faithful-backtest-v1`

Use `--wandb-mode offline` when network logging is not wanted. All local W&B
files are placed beneath the evaluation result directory.

An existing result can also be logged separately:

```bash
/root/miniconda3/envs/zeus-eval/bin/python \
  tools/log_evaluation_wandb.py \
  --result data/evaluation/<cycle>/<state-key>/evaluation.json
```

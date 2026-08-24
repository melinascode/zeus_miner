"""Pack an ENS-mean .npy cycle into the serving bundle.npz layout.

Primary t2m/u100/v100 are the ensemble mean. AIFS Single is stored as
fallback, plus SSRD from the Single GRIB when present. A sidecar records
the downscaler checkpoint the miner should apply after hourly interpolation.

Usage:
  python tools/build_ens_npy_bundle.py --cycle 20260727T000000Z \\
      --checkpoint /Zeus/data/evaluation/training/aifs_downscaler_v2_ens.pt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import importlib.util
import numpy as np

_bundle_path = Path(__file__).resolve().parent / "build_ecmwf_bundle.py"
_spec = importlib.util.spec_from_file_location("build_ecmwf_bundle", _bundle_path)
_bundle = importlib.util.module_from_spec(_spec)
assert _spec.loader is not None
_spec.loader.exec_module(_bundle)
STEPS = _bundle.STEPS
read_grib = _bundle.read_grib
stack = _bundle.stack

ENS_ROOT = Path("/Zeus/data/evaluation/aifs_ens_mean")
SINGLE_ROOT = Path("/Zeus/data/evaluation/aifs_single")
OUT_ROOT = Path("/Zeus/data/evaluation/ecmwf_live")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycle", required=True)
    parser.add_argument("--ens-root", type=Path, default=ENS_ROOT)
    parser.add_argument("--single-root", type=Path, default=SINGLE_ROOT)
    parser.add_argument("--out-root", type=Path, default=OUT_ROOT)
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()

    npy_path = args.ens_root / f"{args.cycle}.npy"
    ens = np.load(npy_path)
    if ens.shape != (len(STEPS), 3, 721, 1440):
        raise SystemExit(f"unexpected ENS shape {ens.shape}")
    t2m = ens[:, 0].astype(np.float32)
    u100 = ens[:, 1].astype(np.float32)
    v100 = ens[:, 2].astype(np.float32)

    bundle = {
        "steps": STEPS.astype(np.int16),
        "t2m": t2m,
        "u100": u100,
        "v100": v100,
    }

    grib = args.single_root / f"{args.cycle}.grib2"
    if grib.is_file():
        print(f"reading AIFS Single {grib}", flush=True)
        single = read_grib(str(grib), ["2t", "100u", "100v", "ssrd"])
        bundle["t2m_single"] = stack(single["2t"], STEPS)
        bundle["u100_single"] = stack(single["100u"], STEPS)
        bundle["v100_single"] = stack(single["100v"], STEPS)
        if single["ssrd"]:
            ssrd_acc = stack(single["ssrd"], STEPS).astype(np.float64)
            ssrd = (np.diff(ssrd_acc, axis=0) / (6 * 3600.0)).astype(np.float32)
            bundle["ssrd"] = np.clip(ssrd, 0.0, None)

    out_dir = args.out_root / args.cycle
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "bundle.npz"
    np.savez(out_path, **bundle)
    meta = {
        "cycle": args.cycle,
        "primary": "aifs-ens-mean",
        "downscaler_checkpoint": args.checkpoint,
        "apply_downscaler": "hourly after linear interpolation of 6h steps",
        "keys": {k: list(v.shape) if hasattr(v, "shape") else None for k, v in bundle.items()},
    }
    (out_dir / "bundle.meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {out_path} ({out_path.stat().st_size / 1e6:.0f} MB)")
    for key, value in bundle.items():
        if hasattr(value, "shape"):
            print(f"  {key:12s} {value.shape} {value.dtype}")
    print(f"wrote {out_dir / 'bundle.meta.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

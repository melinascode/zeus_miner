"""Decide serving composition per horizon: ENS-mean vs Single + specialist.

For each ENS validation cycle and lead, scores over the domain's official
(loss-region-masked) metric:

  ens             interpolated ENS-mean            (what we'd serve without specialists)
  single          interpolated AIFS Single         (specialist's native input)
  single_corr     Single + specialist correction   (matched input, as trained)
  ens_plus_corr   ENS  + Single-derived correction (additive transfer)
  ens_naive_corr  ENS fed through the Single-trained net (the failed path)

Usage:
  PYTHONPATH=/Zeus python tools/score_specialist_composition.py \
      --domain germany --checkpoint .../germany_resunet_pre.pt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, "/Zeus")

from zeus_ml.models.europe_resunet import EuropeResUNet
from zeus_ml.models.germany_resunet import GermanyResUNet
from zeus_ml.train.train_europe_resunet import EuropeCropDataset, build_split

METHODS = ("ens", "single", "single_corr", "ens_plus_corr", "ens_naive_corr")
VAR_WEIGHTS = np.array([0.25, 0.375, 0.375])


def combined(pred: torch.Tensor, truth: torch.Tensor, w: torch.Tensor) -> np.ndarray:
    err = pred - truth
    rmse = torch.sqrt((err.square() * w).mean(dim=(-2, -1)))
    mae = (err.abs() * w).mean(dim=(-2, -1))
    return ((rmse + mae) / 2.0)[0].numpy()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--domain", choices=("germany", "europe"), default="germany")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--data-root", default="/Zeus/data/evaluation/europe_crops")
    ap.add_argument(
        "--statistics",
        default="/Zeus/data/evaluation/training/aifs_downscaler_v2_ens.statistics.json",
    )
    ap.add_argument("--leads", default="0,6,12,24,36,48,72,120,168,240,300,360")
    ap.add_argument("--threads", type=int, default=8)
    args = ap.parse_args()
    torch.set_num_threads(args.threads)

    stats = json.loads(Path(args.statistics).read_text())
    manifest = json.loads((Path(args.data_root) / "manifest.json").read_text())
    single_all = set(manifest["single_cycles"])
    _, ens_val = build_split(manifest["ens_cycles"], "2026-07-11")
    cycles = [c for c in ens_val if c in single_all]
    print(f"validation cycles ({len(cycles)}): {cycles}", flush=True)

    loss_region = "germany" if args.domain == "germany" else "europe"
    common = dict(
        root=args.data_root,
        statistics=stats,
        domain=args.domain,
        loss_region=loss_region,
        cycles=cycles,
    )
    ds_single = EuropeCropDataset(source="single", **common)
    ds_ens = EuropeCropDataset(source="ens", **common)

    ck = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cls = GermanyResUNet if ck.get("domain") == "germany" else EuropeResUNet
    model = cls(
        base_channels=ck["base_channels"],
        blocks_per_stage=ck["blocks_per_stage"],
        in_channels=ck["in_channels"],
        dropout=ck.get("dropout", 0.0),
    )
    model.load_state_dict(ck["model_state"])
    model.eval()
    scales = torch.tensor(ds_single.residual_std, dtype=torch.float32).view(1, 3, 1, 1)

    leads = [int(x) for x in args.leads.split(",")]
    per_lead: dict[int, dict[str, list[np.ndarray]]] = {
        lead: {m: [] for m in METHODS} for lead in leads
    }
    with torch.no_grad():
        for ci, key in enumerate(cycles):
            for lead in leads:
                item_s = ds_single.build_item(ci, lead, None)
                item_e = ds_ens.build_item(ci, lead, None)
                if item_s is None or item_e is None:
                    continue
                w = item_s["metric_weights"].unsqueeze(0)
                truth = item_s["truth"].unsqueeze(0)
                raw_s = item_s["raw"].unsqueeze(0)
                raw_e = item_e["raw"].unsqueeze(0)
                out_s = model(
                    item_s["features"].unsqueeze(0), item_s["context"].unsqueeze(0)
                )
                corr_s = out_s.gate * out_s.correction * scales
                out_e = model(
                    item_e["features"].unsqueeze(0), item_e["context"].unsqueeze(0)
                )
                corr_e = out_e.gate * out_e.correction * scales
                preds = {
                    "ens": raw_e,
                    "single": raw_s,
                    "single_corr": raw_s + corr_s,
                    "ens_plus_corr": raw_e + corr_s,
                    "ens_naive_corr": raw_e + corr_e,
                }
                for m, p in preds.items():
                    per_lead[lead][m].append(combined(p, truth, w))
            print(f"  cycle {key} done", flush=True)

    def bucket(leads_subset: list[int]) -> dict[str, np.ndarray]:
        out = {}
        for m in METHODS:
            rows = [r for lead in leads_subset for r in per_lead[lead][m]]
            out[m] = np.mean(np.stack(rows), axis=0) if rows else np.full(3, np.nan)
        return out

    print(f"\n=== {args.domain} | official {loss_region}-masked metric ===")
    for name, subset in (
        ("short (<=48h)", [l for l in leads if l <= 48]),
        ("long (>48h)", [l for l in leads if l > 48]),
    ):
        rows = bucket(subset)
        print(f"-- {name} --")
        for m in METHODS:
            r = rows[m]
            print(
                f"  {m:15s} t2m={r[0]:.3f} u100={r[1]:.3f} v100={r[2]:.3f} "
                f"weighted={float((r * VAR_WEIGHTS).sum()):.4f}"
            )
    print("\nper-lead weighted (lower is better):")
    header = "lead  " + " ".join(f"{m:>15s}" for m in METHODS)
    print(header)
    for lead in leads:
        cells = []
        for m in METHODS:
            rows = per_lead[lead][m]
            if rows:
                r = np.mean(np.stack(rows), axis=0)
                cells.append(f"{float((r * VAR_WEIGHTS).sum()):15.4f}")
            else:
                cells.append(f"{'n/a':>15s}")
        print(f"{lead:4d}  " + " ".join(cells))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

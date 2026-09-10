"""Compose ENS-mean + global downscaler + ENS-native t2m specialists.

Serving rules (t2m-only ENS specialists, retrained on 358 daily ENS crops):

  * Everywhere: interpolated ENS-mean + global AIFS downscaler (v2), all vars.
  * Europe crop, t2m only: the ENS-native Europe specialist correction,
    feathered at the crop edge; inside the Germany window the Germany
    specialist correction is feathered over the Europe one.
  * Wind is never touched by specialists: three experiments showed
    deterministic correctors cannot beat the ensemble-mean wind.

Specialists were trained on raw ENS interpolation with the same-day Single
run as the paired forecast, so they are applied here with the same inputs.
Two composition variants exist because the global CNN also corrects t2m:
  composed_add     v2 field + specialist correction (may double-correct)
  composed_replace specialist(ENS) field replaces v2 inside the crop
Score both; serve the winner.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import torch

from zeus_ml.datasets.aifs_downscale_dataset import AifsCycleReader, EnsMeanCycleReader
from zeus_ml.models.aifs_downscaler_cnn import (
    FULL_HEIGHT,
    FULL_WIDTH,
    MAX_LEAD_HOURS,
    DownscalerStatistics,
    aifs_downscaler_from_checkpoint,
    bracket_for_lead,
    build_downscaler_context,
    build_downscaler_static_features,
    load_static_maps,
    zenith_triplet,
)
from zeus_ml.models.europe_resunet import (
    IN_CHANNELS,
    EuropeResUNet,
    build_context,
    cosine_solar_zenith,
    evaluate_climatology,
)
from zeus_ml.models.germany_resunet import (
    GERMANY_HEIGHT,
    GERMANY_LAT_SLICE,
    GERMANY_LON_SLICE,
    GERMANY_WIDTH,
    GermanyResUNet,
)
from zeus_ml.evaluate.evaluate_aifs_downscaler import cycle_maps

# Europe crop on the Zeus 721x1440 grid (tools/preprocess_europe_crops.py).
EUROPE_LAT = slice(472, 680)
EUROPE_LON = slice(560, 928)
EUROPE_HEIGHT = 208
EUROPE_WIDTH = 368
# Germany window inside that crop, also as absolute globe slices.
GERMANY_LAT = slice(472 + 56, 472 + 136)
GERMANY_LON = slice(560 + 152, 560 + 248)

def _feather_2d(height: int, width: int, fade: int = 8) -> torch.Tensor:
    mask = torch.ones((height, width), dtype=torch.float32)
    if fade <= 0:
        return mask
    ramp = torch.linspace(0.0, 1.0, fade)
    mask[:fade] *= ramp[:, None]
    mask[-fade:] *= ramp.flip(0)[:, None]
    mask[:, :fade] *= ramp[None, :]
    mask[:, -fade:] *= ramp.flip(0)[None, :]
    return mask


def _interpolate(steps: np.ndarray, lead: int) -> np.ndarray:
    left, right, fraction = bracket_for_lead(lead)
    return (
        (1.0 - fraction) * steps[left].astype(np.float32)
        + fraction * steps[right].astype(np.float32)
    )


def _zscore(field: np.ndarray, ref: np.ndarray) -> np.ndarray:
    return (field - ref.mean()) / max(float(ref.std()), 1e-6)


@dataclass
class ComposerConfig:
    ens_root: str = "/Zeus/data/evaluation/aifs_ens_mean"
    single_root: str = "/Zeus/data/evaluation/aifs_single"
    crop_static: str = "/Zeus/data/evaluation/europe_crops/static.npz"
    global_static: str = "/Zeus/data/evaluation/training/aifs_static"
    statistics: str = (
        "/Zeus/data/evaluation/training/aifs_downscaler_v2_ens.statistics.json"
    )
    # Override with ZEUS_GLOBAL_CHECKPOINT (bundle builder only). Default is
    # the live v2 ENS CNN; unset/empty keeps this path so a missing flag is a
    # no-op revert.
    global_checkpoint: str = field(
        default_factory=lambda: os.environ.get(
            "ZEUS_GLOBAL_CHECKPOINT",
            "/Zeus/data/evaluation/training/aifs_downscaler_v2_ens.pt",
        )
        or "/Zeus/data/evaluation/training/aifs_downscaler_v2_ens.pt"
    )
    europe_checkpoint: str = (
        "/Zeus/data/evaluation/training/europe_resunet_ens_t2m.pt"
    )
    germany_checkpoint: str = (
        "/Zeus/data/evaluation/training/germany_resunet_ens_t2m.pt"
    )
    device: str = "cpu"


class ForecastComposer:
    """Hourly 721x1440 t2m/u100/v100 composer. Call ``predict(cycle, lead)``."""

    def __init__(self, config: ComposerConfig | None = None) -> None:
        self.config = config or ComposerConfig()
        self.device = torch.device(self.config.device)
        self.ens = EnsMeanCycleReader(self.config.ens_root, cache_size=1)
        self.single = AifsCycleReader(self.config.single_root, cache_size=2)
        stats = json.loads(Path(self.config.statistics).read_text(encoding="utf-8"))
        self.state_mean = np.asarray(stats["state_mean"], np.float32)[:, None, None]
        self.state_std = np.asarray(stats["state_std"], np.float32)[:, None, None]
        self.delta_std = np.asarray(stats["delta_std"], np.float32)[:, None, None]
        self.spec_residual = torch.tensor(
            stats["residual_std"], dtype=torch.float32
        ).view(1, 3, 1, 1)

        crop = np.load(self.config.crop_static)
        land = crop["land"]
        oro = crop["orography"]
        rough = crop["roughness"]
        cosine = crop["cosine_latitude"]
        temp_s = np.log1p(crop["temp_scalar"])
        wind_s = np.log1p(crop["wind_scalar"])
        self.clim = crop["climatology_coefficients"]
        self.lat_eu = torch.from_numpy(np.ascontiguousarray(crop["latitudes"]))
        self.lon_eu = torch.from_numpy(np.ascontiguousarray(crop["longitudes"]))
        self.static_europe = torch.from_numpy(
            np.stack(
                [
                    land,
                    _zscore(oro, oro),
                    _zscore(rough, rough),
                    cosine,
                    _zscore(temp_s, temp_s),
                    _zscore(wind_s, wind_s),
                ]
            ).astype(np.float32)
        )
        de_lat, de_lon = GERMANY_LAT_SLICE, GERMANY_LON_SLICE
        self.static_germany = torch.from_numpy(
            np.stack(
                [
                    land,
                    _zscore(oro, oro[de_lat, de_lon]),
                    _zscore(rough, rough[de_lat, de_lon]),
                    cosine,
                    _zscore(temp_s, temp_s[de_lat, de_lon]),
                    _zscore(wind_s, wind_s[de_lat, de_lon]),
                ]
            ).astype(np.float32)
        )[:, de_lat, de_lon]
        self.clim_germany = self.clim[:, :, de_lat, de_lon]
        self.lat_de = self.lat_eu[de_lat]
        self.lon_de = self.lon_eu[de_lon]
        self.europe_feather = _feather_2d(EUROPE_HEIGHT, EUROPE_WIDTH, 8)
        self.germany_feather = _feather_2d(GERMANY_HEIGHT, GERMANY_WIDTH, 8)

        self.europe = self._load_specialist(
            self.config.europe_checkpoint, EuropeResUNet
        )
        self.germany = self._load_specialist(
            self.config.germany_checkpoint, GermanyResUNet
        )
        self._load_global()
        self.latitudes = torch.linspace(-90.0, 90.0, FULL_HEIGHT)
        self.longitudes = torch.arange(-180.0, 180.0, 0.25)

    def _load_specialist(self, path: str, cls):
        ck = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(
            base_channels=ck["base_channels"],
            blocks_per_stage=ck["blocks_per_stage"],
            in_channels=ck.get("in_channels", IN_CHANNELS),
            dropout=ck.get("dropout", 0.0),
        )
        model.load_state_dict(ck["model_state"])
        model.eval()
        return model.to(self.device)

    def _load_global(self) -> None:
        ck = torch.load(
            self.config.global_checkpoint, map_location="cpu", weights_only=False
        )
        self.g_stats = DownscalerStatistics.from_dict(ck["statistics"])
        self.g_mean, self.g_std, self.g_delta, self.g_resid = self.g_stats.tensors()
        self.g_lagged = bool(ck.get("use_lagged", False))
        self.global_model = aifs_downscaler_from_checkpoint(ck)
        self.global_model.eval()
        self.land, self.orography, self.roughness = load_static_maps(
            self.config.global_static
        )

    def _specialist_features(
        self,
        interpolated: np.ndarray,
        delta: np.ndarray,
        lagged: np.ndarray,
        clim: np.ndarray,
        latitudes: torch.Tensor,
        longitudes: torch.Tensor,
        static: torch.Tensor,
        valid: datetime,
        cycle_time: datetime,
        lead: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        zenith_now = cosine_solar_zenith(latitudes, longitudes, valid)
        zenith_prev = cosine_solar_zenith(
            latitudes, longitudes, valid - timedelta(hours=2)
        )
        weather = np.concatenate(
            [
                (interpolated - self.state_mean) / self.state_std,
                delta / self.delta_std,
                (lagged - self.state_mean) / self.state_std,
                (lagged - interpolated) / self.delta_std,
                (interpolated - clim) / self.state_std,
            ]
        ).astype(np.float32)
        features = torch.cat(
            [
                torch.from_numpy(weather),
                zenith_now.clamp_min(0.0).unsqueeze(0),
                (zenith_now - zenith_prev).unsqueeze(0),
                static,
            ]
        ).unsqueeze(0)
        context = build_context(cycle_time, lead).unsqueeze(0)
        return features.to(self.device), context.to(self.device)

    def _apply_specialist(
        self,
        model: torch.nn.Module,
        ens_crop: np.ndarray,
        single_crop: np.ndarray | None,
        lead: int,
        cycle_time: datetime,
        *,
        germany: bool,
    ) -> np.ndarray:
        """Gated correction (3, H, W) for the ENS interpolation on a crop.

        Matches training exactly: state/delta from ENS-mean steps, the lagged
        pair is the same-day Single run at the same lead (fallback to the
        interpolation itself when Single is missing).
        """
        left, right, _ = bracket_for_lead(lead)
        interpolated = _interpolate(ens_crop, lead)
        delta = ens_crop[right].astype(np.float32) - ens_crop[left].astype(
            np.float32
        )
        lagged = (
            _interpolate(single_crop, lead)
            if single_crop is not None
            else interpolated
        )
        valid = cycle_time + timedelta(hours=lead)
        if germany:
            clim = evaluate_climatology(self.clim_germany, valid)
            latitudes, longitudes = self.lat_de, self.lon_de
            static = self.static_germany
        else:
            clim = evaluate_climatology(self.clim, valid)
            latitudes, longitudes = self.lat_eu, self.lon_eu
            static = self.static_europe
        features, context = self._specialist_features(
            interpolated,
            delta,
            lagged,
            clim,
            latitudes,
            longitudes,
            static,
            valid,
            cycle_time,
            lead,
        )
        with torch.inference_mode():
            out = model(features, context)
            corr = (out.gate * out.correction * self.spec_residual.to(self.device))[0]
        return corr.cpu().numpy()

    def _apply_global(
        self,
        ens: np.ndarray,
        single: np.ndarray | None,
        lead: int,
        cycle_time: datetime,
    ) -> torch.Tensor:
        left, right, fraction = bracket_for_lead(lead)
        interpolated = torch.from_numpy(_interpolate(ens, lead))
        delta = torch.from_numpy(
            ens[right].astype(np.float32) - ens[left].astype(np.float32)
        )
        blocks = [
            (interpolated - self.g_mean) / self.g_std,
            delta / self.g_delta,
        ]
        if self.g_lagged:
            lagged = (
                torch.from_numpy(_interpolate(single, lead))
                if single is not None
                else interpolated.clone()
            )
            blocks += [
                (lagged - self.g_mean) / self.g_std,
                (lagged - interpolated) / self.g_delta,
            ]
        weather = torch.cat(blocks, dim=0).unsqueeze(0)
        zenith, zenith_anomaly = zenith_triplet(
            self.latitudes, self.longitudes, cycle_time, lead
        )
        geographic, _ = cycle_maps(cycle_time, self.latitudes)
        static = build_downscaler_static_features(
            self.latitudes,
            self.longitudes,
            geographic_weights=geographic,
            land_sea=self.land,
            orography=self.orography,
            roughness=self.roughness,
            zenith=zenith,
            zenith_anomaly=zenith_anomaly,
        ).unsqueeze(0)
        context = build_downscaler_context(
            lead_hour=lead,
            cycle_hour=cycle_time.hour,
            day_of_year=cycle_time.timetuple().tm_yday,
            fraction=fraction,
        ).unsqueeze(0)
        with torch.inference_mode():
            output = self.global_model(weather, context, static)
            corrected = interpolated + output.correction[0] * self.g_resid
        return corrected

    def predict_hour(
        self, cycle_key: str, lead: int
    ) -> dict[str, torch.Tensor]:
        """Return physical fields of shape (3, 721, 1440) for several methods."""
        cycle_time = datetime.strptime(cycle_key, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )
        ens = self.ens.get(cycle_key)
        single = (
            self.single.get(cycle_key)
            if self.single.path_for(cycle_key).is_file()
            else None
        )
        ens_i = torch.from_numpy(_interpolate(ens, lead))
        global_cnn = self._apply_global(ens, single, lead, cycle_time)

        ens_eu = ens[:, :, EUROPE_LAT, EUROPE_LON]
        ens_de = ens[:, :, GERMANY_LAT, GERMANY_LON]
        single_eu = single[:, :, EUROPE_LAT, EUROPE_LON] if single is not None else None
        single_de = single[:, :, GERMANY_LAT, GERMANY_LON] if single is not None else None
        eu_corr = self._apply_specialist(
            self.europe, ens_eu, single_eu, lead, cycle_time, germany=False
        )
        de_corr = self._apply_specialist(
            self.germany, ens_de, single_de, lead, cycle_time, germany=True
        )
        # t2m channel only; Germany correction feathered over the Europe one.
        corr_t2m = torch.from_numpy(eu_corr[0].copy())
        fade_de = self.germany_feather
        corr_t2m[GERMANY_LAT_SLICE, GERMANY_LON_SLICE] = (
            (1.0 - fade_de) * corr_t2m[GERMANY_LAT_SLICE, GERMANY_LON_SLICE]
            + fade_de * torch.from_numpy(de_corr[0])
        )
        feathered = self.europe_feather * corr_t2m

        composed_add = global_cnn.clone()
        composed_add[0, EUROPE_LAT, EUROPE_LON] += feathered

        composed_replace = global_cnn.clone()
        specialist_field = ens_i[0, EUROPE_LAT, EUROPE_LON] + corr_t2m
        composed_replace[0, EUROPE_LAT, EUROPE_LON] = (
            (1.0 - self.europe_feather)
            * composed_replace[0, EUROPE_LAT, EUROPE_LON]
            + self.europe_feather * specialist_field
        )

        return {
            "ens_linear": ens_i,
            "global_cnn": global_cnn,
            "composed_add": composed_add,
            "composed_replace": composed_replace,
        }

    def predict_window(
        self, cycle_key: str, leads: range | list[int], method: str = "composed_add"
    ) -> np.ndarray:
        """Stack one method over leads into (T, 3, 721, 1440)."""
        return np.stack(
            [self.predict_hour(cycle_key, lead)[method].numpy() for lead in leads],
            axis=0,
        )

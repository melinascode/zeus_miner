from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

from forecast.forecast_store import ForecastStore
from forecast.variables import canonicalize_variable_name, supported_variables
from zeus.utils.compression import decompress_prediction


SUPPORTED_HORIZONS = (48, 360)
DEFAULT_SPATIAL_SHAPE = (721, 1440)
EXPECTED_GFS_MODEL = "gfs_native_leads_with_linear_hourly_interpolation"


def expected_state_keys() -> tuple[str, ...]:
    return tuple(
        sorted(
            f"{variable}@0_{horizon}"
            for variable in supported_variables()
            for horizon in SUPPORTED_HORIZONS
        )
    )


@dataclass(frozen=True)
class LoadedForecastArtifact:
    cycle_time: datetime
    state_key: str
    variable: str
    horizon_hours: int
    tensor: torch.Tensor
    valid_times: tuple[datetime, ...]
    latitudes: np.ndarray
    longitudes: np.ndarray
    manifest: dict[str, Any]
    artifact_metadata: dict[str, Any]
    manifest_sha256: str
    manifest_authenticated: bool
    payload_sha256: str
    commitment_hash: str
    commitment_authenticated: bool
    hotkey: str


class ForecastArtifactReader:
    """Read and fully verify one immutable ForecastStore artifact."""

    def __init__(
        self,
        store_directory: str | Path = "data/forecast_store_v2",
        *,
        spatial_shape: tuple[int, int] = DEFAULT_SPATIAL_SHAPE,
        require_complete_bundle: bool = True,
    ) -> None:
        self.store_directory = Path(store_directory)
        if not self.store_directory.is_dir():
            raise FileNotFoundError(
                f"Forecast store does not exist: {self.store_directory}"
            )
        if not (self.store_directory / "bundles").is_dir():
            raise FileNotFoundError(
                f"Forecast store has no bundles directory: {self.store_directory}"
            )
        self.spatial_shape = spatial_shape
        self.require_complete_bundle = require_complete_bundle
        self.store = ForecastStore(self.store_directory)

    def parse_cycle(self, value: str | datetime) -> datetime:
        if isinstance(value, datetime):
            return self._require_exact_cycle(value)
        if value == "latest":
            manifest = self.store.load_latest_manifest(
                expected_state_keys=(
                    expected_state_keys()
                    if self.require_complete_bundle
                    else None
                ),
                verify_files=True,
            )
            return self.store.parse_cycle_key(manifest["cycle_key"])
        try:
            return self.store.parse_cycle_key(value)
        except ValueError:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return self._require_exact_cycle(parsed)

    def read(
        self,
        cycle: str | datetime,
        variable: str,
        horizon_hours: int,
        *,
        expected_hotkey: str | None = None,
        expected_model: str = EXPECTED_GFS_MODEL,
        expected_commitment_hash: str | None = None,
        expected_manifest_sha256: str | None = None,
    ) -> LoadedForecastArtifact:
        canonical_variable = canonicalize_variable_name(variable)
        if horizon_hours not in SUPPORTED_HORIZONS:
            raise ValueError(
                f"horizon_hours must be one of {SUPPORTED_HORIZONS}, "
                f"received {horizon_hours}."
            )

        cycle_time = self.parse_cycle(cycle)
        state_key = f"{canonical_variable}@0_{horizon_hours}"
        manifest = self.store.load_manifest(
            cycle_time,
            expected_state_keys=(
                expected_state_keys()
                if self.require_complete_bundle
                else None
            ),
            verify_files=True,
            expected_hotkey=expected_hotkey,
        )
        manifest_path = (
            self.store.bundle_path(cycle_time) / self.store.MANIFEST_FILENAME
        )
        manifest_sha256 = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
        if expected_manifest_sha256 is not None:
            self._validate_sha256(
                expected_manifest_sha256,
                "Expected manifest SHA-256",
            )
            if manifest_sha256 != expected_manifest_sha256:
                raise ValueError("Trusted manifest SHA-256 mismatch.")

        hotkey = manifest.get("hotkey")
        if not isinstance(hotkey, str) or not hotkey:
            raise ValueError("Forecast manifest has no valid hotkey.")
        if expected_hotkey is not None and hotkey != expected_hotkey:
            raise ValueError("Forecast manifest hotkey does not match expected hotkey.")
        if manifest.get("model") != expected_model:
            raise ValueError(
                f"Forecast bundle model is {manifest.get('model')!r}; "
                f"expected {expected_model!r}."
            )
        if manifest.get("fallback") is not False:
            raise ValueError("Fallback bundles cannot be evaluated as raw GFS.")
        self._validate_source_alignment(manifest, cycle_time)

        try:
            metadata = dict(manifest["artifacts"][state_key])
        except KeyError as exc:
            raise FileNotFoundError(
                f"Bundle {manifest.get('cycle_key')} has no {state_key} artifact."
            ) from exc

        expected_shape = (
            horizon_hours + 1,
            self.spatial_shape[0],
            self.spatial_shape[1],
        )
        self._validate_metadata(
            metadata=metadata,
            state_key=state_key,
            variable=canonical_variable,
            expected_shape=expected_shape,
        )

        payload = self.store.load_artifact(
            cycle_time,
            state_key,
            verify=True,
        )
        payload_sha256 = hashlib.sha256(payload).hexdigest()
        commitment_hash = hashlib.sha256(
            payload + hotkey.encode("utf-8")
        ).hexdigest()
        if commitment_hash != metadata["commitment_hash"]:
            raise ValueError(f"Commitment hash mismatch for {state_key}.")
        if expected_commitment_hash is not None:
            self._validate_sha256(
                expected_commitment_hash,
                "Expected commitment hash",
            )
            if commitment_hash != expected_commitment_hash:
                raise ValueError(
                    f"Trusted commitment hash mismatch for {state_key}."
                )

        tensor = decompress_prediction(payload, torch.Size(expected_shape))
        if tensor is None:
            raise ValueError(f"Could not decompress {state_key}.")
        if tensor.shape != torch.Size(expected_shape):
            raise ValueError(
                f"Decompressed {state_key} shape {tuple(tensor.shape)} does not "
                f"match {expected_shape}."
            )
        if tensor.dtype != torch.float16:
            raise TypeError(
                f"Decompressed {state_key} dtype must be float16, "
                f"received {tensor.dtype}."
            )
        if not tensor.is_contiguous():
            raise ValueError(f"Decompressed {state_key} is not contiguous.")
        if not torch.isfinite(tensor).all():
            raise ValueError(f"Decompressed {state_key} contains NaN or Inf.")

        valid_times = tuple(
            cycle_time + timedelta(hours=hour)
            for hour in range(horizon_hours + 1)
        )
        latitudes = np.linspace(
            -90.0,
            90.0,
            self.spatial_shape[0],
            dtype=np.float64,
        )
        longitudes = np.arange(
            self.spatial_shape[1],
            dtype=np.float64,
        )
        if self.spatial_shape == DEFAULT_SPATIAL_SHAPE:
            longitudes = -180.0 + longitudes * 0.25

        return LoadedForecastArtifact(
            cycle_time=cycle_time,
            state_key=state_key,
            variable=canonical_variable,
            horizon_hours=horizon_hours,
            tensor=tensor,
            valid_times=valid_times,
            latitudes=latitudes,
            longitudes=longitudes,
            manifest=manifest,
            artifact_metadata=metadata,
            manifest_sha256=manifest_sha256,
            manifest_authenticated=expected_manifest_sha256 is not None,
            payload_sha256=payload_sha256,
            commitment_hash=commitment_hash,
            commitment_authenticated=expected_commitment_hash is not None,
            hotkey=hotkey,
        )

    def _require_exact_cycle(self, value: datetime) -> datetime:
        if value.tzinfo is None:
            utc_value = value.replace(tzinfo=timezone.utc)
        else:
            utc_value = value.astimezone(timezone.utc)
        if (
            utc_value.hour % 6 != 0
            or utc_value.minute != 0
            or utc_value.second != 0
            or utc_value.microsecond != 0
        ):
            raise ValueError(
                "Cycle must be an exact UTC 00/06/12/18 boundary; "
                f"received {value.isoformat()}."
            )
        return utc_value

    @staticmethod
    def _validate_source_alignment(
        manifest: dict[str, Any],
        cycle_time: datetime,
    ) -> None:
        metadata_values = manifest.get("artifacts", {}).values()
        for metadata in metadata_values:
            source_valid_time = metadata.get("source_valid_time_utc")
            if not isinstance(source_valid_time, str):
                raise ValueError("Artifact has no source_valid_time_utc.")
            if ForecastArtifactReader._parse_utc(source_valid_time) != cycle_time:
                raise ValueError(
                    "Artifact source_valid_time_utc does not match target cycle."
                )

        gfs_cycle_value = manifest.get("gfs_common_cycle_utc")
        source_offset = manifest.get("gfs_source_offset_hours")
        if not isinstance(gfs_cycle_value, str):
            raise ValueError("Manifest has no gfs_common_cycle_utc.")
        if (
            not isinstance(source_offset, int)
            or isinstance(source_offset, bool)
            or source_offset < 0
        ):
            raise ValueError("Manifest has no valid gfs_source_offset_hours.")
        gfs_cycle = ForecastArtifactReader._parse_utc(gfs_cycle_value)
        if gfs_cycle + timedelta(hours=source_offset) != cycle_time:
            raise ValueError(
                "GFS source cycle plus offset does not equal target cycle."
            )

    @staticmethod
    def _parse_utc(value: str) -> datetime:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)

    @staticmethod
    def _validate_sha256(value: str, label: str) -> None:
        if len(value) != 64 or any(
            character not in "0123456789abcdef"
            for character in value
        ):
            raise ValueError(f"{label} is not lowercase SHA-256.")

    @staticmethod
    def _validate_metadata(
        *,
        metadata: dict[str, Any],
        state_key: str,
        variable: str,
        expected_shape: tuple[int, int, int],
    ) -> None:
        expected = {
            "filename": f"{state_key}.bin",
            "variable": variable,
            "requested_hours": expected_shape[0],
            "shape": list(expected_shape),
            "dtype": "float16",
            "compression": "blosc2:zstd:bitshuffle:clevel9",
        }
        for key, expected_value in expected.items():
            if metadata.get(key) != expected_value:
                raise ValueError(
                    f"Artifact {state_key} metadata {key!r} is "
                    f"{metadata.get(key)!r}; expected {expected_value!r}."
                )

from __future__ import annotations

import hashlib
import importlib.util
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from forecast.forecast_store import ForecastStore
from forecast.gfs_provider import GFSWeatherDataProvider
from forecast.service import ForecastService
from forecast.variables import (
    VARIABLE_SPECS,
    convert_gfs_to_zeus_target,
    get_variable_spec,
    supported_variables,
)
_compression_spec = importlib.util.spec_from_file_location(
    "zeus_compression_standalone",
    Path("zeus/utils/compression.py"),
)
assert _compression_spec is not None and _compression_spec.loader is not None
_compression_module = importlib.util.module_from_spec(_compression_spec)
_compression_spec.loader.exec_module(_compression_module)
compress_prediction = _compression_module.compress_prediction
decompress_prediction = _compression_module.decompress_prediction


STATE_KEYS = tuple(
    sorted(
        f"{variable}@{start}_{end}"
        for variable in supported_variables()
        for start, end in ((0, 48), (0, 360))
    )
)


def test_registry_contains_exact_four_variables() -> None:
    assert set(VARIABLE_SPECS) == {
        "2m_temperature",
        "100m_u_component_of_wind",
        "100m_v_component_of_wind",
        "surface_solar_radiation_downwards",
    }
    assert len(STATE_KEYS) == 8


def test_target_conversion_fallback_keeps_float16_safe_solar_flux() -> None:
    field = xr.DataArray(
        np.array([[0.0, 500.0], [1000.0, 1200.0]], dtype=np.float32),
        dims=("latitude", "longitude"),
        coords={"latitude": [-1.0, 1.0], "longitude": [0.0, 1.0]},
    )
    converted = convert_gfs_to_zeus_target(
        field,
        get_variable_spec("surface_solar_radiation_downwards"),
    )
    assert converted.shape == field.shape
    assert np.isfinite(converted.values).all()
    assert converted.values.max() <= np.finfo(np.float16).max


def test_exact_gfs_grid_is_reordered_to_zeus_convention() -> None:
    lat = np.linspace(90.0, -90.0, 721)
    lon = np.arange(0.0, 360.0, 0.25)
    # A unique, deterministic field lets us verify longitude/latitude ordering.
    values = (
        lat[:, None].astype(np.float32)
        + lon[None, :].astype(np.float32) / np.float32(1000.0)
    )
    field = xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={"latitude": lat, "longitude": lon},
    )

    normalized = GFSWeatherDataProvider._normalize_coordinates(field)
    assert normalized.shape == (721, 1440)
    np.testing.assert_allclose(
        normalized.latitude.values,
        GFSWeatherDataProvider.TARGET_LATITUDES,
    )
    np.testing.assert_allclose(
        normalized.longitude.values,
        GFSWeatherDataProvider.TARGET_LONGITUDES,
    )
    # lon=-180 maps to original lon=180; lat=-90 maps to the last source row.
    assert normalized.values[0, 0] == pytest.approx(-90.0 + 0.180)
    # lon=0 maps to original lon=0.
    zero_lon_index = int((0.0 - (-180.0)) / 0.25)
    assert normalized.values[-1, zero_lon_index] == pytest.approx(90.0)


def test_persistence_service_generates_distinct_horizon_shapes(monkeypatch) -> None:
    monkeypatch.setattr(ForecastService, "EXPECTED_LATITUDE_SIZE", 3)
    monkeypatch.setattr(ForecastService, "EXPECTED_LONGITUDE_SIZE", 4)

    class Provider:
        def load_history(self, variable_name: str, history_hours: int):
            raise AssertionError("history is passed explicitly")

    service = ForecastService(
        data_provider=Provider(),
        variable_name="2m_temperature",
        history_hours=6,
    )
    latest = np.arange(12, dtype=np.float32).reshape(1, 3, 4)
    history = xr.DataArray(
        latest,
        dims=("valid_time", "latitude", "longitude"),
        coords={
            "valid_time": [np.datetime64("2026-01-01T00:00")],
            "latitude": [-1.0, 0.0, 1.0],
            "longitude": [-1.0, 0.0, 1.0, 2.0],
        },
    )

    short = service.generate_from_history(history, 5)
    long = service.generate_from_history(history, 9)
    assert short.shape == (5, 3, 4)
    assert long.shape == (9, 3, 4)
    assert short.dtype == np.float16
    assert long.flags.c_contiguous
    np.testing.assert_array_equal(short[0], short[-1])
    np.testing.assert_array_equal(long[0], long[-1])


def test_compression_round_trip_is_deterministic() -> None:
    array = np.arange(2 * 3 * 4, dtype=np.float16).reshape(2, 3, 4)
    first = compress_prediction(array)
    second = compress_prediction(array.copy(order="C"))
    assert first == second
    restored = decompress_prediction(first, array.shape)
    np.testing.assert_array_equal(restored.numpy(), array)


def _commitment_hash(payload: bytes, hotkey: str) -> str:
    return hashlib.sha256(payload + hotkey.encode("utf-8")).hexdigest()


def test_store_publishes_and_serves_exact_historical_eight_artifact_bundle(
    tmp_path: Path,
) -> None:
    hotkey = "5TestHotkey"
    store = ForecastStore(tmp_path / "store", retention_days=24)
    cycle = datetime(2026, 7, 24, 6, 45, tzinfo=timezone.utc)

    with store.begin_bundle(cycle, STATE_KEYS) as writer:
        for state_key in STATE_KEYS:
            requested_hours = 49 if state_key.endswith("@0_48") else 361
            variable = state_key.split("@", 1)[0]
            payload = f"payload:{state_key}".encode()
            writer.write_artifact(
                state_key,
                payload,
                shape=(requested_hours, 721, 1440),
                dtype="float16",
                variable=variable,
                requested_hours=requested_hours,
                commitment_hash=_commitment_hash(payload, hotkey),
            )
        manifest = writer.finalize(
            metadata={
                "model": "persistence",
                "hotkey": hotkey,
                "source_variables": {},
            }
        )

    assert manifest["cycle_key"] == "20260724T060000Z"
    loaded = store.load_manifest(
        cycle,
        expected_state_keys=STATE_KEYS,
        expected_hotkey=hotkey,
    )
    assert len(loaded["artifacts"]) == 8

    for state_key in STATE_KEYS:
        expected = f"payload:{state_key}".encode()
        assert store.load_artifact(cycle, state_key) == expected
        assert (
            store.commitment_hashes(loaded)[state_key]
            == _commitment_hash(expected, hotkey)
        )

    next_cycle = cycle + timedelta(hours=6)
    fallback = store.clone_latest_to_cycle(
        next_cycle,
        STATE_KEYS,
        reason="test fallback",
        expected_hotkey=hotkey,
    )
    assert fallback["fallback"] is True
    assert fallback["fallback_from_cycle"] == manifest["cycle_key"]
    for state_key in STATE_KEYS:
        assert store.load_artifact(next_cycle, state_key) == store.load_artifact(
            cycle, state_key
        )


def test_store_rejects_incomplete_bundle(tmp_path: Path) -> None:
    store = ForecastStore(tmp_path / "store")
    cycle = datetime(2026, 7, 24, tzinfo=timezone.utc)
    writer = store.begin_bundle(cycle, STATE_KEYS)
    state_key = STATE_KEYS[0]
    payload = b"only-one"
    writer.write_artifact(
        state_key,
        payload,
        shape=(49, 721, 1440),
        dtype="float16",
        variable=state_key.split("@", 1)[0],
        requested_hours=49,
        commitment_hash=_commitment_hash(payload, "hotkey"),
    )
    with pytest.raises(ValueError, match="incomplete"):
        writer.finalize(metadata={"hotkey": "hotkey"})
    writer.abort()

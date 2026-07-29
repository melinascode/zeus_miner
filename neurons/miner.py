# The MIT License (MIT)
# Copyright © 2023 Yuma Rao
# developer: Eric (Ørpheus A.I.)
# Copyright © 2025 Ørpheus A.I.


import base64
import gc
import logging
import os
import sys
import threading
import time
import typing
from datetime import datetime, timezone
from logging.handlers import RotatingFileHandler
from typing import Any

import bittensor as bt
import numpy as np
import pandas as pd

from forecast.forecast_store import ForecastStore
from forecast.gfs_provider import GFSWeatherDataProvider
from forecast.service import ForecastService
from forecast.variables import canonicalize_variable_name, supported_variables
from zeus import __version__ as zeus_version
from zeus.base.miner import BaseMinerNeuron
from zeus.commitment import ChallengeCommitment, commit_to_chain
from zeus.protocol import PredictionSynapse, TimePredictionSynapse
from zeus.utils.compression import compress_prediction
from zeus.utils.hash import prediction_hash
from zeus.utils.time import to_timestamp
from zeus.validator.challenge_spec import make_state_key
from zeus.validator.constants import (
    CHALLENGE_HASHING_MAX_MINUTE,
    CHALLENGE_REGISTRY,
    LONG_CHALLENGE,
    SHORT_CHALLENGE,
)


def configure_miner_logger() -> logging.Logger:
    """Create an independent console + rotating-file logger."""

    log_directory = os.environ.get("ZEUS_MINER_LOG_DIR", "logs")
    os.makedirs(log_directory, exist_ok=True)

    configured = logging.getLogger("zeus.miner")
    configured.setLevel(logging.INFO)
    configured.propagate = False
    if configured.handlers:
        return configured

    formatter = logging.Formatter(
        fmt=(
            "%(asctime)s | %(levelname)-8s | %(process)d | "
            "%(threadName)s | %(message)s"
        ),
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        filename=os.path.join(log_directory, "miner.log"),
        maxBytes=50 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    configured.addHandler(console_handler)
    configured.addHandler(file_handler)
    return configured


logger = configure_miner_logger()

EXPECTED_STATE_KEYS = tuple(sorted(CHALLENGE_REGISTRY))
EXPECTED_VARIABLES = tuple(sorted(supported_variables()))
EXPECTED_SHAPES = {
    49: (49, 721, 1440),
    361: (361, 721, 1440),
}
WINDOW_BY_HOURS = {
    49: SHORT_CHALLENGE,
    361: LONG_CHALLENGE,
}


def _environment_int(name: str, default: int, minimum: int = 1) -> int:
    raw = os.environ.get(name)
    value = default if raw is None else int(raw)
    if value < minimum:
        raise ValueError(f"{name} must be at least {minimum}; received {value}.")
    return value


def _environment_bool(name: str, default: bool = False) -> bool:
    return os.environ.get(name, str(default)).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def _validate_forecast_array(
    name: str,
    array: np.ndarray,
    expected_shape: tuple[int, int, int],
) -> np.ndarray:
    """Validate one tensor immediately before compression and commitment."""

    if not isinstance(array, np.ndarray):
        raise TypeError(
            f"{name}: expected numpy.ndarray, got {type(array).__name__}."
        )
    if array.shape != expected_shape:
        raise ValueError(
            f"{name}: expected shape {expected_shape}, got {array.shape}."
        )
    if array.dtype != np.float16:
        logger.warning("%s: converting dtype %s -> float16", name, array.dtype)
        array = array.astype(np.float16, copy=False)
    if not np.isfinite(array).all():
        raise ValueError(f"{name}: contains NaN or infinite values.")
    if not array.flags.c_contiguous:
        logger.info("%s: making array C-contiguous", name)
        array = np.ascontiguousarray(array)
    return array


def _utc_datetime_from_timestamp(timestamp: float) -> datetime:
    return pd.Timestamp(timestamp, unit="s", tz="UTC").to_pydatetime()


class Miner(BaseMinerNeuron):
    """Zeus 2.1.1 four-variable, two-horizon protocol miner.

    This patch is deliberately conservative: each variable is still generated
    by a persistence baseline, but all eight challenge artifacts are now
    independent, versioned by challenge cycle, committed independently, and
    routed correctly during reveal and delayed scoring.
    """

    def __init__(self, config=None):
        logger.info("Initializing miner")
        super().__init__(config=config)
        logger.info(
            "Base miner initialized | uid=%s | port=%s",
            self.uid,
            self.config.axon.port,
        )

        self._forecast_refresh_lock = threading.Lock()
        self._prepared_bundle_only = _environment_bool(
            "ZEUS_PREPARED_BUNDLE_ONLY",
            False,
        )
        logger.info(
            "Forecast preparation mode | prepared_bundle_only=%s",
            self._prepared_bundle_only,
        )

        self._history_hours = _environment_int(
            "ZEUS_GFS_HISTORY_HOURS", 6
        )
        retention_days = _environment_int(
            "ZEUS_FORECAST_RETENTION_DAYS", 24
        )
        max_run_age_hours = _environment_int(
            "ZEUS_GFS_MAX_RUN_AGE_HOURS", 24, minimum=0
        )

        self.data_provider = GFSWeatherDataProvider(
            cache_directory=os.environ.get(
                "ZEUS_GFS_CACHE_DIR", "data/gfs_cache"
            ),
            max_run_age_hours=max_run_age_hours,
        )
        self.forecast_services = {
            variable: ForecastService(
                data_provider=self.data_provider,
                variable_name=variable,
                history_hours=self._history_hours,
            )
            for variable in EXPECTED_VARIABLES
        }
        self.forecast_store = ForecastStore(
            directory=os.environ.get(
                "ZEUS_FORECAST_STORE_DIR", "data/forecast_store_v2"
            ),
            retention_days=retention_days,
        )

        bt.logging.info("Attaching reveal handler to miner axon.")
        self.axon.attach(
            forward_fn=self._forward_unhashed_predictions,
            blacklist_fn=self._blacklist_time,
            priority_fn=self._priority_time,
        )
        logger.info("Axon handlers attached | axon=%s", self.axon)

        self._log_store_startup_state()
        if _environment_bool("ZEUS_PRECOMPUTE_ON_STARTUP", False):
            cycle = datetime.now(timezone.utc)
            try:
                manifest, refreshed = self.pre_compute_predictions(cycle)
                logger.info(
                    "Startup precompute finished | cycle=%s | refreshed=%s",
                    manifest["cycle_key"],
                    refreshed,
                )
            except Exception:
                logger.exception(
                    "Startup precompute failed; miner will retry at challenge time"
                )

    @property
    def hotkey(self) -> str:
        return self.wallet.hotkey.ss58_address

    def _log_store_startup_state(self) -> None:
        try:
            manifest = self.forecast_store.load_latest_manifest(
                expected_state_keys=EXPECTED_STATE_KEYS,
                verify_files=True,
                expected_hotkey=self.hotkey,
            )
            logger.info(
                "Loaded latest complete V2 bundle | cycle=%s | artifacts=%d",
                manifest["cycle_key"],
                len(manifest["artifacts"]),
            )
        except FileNotFoundError:
            logger.warning(
                "No V2 forecast bundle exists yet; the first challenge will "
                "generate all eight artifacts"
            )
        except Exception:
            logger.exception(
                "Latest V2 bundle is not usable by the active hotkey"
            )

    def pre_compute_predictions(
        self,
        challenge_time: datetime,
    ) -> tuple[dict[str, Any], bool]:
        """Generate and atomically publish all 4 × 2 challenge artifacts.

        A complete immutable bundle is keyed by the 00/06/12/18 UTC cycle start.
        If live generation fails, a byte-identical copy of the last complete
        bundle is published for the new cycle. No partial or mixed bundle can
        become visible or be committed.
        """

        cycle_time = self.forecast_store.normalize_cycle_time(challenge_time)
        cycle_key = self.forecast_store.cycle_key(cycle_time)

        with self._forecast_refresh_lock:
            try:
                existing = self.forecast_store.load_manifest(
                    cycle_time,
                    expected_state_keys=EXPECTED_STATE_KEYS,
                    verify_files=True,
                    expected_hotkey=self.hotkey,
                )
                logger.info(
                    "Reusing existing complete forecast bundle | cycle=%s",
                    cycle_key,
                )
                return existing, False
            except FileNotFoundError:
                pass
            except ValueError:
                logger.exception(
                    "Existing bundle for %s is invalid and cannot be reused",
                    cycle_key,
                )
                raise

            if self._prepared_bundle_only:
                logger.warning(
                    "Prepared target bundle missing at challenge time | "
                    "cycle=%s | challenge-time generation disabled",
                    cycle_key,
                )

                try:
                    fallback = self.forecast_store.clone_latest_to_cycle(
                        target_cycle_time=cycle_time,
                        expected_state_keys=EXPECTED_STATE_KEYS,
                        expected_hotkey=self.hotkey,
                        reason=(
                            "Prepared target bundle missing at challenge time; "
                            "challenge-time generation disabled"
                        ),
                    )
                except Exception as fallback_error:
                    raise RuntimeError(
                        "Prepared target bundle is missing and no valid "
                        "last-good bundle is available. Challenge-time "
                        "generation is disabled."
                    ) from fallback_error

                logger.warning(
                    "Published byte-identical prepared-mode fallback bundle | "
                    "cycle=%s | source_cycle=%s",
                    fallback["cycle_key"],
                    fallback.get("fallback_from_cycle"),
                )

                return fallback, False

            refresh_started = time.monotonic()
            logger.info(
                "Forecast bundle refresh started | cycle=%s | variables=%d | "
                "artifacts=%d",
                cycle_key,
                len(EXPECTED_VARIABLES),
                len(EXPECTED_STATE_KEYS),
            )

            try:
                common_gfs_cycle = self.data_provider.find_common_available_cycle(
                    EXPECTED_VARIABLES
                )
                logger.info(
                    "Common GFS analysis cycle selected | cycle=%s",
                    common_gfs_cycle.isoformat(),
                )

                source_variables: dict[str, Any] = {}
                with self.forecast_store.begin_bundle(
                    cycle_time,
                    EXPECTED_STATE_KEYS,
                ) as writer:
                    for variable in EXPECTED_VARIABLES:
                        service = self.forecast_services[variable]
                        history_started = time.monotonic()
                        history = self.data_provider.load_history_at_cycle(
                            variable_name=variable,
                            history_hours=self._history_hours,
                            latest_cycle=common_gfs_cycle,
                        )
                        source_valid_time = pd.Timestamp(
                            history.valid_time.values[-1]
                        ).isoformat()
                        source_variables[variable] = {
                            "latest_valid_time_utc": source_valid_time,
                            "units": history.attrs.get("units"),
                            "target_conversion": history.attrs.get(
                                "target_conversion"
                            ),
                            "source": history.attrs.get("source"),
                            "source_product": history.attrs.get(
                                "source_product"
                            ),
                        }
                        logger.info(
                            "History loaded | variable=%s | shape=%s | "
                            "elapsed=%.2fs | target_conversion=%s",
                            variable,
                            tuple(history.shape),
                            time.monotonic() - history_started,
                            history.attrs.get("target_conversion"),
                        )

                        for requested_hours in (49, 361):
                            window = WINDOW_BY_HOURS[requested_hours]
                            state_key = make_state_key(
                                variable,
                                window[0],
                                window[1],
                            )
                            artifact_started = time.monotonic()
                            logger.info(
                                "Generating artifact | state_key=%s",
                                state_key,
                            )
                            forecast = service.generate_from_history(
                                history=history,
                                forecast_steps=requested_hours,
                            )
                            forecast = _validate_forecast_array(
                                state_key,
                                forecast,
                                EXPECTED_SHAPES[requested_hours],
                            )
                            compressed = compress_prediction(forecast)
                            commitment_hash = prediction_hash(
                                compressed,
                                self.hotkey,
                            )
                            writer.write_artifact(
                                state_key,
                                compressed,
                                shape=forecast.shape,
                                dtype=str(forecast.dtype),
                                variable=variable,
                                requested_hours=requested_hours,
                                commitment_hash=commitment_hash,
                                source_valid_time_utc=source_valid_time,
                            )
                            logger.info(
                                "Artifact ready | state_key=%s | "
                                "compressed_bytes=%d | hash=%s... | "
                                "elapsed=%.2fs",
                                state_key,
                                len(compressed),
                                commitment_hash[:16],
                                time.monotonic() - artifact_started,
                            )
                            del forecast, compressed
                            gc.collect()

                        del history
                        gc.collect()

                    manifest = writer.finalize(
                        metadata={
                            "model": "persistence",
                            "hotkey": self.hotkey,
                            "fallback": False,
                            "gfs_common_cycle_utc": common_gfs_cycle.isoformat(),
                            "history_hours": self._history_hours,
                            "source_variables": source_variables,
                            "zeus_version": zeus_version,
                        }
                    )

                removed = self.forecast_store.prune(
                    protect_cycle_keys=(manifest["cycle_key"],)
                )
                logger.info(
                    "Forecast bundle refresh completed successfully | "
                    "cycle=%s | artifacts=%d | removed_old_bundles=%d | "
                    "total_elapsed=%.2fs",
                    manifest["cycle_key"],
                    len(manifest["artifacts"]),
                    len(removed),
                    time.monotonic() - refresh_started,
                )
                return manifest, True

            except Exception as refresh_error:
                logger.exception(
                    "Forecast bundle refresh failed | cycle=%s",
                    cycle_key,
                )
                try:
                    fallback = self.forecast_store.clone_latest_to_cycle(
                        target_cycle_time=cycle_time,
                        expected_state_keys=EXPECTED_STATE_KEYS,
                        expected_hotkey=self.hotkey,
                        reason=(
                            f"{type(refresh_error).__name__}: {refresh_error}"
                        ),
                    )
                except Exception as fallback_error:
                    raise RuntimeError(
                        "Could not generate a complete Zeus V2 bundle and no "
                        "valid last-good bundle is available for fallback."
                    ) from fallback_error

                logger.warning(
                    "Published byte-identical last-good fallback bundle | "
                    "cycle=%s | source_cycle=%s",
                    fallback["cycle_key"],
                    fallback.get("fallback_from_cycle"),
                )
                return fallback, False

    def on_challenge_block(
        self,
        challenge_time: datetime,
        subtensor=None,
    ) -> None:
        """Build/restore the exact eight-artifact bundle and commit its hashes."""

        sub = subtensor or self.subtensor
        cycle_time = self.forecast_store.normalize_cycle_time(challenge_time)
        logger.info(
            "Challenge triggered | challenge_time=%s | cycle_start=%s",
            challenge_time,
            cycle_time,
        )

        manifest, refreshed = self.pre_compute_predictions(cycle_time)
        hashes = self.forecast_store.commitment_hashes(manifest)
        if set(hashes) != set(EXPECTED_STATE_KEYS):
            raise RuntimeError("Refusing to commit an incomplete hash bundle.")

        commitment = ChallengeCommitment()
        for state_key in EXPECTED_STATE_KEYS:
            commitment.hashes[state_key] = hashes[state_key]
            logger.info(
                "Commitment artifact | state_key=%s | hash=%s...",
                state_key,
                hashes[state_key][:16],
            )

        logger.info(
            "Uploading commitment | challenge_time=%s | cycle=%s | "
            "refreshed=%s",
            challenge_time,
            manifest["cycle_key"],
            refreshed,
        )
        commit_to_chain(
            sub,
            self.wallet,
            netuid=self.config.netuid,
            commitment=commitment,
        )
        logger.info(
            "Commitment uploaded | challenge_time=%s | cycle=%s | "
            "artifact_count=%d",
            challenge_time,
            manifest["cycle_key"],
            len(hashes),
        )

    @staticmethod
    def _validate_reveal_request(
        synapse: TimePredictionSynapse,
    ) -> tuple[str, int, str, datetime]:
        variable = canonicalize_variable_name(synapse.variable)
        if variable != synapse.variable:
            raise ValueError(
                f"Validator requested non-canonical variable {synapse.variable!r}."
            )

        requested_hours = int(synapse.requested_hours)
        if requested_hours not in EXPECTED_SHAPES:
            raise ValueError(
                f"Unsupported requested_hours={requested_hours}; expected 49 or 361."
            )
        if int(synapse.step_size) != 1:
            raise ValueError("Zeus V2 requests must use step_size=1 hour.")

        expected_bbox = (-90.0, 90.0, -180.0, 179.75)
        actual_bbox = (
            float(synapse.latitude_start),
            float(synapse.latitude_end),
            float(synapse.longitude_start),
            float(synapse.longitude_end),
        )
        if not np.allclose(actual_bbox, expected_bbox, atol=1e-8):
            raise ValueError(
                f"Unsupported request bounding box {actual_bbox}; expected "
                f"{expected_bbox}."
            )

        start = to_timestamp(synapse.start_time)
        end = to_timestamp(synapse.end_time)
        expected_end = start + pd.Timedelta(hours=requested_hours - 1)
        if end != expected_end:
            raise ValueError(
                f"Request time window is inconsistent: start={start}, end={end}, "
                f"requested_hours={requested_hours}."
            )
        if start != start.floor("6h"):
            raise ValueError(
                f"Request start time {start} is not aligned to a six-hour cycle."
            )

        window = WINDOW_BY_HOURS[requested_hours]
        state_key = make_state_key(variable, window[0], window[1])
        if state_key not in CHALLENGE_REGISTRY:
            raise ValueError(f"Unknown Zeus challenge state key {state_key}.")
        cycle_time = _utc_datetime_from_timestamp(synapse.start_time)
        return variable, requested_hours, state_key, cycle_time

    async def _forward_unhashed_predictions(
        self,
        synapse: TimePredictionSynapse,
    ) -> TimePredictionSynapse:
        """Return exact historical bytes for reveal or delayed scoring."""

        sender = getattr(getattr(synapse, "dendrite", None), "hotkey", None)
        logger.info(
            "Reveal request received | hotkey=%s | variable=%s | "
            "requested_hours=%s | start=%s | end=%s",
            sender,
            synapse.variable,
            synapse.requested_hours,
            synapse.start_time,
            synapse.end_time,
        )
        synapse.version = zeus_version

        now = pd.Timestamp.now("UTC").replace(tzinfo=None)
        request_end = to_timestamp(synapse.end_time)
        if (
            now.hour % 6 == 0
            and now.minute >= CHALLENGE_HASHING_MAX_MINUTE
            and request_end > now - pd.Timedelta(days=4)
        ):
            logger.warning(
                "Reveal intentionally withheld by anti-relay guard | "
                "utc_now=%s | variable=%s | hours=%s | end=%s",
                now,
                synapse.variable,
                synapse.requested_hours,
                request_end,
            )
            return synapse

        try:
            variable, requested_hours, state_key, cycle_time = (
                self._validate_reveal_request(synapse)
            )
            manifest = self.forecast_store.load_manifest(
                cycle_time,
                expected_state_keys=EXPECTED_STATE_KEYS,
                verify_files=False,
                expected_hotkey=self.hotkey,
            )
            metadata = manifest["artifacts"][state_key]
            if metadata["variable"] != variable:
                raise ValueError("Stored artifact variable does not match request.")
            if metadata["requested_hours"] != requested_hours:
                raise ValueError("Stored artifact horizon does not match request.")

            compressed = self.forecast_store.load_artifact(
                cycle_time,
                state_key,
                verify=True,
            )
            computed_hash = prediction_hash(compressed, self.hotkey)
            if computed_hash != metadata["commitment_hash"]:
                raise ValueError(
                    "Stored artifact no longer matches its commitment hash."
                )

            synapse.predictions = base64.b64encode(compressed).decode("ascii")
            logger.info(
                "Reveal response ready | cycle=%s | state_key=%s | "
                "compressed_bytes=%d | base64_chars=%d | hash=%s...",
                manifest["cycle_key"],
                state_key,
                len(compressed),
                len(synapse.predictions),
                computed_hash[:16],
            )
        except Exception:
            # Return an empty prediction rather than substituting the latest
            # bundle: substitution would fail hash verification and is unsafe.
            synapse.predictions = None
            logger.exception(
                "Reveal request could not be served exactly | variable=%s | "
                "requested_hours=%s | start=%s",
                synapse.variable,
                synapse.requested_hours,
                synapse.start_time,
            )
        return synapse

    async def _blacklist_time(
        self,
        synapse: TimePredictionSynapse,
    ) -> typing.Tuple[bool, str]:
        return await self.blacklist(synapse)

    async def _priority_time(self, synapse: TimePredictionSynapse) -> float:
        return await self.priority(synapse)

    async def blacklist(
        self,
        synapse: PredictionSynapse,
    ) -> typing.Tuple[bool, str]:
        return await self._blacklist(synapse)

    async def priority(self, synapse: PredictionSynapse) -> float:
        return await self._priority(synapse)


if __name__ == "__main__":
    try:
        logger.info("Miner process starting | pid=%d", os.getpid())
        with Miner() as miner:
            logger.info(
                "Miner serving | uid=%s | hotkey=%s | port=%s | "
                "variables=%d | artifacts_per_cycle=%d",
                miner.uid,
                miner.hotkey,
                miner.config.axon.port,
                len(EXPECTED_VARIABLES),
                len(EXPECTED_STATE_KEYS),
            )
            while True:
                stats = miner.forecast_store.stats()
                logger.info(
                    "Heartbeat | uid=%s | latest_cycle=%s | bundles=%d | "
                    "store_bytes=%d",
                    miner.uid,
                    stats["latest_cycle"],
                    stats["bundle_count"],
                    stats["total_bytes"],
                )
                time.sleep(60)
    except KeyboardInterrupt:
        logger.info("Miner stopped by operator")
    except Exception:
        logger.exception("Miner terminated by an unhandled exception")
        raise

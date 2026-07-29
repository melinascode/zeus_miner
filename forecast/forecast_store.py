from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Iterable


_STATE_KEY_PATTERN = re.compile(r"^[A-Za-z0-9_]+@[0-9]+_[0-9]+$")


class ForecastStore:
    """Versioned, atomic storage for Zeus V2 challenge bundles.

    One complete bundle contains every variable × horizon artifact for one
    six-hour challenge cycle. Bundles are immutable after publication so an old
    reveal/scoring request always receives the exact bytes committed on-chain.
    """

    SCHEMA_VERSION = 2
    MANIFEST_FILENAME = "manifest.json"
    LATEST_FILENAME = "latest.json"

    def __init__(
        self,
        directory: str | Path = "data/forecast_store",
        retention_days: int = 24,
    ) -> None:
        if retention_days < 1:
            raise ValueError("retention_days must be at least 1.")

        self.directory = Path(directory)
        self.bundles_directory = self.directory / "bundles"
        self.retention_days = retention_days
        self.bundles_directory.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def normalize_cycle_time(value: datetime) -> datetime:
        """Normalize a wall time to its UTC 00/06/12/18 cycle start."""

        if not isinstance(value, datetime):
            raise TypeError("Cycle time must be a datetime.")
        if value.tzinfo is None:
            utc_value = value.replace(tzinfo=timezone.utc)
        else:
            utc_value = value.astimezone(timezone.utc)
        cycle_hour = (utc_value.hour // 6) * 6
        return utc_value.replace(
            hour=cycle_hour,
            minute=0,
            second=0,
            microsecond=0,
        )

    @classmethod
    def cycle_key(cls, value: datetime) -> str:
        return cls.normalize_cycle_time(value).strftime("%Y%m%dT%H%M%SZ")

    @staticmethod
    def parse_cycle_key(key: str) -> datetime:
        return datetime.strptime(key, "%Y%m%dT%H%M%SZ").replace(
            tzinfo=timezone.utc
        )

    def bundle_path(self, cycle_time: datetime) -> Path:
        return self.bundles_directory / self.cycle_key(cycle_time)

    def begin_bundle(
        self,
        cycle_time: datetime,
        expected_state_keys: Iterable[str],
    ) -> "ForecastBundleWriter":
        return ForecastBundleWriter(
            store=self,
            cycle_time=self.normalize_cycle_time(cycle_time),
            expected_state_keys=tuple(sorted(expected_state_keys)),
        )

    def bundle_exists(
        self,
        cycle_time: datetime,
        expected_state_keys: Iterable[str] | None = None,
    ) -> bool:
        try:
            self.load_manifest(
                cycle_time,
                expected_state_keys=expected_state_keys,
            )
            return True
        except (FileNotFoundError, ValueError, KeyError, json.JSONDecodeError):
            return False

    def load_manifest(
        self,
        cycle_time: datetime,
        expected_state_keys: Iterable[str] | None = None,
        verify_files: bool = True,
        expected_hotkey: str | None = None,
    ) -> dict[str, Any]:
        bundle = self.bundle_path(cycle_time)
        manifest_path = bundle / self.MANIFEST_FILENAME
        if not manifest_path.is_file():
            raise FileNotFoundError(
                f"No forecast manifest for cycle {self.cycle_key(cycle_time)}."
            )

        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        self._validate_manifest(
            bundle=bundle,
            manifest=manifest,
            expected_state_keys=expected_state_keys,
            verify_files=verify_files,
            expected_hotkey=expected_hotkey,
        )
        return manifest

    def load_latest_manifest(
        self,
        expected_state_keys: Iterable[str] | None = None,
        verify_files: bool = True,
        expected_hotkey: str | None = None,
    ) -> dict[str, Any]:
        pointer_path = self.directory / self.LATEST_FILENAME
        if not pointer_path.is_file():
            raise FileNotFoundError("No latest forecast bundle pointer exists.")

        pointer = json.loads(pointer_path.read_text(encoding="utf-8"))
        cycle_key = pointer.get("cycle_key")
        if not isinstance(cycle_key, str):
            raise ValueError("Latest forecast pointer has no cycle_key.")
        return self.load_manifest(
            self.parse_cycle_key(cycle_key),
            expected_state_keys=expected_state_keys,
            verify_files=verify_files,
            expected_hotkey=expected_hotkey,
        )

    def load_artifact(
        self,
        cycle_time: datetime,
        state_key: str,
        verify: bool = True,
    ) -> bytes:
        self._validate_state_key(state_key)
        manifest = self.load_manifest(cycle_time, verify_files=False)
        try:
            metadata = manifest["artifacts"][state_key]
        except KeyError as exc:
            raise FileNotFoundError(
                f"Bundle {manifest.get('cycle_key')} has no artifact {state_key}."
            ) from exc

        artifact_path = self.bundle_path(cycle_time) / metadata["filename"]
        payload = artifact_path.read_bytes()
        if not payload:
            raise ValueError(f"Stored artifact {state_key} is empty.")

        if verify:
            if len(payload) != metadata["compressed_bytes"]:
                raise ValueError(
                    f"Stored artifact {state_key} size does not match manifest."
                )
            digest = hashlib.sha256(payload).hexdigest()
            if digest != metadata["payload_sha256"]:
                raise ValueError(
                    f"Stored artifact {state_key} SHA-256 does not match manifest."
                )
        return payload

    @staticmethod
    def commitment_hashes(manifest: dict[str, Any]) -> dict[str, str]:
        artifacts = manifest.get("artifacts", {})
        hashes: dict[str, str] = {}
        for state_key, metadata in artifacts.items():
            commitment_hash = metadata.get("commitment_hash")
            if (
                not isinstance(commitment_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", commitment_hash)
            ):
                raise ValueError(
                    f"Manifest artifact {state_key} has no valid commitment hash."
                )
            hashes[state_key] = commitment_hash
        return hashes

    def clone_latest_to_cycle(
        self,
        target_cycle_time: datetime,
        expected_state_keys: Iterable[str],
        reason: str,
        expected_hotkey: str | None = None,
    ) -> dict[str, Any]:
        """Publish a byte-identical copy of the latest complete bundle.

        This is a last-good fallback. It preserves protocol correctness when a
        live data refresh fails: the current cycle receives new on-chain hashes
        for exact bytes that remain available for later reveal/scoring.
        """

        expected = tuple(sorted(expected_state_keys))
        try:
            return self.load_manifest(
                target_cycle_time,
                expected_state_keys=expected,
                expected_hotkey=expected_hotkey,
            )
        except FileNotFoundError:
            pass

        source_manifest = self.load_latest_manifest(
            expected, expected_hotkey=expected_hotkey
        )
        source_cycle = self.parse_cycle_key(source_manifest["cycle_key"])
        target_cycle = self.normalize_cycle_time(target_cycle_time)
        if source_cycle == target_cycle:
            return source_manifest

        writer = self.begin_bundle(target_cycle, expected)
        try:
            source_directory = self.bundle_path(source_cycle)
            for state_key in expected:
                metadata = dict(source_manifest["artifacts"][state_key])
                source_file = source_directory / metadata["filename"]
                writer.link_artifact(state_key, source_file, metadata)

            return writer.finalize(
                metadata={
                    "model": source_manifest.get("model", "unknown"),
                    "hotkey": source_manifest.get("hotkey"),
                    "fallback": True,
                    "fallback_from_cycle": source_manifest["cycle_key"],
                    "fallback_reason": reason,
                    "source_variables": source_manifest.get(
                        "source_variables", {}
                    ),
                }
            )
        except Exception:
            writer.abort()
            raise

    def prune(
        self,
        now: datetime | None = None,
        protect_cycle_keys: Iterable[str] = (),
    ) -> list[str]:
        current = now or datetime.now(timezone.utc)
        if current.tzinfo is None:
            current = current.replace(tzinfo=timezone.utc)
        cutoff = current.astimezone(timezone.utc) - timedelta(
            days=self.retention_days
        )

        protected = set(protect_cycle_keys)
        try:
            protected.add(self.load_latest_manifest(verify_files=False)["cycle_key"])
        except Exception:
            pass

        removed: list[str] = []
        for path in self.bundles_directory.iterdir():
            if not path.is_dir() or path.name.startswith(".staging-"):
                continue
            if path.name in protected:
                continue
            try:
                cycle_time = self.parse_cycle_key(path.name)
            except ValueError:
                continue
            if cycle_time < cutoff:
                shutil.rmtree(path)
                removed.append(path.name)
        return removed

    def stats(self) -> dict[str, int | str | None]:
        bundle_count = 0
        total_bytes = 0
        for path in self.bundles_directory.iterdir():
            if not path.is_dir() or path.name.startswith(".staging-"):
                continue
            bundle_count += 1
            for file_path in path.iterdir():
                if file_path.is_file():
                    total_bytes += file_path.stat().st_size

        latest_cycle: str | None = None
        try:
            latest_cycle = self.load_latest_manifest(
                verify_files=False
            )["cycle_key"]
        except Exception:
            pass

        return {
            "bundle_count": bundle_count,
            "total_bytes": total_bytes,
            "latest_cycle": latest_cycle,
        }

    def _publish_latest(self, cycle_key: str) -> None:
        pointer = {
            "schema_version": self.SCHEMA_VERSION,
            "cycle_key": cycle_key,
            "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        self._atomic_write(
            self.directory / self.LATEST_FILENAME,
            json.dumps(pointer, indent=2, sort_keys=True).encode("utf-8"),
        )

    @classmethod
    def _validate_manifest(
        cls,
        bundle: Path,
        manifest: dict[str, Any],
        expected_state_keys: Iterable[str] | None,
        verify_files: bool,
        expected_hotkey: str | None,
    ) -> None:
        if manifest.get("schema_version") != cls.SCHEMA_VERSION:
            raise ValueError("Unsupported forecast manifest schema.")
        if manifest.get("cycle_key") != bundle.name:
            raise ValueError("Forecast manifest cycle does not match directory.")

        if expected_hotkey is not None and manifest.get("hotkey") != expected_hotkey:
            raise ValueError(
                "Forecast bundle hotkey does not match the active wallet hotkey."
            )

        artifacts = manifest.get("artifacts")
        if not isinstance(artifacts, dict) or not artifacts:
            raise ValueError("Forecast manifest contains no artifacts.")

        if expected_state_keys is not None:
            expected = set(expected_state_keys)
            actual = set(artifacts)
            if actual != expected:
                missing = sorted(expected - actual)
                extra = sorted(actual - expected)
                raise ValueError(
                    f"Incomplete forecast bundle; missing={missing}, extra={extra}."
                )

        for state_key, metadata in artifacts.items():
            cls._validate_state_key(state_key)
            if not isinstance(metadata, dict):
                raise ValueError(f"Invalid metadata for {state_key}.")
            filename = metadata.get("filename")
            if filename != f"{state_key}.bin":
                raise ValueError(f"Unexpected artifact filename for {state_key}.")
            if not isinstance(metadata.get("compressed_bytes"), int):
                raise ValueError(f"Missing compressed size for {state_key}.")
            payload_sha256 = metadata.get("payload_sha256")
            commitment_hash = metadata.get("commitment_hash")
            if (
                not isinstance(payload_sha256, str)
                or not re.fullmatch(r"[0-9a-f]{64}", payload_sha256)
            ):
                raise ValueError(f"Missing/invalid payload digest for {state_key}.")
            if (
                not isinstance(commitment_hash, str)
                or not re.fullmatch(r"[0-9a-f]{64}", commitment_hash)
            ):
                raise ValueError(
                    f"Missing/invalid commitment hash for {state_key}."
                )
            if verify_files:
                artifact_path = bundle / filename
                if not artifact_path.is_file():
                    raise FileNotFoundError(
                        f"Forecast artifact is missing: {artifact_path}."
                    )
                if artifact_path.stat().st_size != metadata["compressed_bytes"]:
                    raise ValueError(
                        f"Forecast artifact size mismatch: {artifact_path}."
                    )


    @staticmethod
    def _validate_state_key(state_key: str) -> None:
        if not isinstance(state_key, str) or not _STATE_KEY_PATTERN.fullmatch(
            state_key
        ):
            raise ValueError(f"Invalid Zeus challenge state key: {state_key!r}")

    @staticmethod
    def _atomic_write(destination: Path, content: bytes) -> None:
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(
            f".{destination.name}.{uuid.uuid4().hex}.tmp"
        )
        try:
            with temporary.open("wb") as file:
                file.write(content)
                file.flush()
                os.fsync(file.fileno())
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)


class ForecastBundleWriter:
    """Build one complete bundle in staging, then publish it atomically."""

    def __init__(
        self,
        store: ForecastStore,
        cycle_time: datetime,
        expected_state_keys: tuple[str, ...],
    ) -> None:
        if not expected_state_keys:
            raise ValueError("A bundle must expect at least one artifact.")
        for state_key in expected_state_keys:
            store._validate_state_key(state_key)

        self.store = store
        self.cycle_time = cycle_time
        self.cycle_key = store.cycle_key(cycle_time)
        self.expected_state_keys = expected_state_keys
        self.final_directory = store.bundle_path(cycle_time)
        self.staging_directory = store.bundles_directory / (
            f".staging-{self.cycle_key}-{uuid.uuid4().hex}"
        )
        self.staging_directory.mkdir(parents=False, exist_ok=False)
        self.artifacts: dict[str, dict[str, Any]] = {}
        self._finalized = False

    def __enter__(self) -> "ForecastBundleWriter":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        if not self._finalized:
            self.abort()

    def write_artifact(
        self,
        state_key: str,
        payload: bytes,
        *,
        shape: tuple[int, int, int],
        dtype: str,
        variable: str,
        requested_hours: int,
        commitment_hash: str,
        source_valid_time_utc: str | None = None,
    ) -> dict[str, Any]:
        self.store._validate_state_key(state_key)
        if state_key not in self.expected_state_keys:
            raise ValueError(f"Unexpected artifact {state_key}.")
        if state_key in self.artifacts:
            raise ValueError(f"Artifact {state_key} was written twice.")
        if not isinstance(payload, bytes) or not payload:
            raise ValueError(f"Artifact {state_key} payload is empty.")
        if len(commitment_hash) != 64:
            raise ValueError(f"Invalid commitment hash for {state_key}.")

        filename = f"{state_key}.bin"
        destination = self.staging_directory / filename
        with destination.open("wb") as file:
            file.write(payload)
            file.flush()
            os.fsync(file.fileno())

        metadata: dict[str, Any] = {
            "filename": filename,
            "variable": variable,
            "requested_hours": requested_hours,
            "shape": list(shape),
            "dtype": dtype,
            "compression": "blosc2:zstd:bitshuffle:clevel9",
            "compressed_bytes": len(payload),
            "payload_sha256": hashlib.sha256(payload).hexdigest(),
            "commitment_hash": commitment_hash,
        }
        if source_valid_time_utc is not None:
            metadata["source_valid_time_utc"] = source_valid_time_utc
        self.artifacts[state_key] = metadata
        return metadata

    def link_artifact(
        self,
        state_key: str,
        source_file: Path,
        metadata: dict[str, Any],
    ) -> None:
        self.store._validate_state_key(state_key)
        if state_key not in self.expected_state_keys:
            raise ValueError(f"Unexpected artifact {state_key}.")
        if state_key in self.artifacts:
            raise ValueError(f"Artifact {state_key} was linked twice.")
        if not source_file.is_file():
            raise FileNotFoundError(source_file)

        filename = f"{state_key}.bin"
        destination = self.staging_directory / filename
        try:
            os.link(source_file, destination)
        except OSError:
            shutil.copyfile(source_file, destination)

        copied = dict(metadata)
        copied["filename"] = filename
        if destination.stat().st_size != copied["compressed_bytes"]:
            raise ValueError(f"Linked artifact {state_key} size mismatch.")
        self.artifacts[state_key] = copied

    def finalize(self, metadata: dict[str, Any]) -> dict[str, Any]:
        if self._finalized:
            raise RuntimeError("Bundle writer has already been finalized.")
        actual = set(self.artifacts)
        expected = set(self.expected_state_keys)
        if actual != expected:
            raise ValueError(
                "Cannot publish an incomplete forecast bundle; "
                f"missing={sorted(expected - actual)}, "
                f"extra={sorted(actual - expected)}."
            )
        if self.final_directory.exists():
            raise FileExistsError(
                f"Forecast bundle already exists: {self.final_directory}."
            )

        manifest: dict[str, Any] = {
            "schema_version": self.store.SCHEMA_VERSION,
            "cycle_key": self.cycle_key,
            "cycle_start_utc": self.cycle_time.isoformat(),
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "artifacts": {
                key: self.artifacts[key]
                for key in sorted(self.artifacts)
            },
        }
        manifest.update(metadata)

        manifest_path = self.staging_directory / self.store.MANIFEST_FILENAME
        manifest_content = json.dumps(
            manifest,
            indent=2,
            sort_keys=True,
        ).encode("utf-8")
        with manifest_path.open("wb") as file:
            file.write(manifest_content)
            file.flush()
            os.fsync(file.fileno())

        # Rename on the same filesystem publishes the complete directory in one
        # operation. No partially-written bundle can become visible.
        os.rename(self.staging_directory, self.final_directory)
        self.store._publish_latest(self.cycle_key)
        self._finalized = True
        return manifest

    def abort(self) -> None:
        if self.staging_directory.exists():
            shutil.rmtree(self.staging_directory, ignore_errors=True)

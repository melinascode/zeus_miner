from __future__ import annotations

import json
from pathlib import Path

import torch

from zeus.utils.compression import decompress_prediction


def load_gfs_artifact(
    bundle_dir: str | Path,
    variable: str,
    horizon: int = 360,
) -> torch.Tensor:
    """
    Load raw Zeus GFS ForecastStore artifact.

    Returns:
        torch.Tensor
        shape:
        (horizon + 1, latitude, longitude)

    Example:
        (361, 721, 1440)
    """

    bundle_dir = Path(bundle_dir)

    manifest_file = bundle_dir / "manifest.json"

    if not manifest_file.exists():
        raise FileNotFoundError(
            f"Missing manifest: {manifest_file}"
        )

    with open(manifest_file, "r") as f:
        manifest = json.load(f)

    artifact_key = f"{variable}@0_{horizon}"

    if artifact_key not in manifest["artifacts"]:
        raise KeyError(
            f"Artifact not found: {artifact_key}"
        )

    artifact = manifest["artifacts"][artifact_key]

    filename = artifact["filename"]

    shape = torch.Size(
        artifact["shape"]
    )

    compressed_file = bundle_dir / filename

    if not compressed_file.exists():
        raise FileNotFoundError(
            f"Missing artifact: {compressed_file}"
        )

    compressed_bytes = compressed_file.read_bytes()

    tensor = decompress_prediction(
        compressed_bytes,
        shape,
    )

    if tensor is None:
        raise RuntimeError(
            f"Failed to decode {filename}"
        )

    return tensor
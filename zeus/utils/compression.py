import base64
import logging
from typing import Union

import blosc2
import numpy as np
import torch


def compress_prediction(tensor: Union[torch.Tensor, np.ndarray]) -> bytes:
    """Convert a tensor to deterministic lossless blosc2-compressed float16.

    A contiguous ndarray/memoryview is passed directly to blosc2. This avoids a
    second full-size ``tobytes()`` allocation for 361-hour global tensors.
    """

    if isinstance(tensor, torch.Tensor):
        arr = tensor.detach().cpu().numpy()
    elif isinstance(tensor, np.ndarray):
        arr = tensor
    else:
        raise ValueError(f"Unsupported tensor type: {type(tensor)}")

    if arr.dtype != np.float16 or not arr.flags.c_contiguous:
        arr = np.ascontiguousarray(arr, dtype=np.float16)

    return blosc2.compress(
        memoryview(arr),
        typesize=2,
        clevel=9,
        filter=blosc2.Filter.BITSHUFFLE,
        codec=blosc2.Codec.ZSTD,
    )


def decompress_prediction(compressed_bytes: bytes, shape: torch.Size) -> torch.Tensor:
    """Convert compressed bytes back to a float16 tensor."""

    try:
        raw_buffer = blosc2.decompress(compressed_bytes)
        return torch.from_numpy(
            np.frombuffer(raw_buffer, dtype=np.float16).copy()
        ).reshape(shape)
    except Exception as exc:
        logging.error("Error decompressing prediction: %s", exc)
        return None


def decode_base64_to_compressed(b64_str: str) -> bytes:
    """Decode a synapse base64 string to compressed bytes."""

    try:
        return base64.b64decode(b64_str)
    except Exception as exc:
        logging.error("Error decoding base64 to compressed bytes: %s", exc)
        return None

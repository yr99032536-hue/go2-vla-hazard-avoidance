"""Wire contract for the three-image SmolVLA NBV observation."""

from __future__ import annotations

import numpy as np

from soarm_nbv.safety import decode_nbv_joint_vector


MARKER = b"OBS3"


def _rgb(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError(f"{name} must be HWC RGB, got {value.shape}")
    return np.ascontiguousarray(value[:, :, :3], dtype=np.uint8)


def encode_observation(
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    guidance_rgb: np.ndarray,
    joint_pos_deg: np.ndarray,
) -> list[bytes]:
    """Encode one immutable observation into an eight-frame ZMQ message."""
    arrays = (
        _rgb(front_rgb, "front_rgb"),
        _rgb(wrist_rgb, "wrist_rgb"),
        _rgb(guidance_rgb, "guidance_rgb"),
    )
    parts = [MARKER]
    for array in arrays:
        parts.extend((np.asarray(array.shape, dtype=np.int32).tobytes(), array.tobytes()))
    parts.append(
        decode_nbv_joint_vector(joint_pos_deg, "joint_pos_deg")
        .astype(np.float32)
        .tobytes()
    )
    return parts


def decode_observation(parts: list[bytes]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Decode OBS3 as front, wrist, guidance, seven external motor degrees."""
    if len(parts) != 8 or parts[0] != MARKER:
        raise ValueError(f"Expected OBS3 multipart with 8 frames, got {len(parts)}")
    images = []
    for shape_index, bytes_index, name in ((1, 2, "front"), (3, 4, "wrist"), (5, 6, "guidance")):
        shape = tuple(int(value) for value in np.frombuffer(parts[shape_index], dtype=np.int32))
        if len(shape) != 3 or shape[2] < 3 or any(value <= 0 for value in shape):
            raise ValueError(f"invalid {name} shape: {shape}")
        expected = int(np.prod(shape))
        flat = np.frombuffer(parts[bytes_index], dtype=np.uint8)
        if flat.size != expected:
            raise ValueError(f"invalid {name} byte count: expected {expected}, got {flat.size}")
        images.append(flat.reshape(shape).copy())
    state = decode_nbv_joint_vector(parts[7], "joint_pos_deg").copy()
    return images[0], images[1], images[2], state

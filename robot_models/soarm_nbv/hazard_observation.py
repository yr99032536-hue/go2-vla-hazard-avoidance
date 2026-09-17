"""ZMQ wire format for language-conditioned binary-alley hazard inference."""

from __future__ import annotations

import numpy as np

from soarm_nbv.hazard_vla_contract import (
    normalize_target_side,
    validate_task,
)
from soarm_nbv.safety import decode_nbv_joint_vector


MARKER = b"HAZARD_OBS2"


def validate_event_id(value: str) -> str:
    event_id = str(value).strip()
    if not event_id or len(event_id.encode("utf-8")) > 128:
        raise ValueError("event_id must contain 1..128 UTF-8 bytes")
    return event_id


def _rgb(value: np.ndarray, name: str) -> np.ndarray:
    value = np.asarray(value)
    if value.ndim != 3 or value.shape[2] < 3:
        raise ValueError(f"{name} must be HWC RGB, got {value.shape}")
    return np.ascontiguousarray(value[:, :, :3], dtype=np.uint8)


def encode_hazard_observation(
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    joint_pos_deg: np.ndarray,
    task: str,
    target_side: str,
    event_id: str,
) -> list[bytes]:
    front = _rgb(front_rgb, "front_rgb")
    wrist = _rgb(wrist_rgb, "wrist_rgb")
    state = decode_nbv_joint_vector(joint_pos_deg, "joint_pos_deg")
    return [
        MARKER,
        validate_event_id(event_id).encode("utf-8"),
        validate_task(task).encode("utf-8"),
        normalize_target_side(target_side).encode("ascii"),
        np.asarray(front.shape, dtype=np.int32).tobytes(),
        front.tobytes(),
        np.asarray(wrist.shape, dtype=np.int32).tobytes(),
        wrist.tobytes(),
        state.astype(np.float32, copy=False).tobytes(),
    ]


def _decode_rgb(shape_raw: bytes, pixels_raw: bytes, name: str) -> np.ndarray:
    shape = tuple(int(value) for value in np.frombuffer(shape_raw, dtype=np.int32))
    if len(shape) != 3 or shape[2] < 3 or any(value <= 0 for value in shape):
        raise ValueError(f"invalid {name} shape: {shape}")
    flat = np.frombuffer(pixels_raw, dtype=np.uint8)
    expected = int(np.prod(shape))
    if flat.size != expected:
        raise ValueError(
            f"invalid {name} byte count: expected {expected}, got {flat.size}"
        )
    return flat.reshape(shape).copy()


def decode_hazard_observation(
    parts: list[bytes],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, str, str, str]:
    if len(parts) != 9 or parts[0] != MARKER:
        raise ValueError(f"Expected HAZARD_OBS2 multipart with 9 frames, got {len(parts)}")
    try:
        event_id = validate_event_id(parts[1].decode("utf-8"))
        task = validate_task(parts[2].decode("utf-8"))
        target_side = normalize_target_side(parts[3].decode("ascii"))
    except UnicodeError as error:
        raise ValueError("event/task/target_side wire text is invalid") from error
    front = _decode_rgb(parts[4], parts[5], "front")
    wrist = _decode_rgb(parts[6], parts[7], "wrist")
    state = decode_nbv_joint_vector(parts[8], "joint_pos_deg").copy()
    return front, wrist, state, task, target_side, event_id

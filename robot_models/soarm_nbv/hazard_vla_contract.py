"""Strict data/runtime contract for the binary-alley hazard VLA task.

The robot arm has seven actuators.  The policy has one additional, non-motor
output used by the navigation supervisor:

    action[0:7] -> arm target in external degrees
    action[7]   -> alley decision (-1 hazard, 0 checking, +1 safe)

Keeping this contract separate prevents the legacy six-axis drawer and the
seven-action active-mapping policies from silently accepting hazard actions.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np

from soarm_nbv.safety import NBV_JOINT_ORDER, decode_nbv_joint_vector


HAZARD_STATE_DIM = len(NBV_JOINT_ORDER)
HAZARD_ACTION_DIM = HAZARD_STATE_DIM + 1
DECISION_NAME = "alley_decision"
HAZARD_ACTION_NAMES = (*NBV_JOINT_ORDER, DECISION_NAME)

DECISION_HAZARD = -1
DECISION_CHECKING = 0
DECISION_SAFE = 1
FINAL_DECISIONS = (DECISION_HAZARD, DECISION_SAFE)

TARGET_SIDES = ("left", "right")
OPENING_CASES = ("left_only", "right_only", "both")

TASK_BY_SIDE = {
    "left": (
        "inspect the left alley with the wrist camera and report whether "
        "the red hazard cube is present"
    ),
    "right": (
        "inspect the right alley with the wrist camera and report whether "
        "the red hazard cube is present"
    ),
}


def normalize_target_side(side: str) -> str:
    normalized = str(side).strip().lower()
    if normalized not in TARGET_SIDES:
        raise ValueError(f"target_side must be one of {TARGET_SIDES}, got {side!r}")
    return normalized


def validate_opening_case(opening_case: str, target_side: str) -> str:
    normalized = str(opening_case).strip().lower()
    if normalized not in OPENING_CASES:
        raise ValueError(
            f"opening_case must be one of {OPENING_CASES}, got {opening_case!r}"
        )
    side = normalize_target_side(target_side)
    if normalized == "left_only" and side != "left":
        raise ValueError("right cannot be inspected in a left_only opening")
    if normalized == "right_only" and side != "right":
        raise ValueError("left cannot be inspected in a right_only opening")
    return normalized


def task_for_side(side: str) -> str:
    return TASK_BY_SIDE[normalize_target_side(side)]


def validate_task(task: str) -> str:
    task = str(task).strip()
    encoded = task.encode("utf-8")
    if not encoded or len(encoded) > 256:
        raise ValueError("task must contain 1..256 UTF-8 bytes")
    return task


def validate_training_decision(value: int | float | np.number) -> np.float32:
    numeric = float(value)
    if not np.isfinite(numeric) or numeric not in (
        DECISION_HAZARD,
        DECISION_CHECKING,
        DECISION_SAFE,
    ):
        raise ValueError("training decision must be exactly -1, 0, or +1")
    return np.float32(numeric)


def decode_hazard_action(raw: bytes | np.ndarray, field_name: str = "hazard action") -> np.ndarray:
    expected_bytes = HAZARD_ACTION_DIM * np.dtype(np.float32).itemsize
    if isinstance(raw, np.ndarray):
        vector = raw
    else:
        if len(raw) != expected_bytes:
            raise ValueError(
                f"{field_name} must contain exactly {expected_bytes} bytes "
                f"({HAZARD_ACTION_DIM} float32 values), got {len(raw)}"
            )
        vector = np.frombuffer(raw, dtype=np.float32).copy()
    if vector.dtype != np.float32:
        raise TypeError(f"{field_name} must be float32, got {vector.dtype}")
    if vector.shape != (HAZARD_ACTION_DIM,):
        raise ValueError(
            f"{field_name} must have shape ({HAZARD_ACTION_DIM},), got {vector.shape}"
        )
    if not np.isfinite(vector).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return vector


def compose_hazard_action(
    arm_target_deg: bytes | np.ndarray,
    decision: int | float | np.number,
) -> np.ndarray:
    arm = decode_nbv_joint_vector(arm_target_deg, "hazard arm target").copy()
    signal = validate_training_decision(decision)
    return np.concatenate((arm, np.asarray([signal], dtype=np.float32))).astype(
        np.float32,
        copy=False,
    )


def split_hazard_action(action: bytes | np.ndarray) -> tuple[np.ndarray, float]:
    vector = decode_hazard_action(action)
    arm = decode_nbv_joint_vector(vector[:HAZARD_STATE_DIM].copy(), "hazard arm target")
    return arm, float(vector[-1])


def classify_policy_decision(value: float, threshold: float = 0.5) -> int:
    """Convert the continuous SmolVLA output into a fail-closed signal."""
    value = float(value)
    threshold = float(threshold)
    if not np.isfinite(value):
        raise ValueError("policy decision must be finite")
    if not 0.0 < threshold < 1.0:
        raise ValueError("decision threshold must be within (0, 1)")
    if value <= -threshold:
        return DECISION_HAZARD
    if value >= threshold:
        return DECISION_SAFE
    return DECISION_CHECKING


def stable_final_decision(
    values: Iterable[float],
    *,
    threshold: float = 0.5,
    required_samples: int = 3,
) -> int:
    """Return a final decision only after matching consecutive outputs."""
    if required_samples < 1:
        raise ValueError("required_samples must be positive")
    classified = [classify_policy_decision(value, threshold) for value in values]
    if len(classified) < required_samples:
        return DECISION_CHECKING
    tail = classified[-required_samples:]
    if tail[0] in FINAL_DECISIONS and all(value == tail[0] for value in tail):
        return tail[0]
    return DECISION_CHECKING

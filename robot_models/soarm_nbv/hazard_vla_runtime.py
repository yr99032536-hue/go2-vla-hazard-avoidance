"""Runtime transport and decision gates for the binary-alley SmolVLA policy.

The learned policy returns one exact ``float32[8]`` action at a time.  The
first seven values are SO-Arm targets and the final value is a non-motor
alley decision.  This module keeps that ABI separate from the legacy six-axis
drawer bridge and makes the decision channel fail closed.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np
import zmq

from soarm_nbv.hazard_observation import encode_hazard_observation, validate_event_id
from soarm_nbv.hazard_episode_collector import body_branch_entry_overshoot_m
from soarm_nbv.hazard_vla_contract import (
    DECISION_CHECKING,
    decode_hazard_action,
    normalize_target_side,
    split_hazard_action,
    stable_final_decision,
)
from soarm_nbv.safety import (
    ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG,
    validate_active_reversed_nbv_joint_limits_deg,
)


@dataclass(frozen=True)
class StableHazardDecision:
    """One terminal policy decision authorized for the navigation supervisor."""

    event_id: str
    target_side: str
    signal: int
    raw_decision: float
    stable_samples: int
    peek_valid_consecutive_frames: int


@dataclass(frozen=True)
class HazardActionEnvelope:
    """One policy output correlated to the exact observation lease."""

    event_id: str
    target_side: str
    action: np.ndarray


ACTION_MARKER = b"HAZARD_ACTION1"


def wrist_clears_entry_plane(position_world_m, entry_point_world_m, entry_normal_world) -> bool:
    """Require the optical center beyond the actual corner plane, not a cached reach.

    The signed-plane calculation is shared with the base guard, but here the
    measured position is the camera and the required clearance is zero.
    """
    return body_branch_entry_overshoot_m(
        position_world_m, entry_point_world_m, entry_normal_world
    ) >= 0.0

HAZARD_ARM_MAX_VELOCITY_DEG_S = np.asarray(
    (60.0, 60.0, 90.0, 90.0, 90.0, 120.0, 100.0),
    dtype=np.float64,
)


def encode_hazard_action_envelope(
    action: bytes | np.ndarray,
    *,
    event_id: str,
    target_side: str,
) -> list[bytes]:
    vector = decode_hazard_action(action, "SmolVLA hazard action")
    return [
        ACTION_MARKER,
        validate_event_id(event_id).encode("utf-8"),
        normalize_target_side(target_side).encode("ascii"),
        vector.astype(np.float32, copy=False).tobytes(),
    ]


def decode_hazard_action_envelope(parts: list[bytes]) -> HazardActionEnvelope:
    if len(parts) != 4 or parts[0] != ACTION_MARKER:
        raise ValueError(
            f"expected HAZARD_ACTION1 multipart with 4 frames, got {len(parts)}"
        )
    try:
        event_id = validate_event_id(parts[1].decode("utf-8"))
        target_side = normalize_target_side(parts[2].decode("ascii"))
    except UnicodeError as error:
        raise ValueError("hazard action event/side wire text is invalid") from error
    action = decode_hazard_action(parts[3], "SmolVLA hazard action").copy()
    return HazardActionEnvelope(event_id, target_side, action)


def consume_correlated_hazard_action(
    envelope: HazardActionEnvelope,
    *,
    expected_event_id: str,
    expected_target_side: str,
    gate: "HazardDecisionGate",
    base_paused: bool,
    peek_pose_valid: bool,
) -> tuple[np.ndarray, StableHazardDecision | None]:
    """Validate correlation, split motors/signal, and advance the safety gate."""

    if envelope.event_id != validate_event_id(expected_event_id):
        raise ValueError("policy output event_id does not match the active lease")
    expected_side = normalize_target_side(expected_target_side)
    if envelope.target_side != expected_side:
        raise ValueError("policy output target_side does not match the active lease")
    arm_target_deg, raw_decision = decode_runtime_hazard_action(envelope.action)
    terminal = gate.update(
        raw_decision,
        base_paused=base_paused,
        peek_pose_valid=peek_pose_valid,
    )
    return arm_target_deg, terminal


def decode_runtime_hazard_action(
    action: bytes | np.ndarray,
) -> tuple[np.ndarray, float]:
    """Saturate learned motor predictions to the configured actuator limits.

    Finite regression outputs are bounded before the velocity-limited actuator
    command. Malformed and nonfinite actions remain errors.
    The independent decision value and event/peek requirements are unchanged.
    """

    vector = decode_hazard_action(action, "runtime hazard action")
    arm_target_deg, raw_decision = split_hazard_action(vector)
    clamped = np.clip(
        arm_target_deg,
        ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG[:, 0],
        ACTIVE_REVERSED_NBV_JOINT_LIMITS_ARRAY_DEG[:, 1],
    ).astype(np.float32, copy=False)
    return clamped, float(raw_decision)


def slew_hazard_arm_target(
    previous_target_deg: np.ndarray,
    requested_target_deg: np.ndarray,
    dt_s: float,
) -> np.ndarray:
    """Rate-limit a validated policy target before it reaches the articulation."""

    previous = np.asarray(previous_target_deg, dtype=np.float64).reshape(-1)
    requested = np.asarray(requested_target_deg, dtype=np.float64).reshape(-1)
    if previous.shape != (7,) or requested.shape != (7,):
        raise ValueError("hazard slew targets must both have shape (7,)")
    if not np.isfinite(previous).all() or not np.isfinite(requested).all():
        raise ValueError("hazard slew targets must be finite")
    if not np.isfinite(dt_s) or dt_s <= 0.0:
        raise ValueError("hazard slew dt_s must be positive and finite")
    validate_active_reversed_nbv_joint_limits_deg(
        requested,
        "requested hazard arm target",
    )
    maximum_delta = HAZARD_ARM_MAX_VELOCITY_DEG_S * float(dt_s)
    return (
        previous + np.clip(requested - previous, -maximum_delta, maximum_delta)
    ).astype(np.float32)


class HazardDecisionGate:
    """Accept a VLA decision only during a proven, stationary wrist peek."""

    def __init__(
        self,
        *,
        required_peek_frames: int = 5,
        required_stable_samples: int = 3,
        threshold: float = 0.5,
    ) -> None:
        if required_peek_frames < 1 or required_stable_samples < 1:
            raise ValueError("peek frames and stable samples must be positive")
        if not 0.0 < float(threshold) < 1.0:
            raise ValueError("decision threshold must be within (0, 1)")
        self.required_peek_frames = int(required_peek_frames)
        self.required_stable_samples = int(required_stable_samples)
        self.threshold = float(threshold)
        self._values: deque[float] = deque(maxlen=self.required_stable_samples)
        self.clear()

    def clear(self) -> None:
        self.event_id: str | None = None
        self.target_side: str | None = None
        self.peek_valid_frames = 0
        self.terminal_emitted = False
        self._values.clear()

    def begin(self, event_id: str, target_side: str) -> None:
        event = str(event_id).strip()
        if not event:
            raise ValueError("hazard policy context requires a non-empty event_id")
        side = normalize_target_side(target_side)
        self.clear()
        self.event_id = event
        self.target_side = side

    def update(
        self,
        raw_decision: float,
        *,
        base_paused: bool,
        peek_pose_valid: bool,
    ) -> StableHazardDecision | None:
        if self.event_id is None or self.target_side is None:
            raise RuntimeError("hazard decision received without an active context")
        value = float(raw_decision)
        if not np.isfinite(value):
            raise ValueError("hazard decision must be finite")
        if self.terminal_emitted:
            return None
        if not base_paused or not peek_pose_valid:
            self.peek_valid_frames = 0
            self._values.clear()
            return None

        self.peek_valid_frames += 1
        if self.peek_valid_frames < self.required_peek_frames:
            self._values.clear()
            return None

        self._values.append(value)
        signal = stable_final_decision(
            self._values,
            threshold=self.threshold,
            required_samples=self.required_stable_samples,
        )
        if signal == DECISION_CHECKING:
            return None
        self.terminal_emitted = True
        return StableHazardDecision(
            event_id=self.event_id,
            target_side=self.target_side,
            signal=int(signal),
            raw_decision=value,
            stable_samples=self.required_stable_samples,
            peek_valid_consecutive_frames=self.peek_valid_frames,
        )


class HazardSmolVLAZmqBridge:
    """Exact 7-motor + 1-decision ZMQ bridge for the hazard policy runner."""

    def __init__(self, obs_port: int, action_port: int) -> None:
        self.context = zmq.Context()
        self.obs_pub = self.context.socket(zmq.PUB)
        self.obs_pub.setsockopt(zmq.SNDHWM, 1)
        self.obs_pub.bind(f"tcp://*:{int(obs_port)}")
        self.action_sub = self.context.socket(zmq.SUB)
        self.action_sub.setsockopt(zmq.RCVHWM, 1)
        self.action_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.action_sub.connect(f"tcp://localhost:{int(action_port)}")

    def close(self) -> None:
        self.obs_pub.close(0)
        self.action_sub.close(0)
        self.context.term()

    def send_reset(self) -> None:
        try:
            self.obs_pub.send_multipart([b"RESET"], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def drain_actions(self) -> int:
        """Discard every queued output without attempting to authorize it."""

        discarded = 0
        while True:
            try:
                self.action_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                return discarded
            discarded += 1

    def publish_observation(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        joint_pos_deg: np.ndarray,
        task: str,
        target_side: str,
        event_id: str,
    ) -> bool:
        try:
            self.obs_pub.send_multipart(
                encode_hazard_observation(
                    front_rgb,
                    wrist_rgb,
                    joint_pos_deg,
                    task,
                    target_side,
                    event_id,
                ),
                flags=zmq.NOBLOCK,
            )
            return True
        except zmq.Again:
            return False

    def receive_action(self) -> HazardActionEnvelope | None:
        latest = None
        while True:
            try:
                parts = self.action_sub.recv_multipart(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            latest = decode_hazard_action_envelope(parts)
        return latest

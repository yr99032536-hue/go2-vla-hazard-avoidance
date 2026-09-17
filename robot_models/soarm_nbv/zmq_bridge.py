"""ZMQ transport for SO-Arm NBV observation/action exchange.

This module intentionally contains no Isaac Sim or GR00T imports. It only
serializes numpy arrays over two PUB/SUB channels:

- observation channel: sim -> policy
- action channel: policy -> sim
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from soarm_nbv.safety import decode_joint_vector
import zmq


DEFAULT_OBS_PORT = 5555
DEFAULT_ACTION_PORT = 5556


@dataclass(frozen=True)
class ZmqEndpointConfig:
    obs_host: str = "*"
    action_host: str = "localhost"
    obs_port: int = DEFAULT_OBS_PORT
    action_port: int = DEFAULT_ACTION_PORT


@dataclass
class SoArmObservation:
    """Transport payload from Isaac Sim to policy."""

    room_rgb: np.ndarray
    wrist_rgb: np.ndarray
    joint_pos_deg: np.ndarray

    def validate(self) -> None:
        for name, image in (("room_rgb", self.room_rgb), ("wrist_rgb", self.wrist_rgb)):
            if image.dtype != np.uint8:
                raise TypeError(f"{name} must be uint8, got {image.dtype}")
            if image.ndim != 3 or image.shape[-1] != 3:
                raise ValueError(f"{name} must have shape HxWx3, got {image.shape}")
        if self.joint_pos_deg.dtype != np.float32:
            raise TypeError(f"joint_pos_deg must be float32, got {self.joint_pos_deg.dtype}")
        decode_joint_vector(self.joint_pos_deg, "joint_pos_deg")


@dataclass
class SoArmAction:
    """Transport payload from policy to Isaac Sim."""

    joint_target_deg: np.ndarray

    def validate(self) -> None:
        if self.joint_target_deg.dtype != np.float32:
            raise TypeError(f"joint_target_deg must be float32, got {self.joint_target_deg.dtype}")
        decode_joint_vector(self.joint_target_deg, "joint_target_deg")


class ObservationPublisher:
    """PUB socket owned by the Isaac Sim process."""

    def __init__(self, config: ZmqEndpointConfig):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.bind(f"tcp://{config.obs_host}:{config.obs_port}")

    def publish(self, obs: SoArmObservation) -> None:
        obs.validate()
        self._socket.send_multipart(
            [
                np.asarray(obs.room_rgb.shape, dtype=np.int32).tobytes(),
                obs.room_rgb.tobytes(),
                np.asarray(obs.wrist_rgb.shape, dtype=np.int32).tobytes(),
                obs.wrist_rgb.tobytes(),
                obs.joint_pos_deg.tobytes(),
            ]
        )

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


class ObservationSubscriber:
    """SUB socket owned by the policy process."""

    def __init__(self, config: ZmqEndpointConfig, timeout_ms: int = 1000):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.connect(f"tcp://{config.action_host}:{config.obs_port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

    def receive(self) -> SoArmObservation | None:
        try:
            parts = self._socket.recv_multipart()
        except zmq.Again:
            return None
        if len(parts) != 5:
            raise ValueError(f"Expected 5 observation frames, got {len(parts)}")

        room_shape = tuple(np.frombuffer(parts[0], dtype=np.int32).tolist())
        room_rgb = np.frombuffer(parts[1], dtype=np.uint8).reshape(room_shape).copy()
        wrist_shape = tuple(np.frombuffer(parts[2], dtype=np.int32).tolist())
        wrist_rgb = np.frombuffer(parts[3], dtype=np.uint8).reshape(wrist_shape).copy()
        joint_pos_deg = np.frombuffer(parts[4], dtype=np.float32).copy()
        obs = SoArmObservation(room_rgb=room_rgb, wrist_rgb=wrist_rgb, joint_pos_deg=joint_pos_deg)
        obs.validate()
        return obs

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


class ActionPublisher:
    """PUB socket owned by the policy process."""

    def __init__(self, config: ZmqEndpointConfig):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.PUB)
        self._socket.setsockopt(zmq.SNDHWM, 1)
        if hasattr(zmq, "TCP_NODELAY"):
            self._socket.setsockopt(zmq.TCP_NODELAY, 1)
        self._socket.bind(f"tcp://{config.obs_host}:{config.action_port}")

    def publish(self, action: SoArmAction) -> None:
        action.validate()
        self._socket.send(action.joint_target_deg.tobytes())

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()


class ActionSubscriber:
    """SUB socket owned by the Isaac Sim process."""

    def __init__(self, config: ZmqEndpointConfig, timeout_ms: int = 0):
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.SUB)
        self._socket.setsockopt(zmq.RCVHWM, 1)
        self._socket.setsockopt(zmq.CONFLATE, 1)
        if hasattr(zmq, "TCP_NODELAY"):
            self._socket.setsockopt(zmq.TCP_NODELAY, 1)
        self._socket.connect(f"tcp://{config.action_host}:{config.action_port}")
        self._socket.setsockopt_string(zmq.SUBSCRIBE, "")
        self._socket.setsockopt(zmq.RCVTIMEO, timeout_ms)

    def receive(self) -> SoArmAction | None:
        try:
            raw = self._socket.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        action = SoArmAction(joint_target_deg=decode_joint_vector(raw, "joint_target_deg"))
        action.validate()
        return action

    def close(self) -> None:
        self._socket.close(0)
        self._context.term()

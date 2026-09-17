"""One-shot SmolVLA proposal client for the maze scan lane.

The heavyweight SmolVLA runner lives in the LeRobot environment and speaks the
legacy ZMQ protocol: OBS multipart frames in, one conflated six-value action
frame out. This client bursts a few guided observations and returns the first
fresh action as an advisory proposal. It never commands the arm directly; the
supervisor safety gate validates every proposal and falls back to the oracle.
"""

from __future__ import annotations

import os
import time

import numpy as np


_SIDE_TINT = {
    "left": (40, 40, 0),
    "center": (40, 0, 40),
    "right": (0, 40, 40),
}


def guidance_overlay(front_rgb: np.ndarray, side: str) -> np.ndarray:
    """Tint the target third of camera3 so the proposal points that way."""
    guided = front_rgb.copy()
    width = guided.shape[1]
    band = width // 3
    start = {"left": 0, "center": band, "right": 2 * band}[side]
    tint = np.asarray(_SIDE_TINT[side], dtype=np.uint8)
    region = guided[:, start : start + band]
    guided[:, start : start + band] = np.clip(
        region.astype(np.uint16) + tint, 0, 255
    ).astype(np.uint8)
    return guided


class SmolVlaProposer:
    def __init__(
        self,
        obs_port: int | None = None,
        action_port: int | None = None,
        timeout_s: float | None = None,
    ) -> None:
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError(
                "MAZE_PROPOSER=smolvla requires pyzmq in the supervisor environment"
            ) from error
        self._zmq = zmq
        self.obs_port = int(obs_port or os.environ.get("SMOLVLA_OBS_PORT", "5565"))
        self.action_port = int(action_port or os.environ.get("SMOLVLA_ACTION_PORT", "5566"))
        self.timeout_s = float(timeout_s or os.environ.get("SMOLVLA_PROPOSAL_TIMEOUT_S", "8.0"))
        zmq = self._zmq
        self._context = zmq.Context()
        self._actions = self._context.socket(zmq.SUB)
        self._actions.setsockopt_string(zmq.SUBSCRIBE, "")
        self._actions.setsockopt(zmq.RCVHWM, 4)
        self._actions.connect(f"tcp://localhost:{self.action_port}")
        self._obs = self._context.socket(zmq.PUB)
        self._obs.setsockopt(zmq.SNDHWM, 4)
        self._obs.connect(f"tcp://localhost:{self.obs_port}")
        # SUB/PUB wiring is asynchronous; give the runner a moment to connect.
        time.sleep(0.5)

    def propose(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        joint_deg: np.ndarray,
        side: str,
    ) -> np.ndarray:
        guided = guidance_overlay(front_rgb, side)
        room = np.ascontiguousarray(guided, dtype=np.uint8)
        wrist = np.ascontiguousarray(wrist_rgb, dtype=np.uint8)
        room_shape = np.asarray(room.shape, dtype=np.int32)
        wrist_shape = np.asarray(wrist.shape, dtype=np.int32)
        state = np.asarray(joint_deg, dtype=np.float64)

        deadline = time.monotonic() + self.timeout_s
        # Drain stale actions published before this request.
        while True:
            try:
                self._actions.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break

        sent_at = time.monotonic()
        last_action = None
        while time.monotonic() < deadline:
            self._obs.send_multipart(
                [
                    b"OBS",
                    room_shape.tobytes(),
                    room.tobytes(),
                    wrist_shape.tobytes(),
                    wrist.tobytes(),
                    state.tobytes(),
                ],
                flags=zmq.NOBLOCK,
            )
            self._actions.setsockopt(zmq.RCVTIMEO, 200)
            try:
                payload = self._actions.recv()
                if len(payload) == 6 * 4:
                    candidate = np.frombuffer(payload, dtype=np.float32).astype(np.float64)
                    last_action = candidate
                    if time.monotonic() - sent_at > 0.2:
                        break
            except zmq.Again:
                pass
            time.sleep(0.1)
        if last_action is None:
            raise TimeoutError("no SmolVLA action arrived before the proposal deadline")
        return last_action

    def close(self) -> None:
        self._actions.close(0)
        self._obs.close(0)
        self._context.term()

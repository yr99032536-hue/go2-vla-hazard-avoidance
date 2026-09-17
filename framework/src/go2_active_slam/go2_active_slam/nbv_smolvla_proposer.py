"""Advisory SmolVLA client for the three-image NBV runner."""

from __future__ import annotations

import os
import time

import numpy as np


def encode_obs3(
    front_rgb: np.ndarray,
    wrist_rgb: np.ndarray,
    guidance_rgb: np.ndarray,
    state_external_deg: np.ndarray,
) -> list[bytes]:
    images = []
    for name, image in (
        ("front", front_rgb),
        ("wrist", wrist_rgb),
        ("guidance", guidance_rgb),
    ):
        value = np.asarray(image)
        if value.ndim != 3 or value.shape[2] < 3:
            raise ValueError(f"{name} must be HWC RGB")
        images.append(np.ascontiguousarray(value[:, :, :3], dtype=np.uint8))
    state = np.asarray(state_external_deg, dtype=np.float32)
    if state.shape != (7,) or not np.all(np.isfinite(state)):
        raise ValueError("state_external_deg must contain seven finite values")
    parts = [b"OBS3"]
    for image in images:
        parts.extend((np.asarray(image.shape, dtype=np.int32).tobytes(), image.tobytes()))
    parts.append(state.tobytes())
    return parts


class NbVlaProposer:
    """Send a frozen snapshot and return a proposal; never command the arm."""

    def __init__(self, obs_port: int | None = None, action_port: int | None = None, timeout_s: float = 8.0) -> None:
        try:
            import zmq
        except ImportError as error:
            raise RuntimeError("NBV SmolVLA proposal mode requires pyzmq") from error
        self._zmq = zmq
        self.timeout_s = float(timeout_s)
        self._context = zmq.Context()
        self._actions = self._context.socket(zmq.SUB)
        self._actions.setsockopt_string(zmq.SUBSCRIBE, "")
        self._actions.setsockopt(zmq.RCVHWM, 1)
        self._actions.connect(f"tcp://localhost:{int(action_port or os.environ.get('NBV_ACTION_PORT', '5576'))}")
        self._observations = self._context.socket(zmq.PUB)
        self._observations.setsockopt(zmq.SNDHWM, 1)
        self._observations.connect(f"tcp://localhost:{int(obs_port or os.environ.get('NBV_OBS_PORT', '5575'))}")
        time.sleep(0.5)

    def propose(
        self,
        front_rgb: np.ndarray,
        wrist_rgb: np.ndarray,
        guidance_rgb: np.ndarray,
        state_external_deg: np.ndarray,
    ) -> np.ndarray:
        parts = encode_obs3(front_rgb, wrist_rgb, guidance_rgb, state_external_deg)
        while True:
            try:
                self._actions.recv(flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                break
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            try:
                self._observations.send_multipart(parts, flags=self._zmq.NOBLOCK)
            except self._zmq.Again:
                pass
            self._actions.setsockopt(self._zmq.RCVTIMEO, 200)
            try:
                payload = self._actions.recv()
                if len(payload) == 28:
                    proposal = np.frombuffer(payload, dtype=np.float32).astype(np.float64)
                    if np.all(np.isfinite(proposal)):
                        return proposal
            except self._zmq.Again:
                pass
            time.sleep(0.05)
        raise TimeoutError("no fresh NBV SmolVLA proposal arrived before deadline")

    def reset(self) -> None:
        self._observations.send_multipart([b"RESET"])

    def close(self) -> None:
        self._actions.close(0)
        self._observations.close(0)
        self._context.term()

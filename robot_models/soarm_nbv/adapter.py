"""Adapters between SO-Arm simulation payloads and GR00T policy payloads."""

from __future__ import annotations

import numpy as np

from soarm_nbv.safety import clamp_joint_targets_deg, decode_joint_vector
from soarm_nbv.zmq_bridge import SoArmAction, SoArmObservation


DEFAULT_TASK_PROMPT = "Move the wrist camera to inspect the hidden area and improve visibility."


def add_batch_time(value: np.ndarray) -> np.ndarray:
    return value[np.newaxis, np.newaxis, ...]


def observation_to_gr00t(obs: SoArmObservation, task_prompt: str = DEFAULT_TASK_PROMPT) -> dict:
    """Convert transport observation to the GR00T SO100/SO_ARM policy API shape."""
    obs.validate()
    single_arm = obs.joint_pos_deg[:5]
    gripper = obs.joint_pos_deg[5:6]

    return {
        "video": {
            "room": add_batch_time(obs.room_rgb.astype(np.uint8)),
            "wrist": add_batch_time(obs.wrist_rgb.astype(np.uint8)),
        },
        "state": {
            "single_arm": add_batch_time(single_arm),
            "gripper": add_batch_time(gripper),
        },
        "language": {
            "annotation.human.task_description": [[task_prompt]],
        },
    }


def observation_to_gr00t_n1_5(obs: SoArmObservation, task_prompt: str = DEFAULT_TASK_PROMPT) -> dict:
    """Convert transport observation to the GR00T N1.5 flat policy API shape."""
    obs.validate()
    single_arm = obs.joint_pos_deg[:5]
    gripper = obs.joint_pos_deg[5:6]

    return {
        "video.room": obs.room_rgb[np.newaxis, ...].astype(np.uint8),
        "video.wrist": obs.wrist_rgb[np.newaxis, ...].astype(np.uint8),
        "state.single_arm": single_arm[np.newaxis, ...],
        "state.gripper": gripper[np.newaxis, ...],
        "annotation.human.task_description": np.asarray([task_prompt]),
    }


def _first_action_component(value: object, field_name: str, width: int) -> np.ndarray:
    """Return the first batch/timestep component from supported GR00T output shapes."""
    component = np.asarray(value)
    if component.ndim == 2:
        component = component[np.newaxis, ...]
    elif component.ndim != 3:
        raise ValueError(f"{field_name} must have 2 or 3 dimensions, got shape {component.shape}")
    if component.shape[-1] != width:
        raise ValueError(f"{field_name} must contain exactly {width} values, got shape {component.shape}")
    selected = component[0, 0]
    if not np.isfinite(selected).all():
        raise ValueError(f"{field_name} must contain only finite values")
    return np.asarray(selected, dtype=np.float32)


def gr00t_action_to_soarm(action_chunk: dict) -> SoArmAction:
    """Convert a GR00T action chunk to a single SO-Arm joint target vector."""
    single_arm = action_chunk.get("single_arm")
    gripper = action_chunk.get("gripper")
    if single_arm is None:
        single_arm = action_chunk.get("action.single_arm")
    if gripper is None:
        gripper = action_chunk.get("action.gripper")
    if single_arm is None:
        raise ValueError("GR00T action is missing single_arm or action.single_arm")
    if gripper is None:
        raise ValueError("GR00T action is missing gripper or action.gripper")

    arm_values = _first_action_component(single_arm, "single_arm", 5)
    gripper_value = _first_action_component(gripper, "gripper", 1)
    joint_target = np.concatenate((arm_values, gripper_value)).astype(np.float32, copy=False)
    joint_target = decode_joint_vector(joint_target, "joint_target_deg")
    action = SoArmAction(joint_target_deg=clamp_joint_targets_deg(joint_target))
    action.validate()
    return action


def neutral_action_deg() -> SoArmAction:
    """Neutral-ish SO-Arm target used for dry-run transport tests."""
    return SoArmAction(joint_target_deg=np.array([0.0, -28.0, 57.0, 35.0, -50.0, 5.0], dtype=np.float32))

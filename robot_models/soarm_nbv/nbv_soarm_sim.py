"""Minimal Isaac Sim SO-Arm NBV process.

This simulation intentionally excludes Go2, warehouse assets, locomotion policy,
and unrelated scene complexity. It only owns:

- SO-101 arm
- wrist camera
- ZMQ observation publish
- ZMQ action receive
- optional action application
"""

from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import torch

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROBOT_MODELS_DIR = os.path.dirname(SCRIPT_DIR)
if ROBOT_MODELS_DIR not in sys.path:
    sys.path.insert(0, ROBOT_MODELS_DIR)

from isaaclab.app import AppLauncher

from soarm_nbv.safety import SOARM_JOINT_ORDER, ActionSmoother, clamp_joint_targets_deg, deg_to_rad, rad_to_deg
from soarm_nbv.zmq_bridge import ActionSubscriber, ObservationPublisher, SoArmObservation, ZmqEndpointConfig


parser = argparse.ArgumentParser(description="Minimal SO-Arm NBV Isaac Sim process.")
parser.add_argument("--obs-port", type=int, default=5555)
parser.add_argument("--action-port", type=int, default=5556)
parser.add_argument("--apply-actions", action="store_true")
parser.add_argument("--smooth-alpha", type=float, default=0.25)
parser.add_argument("--publish-every", type=int, default=2, help="Publish every N sim steps.")
parser.add_argument("--target-box", action="store_true", help="Spawn a small yellow target box for visual debugging.")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app


import isaaclab.sim as sim_utils  # noqa: E402
from isaaclab.assets import ArticulationCfg, AssetBaseCfg  # noqa: E402
from isaaclab.actuators import ImplicitActuatorCfg  # noqa: E402
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg  # noqa: E402
from isaaclab.sensors.camera import CameraCfg  # noqa: E402
from isaaclab.sim.views import XformPrimView  # noqa: E402
from isaaclab.utils import configclass, math as math_utils  # noqa: E402


SO101_URDF = "/home/iy/robot_models/SO-ARM100/Simulation/SO101/so101_new_calib.urdf"


@configclass
class SoArmNbvSceneCfg(InteractiveSceneCfg):
    ground = AssetBaseCfg(prim_path="/World/ground", spawn=sim_utils.GroundPlaneCfg())
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))

    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=SO101_URDF,
            fix_base=True,
            make_instanceable=False,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=400.0, damping=20.0)
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.1),
            joint_pos={
                "shoulder_pan": 0.0,
                "shoulder_lift": -0.49,
                "elbow_flex": 0.99,
                "wrist_flex": 0.61,
                "wrist_roll": -0.87,
                "gripper": 0.09,
            },
        ),
        actuators={
            "arm": ImplicitActuatorCfg(
                joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
                velocity_limit=2.0,
                effort_limit=100.0,
                stiffness=400.0,
                damping=20.0,
            ),
            "gripper": ImplicitActuatorCfg(
                joint_names_expr=["gripper"],
                velocity_limit=2.0,
                effort_limit=100.0,
                stiffness=100.0,
                damping=2.0,
            ),
        },
    )

    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_camera",
        update_period=0.0,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=400.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=(0.0, 0.0, 0.0), rot=(0.5, -0.5, 0.5, -0.5), convention="ros"),
        update_latest_camera_pose=False,
    )

    target_box = AssetBaseCfg(
        prim_path="/World/TargetBox",
        spawn=sim_utils.MeshCuboidCfg(
            size=(0.05, 0.1, 0.05),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 0.0)),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.4, 0.0, 0.05)),
    )


def setup_camera_pose_sync(device: str):
    wrist_view = XformPrimView("/World/envs/env_0/Robot/gripper_link/wrist_camera", device="cpu")
    wrist_offset_pos = torch.tensor([[0.0, 0.0, 0.0]], device="cpu")
    wrist_offset_rot = torch.tensor([[0.5, -0.5, 0.5, -0.5]], device="cpu")
    return wrist_view, wrist_offset_pos, wrist_offset_rot


def main() -> None:
    sim_cfg = sim_utils.SimulationCfg(dt=0.01, device=args_cli.device)
    sim = sim_utils.SimulationContext(sim_cfg)
    scene_cfg = SoArmNbvSceneCfg(num_envs=1, env_spacing=2.0)
    if not args_cli.target_box:
        scene_cfg.target_box = None
    scene = InteractiveScene(scene_cfg)
    sim.reset()

    robot = scene["robot"]
    wrist_cam = scene.sensors["wrist_camera"]

    joint_ids, joint_names = robot.find_joints(list(SOARM_JOINT_ORDER), preserve_order=True)
    if tuple(joint_names) != SOARM_JOINT_ORDER:
        raise RuntimeError(f"Unexpected SO-Arm joint order: {joint_names}")
    joint_ids_tensor = torch.tensor(joint_ids, device=sim.device, dtype=torch.long)
    ee_body_idx = robot.find_bodies("gripper_link")[0][0]

    wrist_view, wrist_pos, wrist_rot = setup_camera_pose_sync(sim.device)

    endpoints = ZmqEndpointConfig(obs_port=args_cli.obs_port, action_port=args_cli.action_port)
    obs_pub = ObservationPublisher(endpoints)
    action_sub = ActionSubscriber(endpoints)
    smoother = ActionSmoother(alpha=args_cli.smooth_alpha)
    joint_targets = robot.data.default_joint_pos.clone()

    print("\n>>> SO-Arm NBV minimal sim started")
    print(f">>> URDF: {SO101_URDF}")
    print(f">>> Observation PUB: tcp://*:{args_cli.obs_port}")
    print(f">>> Action SUB: tcp://localhost:{args_cli.action_port}")
    print(f">>> Apply actions: {args_cli.apply_actions}")
    print(f">>> Joint order: {', '.join(joint_names)}")
    print(">>> Scene: SO-Arm + wrist camera only")
    print(">>> GR00T room input is filled by duplicating wrist camera frames\n")

    step_count = 0
    try:
        while simulation_app.is_running():
            action = action_sub.receive()
            if action is not None:
                target_deg = clamp_joint_targets_deg(action.joint_target_deg)
                target_deg = smoother.update(target_deg)
                if args_cli.apply_actions:
                    target_rad = torch.as_tensor(deg_to_rad(target_deg), device=sim.device, dtype=joint_targets.dtype)
                    joint_targets[:, joint_ids_tensor] = target_rad
                    robot.set_joint_position_target(joint_targets)
                print(f">>> action {'applied' if args_cli.apply_actions else 'observed'} deg: {target_deg}")

            scene.write_data_to_sim()
            sim.step()

            with torch.no_grad():
                ee_pos = robot.data.body_state_w[:, ee_body_idx, :3].cpu()
                ee_quat = robot.data.body_state_w[:, ee_body_idx, 3:7].cpu()
                wrist_w_pos, wrist_w_quat = math_utils.combine_frame_transforms(ee_pos, ee_quat, wrist_pos, wrist_rot)
                wrist_view.set_world_poses(wrist_w_pos, wrist_w_quat)

            scene.update(sim_cfg.dt)

            if step_count % args_cli.publish_every == 0:
                wrist_out = wrist_cam.data.output
                if "rgb" in wrist_out:
                    wrist_rgb = wrist_out["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                    joint_pos_rad = robot.data.joint_pos[0, joint_ids_tensor].detach().cpu().numpy().astype(np.float32)
                    obs_pub.publish(
                        SoArmObservation(
                            room_rgb=wrist_rgb,
                            wrist_rgb=wrist_rgb,
                            joint_pos_deg=rad_to_deg(joint_pos_rad),
                        )
                    )
            step_count += 1
    finally:
        obs_pub.close()
        action_sub.close()
        simulation_app.close()


if __name__ == "__main__":
    main()

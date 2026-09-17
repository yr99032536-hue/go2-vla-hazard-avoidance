import argparse
import struct
import json
import zlib
import os
import math
import sys
import numpy as np
import torch
import zmq
import subprocess
import signal
import atexit
import time
from pathlib import Path

ROBOT_MODELS_ROOT = Path(__file__).resolve().parents[2]
if str(ROBOT_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_MODELS_ROOT))
from soarm_nbv.robot_model_profile import (
    ROBOT_MODEL_PROFILES,
    NBV_EXTERNAL_JOINT_ORDER,
    CUSTOM_LEADER_CONTROL_ORDER as LEADER_CONTROL_JOINT_ORDER,
    clip_custom_leader_action as clip_leader_action,
    custom_leader_action_deg_to_reversed_sim_rad as leader_action_deg_to_sim_arm_rad,
    external_deg_to_sim_rad,
    nbv_external_deg_to_sim_rad,
    nbv_sim_rad_to_external_deg,
    sim_rad_to_external_deg,
    reversed_sim_rad_to_custom_leader_action_deg as sim_arm_rad_to_leader_action_deg,
    validate_elbow_rotate_deg,
)
from soarm_nbv.locomotion_failure_detector import (
    LocomotionFailureConfig,
    LocomotionFailureDetector,
)

# MCP 익스텐션 자동 활성화
mcp_ext_path = "/home/iy/Documents/isaac-sim-mcp"
if os.path.exists(mcp_ext_path) and "--kit_args" not in sys.argv:
    sys.argv.append("--kit_args")
    sys.argv.append(f"--ext-folder={mcp_ext_path} --enable=isaac.sim.mcp_extension")

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description="Go2 + SO-Arm 시뮬레이션 (기본 평면 맵)")
parser.add_argument('--maze_visual_appearance', action='store_true', help='Apply reviewed yellow-arm/black-motor visual materials.')
parser.add_argument('--maze_inspection_layout', type=Path, help='Ordinary obstacle-maze geometry for physical stop-and-look teacher.')
parser.add_argument('--maze_locomotion_only', action='store_true', help='Known-map collision-only locomotion diagnostic; no arm inspection or visual decisions.')
parser.add_argument('--maze_nav2', action='store_true', help='Actual Nav2 Smac lattice and MPPI known-map locomotion; arm folded.')
parser.add_argument('--maze_nav2_inspection', action='store_true', help='Pause Nav2 at sensed corners for physical arm RGB-D inspection.')
parser.add_argument('--maze_observation_navigation', action='store_true', help='Authorize Nav2 motion only through camera-observed space, not detected alleys.')
parser.add_argument('--maze_observed_navigation', action='store_true', help='Plan with known walls but only RGB-D-observed cube obstacles.')
parser.add_argument("--enable_gr00t", action="store_true", help="Enable ZMQ bridge to a GR00T SO-Arm policy node.")
parser.add_argument("--gr00t_obs_port", type=int, default=5555, help="ZMQ PUB port for camera/state observations.")
parser.add_argument("--gr00t_action_port", type=int, default=5556, help="ZMQ SUB port for GR00T arm actions.")
parser.add_argument(
    "--show_camera_viewport",
    action="store_true",
    help="Show a visible wrist camera viewport. Camera observations remain active without this flag.",
)
parser.add_argument(
    "--show_free_camera_viewport",
    action="store_true",
    help="Show an independent user-controlled perspective viewport in addition to sensor views.",
)
parser.add_argument(
    "--disable_wrist_camera",
    action="store_true",
    help="Do not spawn the SO-ARM wrist camera; the Go2 front camera remains available.",
)
parser.add_argument(
    "--gr00t_apply_actions",
    action="store_true",
    help="Apply received GR00T actions to SO-Arm joint targets. Without this, observations are published only.",
)
parser.add_argument("--enable_smolvla_policy", action="store_true", help="Enable 20k-step SmolVLA policy inference; RIGHT toggles policy control.")
parser.add_argument("--smolvla_obs_port", type=int, default=5565, help="ZMQ PUB port for SmolVLA camera/state observations.")
parser.add_argument("--smolvla_action_port", type=int, default=5566, help="ZMQ SUB port for SmolVLA arm actions.")
parser.add_argument("--smolvla_policy_path", default="/home/iy/Isaac/Robotics/data/smolvla_runs/go2_soarm_open_top_drawer_20k/checkpoints/020000/pretrained_model")
parser.add_argument("--smolvla_task", default="open the top drawer", help="Language instruction sent to the SmolVLA policy.")
parser.add_argument("--smolvla_fps", type=float, default=30.0, help="Observation publish FPS for SmolVLA policy inference.")
parser.add_argument("--smolvla_device", default="cuda", help="SmolVLA inference device used by the LeRobot subprocess.")
parser.add_argument("--smolvla_no_amp", action="store_true", help="Disable CUDA bfloat16 autocast in the SmolVLA subprocess.")
parser.add_argument("--smolvla_action_log_every", type=int, default=20, help="Print every N received SmolVLA actions. 0 disables per-action logs.")
parser.add_argument("--smolvla_script", default="/home/iy/Isaac/Robotics/robot_models/soarm_nbv/smolvla_policy_runner.py")
parser.add_argument("--smolvla_python", default="/home/iy/miniconda3/envs/lerobot/bin/python")
parser.add_argument("--smolvla_log", default="/tmp/soarm_smolvla_policy.log")
parser.add_argument(
    "--enable_hazard_smolvla_policy",
    action="store_true",
    help=(
        "Enable the binary-alley SmolVLA runtime with an exact seven-motor "
        "+ one-decision action contract. Activation follows the supervisor context."
    ),
)
parser.add_argument("--hazard_smolvla_obs_port", type=int, default=5585)
parser.add_argument("--hazard_smolvla_action_port", type=int, default=5586)
parser.add_argument("--hazard_smolvla_policy_path", default="")
parser.add_argument("--hazard_smolvla_fps", type=float, default=30.0)
parser.add_argument("--hazard_smolvla_device", default="cuda")
parser.add_argument("--hazard_smolvla_no_amp", action="store_true")
parser.add_argument("--hazard_smolvla_action_log_every", type=int, default=15)
parser.add_argument(
    "--hazard_smolvla_script",
    default=(
        "/home/iy/Isaac/Robotics/robot_models/soarm_nbv/"
        "smolvla_hazard_policy_runner.py"
    ),
)
parser.add_argument(
    "--hazard_smolvla_python",
    default="/home/iy/miniconda3/envs/lerobot/bin/python",
)
parser.add_argument(
    "--hazard_smolvla_log",
    default="/tmp/soarm_smolvla_hazard_policy.log",
)
parser.add_argument(
    "--go2_policy_path",
    default="/home/iy/Isaac/IsaacLab/logs/rsl_rl/unitree_go2_so101_7motor_reversed_flat/2026-07-26_07-04-08/exported/policy.pt",
    help="Go2 leg policy TorchScript path. Defaults to the verified flat model_7000 export.",
)
parser.add_argument(
    "--go2_policy_obs_mode",
    choices=("rough", "flat"),
    default="flat",
    help="Observation layout for the Go2 policy: rough=247-dim height-scan policy, flat=48-dim default flat policy.",
)
parser.add_argument(
    "--robot_model",
    choices=("legacy", "so101_7motor", "so101_7motor_reversed"),
    default="so101_7motor_reversed",
    help="Robot model profile. Defaults to the model used by the verified mixed walking policy.",
)
parser.add_argument(
    "--no_arm",
    action="store_true",
    help="Spawn the pure Go2 URDF with no arm, wrist camera, or arm actuators.",
)
parser.add_argument(
    "--elbow_rotate_deg",
    type=float,
    default=0.0,
    help="Local held elbow_rotate target in degrees for so101_7motor.",
)
parser.add_argument('--leader_auto', action='store_true', help='Auto-start leader_teleop_bridge.py as background subprocess (teleoperation).')
parser.add_argument(
    '--leader_apply_only_when_base_paused',
    action='store_true',
    help=(
        'Buffer physical leader-arm input while the Go2 base is moving and apply it only '
        'after /active_slam/base_pause is true. Intended for supervised manual NBV pose setup.'
    ),
)
parser.add_argument('--leader_port_dev', default='/dev/ttyACM0', help='Serial port of the physical SO leader arm.')
parser.add_argument('--leader_type', default='so101_leader', choices=('so100_leader', 'so101_leader'))
parser.add_argument('--leader_id', default='teleop_leader_v1', help='Calibration id (MUST match a file in calibration/teleoperators/so_leader/).')
parser.add_argument('--leader_fps', type=float, default=30.0)
parser.add_argument('--leader_action_log_every', type=int, default=20, help='Print every N received leader actions. 0 disables per-action logs.')
parser.add_argument('--leader_script', default='/home/iy/Isaac/Robotics/robot_models/soarm_nbv/leader_bridge_7dof.py')
parser.add_argument('--leader_python', default='/home/iy/miniconda3/envs/lerobot/bin/python')
parser.add_argument('--leader_log', default='/tmp/soarm_leader_teleop.log')
parser.add_argument('--leader_calibration_dir', default='/home/iy/lerobot/calibration/teleoperators/so_leader')
parser.add_argument('--runtime_offset_json', default='/tmp/soarm_runtime_offset.json')
parser.add_argument('--haptic_port', type=int, default=5557, help='ZMQ PUB port for grip force feedback to leader arm')
parser.add_argument('--gripper_port', type=int, default=5558, help='ZMQ SUB port carrying the raw leader gripper motor position')
parser.add_argument('--haptic_max_torque', type=float, default=0.5, help='Grip torque (N·m) mapped to full haptic resistance on leader')
parser.add_argument('--haptic_min_torque', type=float, default=0.05, help='Minimum grip torque (N·m) required before haptic feedback.')
parser.add_argument('--haptic_debug_every', type=int, default=0, help='Print haptic diagnostics every N sim steps. 0 disables.')
parser.add_argument('--enable_haptic_feedback', action='store_true', help='Enable experimental force feedback writes to the physical leader gripper. Disabled by default.')

parser.add_argument('--collect', action='store_true', help='데이터 수집 모드 (SmolVLA 학습용 에피소드 저장)')
parser.add_argument('--task', default='open the drawer', help='수집 데모의 언어 명령(task)')
parser.add_argument('--collect_out_dir', default='/home/iy/Isaac/Robotics/data/go2_soarm_drawer', help='수집 데이터 저장 디렉토리')
parser.add_argument('--collect_every', type=int, default=0, help='DEPRECATED: use --collect_fps. 0 uses FPS-based sampling.')
parser.add_argument('--collect_fps', type=float, default=30.0, help='데이터 수집 FPS. 200Hz sim step에서 시간 누적으로 30Hz에 맞춰 샘플링.')
parser.add_argument('--collect_pause_between_episodes', action='store_true', help='각 episode 저장 후 시뮬레이션 Pause. 시작 상태 재정렬 후 Play로 다음 episode 진행.')
parser.add_argument('--collect_manual_right_arrow', action='store_true', help='RIGHT 순서: 시작 -> 저장 -> 초기화 -> 다음 에피소드 시작.')
parser.add_argument('--episode_len', type=int, default=300, help='에피소드당 프레임 수. 기본 300frame=30Hz 기준 10초')
parser.add_argument('--max_episodes', type=int, default=30, help='최대 에피소드 수')
parser.add_argument(
    '--collect_hazard_vla',
    action='store_true',
    help='GUI human-teacher collection for the binary-alley state(7)/action(8) VLA task.',
)
parser.add_argument(
    '--collect_hazard_nbv_teacher',
    action='store_true',
    help='GUI automatic collection from the scripted wrist-RGB NBV teacher.',
)
parser.add_argument(
    '--hazard_collect_out_dir',
    default='/home/iy/Isaac/Robotics/data/binary_alley_hazard_human',
    help='Output root for accepted/rejected binary-alley human demonstrations.',
)
parser.add_argument('--hazard_collect_fps', type=float, default=30.0)
parser.add_argument(
    '--binary_tree_repeat_reset_status_file',
    type=Path,
    default=None,
    help=(
        'Optional atomic JSON acknowledgement file for reset-in-place binary-tree '
        'collection laps. Used only with --collect_hazard_nbv_teacher.'
    ),
)
parser.add_argument(
    '--hazard_terminal_hold_s',
    type=float,
    default=0.35,
    help='Record this many seconds of the terminal HAZARD/SAFE label before closing the episode.',
)
parser.add_argument('--camera_qa', action='store_true', help='자동 전진으로 wrist/front camera 방향을 캡처하고 종료')
parser.add_argument('--camera_qa_steps', type=int, default=900, help='camera QA 실행 스텝 수')
parser.add_argument('--camera_qa_forward_vel', type=float, default=0.35, help='camera QA 중 Go2 전진 속도 명령')
parser.add_argument('--camera_qa_out_dir', default='/tmp/go2_soarm_camera_qa', help='camera QA 이미지/JSON 저장 디렉토리')
parser.add_argument('--camera_qa_min_travel_m', type=float, default=1.0, help='이 정도 전진하기 전에는 캡처하지 않음 (초기 URDF 자세 방지)')
parser.add_argument('--demo_pan', action='store_true', help='확인용 데모: 팔 정면 자세 + shoulder_pan 좌우 스윕 + Go2 전진')
parser.add_argument('--demo_pan_deg', type=float, default=35.0, help='데모 shoulder_pan 좌우 스윕 진폭(deg)')
parser.add_argument('--demo_pan_period_s', type=float, default=4.0, help='데모 shoulder_pan 한 사이클 주기(초)')
parser.add_argument(
    '--demo_fold_walk',
    action='store_true',
    help='한 마리 GUI QA: 전진 보행 중 팔을 반복해서 펴고 접는다.',
)
parser.add_argument('--demo_fold_period_s', type=float, default=16.0, help='홈-펴기-접기-펴기-홈 한 사이클 주기(초).')
parser.add_argument('--demo_forward_vel', type=float, default=0.15, help='데모 중 Go2 전진 속도 명령')
parser.add_argument('--camera_aim_qa', action='store_true', help='정지 상태에서 카메라 광축/방향 검증 (wrist rot 오버라이드 가능)')
parser.add_argument('--max_steps', type=int, default=0, help='Debug/smoke-test stop after N simulation steps. 0 disables.')
parser.add_argument('--scripted_velocity', nargs=3, type=float, metavar=('VX','VY','WZ'), help='QA-only fixed velocity command instead of keyboard input.')
parser.add_argument('--gait_probe', action='store_true', help='Bounded GUI open-floor command-response diagnostic; no planner.')
parser.add_argument('--scripted_stand_steps', type=int, default=0, help='QA: stand with zero command for N steps before applying --scripted_velocity. Tests idle→walk transition.')
parser.add_argument(
    '--scripted_velocity_duration_steps',
    type=int,
    default=0,
    help='QA: apply --scripted_velocity for this many steps, then release to zero. 0 keeps it active.',
)
parser.add_argument(
    '--scripted_route',
    nargs='+',
    type=float,
    metavar='XY',
    help='Deterministic world-frame XY waypoint route: X1 Y1 X2 Y2 ...',
)
parser.add_argument('--scripted_route_stand_steps', type=int, default=300, help='Stand before following --scripted_route.')
parser.add_argument('--scripted_route_speed', type=float, default=0.30, help='Maximum forward speed for --scripted_route (m/s).')
parser.add_argument('--scripted_route_tolerance', type=float, default=0.35, help='Waypoint reach radius for --scripted_route (m).')
parser.add_argument('--scripted_route_yaw_gain', type=float, default=1.0, help='Proportional heading gain for physical scripted-route tracking.')
parser.add_argument('--scripted_route_max_yaw_rate', type=float, default=0.50, help='Maximum physical scripted-route yaw-rate command (rad/s).')
parser.add_argument('--scripted_route_forward_alignment_rad', type=float, default=0.50, help='Stop forward motion when the heading error exceeds this value.')
parser.add_argument('--scripted_route_linear_slew', type=float, default=0.02, help='Maximum forward-command change per 50 Hz policy step.')
parser.add_argument('--scripted_route_yaw_slew', type=float, default=0.02, help='Maximum yaw-command change per 50 Hz policy step.')
parser.add_argument(
    '--show_scripted_route',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Draw the waypoint polyline, direction arrows, and waypoint dots in the GUI viewport.',
)
parser.add_argument(
    '--abort_on_locomotion_failure',
    action='store_true',
    help='Exit non-zero after a sustained fall or abnormal crossed-leg geometry is detected.',
)
parser.add_argument(
    '--locomotion_failure_grace_steps',
    type=int,
    default=300,
    help='Physics steps ignored by locomotion failure detection after startup.',
)
parser.add_argument(
    '--locomotion_failure_sustain_steps',
    type=int,
    default=75,
    help='Consecutive abnormal physics steps required before locomotion failure abort.',
)
parser.add_argument(
    '--kinematic_scripted_route',
    action='store_true',
    help='Move the robot root continuously along --scripted_route, decoupling the sensor trajectory from locomotion falls.',
)
parser.add_argument(
    '--render_interval',
    type=int,
    default=1,
    help='Render once every N physics steps. Values above 1 speed up GUI simulation while preserving the 200 Hz physics step.',
)
parser.add_argument(
    '--exit_on_route_complete',
    action='store_true',
    help='Exit cleanly as soon as the final scripted-route waypoint is reached.',
)
parser.add_argument(
    '--idle_stance_fallback',
    action=argparse.BooleanOptionalAction,
    default=False,
    help='Optional startup-only nominal stance hold. Disabled by default so WASD commands reach the verified policy directly.',
)
parser.add_argument('--state_debug_every', type=int, default=0, help='Print root/tilt/action runtime telemetry every N simulation steps. 0 disables.')
parser.add_argument('--wrist_rot_override', nargs=4, type=float, metavar=('QW','QX','QY','QZ'), help='wrist 카메라 rot 쿼터니언(opengl/USD convention) 임시 오버라이드')
parser.add_argument('--wrist_pos_override', nargs=3, type=float, metavar=('X','Y','Z'), help='wrist 카메라 로컬 위치 임시 오버라이드')
parser.add_argument('--static_spawn', action='store_true', help='움직임/보행정책 없이 스폰 자세만 유지한 채 시뮬레이션 렌더 (카메라 위치 점검용)')
parser.add_argument(
    '--robot_spawn_xy',
    nargs=2,
    type=float,
    metavar=('X', 'Y'),
    help='로봇 스폰 world XY. 미로 시작 패드 등 환경 좌표에 맞출 때 사용. 기본 (0, 0).',
)
parser.add_argument(
    '--robot_spawn_yaw_deg',
    type=float,
    default=0.0,
    help='로봇 스폰 yaw(도, +Z 기준 반시계). 0도는 +X 전방.',
)
parser.add_argument(
    '--viewport_topdown',
    action='store_true',
    help='뷰포트를 환경 USD의 TopDownCamera(정사영, 미로 위 내려다보기)로 전환.',
)
parser.add_argument(
    '--viewport_camera',
    help='뷰포트 활성 카메라 프림 경로 직접 지정. --viewport_topdown보다 우선.',
)
parser.add_argument(
    '--viewport_follow_robot',
    action='store_true',
    help='GUI perspective camera follows Go2 from behind and above.',
)
parser.add_argument(
    '--viewport_focus_robot_once',
    action='store_true',
    help='Aim the GUI perspective camera at Go2 once, then leave it user-controlled.',
)
parser.add_argument('--camera_pose_debug_every', type=int, default=0, help='정확한 카메라 optical center/world ray를 N sim step마다 출력. 0 disables.')
parser.add_argument('--show_camera_pose_markers', action='store_true', help='Stage의 USD 카메라 아이콘 대신 정확한 optical center 위치에 작은 시각 마커 표시.')
parser.add_argument('--disable_fabric', action='store_true', help='Stage에서 카메라 마커 Transform을 직접 편집할 수 있도록 Fabric 동기화를 비활성화')
parser.add_argument('--wrist_target_world_pos', nargs=3, type=float, metavar=('X','Y','Z'), help='목표 world 위치를 wrist 카메라 local offset으로 환산해서 출력.')
parser.add_argument(
    '--environment_usd',
    type=Path,
    help='Optional local USD environment. When set, the desk/drawer training scene is disabled.',
)
parser.add_argument(
    '--hide_hospital_small_rooms',
    action='store_true',
    help='Hide only NVIDIA Hospital Geo_M1 patient-room structure; preserve all props and equipment.',
)
parser.add_argument(
    '--slam_rgbd',
    action='store_true',
    help='Enable front and wrist RGB-D rendering for SLAM simulation.',
)
parser.add_argument(
    '--slam_ros2',
    action='store_true',
    help='Publish front RGB-D, camera info, clock, odometry, and TF through Isaac ROS 2.',
)
parser.add_argument(
    '--lidar_slam',
    action='store_true',
    help='Go2 L1-style 360-degree PhysX LiDAR on /utlidar/scan for laser SLAM.',
)
parser.add_argument(
    '--lidar_physx',
    action='store_true',
    help='Raycast the live PhysX collision scene instead of maze Cube AABBs.',
)
parser.add_argument(
    '--lidar_rtx',
    action='store_true',
    help='Use a rendered 32-channel RTX LiDAR so visual-only USD geometry also produces 3D returns.',
)
parser.add_argument(
    '--active_gap_arm',
    action='store_true',
    help='Enable the sole authorized Isaac arm action server for SIM_FRONTIER_V0 observation.',
)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
if args_cli.maze_observation_navigation and not args_cli.maze_nav2_inspection:
    parser.error('--maze_observation_navigation requires --maze_nav2_inspection')
if args_cli.gait_probe:
    from soarm_nbv.gait_probe import DURATION as gait_probe_duration, command_at as gait_probe_command
    if (args_cli.environment_usd or args_cli.maze_inspection_layout or args_cli.scripted_route
            or args_cli.scripted_velocity or args_cli.active_gap_arm or args_cli.headless):
        parser.error('--gait_probe requires GUI open floor without another motion controller')
    args_cli.max_steps = int(gait_probe_duration * 200)
    args_cli.abort_on_locomotion_failure = True
if args_cli.maze_nav2_inspection and not (args_cli.maze_nav2 and args_cli.maze_inspection_layout
        and args_cli.enable_cameras and not args_cli.disable_wrist_camera and not args_cli.no_arm):
    parser.error('--maze_nav2_inspection requires Nav2, maze layout and both enabled robot cameras')
if args_cli.scripted_route is not None:
    if len(args_cli.scripted_route) < 4 or len(args_cli.scripted_route) % 2 != 0:
        parser.error('--scripted_route requires at least two XY pairs')
    if args_cli.scripted_velocity is not None:
        parser.error('--scripted_route and --scripted_velocity are mutually exclusive')
    if args_cli.scripted_route_speed <= 0.0:
        parser.error('--scripted_route_speed must be positive')
    if args_cli.scripted_route_tolerance <= 0.0:
        parser.error('--scripted_route_tolerance must be positive')
    if args_cli.scripted_route_yaw_gain <= 0.0:
        parser.error('--scripted_route_yaw_gain must be positive')
    if args_cli.scripted_route_max_yaw_rate <= 0.0:
        parser.error('--scripted_route_max_yaw_rate must be positive')
    if args_cli.scripted_route_forward_alignment_rad <= 0.0:
        parser.error('--scripted_route_forward_alignment_rad must be positive')
    if args_cli.scripted_route_linear_slew <= 0.0 or args_cli.scripted_route_yaw_slew <= 0.0:
        parser.error('--scripted_route_linear_slew and --scripted_route_yaw_slew must be positive')
if args_cli.render_interval < 1:
    parser.error('--render_interval must be at least 1')
if args_cli.lidar_physx and not args_cli.lidar_slam:
    parser.error('--lidar_physx requires --lidar_slam')
if args_cli.lidar_rtx and not args_cli.lidar_slam:
    parser.error('--lidar_rtx requires --lidar_slam')
if args_cli.lidar_physx and args_cli.lidar_rtx:
    parser.error('--lidar_physx and --lidar_rtx are mutually exclusive')
if args_cli.slam_ros2 and not args_cli.environment_usd:
    parser.error("--slam_ros2 requires --environment_usd")
if args_cli.slam_ros2:
    args_cli.slam_rgbd = True
if args_cli.active_gap_arm:
    if not args_cli.slam_ros2:
        parser.error("--active_gap_arm requires --slam_ros2")
    if args_cli.robot_model != "so101_7motor_reversed":
        parser.error("--active_gap_arm requires --robot_model so101_7motor_reversed")
    for required_environment in (
        "ACTIVE_SLAM_TRANSACTION_CONTRACT",
        "ACTIVE_SLAM_HMAC_SECRET",
    ):
        if not os.environ.get(required_environment):
            parser.error(f"--active_gap_arm requires {required_environment}")
if args_cli.leader_apply_only_when_base_paused and not (
    args_cli.active_gap_arm and args_cli.leader_auto
):
    parser.error(
        "--leader_apply_only_when_base_paused requires --active_gap_arm and --leader_auto"
    )
if args_cli.no_arm:
    incompatible_no_arm_modes = {
        "--active_gap_arm": args_cli.active_gap_arm,
        "--collect": args_cli.collect,
        "--demo_pan": args_cli.demo_pan,
        "--demo_fold_walk": args_cli.demo_fold_walk,
        "--enable_gr00t": args_cli.enable_gr00t,
        "--gr00t_apply_actions": args_cli.gr00t_apply_actions,
        "--leader_auto": args_cli.leader_auto,
        "--enable_smolvla_policy": args_cli.enable_smolvla_policy,
        "--enable_hazard_smolvla_policy": args_cli.enable_hazard_smolvla_policy,
        "--camera_qa": args_cli.camera_qa,
        "--camera_aim_qa": args_cli.camera_aim_qa,
        "--show_camera_viewport": args_cli.show_camera_viewport,
    }
    enabled_no_arm_modes = [
        name for name, enabled in incompatible_no_arm_modes.items() if enabled
    ]
    if enabled_no_arm_modes:
        parser.error(
            "--no_arm forbids arm/wrist modes: "
            f"{', '.join(enabled_no_arm_modes)}"
        )
if args_cli.environment_usd:
    args_cli.environment_usd = args_cli.environment_usd.expanduser().resolve()
    if not args_cli.environment_usd.is_file():
        parser.error(f"--environment_usd does not exist: {args_cli.environment_usd}")
    supervised_manual_arm_teleop = bool(
        args_cli.active_gap_arm
        and args_cli.leader_auto
        and args_cli.leader_apply_only_when_base_paused
    )
    forbidden_warehouse_modes = {
        "--collect": args_cli.collect,
        "--demo_pan": args_cli.demo_pan,
        "--enable_gr00t": args_cli.enable_gr00t,
        "--gr00t_apply_actions": args_cli.gr00t_apply_actions,
        "--leader_auto": args_cli.leader_auto and not supervised_manual_arm_teleop,
        "--enable_smolvla_policy": args_cli.enable_smolvla_policy,
    }
    enabled_warehouse_modes = [
        name for name, enabled in forbidden_warehouse_modes.items() if enabled
    ]
    if enabled_warehouse_modes:
        parser.error(
            "--environment_usd forbids "
            f"legacy policy/collection modes: {', '.join(enabled_warehouse_modes)}"
        )
elif args_cli.hide_hospital_small_rooms:
    parser.error("--hide_hospital_small_rooms requires --environment_usd")
if args_cli.enable_smolvla_policy and args_cli.collect_manual_right_arrow:
    parser.error("--enable_smolvla_policy uses RIGHT to toggle policy control and cannot be combined with --collect_manual_right_arrow.")
if args_cli.enable_hazard_smolvla_policy:
    if not args_cli.active_gap_arm or not args_cli.environment_usd:
        parser.error(
            "--enable_hazard_smolvla_policy requires --active_gap_arm and --environment_usd"
        )
    if args_cli.enable_smolvla_policy:
        parser.error(
            "hazard and legacy drawer SmolVLA runtimes are mutually exclusive"
        )
    if args_cli.leader_auto or args_cli.collect_hazard_vla or args_cli.collect:
        parser.error(
            "hazard SmolVLA runtime cannot share arm authority with leader/data collection modes"
        )
    if args_cli.hazard_smolvla_fps <= 0.0:
        parser.error("--hazard_smolvla_fps must be positive")
    if not args_cli.hazard_smolvla_policy_path:
        parser.error("--hazard_smolvla_policy_path is required")
    if not Path(args_cli.hazard_smolvla_policy_path).expanduser().is_dir():
        parser.error(
            "hazard SmolVLA policy directory does not exist: "
            f"{args_cli.hazard_smolvla_policy_path}"
        )
    args_cli.enable_cameras = True
if args_cli.collect_hazard_vla and args_cli.collect_hazard_nbv_teacher:
    parser.error("human and scripted NBV teacher collection modes are mutually exclusive")
if args_cli.collect_hazard_vla:
    if not (
        args_cli.active_gap_arm
        and args_cli.leader_auto
        and args_cli.leader_apply_only_when_base_paused
    ):
        parser.error(
            "--collect_hazard_vla requires --active_gap_arm --leader_auto "
            "--leader_apply_only_when_base_paused"
        )
    if args_cli.headless:
        parser.error("--collect_hazard_vla is GUI-only by workspace safety rule")
    if args_cli.collect or args_cli.enable_smolvla_policy:
        parser.error(
            "--collect_hazard_vla cannot be combined with legacy --collect or "
            "--enable_smolvla_policy"
        )
    if args_cli.hazard_collect_fps <= 0.0:
        parser.error("--hazard_collect_fps must be positive")
    if not 0.10 <= args_cli.hazard_terminal_hold_s <= 2.0:
        parser.error("--hazard_terminal_hold_s must be within 0.10..2.0")
    args_cli.enable_cameras = True
if args_cli.collect_hazard_nbv_teacher:
    if not (args_cli.active_gap_arm and args_cli.environment_usd):
        parser.error(
            "--collect_hazard_nbv_teacher requires --active_gap_arm and --environment_usd"
        )
    if args_cli.headless:
        parser.error(
            "--collect_hazard_nbv_teacher is GUI-only by workspace safety rule"
        )
    if args_cli.leader_auto or args_cli.collect or args_cli.enable_smolvla_policy:
        parser.error(
            "scripted NBV collection cannot share arm authority with leader or legacy modes"
        )
    if args_cli.enable_hazard_smolvla_policy:
        parser.error("scripted NBV collection and trained VLA inference are mutually exclusive")
    if args_cli.hazard_collect_fps <= 0.0:
        parser.error("--hazard_collect_fps must be positive")
    if args_cli.binary_tree_repeat_reset_status_file is not None:
        args_cli.binary_tree_repeat_reset_status_file = (
            args_cli.binary_tree_repeat_reset_status_file.expanduser().resolve()
        )
elif args_cli.binary_tree_repeat_reset_status_file is not None:
    parser.error(
        "--binary_tree_repeat_reset_status_file requires --collect_hazard_nbv_teacher"
    )
    args_cli.enable_cameras = True
if args_cli.camera_qa:
    args_cli.enable_cameras = True
if args_cli.demo_pan:
    args_cli.enable_cameras = True
if args_cli.demo_pan and args_cli.demo_fold_walk:
    parser.error("--demo_pan and --demo_fold_walk are mutually exclusive")
if args_cli.demo_fold_period_s <= 0.0:
    parser.error("--demo_fold_period_s must be positive")

# 환경 USD(예: GJC 미로)의 시작 패드에 로봇을 맞춰 스폰할 때 사용.
ROBOT_SPAWN_XY = tuple(args_cli.robot_spawn_xy) if args_cli.robot_spawn_xy else (0.0, 0.0)
_SPAWN_YAW_HALF_RAD = math.radians(args_cli.robot_spawn_yaw_deg) / 2.0
ROBOT_SPAWN_ROT = (
    math.cos(_SPAWN_YAW_HALF_RAD),
    0.0,
    0.0,
    math.sin(_SPAWN_YAW_HALF_RAD),
)
if args_cli.camera_aim_qa:
    args_cli.enable_cameras = True
if args_cli.enable_smolvla_policy:
    args_cli.enable_cameras = True
if args_cli.enable_hazard_smolvla_policy:
    args_cli.enable_cameras = True
if args_cli.slam_rgbd:
    args_cli.enable_cameras = True
if args_cli.leader_auto:
    args_cli.gr00t_apply_actions = True
ROBOT_MODEL_PROFILE = ROBOT_MODEL_PROFILES[args_cli.robot_model]
try:
    ELBOW_ROTATE_HOLD_DEG = validate_elbow_rotate_deg(ROBOT_MODEL_PROFILE, args_cli.elbow_rotate_deg)
except ValueError as error:
    parser.error(str(error))
ELBOW_ROTATE_HOLD_RAD = math.radians(ELBOW_ROTATE_HOLD_DEG)
ROBOT_INITIAL_JOINT_POS = dict(ROBOT_MODEL_PROFILE.initial_arm_joint_pos_rad)
if "elbow_rotate" in ROBOT_INITIAL_JOINT_POS:
    ROBOT_INITIAL_JOINT_POS["elbow_rotate"] = ELBOW_ROTATE_HOLD_RAD
if args_cli.no_arm:
    ROBOT_INITIAL_JOINT_POS = {}
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import carb
import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.devices import Se2Keyboard, Se2KeyboardCfg
from isaaclab.utils import configclass
from isaaclab.actuators import DCMotorCfg, ImplicitActuatorCfg
from isaaclab.sensors.camera import CameraCfg
from isaaclab.sensors import ContactSensorCfg, RayCasterCfg, patterns

from isaaclab.utils import math as math_utils

from soarm_nbv.safety import decode_joint_vector
from soarm_nbv.zmq_bridge import ActionSubscriber, ZmqEndpointConfig


ROUGH_POLICY_JOINT_ORDER = [
    "FL_hip_joint", "FL_thigh_joint", "FL_calf_joint",
    "FR_hip_joint", "FR_thigh_joint", "FR_calf_joint",
    "RL_hip_joint", "RL_thigh_joint", "RL_calf_joint",
    "RR_hip_joint", "RR_thigh_joint", "RR_calf_joint",
    "gripper", "wrist_roll", "wrist_flex", "elbow_flex", "shoulder_lift", "shoulder_pan",
]
MIXED_POLICY_ACTION_JOINT_ORDER = [
    "FL_hip_joint", "FR_hip_joint", "RL_hip_joint", "RR_hip_joint",
    "FL_thigh_joint", "FR_thigh_joint", "RL_thigh_joint", "RR_thigh_joint",
    "FL_calf_joint", "FR_calf_joint", "RL_calf_joint", "RR_calf_joint",
]
GO2_ACTION_SCALE = 0.25
GO2_POLICY_ACTION_LIMIT = 4.0
GO2_POLICY_OBS_DIM = 247
GO2_POLICY_OBS_DIMS = {"rough": GO2_POLICY_OBS_DIM, "flat": 48}
STARTUP_SETTLE_STEPS = 0
LEG_POLICY_RAMP_STEPS = 20
GR00T_FULL_JOINT_ORDER = list(ROBOT_MODEL_PROFILE.external_joint_order)
# wrist 카메라 로컬 오프셋 (gripper_link 기준)
# 실제 카메라 prim은 gripper_link 밑에 두어 로봇팔의 위치와 회전을 그대로 따라간다.
# pos: gripper_link 기준 미터 단위 local offset / rot: gripper_link 기준 opengl/USD camera convention 쿼터니언
WRIST_CAMERA_LOCAL_POS = (0.06499, -0.03017, -0.00565)
WRIST_CAMERA_LOCAL_ROT = (0.0677732, 0.0677732, 0.7038514, 0.7038514)
FRONT_CAMERA_LOCAL_POS = (0.33357, -0.00215, 0.12349)
FRONT_CAMERA_LOCAL_ROT = (1.0, 0.0, 0.0, 0.0)
WRIST_CAMERA_HEIGHT = 240
WRIST_CAMERA_WIDTH = 320
FRONT_CAMERA_HEIGHT = 480
FRONT_CAMERA_WIDTH = 640
SLAM_RGBD_VALIDATION_INTERVAL_STEPS = 200
if getattr(args_cli, "wrist_pos_override", None) is not None:
    WRIST_CAMERA_LOCAL_POS = tuple(args_cli.wrist_pos_override)
if getattr(args_cli, "wrist_rot_override", None) is not None:
    WRIST_CAMERA_LOCAL_ROT = tuple(args_cli.wrist_rot_override)
print(f">>> wrist camera local offset: pos={WRIST_CAMERA_LOCAL_POS} rot={WRIST_CAMERA_LOCAL_ROT}", flush=True)
GR00T_ACTION_CLIP_DEG = {
    "shoulder_pan": (-110.0, 110.0),
    "shoulder_lift": (-100.0, 100.0),
    "elbow_flex": (-96.8, 96.8),
    "wrist_flex": (-95.0, 95.0),
    "wrist_roll": (-157.2, 162.8),
    "gripper": (-10.0, 100.0),
}
if ROBOT_MODEL_PROFILE.key == "so101_7motor":
    GR00T_ACTION_CLIP_DEG["elbow_flex"] = (-2.0, 96.8)
elif ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed":
    GR00T_ACTION_CLIP_DEG["elbow_flex"] = (-90.0, 90.0)
SO_ARM_LINKS = ROBOT_MODEL_PROFILE.collision_link_names
SO_ARM_GRIPPER_LINKS = set(ROBOT_MODEL_PROFILE.gripper_link_names)
GO2_BASE_URDF = ROBOT_MODELS_ROOT / "assets/urdf/go2.urdf"


def export_stage(path: str) -> None:
    if not path:
        return

    import omni.usd

    os.makedirs(os.path.dirname(path), exist_ok=True)
    stage = omni.usd.get_context().get_stage()
    stage.GetRootLayer().Export(path)
    print(f">>> Stage exported: {path}")


def hide_hospital_small_room_structure() -> None:
    """Hide only NVIDIA Hospital's Geo_M1 patient-room shell modules.

    Furniture and equipment are separate top-level prims. Limiting the filter
    to the structural Geo_M1 namespace removes unused small room shells without
    deleting beds, carts, desks, machines, or other SLAM landmarks. Overrides
    live in the session layer, so the referenced NVIDIA asset stays untouched.
    """
    if not args_cli.hide_hospital_small_rooms:
        return

    import omni.usd

    stage = omni.usd.get_context().get_stage()
    environment_prefix = "/World/environment/"
    paths_to_hide = []
    for prim in stage.Traverse():
        path = str(prim.GetPath())
        if not path.startswith(environment_prefix):
            continue
        # The NVIDIA asset is flat at its default prim. Filtering only direct
        # children avoids collecting descendants that become invalid after a
        # parent structural module is deactivated.
        if str(prim.GetParent().GetPath()) != environment_prefix.rstrip("/"):
            continue
        name = prim.GetName()
        is_small_room_structure = name.startswith("Geo_M1_")
        is_small_room_threshold = False
        if name.startswith("Geo_M_DoorFloor"):
            suffix = name.removeprefix("Geo_M_DoorFloor").split("_", 1)[0]
            is_small_room_threshold = suffix.isdigit() and 11 <= int(suffix) <= 23
        if is_small_room_structure or is_small_room_threshold:
            paths_to_hide.append(prim.GetPath())

    previous_edit_target = stage.GetEditTarget()
    try:
        stage.SetEditTarget(stage.GetSessionLayer())
        for path in paths_to_hide:
            prim = stage.GetPrimAtPath(path)
            if prim.IsValid():
                prim.SetActive(False)
    finally:
        stage.SetEditTarget(previous_edit_target)

    print(
        f">>> Hospital small rooms hidden: {len(paths_to_hide)} structural prims; "
        "furniture/equipment preserved",
        flush=True,
    )


class WasdKeyboard(Se2Keyboard):
    def _create_key_bindings(self):
        self._INPUT_KEY_MAPPING = {
            "W": np.asarray([1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "S": np.asarray([-1.0, 0.0, 0.0]) * self.v_x_sensitivity,
            "A": np.asarray([0.0, 1.0, 0.0]) * self.v_y_sensitivity,
            "D": np.asarray([0.0, -1.0, 0.0]) * self.v_y_sensitivity,
            "Q": np.asarray([0.0, 0.0, 1.0]) * self.omega_z_sensitivity,
            "E": np.asarray([0.0, 0.0, -1.0]) * self.omega_z_sensitivity,
        }


def setup_camera_viewports(show_free_camera: bool = False, camera_sensors=()):
    import carb
    from omni.kit.viewport.utility import create_viewport_window

    settings = carb.settings.get_settings()
    settings.set("/app/viewport/show/camera", True)
    settings.set("/app/viewport/createCameraModelRep", True)

    camera_views = [
        ("Wrist Camera", "/World/envs/env_0/Robot/gripper_link/wrist_camera"),
        ("Front RGB-D Camera", "/World/envs/env_0/Robot/base/front_camera_sensor"),
    ]
    viewports = []
    vp_y = 740
    for idx, (title, camera_path) in enumerate(camera_views):
        shared_sensor=(camera_sensors[idx] if args_cli.maze_nav2 and camera_sensors
                       and os.environ.get('MAZE_SHARED_CAMERA_RENDER','1')=='1' else None)
        print('MAZE_CAMERA_WINDOW creating',title,flush=True)
        if shared_sensor is not None:
            if '/home/iy/Isaac/maze_visual_demo' not in sys.path:
                sys.path.insert(0,'/home/iy/Isaac/maze_visual_demo')
            from sensor_viewports import SensorImageWindow
            window=SensorImageWindow(title,shared_sensor)
        else:
            window = create_viewport_window(
                title, width=420, height=320, visible=True,
                position_x=10 + idx * 430, position_y=vp_y,
            )
            window.viewport_api.set_active_camera(camera_path)
        print('MAZE_CAMERA_WINDOW ready',title,flush=True)
        viewports.append(window)
    if args_cli.maze_inspection_layout or args_cli.gait_probe:
        import asyncio
        import omni.ui as ui
        import omni.kit.app
        if '/home/iy/Isaac/maze_visual_demo' not in sys.path:
            sys.path.insert(0, '/home/iy/Isaac/maze_visual_demo')
        from camera_layout import dock_sensor_views
        # Keep the one-shot task alive; no per-frame layout enforcement.
        viewports[0]._maze_dock_task = asyncio.ensure_future(
            dock_sensor_views(ui, omni.kit.app.get_app()))
    if show_free_camera:
        free_window = create_viewport_window(
            "Free Camera", width=420, height=320, visible=True,
            position_x=870, position_y=vp_y,
        )
        try:
            free_window.viewport_api.set_active_camera("/OmniverseKit_Persp")
            print(">>> Free Camera viewport: ENABLED (RMB+WASD fly, Alt+LMB orbit, MMB pan)")
        except Exception as _e:
            print(f">>> Free Camera viewport camera binding skipped: {_e}", flush=True)
        viewports.append(free_window)
    print(">>> Live Camera Viewports: ENABLED (Main + Wrist + Front RGB-D"
          + (" + Free Camera)" if show_free_camera else ")"))
    return viewports
def setup_perspective_camera():
    """관찰용 원근 카메라 prim 생성만 담당 (viewport 전환은 setup_camera_viewports에서 처리)."""
    try:
        import omni.usd
        from pxr import UsdGeom, Gf

        TRANSLATE = (1.0488, -1.33868, 3.05266)
        ROTATE = (28.76521, 0.0, -1.21729)

        stage = omni.usd.get_context().get_stage()
        cam_path = "/World/perspective_camera"
        if stage.GetPrimAtPath(cam_path).IsValid():
            stage.RemovePrim(cam_path)

        UsdGeom.Camera.Define(stage, cam_path)
        cam = UsdGeom.Xformable(stage.GetPrimAtPath(cam_path))
        cam.GetXformOpOrderAttr().Clear()
        cam.AddTranslateOp().Set(Gf.Vec3d(*TRANSLATE))
        cam.AddRotateXYZOp().Set(Gf.Vec3f(*ROTATE))
        print(f">>> Perspective camera prim created: translate={TRANSLATE} rotate={ROTATE}")
    except Exception as _e:
        print(f">>> Perspective camera prim creation skipped: {_e}", flush=True)
def _fmt_xyz(values, digits: int = 5) -> str:
    if hasattr(values, "detach"):
        values = values.detach().cpu().tolist()
    return "(" + ", ".join(f"{float(v):.{digits}f}" for v in values) + ")"


def create_camera_pose_markers() -> None:
    """Stage 카메라 아이콘 대신 정확한 optical center 위치를 표시하는 visual-only 큐브 마커."""
    import omni.usd
    from pxr import Gf, UsdGeom

    stage = omni.usd.get_context().get_stage()
    marker_specs = [
        (
            "/World/envs/env_0/Robot/gripper_link/wrist_camera_exact_optical_center",
            WRIST_CAMERA_LOCAL_POS,
            0.012,
            (1.0, 0.0, 1.0),
        ),
        (
            "/World/envs/env_0/Robot/base/front_camera_exact_optical_center",
            FRONT_CAMERA_LOCAL_POS,
            0.020,
            (0.0, 1.0, 1.0),
        ),
    ]
    for marker_path, local_pos, size, color in marker_specs:
        cube = UsdGeom.Cube.Define(stage, marker_path)
        cube.CreateSizeAttr(size)
        cube.CreateDisplayColorAttr([Gf.Vec3f(*color)])
        xform = UsdGeom.Xformable(cube.GetPrim())
        xform.ClearXformOpOrder()
        xform.AddTranslateOp().Set(Gf.Vec3d(*local_pos))
    print(
        ">>> Camera exact optical-center cube markers: ENABLED "
        "(magenta cube=wrist under gripper_link, cyan cube=front under base; use Transform Translate as the calibration value)",
        flush=True,
    )


def print_camera_pose_debug(
    policy_step: int,
    robot,
    gripper_link_idx: int,
    base_link_idx: int,
    wrist_camera_local_pos: torch.Tensor,
    wrist_camera_local_rot: torch.Tensor,
    front_camera_local_pos: torch.Tensor,
    front_camera_local_rot: torch.Tensor,
) -> None:
    """정확한 카메라 optical center와 광축을 parent link 기준 수치로 출력."""
    opengl_forward = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=wrist_camera_local_pos.device)
    opengl_up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=wrist_camera_local_pos.device)

    def _describe(
        label: str,
        parent_name: str,
        parent_idx: int,
        local_pos: torch.Tensor,
        local_rot: torch.Tensor,
        convention: str,
    ):
        parent_pos = robot.data.body_state_w[:, parent_idx, :3]
        parent_quat = robot.data.body_state_w[:, parent_idx, 3:7]
        cam_pos, cam_quat = math_utils.combine_frame_transforms(parent_pos, parent_quat, local_pos, local_rot)
        cam_quat_opengl = (
            cam_quat
            if convention == "opengl"
            else math_utils.convert_camera_frame_orientation_convention(
                cam_quat,
                origin=convention,
                target="opengl",
            )
        )
        ray_dir = torch.nn.functional.normalize(math_utils.quat_apply(cam_quat_opengl, opengl_forward), dim=-1)
        ray_up = torch.nn.functional.normalize(math_utils.quat_apply(cam_quat_opengl, opengl_up), dim=-1)
        ray_10cm = cam_pos + ray_dir * 0.10
        print(f">>> [camera_pose] step={policy_step} {label}", flush=True)
        print(f"    parent={parent_name} parent_pos_w={_fmt_xyz(parent_pos[0])}", flush=True)
        print(f"    local_offset_for_code={_fmt_xyz(local_pos[0])}", flush=True)
        print(f"    optical_center_pos_w={_fmt_xyz(cam_pos[0])}", flush=True)
        print(f"    camera_rot_{convention}_wxyz={_fmt_xyz(cam_quat[0])}", flush=True)
        print(f"    optical_forward_dir_w={_fmt_xyz(ray_dir[0])}  +10cm={_fmt_xyz(ray_10cm[0])}", flush=True)
        print(f"    optical_up_dir_w={_fmt_xyz(ray_up[0])}", flush=True)
        return parent_pos, parent_quat

    wrist_parent_pos, wrist_parent_quat = _describe(
        "wrist",
        "gripper_link",
        gripper_link_idx,
        wrist_camera_local_pos,
        wrist_camera_local_rot,
        "opengl",
    )
    _describe("front", "base", base_link_idx, front_camera_local_pos, front_camera_local_rot, "world")

    if args_cli.wrist_target_world_pos is not None:
        target_w = torch.tensor([args_cli.wrist_target_world_pos], dtype=torch.float32, device=wrist_camera_local_pos.device)
        required_local = math_utils.quat_apply_inverse(wrist_parent_quat, target_w - wrist_parent_pos)
        print(
            ">>> [camera_pose] target world -> code value: "
            f"WRIST_CAMERA_LOCAL_POS = {_fmt_xyz(required_local[0])}",
            flush=True,
        )



class Gr00tZmqBridge:
    def __init__(self, obs_port: int, action_port: int):
        self.context = zmq.Context()
        self.obs_pub = self.context.socket(zmq.PUB)
        self.obs_pub.bind(f"tcp://*:{obs_port}")
        self.action_sub = self.context.socket(zmq.SUB)
        self.action_sub.connect(f"tcp://localhost:{action_port}")
        self.action_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.action_sub.setsockopt(zmq.RCVTIMEO, 0)
        self.obs_port = obs_port
        self.action_port = action_port
        print(f">>> GR00T ZMQ Bridge: OBS PUB tcp://*:{obs_port}, ACTION SUB tcp://localhost:{action_port}")

    def close(self):
        self.obs_pub.close(0)
        self.action_sub.close(0)
        self.context.term()

    def publish_observation(self, wrist_rgb: np.ndarray, joint_pos: np.ndarray):
        # GR00T SO_ARM Starter expects room+wrist streams. In minimal mode nbv_v4 has
        # only one real camera, so duplicate wrist into the room stream contract.
        room_rgb = wrist_rgb
        self.obs_pub.send_multipart(
            [
                np.array(room_rgb.shape, dtype=np.int32).tobytes(),
                room_rgb.tobytes(),
                np.array(wrist_rgb.shape, dtype=np.int32).tobytes(),
                wrist_rgb.tobytes(),
                joint_pos.astype(np.float32).tobytes(),
            ]
        )

    def receive_action(self) -> np.ndarray | None:
        try:
            action_raw = self.action_sub.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        return decode_joint_vector(action_raw, "GR00T action")


class SmolVLAZmqBridge:
    """Low-latency ZMQ bridge for the external LeRobot SmolVLA runner."""

    def __init__(self, obs_port: int, action_port: int):
        self.context = zmq.Context()
        self.obs_pub = self.context.socket(zmq.PUB)
        self.obs_pub.setsockopt(zmq.SNDHWM, 1)
        self.obs_pub.bind(f"tcp://*:{obs_port}")
        self.action_sub = self.context.socket(zmq.SUB)
        self.action_sub.setsockopt(zmq.RCVHWM, 1)
        self.action_sub.setsockopt(zmq.CONFLATE, 1)
        self.action_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        self.action_sub.connect(f"tcp://localhost:{action_port}")
        self.obs_port = obs_port
        self.action_port = action_port
        print(f">>> SmolVLA ZMQ Bridge: OBS PUB tcp://*:{obs_port}, ACTION SUB tcp://localhost:{action_port}")

    def close(self):
        self.obs_pub.close(0)
        self.action_sub.close(0)
        self.context.term()

    def send_reset(self) -> None:
        try:
            self.obs_pub.send_multipart([b"RESET"], flags=zmq.NOBLOCK)
        except zmq.Again:
            pass

    def publish_observation(self, room_rgb: np.ndarray, wrist_rgb: np.ndarray, joint_pos: np.ndarray) -> bool:
        room_rgb = np.ascontiguousarray(room_rgb[:, :, :3], dtype=np.uint8)
        wrist_rgb = np.ascontiguousarray(wrist_rgb[:, :, :3], dtype=np.uint8)
        try:
            self.obs_pub.send_multipart(
                [
                    b"OBS",
                    np.array(room_rgb.shape, dtype=np.int32).tobytes(),
                    room_rgb.tobytes(),
                    np.array(wrist_rgb.shape, dtype=np.int32).tobytes(),
                    wrist_rgb.tobytes(),
                    joint_pos.astype(np.float32).tobytes(),
                ],
                flags=zmq.NOBLOCK,
            )
            return True
        except zmq.Again:
            return False

    def receive_action(self) -> np.ndarray | None:
        latest = None
        while True:
            try:
                action_raw = self.action_sub.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            latest = decode_joint_vector(action_raw, "SmolVLA action")
        return latest


def clip_gr00t_action(action: np.ndarray) -> np.ndarray:
    clipped = action.astype(np.float32).copy()
    for i, joint_name in enumerate(GR00T_FULL_JOINT_ORDER):
        lo, hi = GR00T_ACTION_CLIP_DEG[joint_name]
        clipped[i] = np.clip(clipped[i], lo, hi)
    return clipped




def validate_slam_rgbd_frame(
    camera_name: str,
    output: dict[str, torch.Tensor],
    expected_height: int,
    expected_width: int,
) -> tuple[float, float, int]:
    required = {"rgb", "distance_to_image_plane"}
    missing = required.difference(output)
    if missing:
        raise RuntimeError(f"{camera_name} RGB-D output missing channels: {sorted(missing)}")
    rgb = output["rgb"]
    depth = output["distance_to_image_plane"]
    if tuple(rgb.shape[1:3]) != (expected_height, expected_width):
        raise RuntimeError(f"{camera_name} RGB shape mismatch: {tuple(rgb.shape)}")
    if tuple(depth.shape[1:3]) != (expected_height, expected_width):
        raise RuntimeError(f"{camera_name} depth shape mismatch: {tuple(depth.shape)}")
    finite_positive = torch.isfinite(depth) & (depth > 0.0)
    valid_count = int(finite_positive.sum().item())
    if valid_count == 0:
        raise RuntimeError(f"{camera_name} depth has no finite positive samples")
    valid_depth = depth[finite_positive]
    return float(valid_depth.min().item()), float(valid_depth.max().item()), valid_count

def first_nonfinite_state(
    robot,
    sim_arm_joint_ids: torch.Tensor | None,
    gripper_link_idx: int | None,
) -> str | None:
    state_checks = [
        ("robot_joint_pos", robot.data.joint_pos),
        ("robot_joint_vel", robot.data.joint_vel),
        ("robot_root_state_w", robot.data.root_state_w),
    ]
    if sim_arm_joint_ids is not None and sim_arm_joint_ids.numel() > 0:
        state_checks.extend((
            ("soarm_joint_pos", robot.data.joint_pos[:, sim_arm_joint_ids]),
            ("soarm_joint_vel", robot.data.joint_vel[:, sim_arm_joint_ids]),
        ))
    if gripper_link_idx is not None:
        state_checks.extend((
            ("gripper_link_body_state_w", robot.data.body_state_w[:, gripper_link_idx]),
            ("gripper_link_body_pos_w", robot.data.body_pos_w[:, gripper_link_idx]),
        ))
    for name, tensor in state_checks:
        if not torch.isfinite(tensor).all():
            return name
    return None




def flush_initial_robot_pose(sim, scene, robot, joint_targets: torch.Tensor) -> None:
    root_state = robot.data.default_root_state.clone()
    try:
        root_state[:, :3] += scene.env_origins
    except Exception:
        pass
    robot.write_root_pose_to_sim(root_state[:, :7])
    robot.write_root_velocity_to_sim(torch.zeros_like(root_state[:, 7:]))
    robot.write_joint_state_to_sim(joint_targets, torch.zeros_like(joint_targets))
    robot.set_joint_position_target(joint_targets)
    scene.write_data_to_sim()
    sim.forward()
    scene.update(0.0)


def raise_robot_to_foot_clearance(
    sim,
    scene,
    robot,
    target_foot_z: float = 0.022,
    label: str = "spawn",
) -> tuple[list[str], list[int]]:
    """발 collision sphere가 지면에 박힌 경우에만 root를 올린다."""
    foot_names = ["FL_foot", "FR_foot", "RL_foot", "RR_foot"]
    foot_indices = [robot.find_bodies(name)[0][0] for name in foot_names]
    try:
        scene.update(0.0)
        foot_zs = [float(robot.data.body_pos_w[0, idx, 2].cpu()) for idx in foot_indices]
        raise_amount = target_foot_z - min(foot_zs)
        if raise_amount > 1e-3:
            root_pos = robot.data.root_pos_w[0].clone()
            root_quat = robot.data.root_quat_w[0].clone()
            root_pos[2] += raise_amount
            robot.write_root_pose_to_sim(torch.cat([root_pos, root_quat]).unsqueeze(0))
            robot.write_root_velocity_to_sim(torch.zeros((1, 6), dtype=root_pos.dtype, device=sim.device))
            sim.forward()
            scene.update(0.0)
            print(
                f">>> [{label}] feet_z={dict(zip(foot_names, [round(z, 3) for z in foot_zs]))} "
                f"-> root +{raise_amount:.3f}m (raise-only foot clearance)",
                flush=True,
            )
        else:
            print(
                f">>> [{label}] feet_z={dict(zip(foot_names, [round(z, 3) for z in foot_zs]))} "
                "-> no root lowering",
                flush=True,
            )
    except Exception as exc:
        print(f">>> [{label}] foot 접지 보정 스킵: {exc}", flush=True)
    return foot_names, foot_indices


def print_robot_pose_summary(robot, foot_names: list[str], foot_indices: list[int], label: str) -> None:
    try:
        root_pos = [round(float(v), 3) for v in robot.data.root_pos_w[0].detach().cpu().tolist()]
        root_quat = [round(float(v), 3) for v in robot.data.root_quat_w[0].detach().cpu().tolist()]
        foot_zs = [round(float(robot.data.body_pos_w[0, idx, 2].cpu()), 3) for idx in foot_indices]
        print(
            f">>> [{label}] root_pos={root_pos} root_quat={root_quat} "
            f"feet_z={dict(zip(foot_names, foot_zs))}",
            flush=True,
        )
    except Exception:
        pass


def settle_startup_pose(sim, scene, robot, joint_targets: torch.Tensor, dt: float) -> None:
    if STARTUP_SETTLE_STEPS <= 0:
        return
    for _ in range(STARTUP_SETTLE_STEPS):
        robot.set_joint_position_target(joint_targets)
        scene.write_data_to_sim()
        sim.step(render=False)
        scene.update(dt)
    sim.render()


def _load_utlidar_static_boxes(environment_usd: Path) -> np.ndarray:
    """Load the maze's authored Cube geometry as world-space AABBs."""
    from pxr import Usd, UsdGeom

    stage = Usd.Stage.Open(str(environment_usd))
    if stage is None:
        raise RuntimeError(f"failed to open LiDAR environment USD: {environment_usd}")
    bounds = UsdGeom.BBoxCache(
        Usd.TimeCode.Default(),
        [UsdGeom.Tokens.default_],
        useExtentsHint=True,
    )
    boxes: list[tuple[float, float, float, float, float, float]] = []
    for prim in stage.Traverse():
        if not prim.IsA(UsdGeom.Cube):
            continue
        aligned = bounds.ComputeWorldBound(prim).ComputeAlignedRange()
        minimum = aligned.GetMin()
        maximum = aligned.GetMax()
        boxes.append(
            (
                float(minimum[0]),
                float(maximum[0]),
                float(minimum[1]),
                float(maximum[1]),
                float(minimum[2]),
                float(maximum[2]),
            )
        )
    if not boxes:
        raise RuntimeError(f"LiDAR environment contains no Cube geometry: {environment_usd}")
    return np.asarray(boxes, dtype=np.float32)


def _raycast_utlidar_boxes(
    origin_xyz: np.ndarray,
    world_angles: np.ndarray,
    boxes_xyz: np.ndarray,
    max_distance_m: float,
) -> np.ndarray:
    """Intersect a horizontal ray ring with static world-space box geometry."""
    origin = np.asarray(origin_xyz, dtype=np.float32)
    boxes = np.asarray(boxes_xyz, dtype=np.float32)
    height_mask = (boxes[:, 4] <= origin[2]) & (origin[2] <= boxes[:, 5])
    boxes = boxes[height_mask]
    if boxes.size == 0:
        return np.full(world_angles.shape, np.inf, dtype=np.float32)

    dx = np.cos(world_angles, dtype=np.float32)[:, None]
    dy = np.sin(world_angles, dtype=np.float32)[:, None]
    epsilon = 1.0e-7
    with np.errstate(divide="ignore", invalid="ignore"):
        tx_a = (boxes[None, :, 0] - origin[0]) / dx
        tx_b = (boxes[None, :, 1] - origin[0]) / dx
        ty_a = (boxes[None, :, 2] - origin[1]) / dy
        ty_b = (boxes[None, :, 3] - origin[1]) / dy

    parallel_x = np.abs(dx) <= epsilon
    outside_x = (origin[0] < boxes[None, :, 0]) | (origin[0] > boxes[None, :, 1])
    tx_near = np.where(parallel_x, np.where(outside_x, np.inf, -np.inf), np.minimum(tx_a, tx_b))
    tx_far = np.where(parallel_x, np.where(outside_x, -np.inf, np.inf), np.maximum(tx_a, tx_b))

    parallel_y = np.abs(dy) <= epsilon
    outside_y = (origin[1] < boxes[None, :, 2]) | (origin[1] > boxes[None, :, 3])
    ty_near = np.where(parallel_y, np.where(outside_y, np.inf, -np.inf), np.minimum(ty_a, ty_b))
    ty_far = np.where(parallel_y, np.where(outside_y, -np.inf, np.inf), np.maximum(ty_a, ty_b))

    entry = np.maximum(tx_near, ty_near)
    exit_distance = np.minimum(tx_far, ty_far)
    valid = (exit_distance >= np.maximum(entry, 0.0)) & (entry <= max_distance_m)
    distances = np.where(valid, np.where(entry >= 0.0, entry, exit_distance), np.inf)
    return np.min(distances, axis=1).astype(np.float32)


def _setup_utlidar_l1(
    environment_usd: Path,
    *,
    use_physx: bool = False,
    use_rtx: bool = False,
):
    """Create deterministic 2D navigation and 3D mapping LiDAR publishers.

    ``/utlidar/scan`` remains a horizontal 360 degree LaserScan for navigation.
    ``/utlidar/points`` is an independent multi-elevation PointCloud2 used by
    RTAB-Map's 3D occupancy pipeline. The latter deliberately uses a lower
    update rate to keep the Python PhysX query load bounded.
    """
    import rclpy
    from rclpy.parameter import Parameter
    from geometry_msgs.msg import TransformStamped
    from sensor_msgs.msg import LaserScan, PointCloud2
    from tf2_ros import TransformBroadcaster

    if not rclpy.ok():
        rclpy.init()
    node = rclpy.create_node("utlidar_scan_publisher")
    node.set_parameters([Parameter("use_sim_time", value=True)])
    publisher = node.create_publisher(LaserScan, "/utlidar/scan", 10)
    cloud_publisher = node.create_publisher(PointCloud2, "/utlidar/points", 5)
    state = {
        "node": node,
        "publisher": publisher,
        "cloud_publisher": cloud_publisher,
        "tf_broadcaster": TransformBroadcaster(node),
        "transform_type": TransformStamped,
        "angles": np.linspace(-math.pi, math.pi, 360, endpoint=False, dtype=np.float32),
        "cloud_azimuths": np.linspace(-math.pi, math.pi, 180, endpoint=False, dtype=np.float32),
        "cloud_elevations": np.deg2rad(
            np.asarray((-20, -15, -10, -5, 0, 5, 10, 15, 20), dtype=np.float32)
        ),
        "last_pub_step": -10_000,
        "last_cloud_pub_step": -10_000,
        "scan_count": 0,
        "cloud_count": 0,
    }
    if use_rtx:
        from isaacsim.sensors.rtx import LidarRtx

        lidar = LidarRtx(
            prim_path="/World/envs/env_0/Robot/base/utlidar_lidar",
            name="utlidar_l1_rtx",
            translation=np.asarray((0.30, 0.0, 0.15), dtype=np.float32),
            config_file_name="HESAI_XT32_SD10",
            **{"omni:sensor:Core:outputFrameOfReference": "SENSOR"},
        )
        lidar.initialize()
        lidar.attach_writer(
            "RtxLidarROS2PublishPointCloudBuffer",
            frameId="utlidar_lidar",
            nodeNamespace="",
            queueSize=5,
            topicName="/utlidar/points",
            context=0,
            qosProfile="",
        )
        state["rtx_lidar"] = lidar
        print(
            ">>> Utlidar 3D RTX: ENABLED "
            "(/utlidar/points, Hesai XT32 32-channel full scans at 10 Hz, sensor frame)",
            flush=True,
        )
    elif use_physx:
        import omni.physx

        state["scene_query"] = omni.physx.get_physx_scene_query_interface()
        print(
            ">>> Utlidar 3D emulation: ENABLED "
            "(/utlidar/scan 360x1 at 10 Hz + /utlidar/points 180x9 at 2 Hz, "
            "live PhysX collision rays, vertical FOV -20..+20 deg, 30 m)",
            flush=True,
        )
    else:
        static_boxes = _load_utlidar_static_boxes(environment_usd)
        state["static_boxes"] = static_boxes
        print(
            ">>> Utlidar 3D emulation: ENABLED "
            f"(/utlidar/scan + /utlidar/points, {len(static_boxes)} static boxes, 30 m)",
            flush=True,
        )
    return state


def _quat_wxyz_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    """Return a float32 3x3 rotation matrix for a normalized WXYZ quaternion."""
    w, x, y, z = (float(value) for value in quaternion)
    norm = math.sqrt(w * w + x * x + y * y + z * z)
    if norm <= 1.0e-12:
        return np.eye(3, dtype=np.float32)
    w, x, y, z = w / norm, x / norm, y / norm, z / norm
    return np.asarray(
        (
            (1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)),
            (2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)),
            (2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)),
        ),
        dtype=np.float32,
    )


def _raycast_utlidar_directions(
    state,
    origin: np.ndarray,
    directions_world: np.ndarray,
    max_distance_m: float,
) -> np.ndarray:
    """Raycast arbitrary 3D unit directions and return finite hit distances."""
    distances = np.full(directions_world.shape[0], np.inf, dtype=np.float32)
    if "scene_query" in state:
        ray_origin = tuple(float(value) for value in origin)
        for index, direction in enumerate(directions_world):
            hit = state["scene_query"].raycast_closest(
                ray_origin,
                tuple(float(value) for value in direction),
                max_distance_m,
            )
            if not hit["hit"]:
                continue
            distance = float(hit["distance"])
            # Reject the chassis/arm immediately surrounding the sensor.
            if distance >= 0.40:
                distances[index] = distance
        return distances

    # Slab intersection fallback for authored Cube AABBs.
    boxes = np.asarray(state["static_boxes"], dtype=np.float32)
    box_min = boxes[:, (0, 2, 4)]
    box_max = boxes[:, (1, 3, 5)]
    epsilon = 1.0e-7
    for index, direction in enumerate(directions_world):
        with np.errstate(divide="ignore", invalid="ignore"):
            inverse = np.where(np.abs(direction) > epsilon, 1.0 / direction, np.inf)
            t0 = (box_min - origin) * inverse
            t1 = (box_max - origin) * inverse
        near = np.max(np.minimum(t0, t1), axis=1)
        far = np.min(np.maximum(t0, t1), axis=1)
        valid = (far >= np.maximum(near, 0.0)) & (near <= max_distance_m)
        if np.any(valid):
            candidates = np.where(valid, np.where(near >= 0.0, near, far), np.inf)
            distance = float(np.min(candidates))
            if distance >= 0.40:
                distances[index] = distance
    return distances


def _publish_xyz_cloud(state, sim_time_s: float, points_sensor: np.ndarray) -> None:
    """Publish a compact XYZ-only PointCloud2 without sensor_msgs_py."""
    from sensor_msgs.msg import PointCloud2, PointField

    points = np.ascontiguousarray(points_sensor, dtype=np.float32).reshape(-1, 3)
    message = PointCloud2()
    message.header.frame_id = "utlidar_lidar"
    message.header.stamp.sec = int(sim_time_s)
    message.header.stamp.nanosec = int((sim_time_s - int(sim_time_s)) * 1_000_000_000)
    message.height = 1
    message.width = int(points.shape[0])
    message.fields = [
        PointField(name="x", offset=0, datatype=PointField.FLOAT32, count=1),
        PointField(name="y", offset=4, datatype=PointField.FLOAT32, count=1),
        PointField(name="z", offset=8, datatype=PointField.FLOAT32, count=1),
    ]
    message.is_bigendian = False
    message.point_step = 12
    message.row_step = message.point_step * message.width
    message.data = points.tobytes()
    message.is_dense = True
    state["cloud_publisher"].publish(message)


def _publish_wrist_camera_tf(
    state,
    sim_time_s: float,
    robot,
    base_link_idx: int,
    wrist_camera_sensor,
) -> None:
    """Publish the moving base_link->wrist optical transform for RGB-D fusion."""
    if wrist_camera_sensor is None:
        return
    base_position_w = robot.data.body_pos_w[:, base_link_idx]
    base_quaternion_w = robot.data.body_quat_w[:, base_link_idx]
    wrist_position_w, wrist_quaternion_opengl_w = wrist_camera_sensor._view.get_world_poses()
    wrist_quaternion_ros_w = math_utils.convert_camera_frame_orientation_convention(
        wrist_quaternion_opengl_w,
        origin="opengl",
        target="ros",
    )
    wrist_position_base, wrist_quaternion_base = math_utils.subtract_frame_transforms(
        base_position_w,
        base_quaternion_w,
        wrist_position_w,
        wrist_quaternion_ros_w,
    )
    position = wrist_position_base[0].detach().cpu().numpy()
    quaternion = wrist_quaternion_base[0].detach().cpu().numpy()
    transform = state["transform_type"]()
    transform.header.stamp.sec = int(sim_time_s)
    transform.header.stamp.nanosec = int((sim_time_s - int(sim_time_s)) * 1_000_000_000)
    transform.header.frame_id = "base_link"
    transform.child_frame_id = "wrist_camera_optical_frame"
    transform.transform.translation.x = float(position[0])
    transform.transform.translation.y = float(position[1])
    transform.transform.translation.z = float(position[2])
    transform.transform.rotation.w = float(quaternion[0])
    transform.transform.rotation.x = float(quaternion[1])
    transform.transform.rotation.y = float(quaternion[2])
    transform.transform.rotation.z = float(quaternion[3])
    state["tf_broadcaster"].sendTransform(transform)


def _publish_utlidar_scan(
    state,
    sim_time_s: float,
    sim_step: int,
    robot,
    base_link_idx: int,
    wrist_camera_sensor=None,
) -> None:
    """Publish 2D navigation scan, dynamic wrist TF, and throttled 3D cloud."""
    if sim_step - state["last_pub_step"] < 20:  # 200 Hz physics -> 10 Hz scan
        return
    state["last_pub_step"] = sim_step

    # The RTX writer owns /utlidar/points. Hospital walls are render meshes
    # without PhysX collision, so image-space RTX ranging is required there.
    if "rtx_lidar" in state:
        _publish_wrist_camera_tf(
            state,
            sim_time_s,
            robot,
            base_link_idx,
            wrist_camera_sensor,
        )
        return

    from sensor_msgs.msg import LaserScan

    base_position = robot.data.body_pos_w[0, base_link_idx].detach().cpu().numpy()
    base_quaternion = robot.data.body_quat_w[0, base_link_idx].detach().cpu().numpy()
    rotation_world_from_base = _quat_wxyz_to_matrix(base_quaternion)
    sensor_offset_base = np.asarray((0.30, 0.0, 0.15), dtype=np.float32)
    origin = base_position.astype(np.float32) + rotation_world_from_base @ sensor_offset_base
    scan_directions_sensor = np.column_stack(
        (
            np.cos(state["angles"]),
            np.sin(state["angles"]),
            np.zeros_like(state["angles"]),
        )
    ).astype(np.float32)
    scan_directions_world = scan_directions_sensor @ rotation_world_from_base.T
    distances = _raycast_utlidar_directions(state, origin, scan_directions_world, 30.0)
    ranges = np.where((distances >= 0.15) & (distances <= 30.0), distances, np.inf).tolist()

    count = len(ranges)
    message = LaserScan()
    message.header.frame_id = "utlidar_lidar"
    stamp_s = float(sim_time_s)
    message.header.stamp.sec = int(stamp_s)
    message.header.stamp.nanosec = int((stamp_s - int(stamp_s)) * 1_000_000_000)
    message.angle_min = -math.pi
    message.angle_max = math.pi - 2.0 * math.pi / count
    message.angle_increment = 2.0 * math.pi / count
    message.time_increment = 0.0
    message.scan_time = 0.1
    message.range_min = 0.15
    message.range_max = 30.0
    message.ranges = ranges
    state["publisher"].publish(message)
    _publish_wrist_camera_tf(
        state,
        sim_time_s,
        robot,
        base_link_idx,
        wrist_camera_sensor,
    )
    state["scan_count"] += 1
    if state["scan_count"] % 50 == 0:
        valid_count = int(np.isfinite(distances).sum())
        print(f">>> Utlidar scan health: valid={valid_count}/{count}", flush=True)

    # 200 Hz physics -> 2 Hz 3D cloud. Nine 180-point rings preserve enough
    # structure for registration without dominating the RGB-D renderers.
    if sim_step - state["last_cloud_pub_step"] < 100:
        return
    state["last_cloud_pub_step"] = sim_step
    azimuth_grid, elevation_grid = np.meshgrid(
        state["cloud_azimuths"],
        state["cloud_elevations"],
    )
    cos_elevation = np.cos(elevation_grid)
    cloud_directions_sensor = np.column_stack(
        (
            (cos_elevation * np.cos(azimuth_grid)).reshape(-1),
            (cos_elevation * np.sin(azimuth_grid)).reshape(-1),
            np.sin(elevation_grid).reshape(-1),
        )
    ).astype(np.float32)
    cloud_directions_world = cloud_directions_sensor @ rotation_world_from_base.T
    cloud_distances = _raycast_utlidar_directions(
        state,
        origin,
        cloud_directions_world,
        30.0,
    )
    valid = np.isfinite(cloud_distances) & (cloud_distances >= 0.40) & (cloud_distances <= 30.0)
    points_sensor = cloud_directions_sensor[valid] * cloud_distances[valid, None]
    _publish_xyz_cloud(state, sim_time_s, points_sensor)
    state["cloud_count"] += 1
    if state["cloud_count"] % 10 == 0 and points_sensor.size:
        print(
            f">>> Utlidar 3D cloud health: valid={int(valid.sum())}/{len(valid)} "
            f"z=[{float(points_sensor[:, 2].min()):.2f},{float(points_sensor[:, 2].max()):.2f}]m",
            flush=True,
        )


def configure_so_arm_collision():
    """SO-Arm의 collision만 관리한다.

    보행 정책(model_3150)은 팔의 실제 질량/중력이 포함된 Go2+SO-Arm 통합 로봇으로
    학습됐다(02. 강화학습 자세 참고). 따라서 팔을 무중력/무질량화하면 mass·CoM·관성이
    학습 분포를 벗어나 정책이 과동작해 시동 직후 튀어오른다. 그래서 질량·중력은 URDF
    실측값 그대로 두고, 텔레옵 안정화에 필요한 collision 처리만 수행한다.
      - 팔 본체 링크: collision OFF (Go2 몸체/자기 충돌로 인한 지터 방지)
      - 그리퍼 링크: collision ON  (서랍 핸들 등 물리 파지)
    """
    import omni.usd
    from pxr import UsdPhysics, Usd

    stage = omni.usd.get_context().get_stage()
    arm_collision_off = []
    gripper_kept = []
    missing_links = []
    for link_name in SO_ARM_LINKS:
        prim_path = f"/World/envs/env_0/Robot/{link_name}"
        prim = stage.GetPrimAtPath(prim_path)
        if not prim.IsValid():
            missing_links.append(link_name)
            continue
        if link_name in SO_ARM_GRIPPER_LINKS:
            gripper_kept.append(link_name)
            continue
        for child in Usd.PrimRange(prim):
            if child.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI(child).CreateCollisionEnabledAttr(False)
        arm_collision_off.append(link_name)
    if missing_links:
        raise RuntimeError(f"SO-Arm collision links missing from imported runtime asset: {missing_links}")
    print(
        f">>> SO-Arm collision: arm OFF={arm_collision_off} | gripper KEPT={gripper_kept} "
        f"| mass/gravity = URDF 실측 유지 (보행 정책 학습 분포 일치)"
    )


@configclass
class NBVv4SceneCfg(InteractiveSceneCfg):
    # 기본 환경
    ground = None if args_cli.environment_usd else AssetBaseCfg(
        prim_path="/World/ground",
        spawn=(
            sim_utils.GroundPlaneCfg(
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=1.0,
                    dynamic_friction=1.0,
                    restitution=0.0,
                )
            )
            if ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed"
            else sim_utils.GroundPlaneCfg()
        ),
    )
    light = AssetBaseCfg(prim_path="/World/light", spawn=sim_utils.DomeLightCfg(intensity=3000.0))

    warehouse = (
        AssetBaseCfg(
            prim_path="/World/environment",
            spawn=sim_utils.UsdFileCfg(usd_path=str(args_cli.environment_usd)),
        )
        if args_cli.environment_usd
        else None
    )

    # 오피스 데스크 (go2 전방 3m, glb→obj→URDF 변환, fix_base 고정)
    desk = None if args_cli.environment_usd else AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Desk",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/iy/robot_models/desk_urdf/med_office_desk.urdf",
            make_instanceable=False,
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(1.5, 0.6, 0.0), rot=(0.0, 0.0, 0.0, 1.0)),  # go2 전방 1.5m (기존 2.0m에서 0.5m 단축)
    )

    # 서랍장 (데스크 위, go2 전방 3m) — fix_base 로 데스크 상면 위에 고정, 서랍 관절은 position 제어
    drawer = None if args_cli.environment_usd else ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Drawer",
        spawn=sim_utils.UrdfFileCfg(
            asset_path="/home/iy/robot_models/drawer_urdf/drawer.urdf",
            make_instanceable=False,
            fix_base=True,
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=100.0, damping=10.0),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(pos=(1.6, 0.1, 0.635), rot=(0.707, 0.0, 0.0, -0.707)),  # 책상 1.5m 이동에 맞춤. 서랍 body 반높이 0.112 -> z=0.5231+0.112=0.635
        actuators={
            "drawers": ImplicitActuatorCfg(
                joint_names_expr=[".*"],
                stiffness=0.0,
                damping=5.0,
            ),
        },
    )

    # 조작 대상 큐브 2개 (책상 상면 위, drawer 양옆, 2.7cm) — Go2 쪽으로 당기되 책상 x-min(≈1.994m)을 벗어나지 않게 center x=2.03 유지
    orange_cube = None if args_cli.environment_usd else RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/OrangeCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.027, 0.027, 0.027),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.30, 0.0)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_angular_velocity=1000.0, max_linear_velocity=1000.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.53, -0.158, 0.610)),
    )
    green_cube = None if args_cli.environment_usd else RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/GreenCube",
        spawn=sim_utils.CuboidCfg(
            size=(0.027, 0.027, 0.027),
            visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(0.1, 0.8, 0.1)),
            collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True, contact_offset=0.005, rest_offset=0.0),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(solver_position_iteration_count=16, solver_velocity_iteration_count=1, max_angular_velocity=1000.0, max_linear_velocity=1000.0),
            mass_props=sim_utils.MassPropertiesCfg(mass=0.08),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(1.53, -0.103, 0.610)),
    )

    # 로봇 (Go2 + SO-Arm), instanceable=False 로 GUI 조작 가능
    robot = ArticulationCfg(
        prim_path="{ENV_REGEX_NS}/Robot",
        spawn=sim_utils.UrdfFileCfg(
            asset_path=str(GO2_BASE_URDF if args_cli.no_arm else ROBOT_MODELS_ROOT / ROBOT_MODEL_PROFILE.asset_path),
            make_instanceable=args_cli.no_arm or ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed",
            fix_base=False,
            activate_contact_sensors=ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed",
            self_collision=ROBOT_MODEL_PROFILE.key == "so101_7motor",
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=ROBOT_MODEL_PROFILE.key == "so101_7motor",
                solver_position_iteration_count=8 if (args_cli.active_gap_arm or args_cli.maze_inspection_layout) else 4,
                solver_velocity_iteration_count=4 if (args_cli.active_gap_arm or args_cli.maze_inspection_layout) else 0,
            ),
            joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                drive_type="force",
                target_type="position",
                gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                    stiffness=400.0,
                    damping=160.0 if args_cli.active_gap_arm else 20.0,
                ),
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(ROBOT_SPAWN_XY[0], ROBOT_SPAWN_XY[1], 0.4),
            rot=ROBOT_SPAWN_ROT,
            joint_pos={
                ".*L_hip_joint": 0.1,
                ".*R_hip_joint": -0.1,
                "F[L,R]_thigh_joint": 0.8,
                "R[L,R]_thigh_joint": 1.0,
                ".*calf_joint": -1.5,
                **ROBOT_INITIAL_JOINT_POS,
            },
        ),
        soft_joint_pos_limit_factor=1.0,
        actuators={
            "legs": DCMotorCfg(
                joint_names_expr=[".*_hip_joint", ".*_thigh_joint", ".*_calf_joint"],
                effort_limit=23.5,
                saturation_effort=23.5,
                velocity_limit=30.0,
                stiffness=25.0,
                damping=0.5,
                friction=0.0,
            ),
            **({} if args_cli.no_arm else {"arm": ImplicitActuatorCfg(
                joint_names_expr=(
                    list(ROBOT_MODEL_PROFILE.simulation_arm_joint_order)
                    if ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed"
                    else [
                        joint_name
                        for joint_name in ROBOT_MODEL_PROFILE.simulation_arm_joint_order
                        if joint_name != "gripper"
                    ]
                ),
                effort_limit_sim=300.0 if ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed" else 100.0,
                stiffness=400.0,
                # Keep the locomotion/training actuator distribution.  The
                # previous active-arm-only value 160 caused a persistent
                # discrete-time elbow velocity oscillation (~0.2 rad/s), so
                # authorized trajectories could never satisfy the settle gate.
                damping=20.0,
            )}),
            **(
                {}
                if args_cli.no_arm or ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed"
                else {
                    "gripper": ImplicitActuatorCfg(
                        joint_names_expr=["gripper"],
                        effort_limit_sim=100.0,
                        stiffness=200.0,
                        damping=20.0,
                    )
                }
            ),
        },
    )

    # 카메라
    wrist_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/gripper_link/wrist_camera",
        update_period=0.0,
        height=WRIST_CAMERA_HEIGHT,
        width=WRIST_CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"] if (args_cli.slam_rgbd or args_cli.maze_nav2_inspection or
            (args_cli.maze_visual_appearance and not (args_cli.maze_nav2 or args_cli.maze_locomotion_only))) else ["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=14.0,
            focus_distance=0.35,
            horizontal_aperture=20.955,
            clipping_range=(0.05, 20.0) if args_cli.slam_rgbd else ((0.01, 5.0) if args_cli.maze_nav2_inspection else (0.01, 2.0)),
        ),
        offset=CameraCfg.OffsetCfg(
            pos=WRIST_CAMERA_LOCAL_POS,
            rot=WRIST_CAMERA_LOCAL_ROT,
            convention="opengl",
        ),
        update_latest_camera_pose=args_cli.maze_observed_navigation or args_cli.maze_visual_appearance,
    )

    front_camera = CameraCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base/front_camera_sensor",
        update_period=1.0 / 30.0,
        height=FRONT_CAMERA_HEIGHT,
        width=FRONT_CAMERA_WIDTH,
        data_types=["rgb", "distance_to_image_plane"] if (args_cli.slam_rgbd or args_cli.maze_observation_navigation or
            (args_cli.maze_visual_appearance and not (args_cli.maze_nav2 or args_cli.maze_locomotion_only))) else ["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=24.0,
            focus_distance=2.0,
            horizontal_aperture=20.955,
            clipping_range=(0.01, 100.0),
        ),
        offset=CameraCfg.OffsetCfg(pos=FRONT_CAMERA_LOCAL_POS, rot=FRONT_CAMERA_LOCAL_ROT, convention="world"),
        update_latest_camera_pose=args_cli.maze_observed_navigation or args_cli.maze_visual_appearance,
    )

    if args_cli.maze_observation_navigation and os.environ.get('MAZE_LOCAL_OCCUPANCY')=='1':
        for mapping_camera in (wrist_camera,front_camera):
            mapping_camera.data_types.append('instance_id_segmentation_fast')
            mapping_camera.colorize_instance_id_segmentation=False
        del mapping_camera  # Config class attributes are instantiated as scene entities.

    arm_contact_sensor = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*",
        update_period=0.0,
        history_length=2,
        track_air_time=False,
    ) if (args_cli.active_gap_arm or args_cli.maze_inspection_layout) and not args_cli.no_arm else None

    height_scanner = None if args_cli.environment_usd else RayCasterCfg(
        prim_path="{ENV_REGEX_NS}/Robot/base",
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 20.0)),
        ray_alignment="yaw",
        pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[1.6, 1.0]),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )
    # 햅틱 피드백: ContactSensor는 GPU PhysX에서 접촉력이 0으로 나오는 문제가 있어
    # 그리퍼 모터 joint 상태(target vs actual + velocity)로 직접 파지 저항을 계산한다.




_leader_proc = None
_smolvla_proc = None


def stop_existing_leader_processes(leader_script: Path) -> None:
    leader_script_path = str(leader_script)
    try:
        proc_list = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f">>> [leader_auto] process scan warning: {exc}")
        return

    current_pid = os.getpid()
    stopped = []
    for line in proc_list.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_text, command, arguments = parts
        if "python" not in command or leader_script_path not in arguments:
            continue
        pid = int(pid_text)
        if pid == current_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            continue
    if stopped:
        print(f">>> [leader_auto] stopped stale leader_bridge process(es): {stopped}")
        time.sleep(1.0)


def start_leader_subprocess(args_ns) -> None:
    """leader_teleop_bridge.py를 백그라운드 subprocess로 시작 (텔레오퍼레이션 자동 연결).

    핵심 방어:
      1) --no-calibrate 강제 → calibrate()/input() 진입 차단 (EOF 원천 차단)
      2) --leader-id teleop_leader_v1 → 실제 캘리브 파일 매칭 (self.calibration={} 방지)
    """
    global _leader_proc
    if not getattr(args_ns, 'leader_auto', False):
        return
    leader_script = Path(args_ns.leader_script)
    if not leader_script.is_file():
        print(f">>> [leader_auto] leader script not found, skipping: {leader_script}")
        return
    stop_existing_leader_processes(leader_script)
    # USB 포트가 자주 바뀌므로 실제 존재하는 포트를 자동 탐색한다.
    _port = args_ns.leader_port_dev
    if not Path(_port).exists():
        for _candidate in ("/dev/ttyACM0", "/dev/ttyACM1", "/dev/ttyACM2", "/dev/ttyUSB0", "/dev/ttyUSB1"):
            if Path(_candidate).exists():
                _port = _candidate
                print(f">>> [leader_auto] 지정 포트({args_ns.leader_port_dev}) 없음, 자동 감지: {_port}")
                break
    log_fh = open(args_ns.leader_log, 'ab', buffering=0)
    cmd = [
        args_ns.leader_python, '-u', str(leader_script),
        '--leader-port', _port,
        '--leader-type', args_ns.leader_type,
        '--leader-id', args_ns.leader_id,
        '--calibration-dir', args_ns.leader_calibration_dir,
        '--action-port', str(args_ns.gr00t_action_port),
        '--fps', str(args_ns.leader_fps),
        '--log-every', str(args_ns.leader_action_log_every),
        '--runtime-offset-json', args_ns.runtime_offset_json,
        '--no-calibrate',
    ]
    if getattr(args_ns, 'enable_haptic_feedback', False):
        cmd.extend([
            '--feedback-port', str(args_ns.haptic_port),
            '--enable-haptic-feedback',
        ])
    _leader_proc = subprocess.Popen(
        cmd, stdout=log_fh, stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL, start_new_session=True,
    )
    atexit.register(stop_leader_subprocess)
    try:
        signal.signal(signal.SIGTERM, lambda *_: (stop_leader_subprocess(), sys.exit(0)))
    except (ValueError, OSError):
        pass
    print(f">>> [leader_auto] leader_bridge PID={_leader_proc.pid} started")
    print(f">>> [leader_auto] log: {args_ns.leader_log}")
    time.sleep(1.5)
    rc = _leader_proc.poll()
    if rc is not None:
        print(f">>> [leader_auto] WARNING: leader_bridge exited early (code={rc}). Check: tail -50 {args_ns.leader_log}")
        print(f">>> [leader_auto] If calibration missing: run calibrate_leader.sh first.")
        _leader_proc = None


def stop_leader_subprocess() -> None:
    """leader 자식 프로세스 정리 (graceful SIGTERM → 강제 kill)."""
    global _leader_proc
    proc = _leader_proc
    _leader_proc = None
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(">>> [leader_auto] leader_bridge stopped")
    except Exception as e:
        print(f">>> [leader_auto] stop warning: {e}")


def stop_existing_smolvla_processes(runner_script: Path) -> None:
    runner_script_path = str(runner_script)
    try:
        proc_list = subprocess.run(
            ["ps", "-eo", "pid=,comm=,args="],
            check=True,
            capture_output=True,
            text=True,
        )
    except subprocess.CalledProcessError as exc:
        print(f">>> [smolvla] process scan warning: {exc}")
        return

    current_pid = os.getpid()
    stopped = []
    for line in proc_list.stdout.splitlines():
        parts = line.strip().split(maxsplit=2)
        if len(parts) < 3:
            continue
        pid_text, command, arguments = parts
        if "python" not in command or runner_script_path not in arguments:
            continue
        pid = int(pid_text)
        if pid == current_pid:
            continue
        try:
            os.kill(pid, signal.SIGTERM)
            stopped.append(pid)
        except ProcessLookupError:
            continue
    if stopped:
        print(f">>> [smolvla] stopped stale policy runner process(es): {stopped}")
        time.sleep(1.0)


def start_smolvla_subprocess(args_ns) -> None:
    """Start the SmolVLA runner in the LeRobot environment."""
    global _smolvla_proc
    if not getattr(args_ns, "enable_smolvla_policy", False):
        return
    runner_script = Path(args_ns.smolvla_script)
    if not runner_script.is_file():
        print(f">>> [smolvla] policy runner script not found, skipping: {runner_script}")
        return
    policy_path = Path(args_ns.smolvla_policy_path)
    if not policy_path.is_dir():
        print(f">>> [smolvla] policy directory not found, skipping: {policy_path}")
        return

    stop_existing_smolvla_processes(runner_script)
    log_fh = open(args_ns.smolvla_log, "ab", buffering=0)
    cmd = [
        args_ns.smolvla_python,
        "-u",
        str(runner_script),
        "--policy-path",
        str(policy_path),
        "--task",
        args_ns.smolvla_task,
        "--obs-port",
        str(args_ns.smolvla_obs_port),
        "--action-port",
        str(args_ns.smolvla_action_port),
        "--device",
        args_ns.smolvla_device,
        "--log-every",
        str(args_ns.smolvla_action_log_every),
    ]
    if getattr(args_ns, "smolvla_no_amp", False):
        cmd.append("--no-amp")

    env = os.environ.copy()
    python_path = Path(args_ns.smolvla_python)
    conda_env_dir = python_path.parent.parent
    conda_lib = conda_env_dir / "lib"
    if conda_lib.is_dir():
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = f"{conda_lib}:{old_ld}" if old_ld else str(conda_lib)
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HUB_OFFLINE", "1")

    _smolvla_proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    atexit.register(stop_smolvla_subprocess)
    try:
        signal.signal(signal.SIGTERM, lambda *_: (stop_smolvla_subprocess(), stop_leader_subprocess(), sys.exit(0)))
    except (ValueError, OSError):
        pass
    print(f">>> [smolvla] policy runner PID={_smolvla_proc.pid} started")
    print(f">>> [smolvla] log: {args_ns.smolvla_log}")
    time.sleep(0.5)
    rc = _smolvla_proc.poll()
    if rc is not None:
        print(f">>> [smolvla] WARNING: policy runner exited early (code={rc}). Check log: {args_ns.smolvla_log}")
        _smolvla_proc = None


def start_hazard_smolvla_subprocess(args_ns) -> None:
    """Start the exact 7-motor + 1-decision SmolVLA runner."""
    global _smolvla_proc
    if not getattr(args_ns, "enable_hazard_smolvla_policy", False):
        return
    runner_script = Path(args_ns.hazard_smolvla_script).expanduser().resolve()
    policy_path = Path(args_ns.hazard_smolvla_policy_path).expanduser().resolve()
    if not runner_script.is_file():
        raise FileNotFoundError(f"hazard SmolVLA runner is missing: {runner_script}")
    if not policy_path.is_dir():
        raise FileNotFoundError(f"hazard SmolVLA policy is missing: {policy_path}")

    stop_existing_smolvla_processes(runner_script)
    log_fh = open(args_ns.hazard_smolvla_log, "ab", buffering=0)
    cmd = [
        args_ns.hazard_smolvla_python,
        "-u",
        str(runner_script),
        "--policy-path",
        str(policy_path),
        "--obs-port",
        str(args_ns.hazard_smolvla_obs_port),
        "--action-port",
        str(args_ns.hazard_smolvla_action_port),
        "--device",
        args_ns.hazard_smolvla_device,
        "--log-every",
        str(args_ns.hazard_smolvla_action_log_every),
    ]
    if args_ns.hazard_smolvla_no_amp:
        cmd.append("--no-amp")

    env = os.environ.copy()
    python_path = Path(args_ns.hazard_smolvla_python)
    conda_lib = python_path.parent.parent / "lib"
    if conda_lib.is_dir():
        old_ld = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = (
            f"{conda_lib}:{old_ld}" if old_ld else str(conda_lib)
        )
    env["PYTHONUNBUFFERED"] = "1"
    env.setdefault("HF_HUB_OFFLINE", "1")

    _smolvla_proc = subprocess.Popen(
        cmd,
        stdout=log_fh,
        stderr=subprocess.STDOUT,
        stdin=subprocess.DEVNULL,
        start_new_session=True,
        env=env,
    )
    atexit.register(stop_smolvla_subprocess)
    try:
        signal.signal(
            signal.SIGTERM,
            lambda *_: (
                stop_smolvla_subprocess(),
                stop_leader_subprocess(),
                sys.exit(0),
            ),
        )
    except (ValueError, OSError):
        pass
    print(
        f">>> [smolvla_hazard] policy runner PID={_smolvla_proc.pid} started",
        flush=True,
    )
    print(f">>> [smolvla_hazard] log: {args_ns.hazard_smolvla_log}", flush=True)
    time.sleep(0.5)
    rc = _smolvla_proc.poll()
    if rc is not None:
        _smolvla_proc = None
        raise RuntimeError(
            f"hazard SmolVLA runner exited early (code={rc}); "
            f"check {args_ns.hazard_smolvla_log}"
        )


def stop_smolvla_subprocess() -> None:
    """Stop the external SmolVLA runner."""
    global _smolvla_proc
    proc = _smolvla_proc
    _smolvla_proc = None
    if proc is None:
        return
    try:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=3.0)
            except subprocess.TimeoutExpired:
                proc.kill()
        print(">>> [smolvla] policy runner stopped")
    except Exception as e:
        print(f">>> [smolvla] stop warning: {e}")


def main():
    ros2_slam_handles = None
    active_arm_server = None
    if ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed":
        sim_cfg = sim_utils.SimulationCfg(
            device=args_cli.device,
            dt=0.005,
            # This standalone loop already owns policy decimation.  Keep the
            # underlying World rendering_dt equal to physics_dt so one
            # sim.step() is exactly one 5 ms physics step.  GUI render skipping
            # is applied explicitly at the call site below.
            render_interval=1,
            use_fabric=not args_cli.disable_fabric,
            physics_material=sim_utils.RigidBodyMaterialCfg(
                friction_combine_mode="multiply",
                restitution_combine_mode="multiply",
                static_friction=1.0,
                dynamic_friction=1.0,
                restitution=0.0,
            ),
            physx=sim_utils.PhysxCfg(
                gpu_max_rigid_patch_count=10 * 2**15,
                enable_external_forces_every_iteration=args_cli.active_gap_arm,
            ),
        )
    else:
        sim_cfg = sim_utils.SimulationCfg(
            device=args_cli.device,
            dt=0.005,
            render_interval=1,
            use_fabric=not args_cli.disable_fabric,
            physx=sim_utils.PhysxCfg(
                bounce_threshold_velocity=0.01,
                friction_correlation_distance=0.00625,
                gpu_found_lost_aggregate_pairs_capacity=1024 * 1024 * 4,
                gpu_total_aggregate_pairs_capacity=16 * 1024,
            ),
        )
    sim = sim_utils.SimulationContext(sim_cfg)
    if args_cli.state_debug_every > 0:
        print(f">>> [timing] configured_dt={sim_cfg.dt} physics_dt={sim.get_physics_dt()}", flush=True)

    scene_cfg = NBVv4SceneCfg(num_envs=1, env_spacing=5.0)
    if args_cli.gait_probe:
        # Use local collision geometry, not the default remote Grid USD.
        scene_cfg.desk = scene_cfg.drawer = None
        scene_cfg.orange_cube = scene_cfg.green_cube = None
        # Flat 48-D policy has no terrain-height observation. The default
        # RayCaster expects a mesh ground and must not query a Cube primitive.
        scene_cfg.height_scanner = None
        scene_cfg.robot.spawn.articulation_props.solver_position_iteration_count = 8
        scene_cfg.robot.spawn.articulation_props.solver_velocity_iteration_count = 4
        scene_cfg.ground = AssetBaseCfg(
            prim_path='/World/ground',
            init_state=AssetBaseCfg.InitialStateCfg(pos=(0., 0., -.1)),
            spawn=sim_utils.CuboidCfg(
                size=(30., 30., .2),
                collision_props=sim_utils.CollisionPropertiesCfg(),
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode='multiply', restitution_combine_mode='multiply',
                    static_friction=1., dynamic_friction=1., restitution=0.),
                visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(.45, .45, .45)),
            ),
        )
    if not args_cli.enable_cameras:
        scene_cfg.wrist_camera = None
        scene_cfg.front_camera = None
    elif args_cli.no_arm or args_cli.disable_wrist_camera:
        scene_cfg.wrist_camera = None
    scene = InteractiveScene(scene_cfg)
    hide_hospital_small_room_structure()
    if args_cli.maze_visual_appearance:
        import importlib.util
        appearance_dir = Path('/home/iy/Isaac/maze_visual_demo')
        sys.path.insert(0, str(appearance_dir))
        from physical_appearance import apply as apply_maze_appearance
        import omni.usd
        apply_maze_appearance(omni.usd.get_context().get_stage())

    so_arm_dynamics_label = "URDF DEFAULT"
    if args_cli.leader_auto:
        configure_so_arm_collision()
        so_arm_dynamics_label = "TELEOP (collision-managed; mass/gravity = URDF 실측 유지)"

    sim.reset()
    scene.reset()
    camera_viewports = []

    robot = scene["robot"]
    if ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed":
        material_props = robot.root_physx_view.get_material_properties()
        material_props[..., 0] = 0.8
        material_props[..., 1] = 0.6
        material_props[..., 2] = 0.0
        material_env_ids = torch.arange(material_props.shape[0], dtype=torch.int32, device="cpu")
        robot.root_physx_view.set_material_properties(material_props, material_env_ids)
    height_scanner = scene.sensors.get("height_scanner")
    default_joint_pos = robot.data.default_joint_pos.clone()
    policy = torch.jit.load(args_cli.go2_policy_path, map_location=sim.device).eval()
    print(
        f">>> Go2 policy: mode={args_cli.go2_policy_obs_mode} "
        f"obs_dim={GO2_POLICY_OBS_DIMS[args_cli.go2_policy_obs_mode]} path={args_cli.go2_policy_path}",
        flush=True,
    )
    if args_cli.state_debug_every > 0:
        material_props = robot.root_physx_view.get_material_properties()
        print(
            f">>> [materials] shape={tuple(material_props.shape)} "
            f"static=({float(material_props[..., 0].min()):.3f},{float(material_props[..., 0].max()):.3f}) "
            f"dynamic=({float(material_props[..., 1].min()):.3f},{float(material_props[..., 1].max()):.3f}) "
            f"restitution=({float(material_props[..., 2].min()):.3f},{float(material_props[..., 2].max()):.3f})",
            flush=True,
        )
    requested_rough_joint_order = (
        ROUGH_POLICY_JOINT_ORDER[:12] if args_cli.no_arm else ROUGH_POLICY_JOINT_ORDER
    )
    rough_joint_ids, rough_joint_names = robot.find_joints(requested_rough_joint_order, preserve_order=True)
    if rough_joint_names != requested_rough_joint_order:
        raise RuntimeError(f"Unexpected rough-policy joint order: {rough_joint_names}")
    rough_joint_ids = torch.tensor(rough_joint_ids, device=sim.device, dtype=torch.long)
    leg_joint_ids = rough_joint_ids[:12]
    leg_joint_names = rough_joint_names[:12]
    if ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed" or (
        args_cli.no_arm and ROBOT_MODEL_PROFILE.key == "legacy"
    ):
        policy_action_joint_ids, policy_action_joint_names = robot.find_joints(
            MIXED_POLICY_ACTION_JOINT_ORDER, preserve_order=True
        )
        if policy_action_joint_names != MIXED_POLICY_ACTION_JOINT_ORDER:
            raise RuntimeError(f"Unexpected mixed-policy action joint order: {policy_action_joint_names}")
        policy_action_joint_ids = torch.tensor(policy_action_joint_ids, device=sim.device, dtype=torch.long)
    else:
        policy_action_joint_ids = leg_joint_ids
    base_link_idx = robot.find_bodies("base")[0][0]
    if args_cli.no_arm:
        external_joint_names = []
        external_joint_ids = torch.empty(0, device=sim.device, dtype=torch.long)
        sim_arm_joint_names = []
        sim_arm_joint_ids = torch.empty(0, device=sim.device, dtype=torch.long)
        gripper_link_idx = None
        gripper_joint_idx = None
        sim_arm_only_mask = torch.empty(0, device=sim.device, dtype=torch.bool)
        sim_arm_only_joint_ids = sim_arm_joint_ids
        elbow_rotate_joint_idx = None
    else:
        runtime_external_joint_order = (
            NBV_EXTERNAL_JOINT_ORDER
            if (args_cli.active_gap_arm or args_cli.maze_inspection_layout)
            else ROBOT_MODEL_PROFILE.external_joint_order
        )
        external_joint_ids_raw, external_joint_names = robot.find_joints(
            list(runtime_external_joint_order), preserve_order=True
        )
        if tuple(external_joint_names) != tuple(runtime_external_joint_order):
            raise RuntimeError(f"Unexpected external SO-Arm joint order: {external_joint_names}")
        external_joint_ids = torch.tensor(external_joint_ids_raw, device=sim.device, dtype=torch.long)
        sim_arm_joint_ids_raw, sim_arm_joint_names = robot.find_joints(
            list(ROBOT_MODEL_PROFILE.simulation_arm_joint_order), preserve_order=True
        )
        if tuple(sim_arm_joint_names) != ROBOT_MODEL_PROFILE.simulation_arm_joint_order:
            raise RuntimeError(f"Unexpected simulated SO-Arm joint order: {sim_arm_joint_names}")
        sim_arm_joint_ids = torch.tensor(sim_arm_joint_ids_raw, device=sim.device, dtype=torch.long)
        gripper_link_idx = robot.find_bodies("gripper_link")[0][0]
        gripper_joint_idx = int(external_joint_ids[external_joint_names.index("gripper")].item())
        sim_arm_only_mask = sim_arm_joint_ids != gripper_joint_idx
        sim_arm_only_joint_ids = sim_arm_joint_ids[sim_arm_only_mask]
        elbow_rotate_joint_idx = (
            int(sim_arm_joint_ids[sim_arm_joint_names.index("elbow_rotate")].item())
            if "elbow_rotate" in sim_arm_joint_names
            else None
        )
    elbow_rotate_hold_rad = ELBOW_ROTATE_HOLD_RAD
    if ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed" and not args_cli.no_arm:
        elbow_flex_joint_idx = int(sim_arm_joint_ids[sim_arm_joint_names.index("elbow_flex")].item())
        elbow_limits = torch.tensor([[[-math.pi, math.pi]]], dtype=torch.float32, device=sim.device)
        robot.write_joint_position_limit_to_sim(elbow_limits, joint_ids=[elbow_flex_joint_idx])
        print(">>> Reversed elbow_flex runtime limits expanded to [-180, +180] deg for validated teleoperation")

    wrist_camera_sensor = (
        scene.sensors["wrist_camera"]
        if args_cli.enable_cameras and not args_cli.no_arm and not args_cli.disable_wrist_camera
        else None
    )
    front_camera_sensor = scene.sensors["front_camera"] if args_cli.enable_cameras else None
    arm_contact_sensor_runtime = scene.sensors.get("arm_contact_sensor") if (args_cli.active_gap_arm or args_cli.maze_inspection_layout) else None
    arm_contact_body_ids = []
    foot_contact_body_ids = []
    maze_foot_body_ids = []
    if arm_contact_sensor_runtime is not None:
        arm_contact_body_ids, resolved_contact_names = arm_contact_sensor_runtime.find_bodies(
            list(ROBOT_MODEL_PROFILE.collision_link_names),
            preserve_order=True,
        )
        if tuple(resolved_contact_names) != ROBOT_MODEL_PROFILE.collision_link_names:
            raise RuntimeError(
                f"Active arm contact audit mismatch: {resolved_contact_names}"
            )
        print(
            f">>> Active arm collision audit: {resolved_contact_names}",
            flush=True,
        )
        foot_contact_body_ids, resolved_foot_contact_names = (
            arm_contact_sensor_runtime.find_bodies(
                ["FL_foot", "FR_foot", "RL_foot", "RR_foot"],
                preserve_order=True,
            )
        )
        if tuple(resolved_foot_contact_names) != (
            "FL_foot",
            "FR_foot",
            "RL_foot",
            "RR_foot",
        ):
            raise RuntimeError(
                f"Go2 foot-contact audit mismatch: {resolved_foot_contact_names}"
            )
        if args_cli.maze_inspection_layout:
            maze_foot_body_ids, maze_foot_names = robot.find_bodies(
                list(resolved_foot_contact_names), preserve_order=True)
            if tuple(maze_foot_names) != tuple(resolved_foot_contact_names):
                raise RuntimeError('Maze foot pose/contact ordering mismatch')
    arm_forbidden_contact_ticks = 0
    wrist_camera_local_pos = torch.tensor([WRIST_CAMERA_LOCAL_POS], dtype=torch.float32, device=sim.device)
    wrist_camera_local_rot = torch.tensor([WRIST_CAMERA_LOCAL_ROT], dtype=torch.float32, device=sim.device)
    front_camera_local_pos = torch.tensor([FRONT_CAMERA_LOCAL_POS], dtype=torch.float32, device=sim.device)
    front_camera_local_rot = torch.tensor([FRONT_CAMERA_LOCAL_ROT], dtype=torch.float32, device=sim.device)
    default_leg_joint_pos = default_joint_pos[:, leg_joint_ids].clone()
    default_policy_action_joint_pos = default_joint_pos[:, policy_action_joint_ids].clone()
    default_rough_joint_pos = default_joint_pos[:, rough_joint_ids].clone()
    default_joint_vel = robot.data.default_joint_vel.clone()
    last_action = torch.zeros((1, len(leg_joint_names)), device=sim.device)
    smoothed_vel_cmd_b = torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
    requested_vel_cmd_b = torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
    has_received_motion_command = False
    policy_decimation = 4
    print(
        f">>> Locomotion timing: physics={1.0 / sim_cfg.dt:.0f}Hz "
        f"policy={1.0 / (sim_cfg.dt * policy_decimation):.0f}Hz "
        f"render={1.0 / (sim_cfg.dt * args_cli.render_interval):.0f}Hz "
        f"policy_decimation={policy_decimation} render_skip={args_cli.render_interval}",
        flush=True,
    )
    policy_step = 0
    joint_targets = default_joint_pos.clone()
    if args_cli.active_gap_arm or args_cli.maze_inspection_layout:
        active_home_external_deg = np.asarray(
            (0.0, 17.0, -75.0, 0.0, 0.0, 0.0, 0.0),
            dtype=np.float32,
        )
        active_home_rad = torch.as_tensor(
            nbv_external_deg_to_sim_rad(
                ROBOT_MODEL_PROFILE,
                active_home_external_deg,
            ),
            device=sim.device,
            dtype=joint_targets.dtype,
        )
        default_joint_pos[:, external_joint_ids] = active_home_rad
        joint_targets[:, external_joint_ids] = active_home_rad
        print(
            f">>> Active gap arm procedural HOME external_deg={active_home_external_deg.tolist()}",
            flush=True,
        )
    def hold_elbow_rotate_target(targets: torch.Tensor) -> torch.Tensor:
        if elbow_rotate_joint_idx is not None and not args_cli.active_gap_arm:
            targets[:, elbow_rotate_joint_idx] = elbow_rotate_hold_rad
        return targets

    def runtime_external_deg_to_sim_rad(values: np.ndarray) -> np.ndarray:
        if args_cli.active_gap_arm:
            return nbv_external_deg_to_sim_rad(ROBOT_MODEL_PROFILE, values)
        return external_deg_to_sim_rad(ROBOT_MODEL_PROFILE, values)

    def runtime_sim_rad_to_external_deg(values: np.ndarray) -> np.ndarray:
        if args_cli.active_gap_arm:
            return nbv_sim_rad_to_external_deg(ROBOT_MODEL_PROFILE, values)
        return sim_rad_to_external_deg(ROBOT_MODEL_PROFILE, values)

    def calibrated_leader_action_sim_rad(values: np.ndarray) -> np.ndarray:
        """Keep the physical leader's calibrated simulation-coordinate mapping."""
        return leader_action_deg_to_sim_arm_rad(values)

    joint_targets = hold_elbow_rotate_target(joint_targets)
    latest_gr00t_action_deg = None
    latest_leader_action_deg = None
    latest_smolvla_action_deg = None
    # Raw leader gripper (servo ID 7) received over its own ZMQ channel.
    leader_gripper_sub = None
    if args_cli.leader_auto:
        leader_gripper_sub = zmq.Context().socket(zmq.SUB)
        leader_gripper_sub.setsockopt(zmq.RCVHWM, 1)
        leader_gripper_sub.setsockopt(zmq.CONFLATE, 1)
        leader_gripper_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        leader_gripper_sub.bind(f"tcp://*:{args_cli.gripper_port}")
    latest_leader_gripper_raw = None
    def receive_leader_gripper_raw():
        """Return the newest raw uint32 gripper position, or None."""
        if leader_gripper_sub is None:
            return None
        try:
            frame = leader_gripper_sub.recv(flags=zmq.NOBLOCK)
        except zmq.Again:
            return None
        if len(frame) != 12:
            return None
        seq, raw_pos = struct.unpack("<II", frame[:8])
        (crc,) = struct.unpack("<I", frame[8:])
        if zlib.crc32(frame[:8]) & 0xFFFFFFFF != crc:
            return None
        return raw_pos
    arm_action_enabled = bool(args_cli.gr00t_apply_actions)
    nonfinite_stop_reported = False
    camera_qa_out_dir = Path(args_cli.camera_qa_out_dir) if args_cli.camera_qa else None
    if camera_qa_out_dir is not None:
        camera_qa_out_dir.mkdir(parents=True, exist_ok=True)
        print(f">>> [camera_qa] output: {camera_qa_out_dir}")
    camera_qa_start_x = float(robot.data.root_pos_w[0, 0].detach().cpu().item()) if args_cli.camera_qa else 0.0
    camera_qa_travel_reached = not args_cli.camera_qa

    # 자율주행 미션에서는 WASD 키보드 장치를 아예 생성하지 않는다.
    # 수퍼바이저의 /active_slam/base_cmd_vel 만 바퀴 명령의 단일 소스가 된다.
    keyboard = (
        None
        if args_cli.active_gap_arm
        else WasdKeyboard(
            Se2KeyboardCfg(
                v_x_sensitivity=0.8,
                v_y_sensitivity=0.6,
                omega_z_sensitivity=1.2,
                sim_device=sim.device,
            )
        )
    )
    if keyboard is not None:
        keyboard.add_callback("K", keyboard.reset)
    gr00t_bridge = Gr00tZmqBridge(args_cli.gr00t_obs_port, args_cli.gr00t_action_port) if args_cli.enable_gr00t else None
    leader_action_sub = (
        ActionSubscriber(ZmqEndpointConfig(action_port=args_cli.gr00t_action_port)) if args_cli.leader_auto else None
    )
    smolvla_bridge = (
        SmolVLAZmqBridge(args_cli.smolvla_obs_port, args_cli.smolvla_action_port)
        if args_cli.enable_smolvla_policy
        else None
    )
    smolvla_obs_dt = 1.0 / max(float(args_cli.smolvla_fps), 1.0e-6)
    smolvla_state = {
        "active": False,
        "toggle_requested": False,
        "next_obs_time": None,
        "proc_exit_reported": False,
    }
    hazard_smolvla_bridge = None
    hazard_decision_gate = None
    hazard_task_for_side_runtime = None
    hazard_wrist_peek_pose_valid = None
    hazard_slew_arm_target = None
    hazard_consume_correlated_action = None
    hazard_smolvla_obs_dt = 0.0
    hazard_smolvla_state = {
        "context": None,
        "event_id": None,
        "target_side": None,
        "next_obs_time": None,
        "requested_arm_target_deg": None,
        "applied_arm_target_deg": None,
        "rejected_outputs": 0,
    }
    if args_cli.enable_hazard_smolvla_policy:
        from soarm_nbv.hazard_episode_collector import (
            wrist_peek_pose_valid as hazard_wrist_peek_pose_valid,
        )
        from soarm_nbv.hazard_vla_contract import (
            task_for_side as hazard_task_for_side_runtime,
        )
        from soarm_nbv.hazard_vla_runtime import (
            HazardDecisionGate,
            HazardSmolVLAZmqBridge,
            consume_correlated_hazard_action as hazard_consume_correlated_action,
            slew_hazard_arm_target as hazard_slew_arm_target,
            wrist_clears_entry_plane as hazard_wrist_clears_entry_plane,
        )

        hazard_smolvla_bridge = HazardSmolVLAZmqBridge(
            args_cli.hazard_smolvla_obs_port,
            args_cli.hazard_smolvla_action_port,
        )
        hazard_decision_gate = HazardDecisionGate(
            required_peek_frames=5,
            required_stable_samples=3,
            threshold=0.5,
        )
        hazard_smolvla_obs_dt = 1.0 / float(args_cli.hazard_smolvla_fps)
    # 햅틱 피드백은 실물 리더 그리퍼 모터에 쓰기를 발생시키므로 기본값은 OFF.
    haptic_pub = None
    if args_cli.leader_auto and args_cli.enable_haptic_feedback:
        _haptic_ctx = zmq.Context()
        haptic_pub = _haptic_ctx.socket(zmq.PUB)
        haptic_pub.setsockopt(zmq.SNDHWM, 1)
        haptic_pub.setsockopt(zmq.CONFLATE, 1)
        haptic_pub.bind(f"tcp://*:{args_cli.haptic_port}")
        print(f">>> Haptic feedback PUB: port {args_cli.haptic_port}")
    if args_cli.leader_auto:
        start_leader_subprocess(args_cli)
        flush_initial_robot_pose(sim, scene, robot, joint_targets)
        # --- 리더 신호 대기 중 로봇 홀딩 ---
        # sim.reset()이 timeline을 play로 두므로, 단순 time.sleep으로 기다리면
        # 정책 없이 물리가 돌아 로봇이 쓰러진다. PD position target으로 버틴다.
        _hold_deadline = time.monotonic() + 3.0
        _initial_leader_action_deg = None
        while time.monotonic() < _hold_deadline:
            _la = leader_action_sub.receive()
            if _la is not None:
                _initial_leader_action_deg = clip_leader_action(_la.joint_target_deg)
                break
            robot.set_joint_position_target(joint_targets)
            scene.write_data_to_sim()
            sim.step(render=False)
            scene.update(sim_cfg.dt)
        if _initial_leader_action_deg is not None:
            latest_leader_action_deg = _initial_leader_action_deg.copy()
        if (
            _initial_leader_action_deg is not None
            and not args_cli.leader_apply_only_when_base_paused
        ):
            _init_rad = torch.as_tensor(
                calibrated_leader_action_sim_rad(_initial_leader_action_deg),
                device=sim.device,
                dtype=joint_targets.dtype,
            )
            joint_targets[:, sim_arm_joint_ids] = _init_rad
            _init_pos = default_joint_pos.clone()
            _init_pos[:, sim_arm_joint_ids] = _init_rad
            default_joint_pos[:, sim_arm_joint_ids] = _init_rad
            default_rough_joint_pos = default_joint_pos[:, rough_joint_ids].clone()
            robot.write_joint_state_to_sim(_init_pos, default_joint_vel)
            print(f">>> [leader_auto] initial SO-Arm pose synchronized deg: {_initial_leader_action_deg}")
        elif _initial_leader_action_deg is not None:
            print(
                ">>> [leader_auto] initial SO-Arm pose buffered; application is gated "
                "until /active_slam/base_pause=true",
                flush=True,
            )
        else:
            print(">>> [leader_auto] WARNING: no initial leader action received; using default SO-Arm pose")
    if args_cli.enable_smolvla_policy:
        start_smolvla_subprocess(args_cli)
    if args_cli.enable_hazard_smolvla_policy:
        start_hazard_smolvla_subprocess(args_cli)
    flush_initial_robot_pose(sim, scene, robot, joint_targets)
    # 스폰 후 발이 바닥에 박혀 있으면 root를 "올리기만" 한다.
    # 한 발(FL)만 보고 내리면 반대쪽 발/settle 상태에 따라 root가 땅 밑으로 들어가 폭발적으로 튀어오른다.
    _foot_names, _foot_indices = raise_robot_to_foot_clearance(sim, scene, robot, label="spawn")
    settle_startup_pose(sim, scene, robot, joint_targets, sim_cfg.dt)
    print_robot_pose_summary(robot, _foot_names, _foot_indices, "spawn ready")
    # Keep the timeline paused while ROS graphs, render products, viewports,
    # teleoperation bridges, and callbacks are constructed.  Several of those
    # setup paths call simulation_app.update(); if playback has already begun,
    # those UI updates advance physics without running the locomotion policy.
    if args_cli.slam_ros2:
        from soarm_nbv.ros2_slam_bridge import setup_slam_publishers

        ros2_slam_handles = setup_slam_publishers(
            simulation_app=simulation_app,
            camera_prim_path="/World/envs/env_0/Robot/base/front_camera_sensor",
            wrist_camera_prim_path=(
                None
                if args_cli.no_arm
                else "/World/envs/env_0/Robot/gripper_link/wrist_camera"
            ),
            chassis_prim_path="/World/envs/env_0/Robot/base",
            width=FRONT_CAMERA_WIDTH,
            height=FRONT_CAMERA_HEIGHT,
            wrist_width=WRIST_CAMERA_WIDTH,
            wrist_height=WRIST_CAMERA_HEIGHT,
        )
        slam_topics = (
            "/camera/color/image_raw, /camera/depth/image_rect_raw, "
            "/camera/camera_info, /clock, /odom, /tf"
        )
        if not args_cli.no_arm:
            slam_topics += ", /wrist_camera/*"
        print(f">>> ROS 2 SLAM publishers: ENABLED ({slam_topics})", flush=True)
    utlidar_state = (
        _setup_utlidar_l1(
            args_cli.environment_usd,
            use_physx=args_cli.lidar_physx,
            use_rtx=args_cli.lidar_rtx,
        )
        if args_cli.lidar_slam
        else None
    )
    if args_cli.active_gap_arm:
        import omni.physx
        from soarm_nbv.isaac_active_arm_trajectory_server import (
            IsaacActiveArmTrajectoryServer,
            arm_link_positions_base_from_external_deg,
        )

        def validate_arm_scene_clearance(goal) -> bool:
            base_position = robot.data.body_pos_w[0, base_link_idx]
            base_quaternion = robot.data.body_quat_w[0, base_link_idx]
            scene_query = omni.physx.get_physx_scene_query_interface()
            robot_path_prefix = "/World/envs/env_0/Robot"
            configurations = [
                np.asarray(goal.validated_start_external_deg, dtype=np.float64)
            ]
            previous_configuration = configurations[0]
            for trajectory_point in goal.points:
                current_configuration = np.asarray(
                    trajectory_point.position_external_deg,
                    dtype=np.float64,
                )
                configurations.extend(
                    previous_configuration
                    + ratio * (current_configuration - previous_configuration)
                    for ratio in np.linspace(0.0, 1.0, 9, endpoint=True)[1:]
                )
                previous_configuration = current_configuration
            for configuration in configurations:
                link_positions_base = arm_link_positions_base_from_external_deg(
                    configuration
                )
                samples_base = []
                for start, end in zip(
                    link_positions_base,
                    link_positions_base[1:],
                ):
                    count = max(
                        2,
                        math.ceil(float(np.linalg.norm(end - start)) / 0.02)
                        + 1,
                    )
                    samples_base.extend(
                        start + ratio * (end - start)
                        for ratio in np.linspace(0.0, 1.0, count)
                    )
                samples_tensor = torch.as_tensor(
                    np.asarray(samples_base),
                    dtype=base_position.dtype,
                    device=base_position.device,
                )
                world_samples = base_position.unsqueeze(0) + math_utils.quat_apply(
                    base_quaternion.unsqueeze(0).expand(samples_tensor.shape[0], -1),
                    samples_tensor,
                )
                for sample in world_samples.detach().cpu().numpy():
                    environment_hit = False
                    environment_hit_path = None

                    def report_hit(hit):
                        nonlocal environment_hit, environment_hit_path
                        hit_path = str(hit.rigid_body)
                        if not hit_path.startswith(robot_path_prefix):
                            environment_hit = True
                            if environment_hit_path is None:
                                environment_hit_path = hit_path
                        return True

                    scene_query.overlap_sphere(
                        0.02,
                        carb.Float3(
                            float(sample[0]),
                            float(sample[1]),
                            float(sample[2]),
                        ),
                        report_hit,
                        False,
                    )
                    if environment_hit:
                        print(
                            ">>> ARM_SCENE_CLEARANCE_REJECT "
                            f"obstacle={environment_hit_path} "
                            f"sample_world_m={[round(float(value), 4) for value in sample]} "
                            f"configuration_external_deg="
                            f"{[round(float(value), 3) for value in configuration]}",
                            flush=True,
                        )
                        return False
            return True

        active_arm_server = IsaacActiveArmTrajectoryServer(
            contract_path=os.environ["ACTIVE_SLAM_TRANSACTION_CONTRACT"],
            secret_path=os.environ["ACTIVE_SLAM_HMAC_SECRET"],
            scene_clearance_validator=validate_arm_scene_clearance,
            enforce_runtime_velocity_limits=(
                os.environ.get("ACTIVE_ARM_ENFORCE_RUNTIME_VELOCITY_LIMITS", "1") == "1"
            ),
        )
        print(
            ">>> Active gap arm authority: ENABLED "
            "(sole ApplyArmTrajectory server; legacy arm sources disabled)",
            flush=True,
        )
    if args_cli.show_camera_pose_markers:
        create_camera_pose_markers()
    maze_main_viewport = None
    if args_cli.maze_nav2_inspection:
        import faulthandler
        faulthandler.dump_traceback_later(45.,repeat=False)
        print('MAZE_UI_SETUP begin',flush=True)
    if args_cli.maze_visual_appearance:
        from omni.kit.viewport.utility import get_active_viewport_window
        maze_main_viewport = get_active_viewport_window()
    camera_viewports = (
        setup_camera_viewports(show_free_camera=args_cli.show_free_camera_viewport,
                               camera_sensors=(wrist_camera_sensor,front_camera_sensor))
        if args_cli.enable_cameras and args_cli.show_camera_viewport
        else []
    )

    maze_display_ui = None
    if args_cli.maze_nav2 and camera_viewports:
        from resource_audit import camera_bindings
        print('MAZE_CAMERA_BINDINGS',json.dumps(camera_bindings(camera_viewports,
              (wrist_camera_sensor,front_camera_sensor))),flush=True)
    if maze_main_viewport is not None:
        from physical_ui import PhysicalUI
        import omni.usd
        maze_lighting = None
        if args_cli.maze_nav2 or args_cli.maze_locomotion_only:
            from scene_lighting import SceneLighting
            print('MAZE_UI_SETUP lights',flush=True)
            maze_lighting = SceneLighting(omni.usd.get_context().get_stage(), [
                '/World/envs/env_0/Robot/base/front_camera_sensor',
                '/World/envs/env_0/Robot/gripper_link/wrist_camera'],
                settings=carb.settings.get_settings())
        maze_display_ui = PhysicalUI(omni.usd.get_context().get_stage(), maze_main_viewport, camera_viewports,
                                     (front_camera_sensor, wrist_camera_sensor), lighting=maze_lighting)
        if args_cli.maze_nav2 or args_cli.maze_locomotion_only:
            # Set only the captured MAIN viewport once. The newly created
            # wrist viewport can otherwise become "active" during setup.
            maze_display_ui.reset_top_view()
    if args_cli.maze_nav2_inspection:
        faulthandler.cancel_dump_traceback_later()
        print('MAZE_UI_SETUP complete',flush=True)

    print("\n>>> Go2 SLAM Simulation Started! <<<" if args_cli.no_arm else "\n>>> Go2 + SO-Arm Simulation Started! <<<")
    import omni.kit.app
    mcp_enabled = omni.kit.app.get_app().get_extension_manager().is_extension_enabled('isaac.sim.mcp_extension')
    print(f">>> MCP Extension: {'ENABLED' if mcp_enabled else 'DISABLED'}")
    map_label = str(args_cli.environment_usd) if args_cli.environment_usd else "기본 평면 (GroundPlane)"
    if args_cli.no_arm:
        camera_label = "ENABLED (Front RGB-D only)" if args_cli.slam_rgbd else "ENABLED (Front RGB only)"
    else:
        camera_label = "ENABLED (Wrist + Front RGB-D)" if args_cli.slam_rgbd else "ENABLED (Wrist + Front RGB)"
    print(f">>> Map: {map_label}")
    print(f">>> Camera Sensors: {camera_label if args_cli.enable_cameras else 'DISABLED'}")
    print(
        f">>> Camera Viewports: "
        f"{'ENABLED (Main + Wrist + Front RGB-D' + (' + Free Camera)' if args_cli.show_free_camera_viewport else ')') if args_cli.enable_cameras and args_cli.show_camera_viewport else 'DISABLED'}"
    )
    print(f">>> SO-Arm Dynamics & Collision: {'ABSENT' if args_cli.no_arm else so_arm_dynamics_label}")
    elbow_rotate_status = (
        "active-NBV-controlled"
        if args_cli.active_gap_arm and elbow_rotate_joint_idx is not None
        else "leader-controlled"
        if args_cli.leader_auto and elbow_rotate_joint_idx is not None
        else f"hold={ELBOW_ROTATE_HOLD_DEG:.2f} deg"
        if elbow_rotate_joint_idx is not None
        else "n/a"
    )
    robot_model_label = "go2_no_arm" if args_cli.no_arm else ROBOT_MODEL_PROFILE.key
    print(f">>> Robot model: {robot_model_label}; elbow_rotate={elbow_rotate_status}", flush=True)
    print(">>> Mustard bottle: REMOVED")
    print(">>> Robot instanceable: FALSE (GUI 조작 가능)\n")
    if args_cli.camera_qa:
        manual_label = "DISABLED (camera QA autonomous velocity)"
    elif args_cli.demo_pan:
        manual_label = "DISABLED (demo pan sweep)"
    elif args_cli.demo_fold_walk:
        manual_label = "DISABLED (fold/unfold walking QA)"
    elif args_cli.active_gap_arm:
        manual_label = "DISABLED (autonomous supervisor /active_slam/base_cmd_vel)"
    elif args_cli.scripted_route is not None:
        manual_label = "DISABLED (deterministic scripted route)"
    else:
        manual_label = "ENABLED"
    print(f">>> Manual Control: {manual_label}")
    print(">>> RL Locomotion Policy: ENABLED")
    print(f">>> GR00T Bridge: {'ENABLED' if args_cli.enable_gr00t else 'DISABLED'}")
    print(f">>> Leader Teleoperation: {'ENABLED' if args_cli.leader_auto else 'DISABLED'}")
    print(f">>> Leader Action Apply: {'ENABLED' if args_cli.leader_auto and args_cli.gr00t_apply_actions else 'DISABLED'}")
    if args_cli.leader_apply_only_when_base_paused:
        print(">>> Leader Apply Gate: BASE-PAUSE ONLY (walking keeps procedural HOME)")
    print(f">>> GR00T Action Apply: {'ENABLED' if args_cli.gr00t_apply_actions else 'DISABLED'}")
    print(f">>> SmolVLA Policy: {'ENABLED (RIGHT toggles)' if args_cli.enable_smolvla_policy else 'DISABLED'}")
    print(
        ">>> Hazard SmolVLA Policy: "
        + (
            "ENABLED (7 motor targets + correlated action[7] decision)"
            if args_cli.enable_hazard_smolvla_policy
            else "DISABLED"
        )
    )
    print(f">>> Policy: {args_cli.go2_policy_path} ({args_cli.go2_policy_obs_mode}, obs_dim={GO2_POLICY_OBS_DIMS[args_cli.go2_policy_obs_mode]})")
    print(f">>> Leg joints: {', '.join(leg_joint_names)}")
    print(f">>> Policy action joints: {', '.join(policy_action_joint_names)}")
    print(f">>> Observation joints ({robot.num_joints}): {', '.join(robot.joint_names)}")
    key_help = (
        ">>> Autonomous mode: base velocity comes only from the supervisor"
        if args_cli.active_gap_arm
        else ">>> Keys: hold W/S forward/back, hold A/D strafe, hold Q/E yaw, K stop"
    )
    if args_cli.enable_smolvla_policy:
        key_help += ", RIGHT toggle SmolVLA"
    print(key_help + "\n")

    # 데이터 수집 (--collect)
    collector = None
    collect_frame_count = 0
    collect_ep_count = 0
    collect_dt = 1.0 / max(float(args_cli.collect_fps), 1.0e-6)
    collect_next_time = None
    collect_episode_start_time = None
    collect_manual = bool(args_cli.collect_manual_right_arrow)
    collect_state = {
        "active": not collect_manual,
        "toggle_requested": False,
        "needs_reset": False,
    }
    if args_cli.collect:
        import sys as _sys
        if "/home/iy/Isaac/Robotics/robot_models" not in _sys.path:
            _sys.path.insert(0, "/home/iy/Isaac/Robotics/robot_models")
        from soarm_nbv.episode_collector import EpisodeCollector
        collector = EpisodeCollector(args_cli.collect_out_dir, task=args_cli.task, fps=args_cli.collect_fps)
        print(
            f">>> [collect] task='{args_cli.task}' max_episodes={args_cli.max_episodes} "
            f"episode_len={args_cli.episode_len} fps={args_cli.collect_fps}"
        )

    # Binary-alley human/scripted-NBV teacher collection. This is deliberately
    # separate from the legacy six-axis drawer collector above.
    hazard_collector = None
    hazard_label_keyboard = None
    hazard_collect_dt = 1.0 / max(float(args_cli.hazard_collect_fps), 1.0e-6)
    hazard_collect_state = {
        "context": None,
        "event_id": None,
        "completed_event_ids": set(),
        "label_request": None,
        "terminal_signal": 0,
        "terminal_start_time": None,
        "episode_start_time": None,
        "next_sample_time": None,
        "peek_valid_samples": 0,
        "peek_validated_samples": 0,
        "latest_peek_valid": False,
        "pending_teacher_labels": {},
    }
    if args_cli.collect_hazard_vla or args_cli.collect_hazard_nbv_teacher:
        from soarm_nbv.hazard_episode_collector import (
            BODY_BRANCH_ENTRY_OVERSHOOT_LIMIT_M,
            HazardEpisodeCollector,
            PEEK_MINIMUM_CONSECUTIVE_FRAMES,
            body_branch_entry_overshoot_m,
            body_crossed_branch_entry_during_inspection,
            wrist_optical_forward_base,
            wrist_peek_pose_valid,
        )
        from soarm_nbv.hazard_vla_contract import (
            DECISION_HAZARD as HAZARD_TEACHER_DECISION_HAZARD,
            DECISION_SAFE as HAZARD_TEACHER_DECISION_SAFE,
            task_for_side as hazard_task_for_side,
        )

        hazard_collector = HazardEpisodeCollector(
            args_cli.hazard_collect_out_dir,
            fps=args_cli.hazard_collect_fps,
            teacher_type=(
                "human"
                if args_cli.collect_hazard_vla
                else "scripted_wrist_rgb_nbv"
            ),
        )
        if args_cli.collect_hazard_vla:
            hazard_label_keyboard = Se2Keyboard(
                Se2KeyboardCfg(
                    v_x_sensitivity=0.0,
                    v_y_sensitivity=0.0,
                    omega_z_sensitivity=0.0,
                    sim_device=sim.device,
                )
            )

        def _request_hazard_label(value):
            hazard_collect_state["label_request"] = value

        if args_cli.collect_hazard_vla:
            hazard_label_keyboard.add_callback(
                "H",
                lambda: _request_hazard_label(HAZARD_TEACHER_DECISION_HAZARD),
            )
            hazard_label_keyboard.add_callback(
                "S",
                lambda: _request_hazard_label(HAZARD_TEACHER_DECISION_SAFE),
            )
            hazard_label_keyboard.add_callback("X", lambda: _request_hazard_label("discard"))
            hazard_label_keyboard.add_callback("R", lambda: _request_hazard_label("start"))
            print(
                ">>> [hazard_collect] GUI human teacher enabled: "
                "R=start, H=HAZARD, S=SAFE, X=discard",
                flush=True,
            )
        else:
            print(
                ">>> [hazard_collect] GUI scripted NBV teacher enabled: "
                "episodes start and receive terminal labels automatically",
                flush=True,
            )
        print(
            f">>> [hazard_collect] output={hazard_collector.session_dir} "
            f"fps={args_cli.hazard_collect_fps} state=7 action=8",
            flush=True,
        )

    def _request_collect_toggle():
        collect_state["toggle_requested"] = True

    def _request_smolvla_toggle():
        smolvla_state["toggle_requested"] = True

    def _finish_collect_episode(reason: str) -> bool:
        nonlocal collect_frame_count, collect_ep_count, collect_next_time, collect_episode_start_time
        if collector is None:
            return False
        if collect_frame_count <= 0:
            collect_state["active"] = False
            collect_state["needs_reset"] = collect_manual
            collect_next_time = None
            collect_episode_start_time = None
            print(f">>> [collect] episode cancelled ({reason}; no frames)", flush=True)
            return False
        collector.flush()
        collect_frame_count = 0
        collect_ep_count += 1
        collect_next_time = None
        collect_episode_start_time = None
        if collect_manual:
            collect_state["active"] = False
            collect_state["needs_reset"] = collect_ep_count < args_cli.max_episodes
        print(f">>> [collect] episode {collect_ep_count}/{args_cli.max_episodes} done ({reason})", flush=True)
        if collect_manual and collect_ep_count < args_cli.max_episodes:
            print(">>> [collect] RIGHT again: reset simulation; RIGHT after reset starts next episode", flush=True)
        if (
            args_cli.collect_pause_between_episodes
            and not collect_manual
            and collect_ep_count < args_cli.max_episodes
        ):
            print(
                ">>> [collect] PAUSED for next episode reset: drawer/robot을 시작 상태로 맞춘 뒤 Play를 눌러 계속 수집",
                flush=True,
            )
            sim.pause()
        if collect_ep_count >= args_cli.max_episodes:
            print(">>> [collect] max_episodes 도달, 수집 종료")
            return True
        return False

    def _reset_collect_scene_for_next_episode():
        nonlocal joint_targets, policy_step, latest_leader_action_deg, latest_gr00t_action_deg
        nonlocal arm_action_enabled, nonfinite_stop_reported, last_action
        if haptic_pub is not None:
            try:
                haptic_pub.send(struct.pack('f', 0.0), flags=zmq.NOBLOCK)
            except zmq.Again:
                pass
        print(">>> [collect] RESET simulation for next episode", flush=True)
        latest_leader_action_deg = None
        latest_gr00t_action_deg = None
        last_action = torch.zeros_like(last_action)
        arm_action_enabled = bool(args_cli.gr00t_apply_actions)
        nonfinite_stop_reported = False

        joint_targets = default_joint_pos.clone()
        joint_targets = hold_elbow_rotate_target(joint_targets)
        _robot_root_state = robot.data.default_root_state.clone()
        try:
            _robot_root_state[:, :3] += scene.env_origins
        except Exception:
            pass
        robot.write_root_pose_to_sim(_robot_root_state[:, :7])
        robot.write_root_velocity_to_sim(torch.zeros_like(_robot_root_state[:, 7:]))
        robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
        robot.set_joint_position_target(joint_targets)
        # 서랍 닫기
        _drawer = scene["drawer"]
        _drawer.write_joint_state_to_sim(
            _drawer.data.default_joint_pos.clone(),
            _drawer.data.default_joint_vel.clone(),
        )
        # 큐브 시작 위치로
        for _cube_name in ("orange_cube", "green_cube"):
            _cube = scene[_cube_name]
            _cube.write_root_pose_to_sim(_cube.data.default_root_state[:, :7].clone())
            _cube.write_root_velocity_to_sim(torch.zeros((1, 6), device=sim.device))
        scene.write_data_to_sim()
        sim.forward()
        scene.update(0.0)
        raise_robot_to_foot_clearance(sim, scene, robot, label="collect-reset")
        flush_initial_robot_pose(sim, scene, robot, joint_targets)
        settle_startup_pose(sim, scene, robot, joint_targets, sim_cfg.dt)
        policy_step = 0
        print(">>> [collect] RESET complete, ready for next episode", flush=True)

    if args_cli.collect and collect_manual:
        keyboard.add_callback("RIGHT", _request_collect_toggle)
        print(
            f">>> [collect] RIGHT ARROW manual sequence: RIGHT=start, RIGHT=save, RIGHT=reset, RIGHT=start next; "
            f"auto-save at {args_cli.episode_len} frames (~{args_cli.episode_len / max(args_cli.collect_fps, 1.0e-6):.1f}s)",
            flush=True,
        )
    if args_cli.enable_smolvla_policy:
        keyboard.add_callback("RIGHT", _request_smolvla_toggle)
        print(">>> [smolvla] RIGHT ARROW toggles 20k SmolVLA policy control.", flush=True)

    loop_count = 0
    external_supervisor_command_available = False
    external_supervisor_motion_started = False
    _prev_command_active = False
    scripted_route_waypoints = (
        np.asarray(args_cli.scripted_route, dtype=np.float32).reshape(-1, 2)
        if args_cli.scripted_route is not None
        else None
    )
    maze_inspector = None
    maze_trial_stop = None
    if args_cli.maze_inspection_layout:
        if args_cli.kinematic_scripted_route or args_cli.no_arm:
            raise ValueError('Maze inspection requires a dynamic robot with arm')
        if ROBOT_MODEL_PROFILE.key != 'so101_7motor_reversed':
            raise ValueError('Maze arm trajectory requires the reviewed reversed seven-motor model')
        if any((args_cli.active_gap_arm,args_cli.leader_auto,args_cli.demo_pan,
                args_cli.demo_fold_walk,args_cli.camera_qa,args_cli.camera_aim_qa)):
            raise ValueError('Maze controller cannot share body/arm control with another mode')
        sys.path.insert(0, '/home/iy/Isaac/maze_visual_demo')
        from sensor_navigation import Inspector, PhysxScan
        from observed_navigation import ObservedPlanner, red_cube_cells, camera_pose_matrix
        from inspection_control import ground_support, depth_world_points
        maze_scan = PhysxScan()
        maze_layout=json.loads(args_cli.maze_inspection_layout.read_text())
        if args_cli.maze_observed_navigation and not (args_cli.enable_cameras and args_cli.slam_rgbd):
            raise ValueError('Observed maze navigation requires RGB-D cameras')
        maze_home={name: float(default_joint_pos[0,idx].item()) for name,idx in zip(sim_arm_joint_names,sim_arm_joint_ids)}
        if args_cli.maze_locomotion_only or args_cli.maze_nav2:
            if args_cli.maze_observed_navigation:
                raise ValueError('Locomotion diagnostic cannot share the visual decision controller')
            if args_cli.maze_nav2:
                from nav2_client import Nav2Locomotion
                if os.environ.get('MAZE_DYNAMIC_MISSION')=='1':
                    from mission_navigation import MissionLocomotion
                    maze_controller_type=MissionLocomotion
                else:
                    maze_controller_type=Nav2Locomotion
                maze_inspector=maze_controller_type(maze_layout,maze_home,
                    arm_convert=(lambda values: nbv_external_deg_to_sim_rad(ROBOT_MODEL_PROFILE,values))
                    if args_cli.maze_nav2_inspection else None,
                    observation_driven=args_cli.maze_observation_navigation)
                from trial_stop import TrialStop
                import signal as process_signals
                maze_trial_stop=TrialStop()
                process_signals.signal(process_signals.SIGINT,maze_trial_stop)
                process_signals.signal(process_signals.SIGTERM,maze_trial_stop)
            else:
                from corridor_locomotion import CorridorLocomotion
                maze_inspector=CorridorLocomotion(maze_layout,maze_home)
            if maze_display_ui is not None:
                maze_display_ui.fov_enabled=True
                maze_display_ui.set_enabled(os.environ.get('MAZE_DARKNESS') == '1')
        else:
            maze_inspector = Inspector(maze_layout,maze_home,
                arm_convert=lambda values: nbv_external_deg_to_sim_rad(ROBOT_MODEL_PROFILE, values),
                navigation=ObservedPlanner(maze_layout) if args_cli.maze_observed_navigation else None)
    scripted_route_index = 0
    scripted_route_complete_reported = False
    scripted_route_draw_interface = None
    kinematic_route_root_z = float(robot.data.root_pos_w[0, 2].item())
    kinematic_route_x = float(robot.data.root_pos_w[0, 0].item())
    kinematic_route_y = float(robot.data.root_pos_w[0, 1].item())
    _initial_route_quat = robot.data.root_quat_w[0]
    _rw, _rx, _ry, _rz = (float(value.item()) for value in _initial_route_quat)
    kinematic_route_yaw = math.atan2(
        2.0 * (_rw * _rz + _rx * _ry),
        1.0 - 2.0 * (_ry * _ry + _rz * _rz),
    )
    kinematic_route_pose_target = None
    stationary_base_pose_target = None
    if scripted_route_waypoints is not None:
        print(
            f">>> Scripted route: {scripted_route_waypoints.tolist()} "
            f"(speed={args_cli.scripted_route_speed:.2f} m/s, "
            f"tolerance={args_cli.scripted_route_tolerance:.2f} m)",
            flush=True,
        )
        if args_cli.show_scripted_route:
            if args_cli.headless:
                print(">>> Scripted route overlay skipped: GUI viewport is not available in headless mode.", flush=True)
            else:
                try:
                    from isaacsim.util.debug_draw import _debug_draw

                    scripted_route_draw_interface = _debug_draw.acquire_debug_draw_interface()
                    route_z = 0.08
                    route_points = [
                        (float(x), float(y), route_z)
                        for x, y in scripted_route_waypoints
                    ]
                    segment_starts = route_points[:-1]
                    segment_ends = route_points[1:]
                    route_color = (0.0, 0.85, 1.0, 1.0)
                    scripted_route_draw_interface.draw_lines(
                        segment_starts,
                        segment_ends,
                        [route_color] * len(segment_starts),
                        [6.0] * len(segment_starts),
                    )

                    # Two short strokes at each segment end make the ordered
                    # waypoint direction legible from the top-down viewport.
                    arrow_starts = []
                    arrow_ends = []
                    for segment_start, segment_end in zip(segment_starts, segment_ends, strict=True):
                        dx = segment_end[0] - segment_start[0]
                        dy = segment_end[1] - segment_start[1]
                        length = math.hypot(dx, dy)
                        if length <= 1.0e-6:
                            continue
                        ux, uy = dx / length, dy / length
                        arrow_length = min(0.22, 0.30 * length)
                        arrow_width = 0.55 * arrow_length
                        back_x = segment_end[0] - arrow_length * ux
                        back_y = segment_end[1] - arrow_length * uy
                        for side in (-1.0, 1.0):
                            arrow_starts.append(segment_end)
                            arrow_ends.append(
                                (
                                    back_x + side * arrow_width * -uy,
                                    back_y + side * arrow_width * ux,
                                    route_z,
                                )
                            )
                    if arrow_starts:
                        scripted_route_draw_interface.draw_lines(
                            arrow_starts,
                            arrow_ends,
                            [route_color] * len(arrow_starts),
                            [5.0] * len(arrow_starts),
                        )

                    waypoint_colors = [(1.0, 0.82, 0.0, 1.0)] * len(route_points)
                    waypoint_sizes = [14.0] * len(route_points)
                    # A loop ends at its start, so magenta denotes the shared
                    # START/FINISH pad and yellow denotes intermediate points.
                    waypoint_colors[0] = (1.0, 0.0, 0.8, 1.0)
                    waypoint_sizes[0] = 28.0
                    if np.allclose(scripted_route_waypoints[0], scripted_route_waypoints[-1]):
                        waypoint_colors[-1] = (1.0, 0.0, 0.8, 1.0)
                        waypoint_sizes[-1] = 28.0
                    else:
                        waypoint_colors[-1] = (1.0, 0.15, 0.0, 1.0)
                        waypoint_sizes[-1] = 24.0
                    scripted_route_draw_interface.draw_points(
                        route_points,
                        waypoint_colors,
                        waypoint_sizes,
                    )
                    print(
                        ">>> Scripted route overlay: cyan arrows=travel direction, "
                        "yellow dots=waypoints, magenta=START/FINISH.",
                        flush=True,
                    )
                except Exception as exc:
                    raise RuntimeError(f"Failed to draw scripted route overlay: {exc}") from exc
    next_slam_rgbd_validation_step = (
        5 if args_cli.slam_ros2 else 0 if args_cli.slam_rgbd else None
    )
    previous_root_linear_velocity_w = robot.data.root_lin_vel_w.clone()
    locomotion_failure_reason = None
    locomotion_failure_detector = None
    if args_cli.abort_on_locomotion_failure:
        locomotion_failure_detector = LocomotionFailureDetector(
            robot.data.body_pos_w[0, base_link_idx].detach().cpu().numpy(),
            robot.data.body_quat_w[0, base_link_idx].detach().cpu().numpy(),
            robot.data.body_pos_w[0, _foot_indices].detach().cpu().numpy(),
            LocomotionFailureConfig(
                grace_samples=args_cli.locomotion_failure_grace_steps,
                sustained_samples=args_cli.locomotion_failure_sustain_steps,
            ),
        )
        print(
            f">>> Locomotion failure detector: ENABLED "
            f"(grace={args_cli.locomotion_failure_grace_steps}, "
            f"sustain={args_cli.locomotion_failure_sustain_steps})",
            flush=True,
        )

    # GUI 모드에서 뷰포트 카메라 전환. 뷰포트 창 초기화가 늦을 수 있어 재시도.
    viewport_camera_target = None
    if not args_cli.headless:
        if args_cli.viewport_camera:
            viewport_camera_target = args_cli.viewport_camera
        elif args_cli.viewport_topdown:
            viewport_camera_target = "/World/environment/TopDownCamera"
    viewport_attempts = 0
    viewport_focus_done = False

    def _try_set_viewport_camera(path: str) -> bool:
        try:
            import omni.kit.viewport.utility as viewport_utility

            window = viewport_utility.get_active_viewport_window()
            if window is None:
                return False
            api = window.viewport_api
            if api.stage is None or not api.stage.GetPrimAtPath(path).IsValid():
                return False
            api.set_active_camera(path)
            return True
        except Exception:
            return False

    follow_camera_reported = False

    def _update_follow_camera() -> None:
        nonlocal follow_camera_reported

        from isaacsim.core.utils.viewports import set_camera_view
        from omni.kit.viewport.utility import get_active_viewport_window

        viewport_window = get_active_viewport_window()
        if viewport_window is None or viewport_window.viewport_api is None:
            raise RuntimeError("active viewport is not ready")
        viewport_api = viewport_window.viewport_api

        # Update the camera that is actually bound to the visible viewport.
        # Assuming /OmniverseKit_Persp here made the pose change invisible when
        # Kit restored another camera from its previous UI state.
        camera_path = str(viewport_api.camera_path)
        if not viewport_api.stage.GetPrimAtPath(camera_path).IsValid():
            camera_path = "/OmniverseKit_Persp"
            viewport_api.set_active_camera(camera_path)

        base_position = robot.data.body_pos_w[0, base_link_idx].detach().cpu().numpy()
        base_quaternion = robot.data.body_quat_w[0, base_link_idx].detach().cpu().numpy()
        w, x, y, z = (float(value) for value in base_quaternion)
        yaw = math.atan2(
            2.0 * (w * z + x * y),
            1.0 - 2.0 * (y * y + z * z),
        )
        forward_x, forward_y = math.cos(yaw), math.sin(yaw)
        eye = [
            float(base_position[0]) - 1.0 * forward_x,
            float(base_position[1]) - 1.0 * forward_y,
            float(base_position[2]) + 0.8,
        ]
        target = [
            float(base_position[0]) + 0.3 * forward_x,
            float(base_position[1]) + 0.3 * forward_y,
            float(base_position[2]) + 0.05,
        ]
        set_camera_view(
            eye=eye,
            target=target,
            camera_prim_path=camera_path,
            viewport_api=viewport_api,
        )
        if not follow_camera_reported:
            print(
                f">>> Viewport follow active: camera={camera_path}, target=Go2 base_link",
                flush=True,
            )
            follow_camera_reported = True

    # All setup that may pump the Kit application is complete.  Start physics
    # immediately before the controlled loop so the first physics step is also
    # the first learned-policy step.
    sim.play()
    print(">>> Simulation: PLAYING (setup complete; policy loop owns first physics step)", flush=True)
    rtf_wall_start_s = time.monotonic()
    rtf_sim_start_s = float(sim.current_time)

    while simulation_app.is_running():
        if maze_trial_stop is not None and maze_trial_stop.requested:
            maze_inspector.termination_reason=f'signal:{maze_trial_stop.signal_number}'
            print('NAV2_GRACEFUL_STOP',maze_inspector.termination_reason,flush=True)
            break
        if (
            args_cli.viewport_focus_robot_once
            and not args_cli.headless
            and not viewport_focus_done
            and loop_count >= 20
        ):
            try:
                _update_follow_camera()
                viewport_focus_done = True
                print(">>> Viewport focused on Go2 once; manual camera control is now free.", flush=True)
            except Exception as exc:
                if loop_count % 60 == 20:
                    print(f">>> One-shot camera focus retry: {exc}", flush=True)
        if args_cli.viewport_follow_robot and not args_cli.headless and loop_count % 20 == 0:
            try:
                _update_follow_camera()
            except Exception as exc:
                if loop_count == 0:
                    print(f">>> Follow camera initialization warning: {exc}", flush=True)
        if viewport_camera_target is not None and loop_count % 60 == 0:
            viewport_attempts += 1
            if _try_set_viewport_camera(viewport_camera_target):
                print(f">>> Viewport camera switched: {viewport_camera_target}", flush=True)
                viewport_camera_target = None
            elif viewport_attempts >= 50:
                print(">>> Viewport camera switch timed out; keeping default view.", flush=True)
                viewport_camera_target = None
        if args_cli.state_debug_every > 0 and loop_count % args_cli.state_debug_every == 0:
            _kb_dbg = keyboard.advance() if keyboard is not None else torch.zeros(3, device=sim.device)
            print(
                f"[dbg] is_playing={sim.is_playing()} loop={loop_count} "
                f"kb_vel={_kb_dbg.cpu().numpy().tolist()} norm={_kb_dbg.norm().item():.2f}",
                flush=True,
            )
        loop_count += 1
        if sim.is_playing():
            requested_collection_lap = (
                active_arm_server.take_episode_reset_request()
                if args_cli.binary_tree_repeat_reset_status_file is not None
                and active_arm_server is not None
                else None
            )
            if requested_collection_lap is not None:
                if hazard_collector is None or hazard_collector.active:
                    locomotion_failure_reason = "unsafe_between_lap_reset_request"
                    print(
                        ">>> SAFETY STOP: between-lap reset arrived while the hazard "
                        "collector was unavailable or still recording",
                        flush=True,
                    )
                    break
                try:
                    active_arm_server.prepare_episode_reset()
                    print(
                        f">>> [batch_collect] resetting robot in-place for lap "
                        f"{requested_collection_lap}",
                        flush=True,
                    )

                    # Keep Kit, rendering, sensors, and the GPU context alive.
                    # Only the scene articulation and control state are reset.
                    scene.reset()
                    joint_targets = default_joint_pos.clone()
                    joint_targets = hold_elbow_rotate_target(joint_targets)
                    reset_root_state = robot.data.default_root_state.clone()
                    reset_root_state[:, :3] += scene.env_origins
                    robot.write_root_pose_to_sim(reset_root_state[:, :7])
                    robot.write_root_velocity_to_sim(
                        torch.zeros_like(reset_root_state[:, 7:])
                    )
                    robot.write_joint_state_to_sim(default_joint_pos, default_joint_vel)
                    robot.set_joint_position_target(joint_targets)
                    scene.write_data_to_sim()
                    sim.forward()
                    scene.update(0.0)
                    _foot_names, _foot_indices = raise_robot_to_foot_clearance(
                        sim,
                        scene,
                        robot,
                        label=f"binary-tree-lap-{requested_collection_lap}-reset",
                    )
                    flush_initial_robot_pose(sim, scene, robot, joint_targets)
                    settle_startup_pose(sim, scene, robot, joint_targets, sim_cfg.dt)

                    # The policy output is produced under torch.inference_mode(),
                    # so ``last_action`` can be an inference tensor.  Such a
                    # tensor cannot be mutated in-place after leaving inference
                    # mode.  Allocate ordinary control tensors for the new lap
                    # instead of calling zero_()/copy_() on stale tensors.
                    last_action = torch.zeros(
                        (1, len(leg_joint_names)),
                        dtype=default_joint_pos.dtype,
                        device=sim.device,
                    )
                    smoothed_vel_cmd_b = torch.zeros(
                        (1, 3), dtype=torch.float32, device=sim.device
                    )
                    requested_vel_cmd_b = torch.zeros(
                        (1, 3), dtype=torch.float32, device=sim.device
                    )
                    if utlidar_state is not None:
                        # The LiDAR throttler compares these values against
                        # ``policy_step``.  Since a new lap resets policy_step
                        # to zero while keeping the ROS publisher alive, stale
                        # counters from the previous lap would suppress every
                        # scan until the old step number was reached again.
                        utlidar_state["last_pub_step"] = -10_000
                        utlidar_state["last_cloud_pub_step"] = -10_000
                        utlidar_state["scan_count"] = 0
                        utlidar_state["cloud_count"] = 0
                    has_received_motion_command = False
                    policy_step = 0
                    arm_forbidden_contact_ticks = 0
                    external_supervisor_command_available = False
                    external_supervisor_motion_started = False
                    _prev_command_active = False
                    stationary_base_pose_target = None
                    nonfinite_stop_reported = False
                    previous_root_linear_velocity_w = robot.data.root_lin_vel_w.clone()
                    locomotion_failure_reason = None
                    scripted_route_index = 0
                    scripted_route_complete_reported = False
                    hazard_collect_state.update(
                        {
                            "context": None,
                            "event_id": None,
                            "label_request": None,
                            "terminal_signal": 0,
                            "terminal_start_time": None,
                            "episode_start_time": None,
                            "next_sample_time": None,
                            "peek_valid_samples": 0,
                            "peek_validated_samples": 0,
                            "latest_peek_valid": False,
                        }
                    )
                    hazard_collect_state["pending_teacher_labels"].clear()
                    if args_cli.abort_on_locomotion_failure:
                        locomotion_failure_detector = LocomotionFailureDetector(
                            robot.data.body_pos_w[0, base_link_idx]
                            .detach()
                            .cpu()
                            .numpy(),
                            robot.data.body_quat_w[0, base_link_idx]
                            .detach()
                            .cpu()
                            .numpy(),
                            robot.data.body_pos_w[0, _foot_indices]
                            .detach()
                            .cpu()
                            .numpy(),
                            LocomotionFailureConfig(
                                grace_samples=args_cli.locomotion_failure_grace_steps,
                                sustained_samples=args_cli.locomotion_failure_sustain_steps,
                            ),
                        )

                    status_path = args_cli.binary_tree_repeat_reset_status_file
                    status_path.parent.mkdir(parents=True, exist_ok=True)
                    status_tmp = status_path.with_name(status_path.name + ".tmp")
                    status_tmp.write_text(
                        json.dumps(
                            {
                                "schema": "binary_tree_repeat_reset.v1",
                                "status": "ready",
                                "ready_lap": int(requested_collection_lap),
                                "sim_time_s": float(sim.current_time),
                            },
                            indent=2,
                            sort_keys=True,
                        ),
                        encoding="utf-8",
                    )
                    status_tmp.replace(status_path)
                    print_robot_pose_summary(
                        robot,
                        _foot_names,
                        _foot_indices,
                        f"lap {requested_collection_lap} reset ready",
                    )
                    print(
                        f">>> [batch_collect] lap {requested_collection_lap} reset ready",
                        flush=True,
                    )
                except Exception as exc:
                    locomotion_failure_reason = "between_lap_reset_failed"
                    print(
                        f">>> SAFETY STOP: in-place reset failed: {exc}",
                        flush=True,
                    )
                    break
            explicit_base_pause = bool(
                args_cli.active_gap_arm
                and active_arm_server is not None
                and active_arm_server.base_paused()
            )
            if args_cli.enable_hazard_smolvla_policy:
                if _smolvla_proc is None:
                    locomotion_failure_reason = "hazard_smolvla_runner_missing"
                    print(
                        ">>> SAFETY STOP: hazard SmolVLA runner is not active",
                        flush=True,
                    )
                    break
                hazard_runner_rc = _smolvla_proc.poll()
                if hazard_runner_rc is not None:
                    locomotion_failure_reason = "hazard_smolvla_runner_exited"
                    print(
                        ">>> SAFETY STOP: hazard SmolVLA runner exited during the task "
                        f"(code={hazard_runner_rc}); check "
                        f"{args_cli.hazard_smolvla_log}",
                        flush=True,
                    )
                    break
            leader_apply_allowed = bool(
                not args_cli.leader_apply_only_when_base_paused
                or explicit_base_pause
            )
            scripted_route_startup_hold = bool(
                scripted_route_waypoints is not None
                and policy_step < args_cli.scripted_route_stand_steps
            )
            # Let the active locomotion policy own zero-command balance. Do not
            # kinematically freeze the root while waiting for the supervisor:
            # that suppresses learned standing actions and creates an
            # out-of-distribution release transient at first motion.
            startup_route_hold = scripted_route_startup_hold
            # base_pause is a logical arm-authorization lease, not a root-pose
            # constraint. Root pinning remains only for the optional pre-route
            # startup carrier.
            stationary_base_hold_active = startup_route_hold
            if stationary_base_hold_active and stationary_base_pose_target is None:
                stationary_base_pose_target = robot.data.root_state_w[:, :7].clone()
            elif not stationary_base_hold_active:
                stationary_base_pose_target = None
            nonfinite_state = first_nonfinite_state(robot, sim_arm_joint_ids, gripper_link_idx)
            if nonfinite_state is not None:
                if not nonfinite_stop_reported:
                    print(
                        f">>> SAFETY STOP: non-finite robot state detected in {nonfinite_state}; "
                        "disabling SO-Arm action application.",
                        flush=True,
                    )
                    nonfinite_stop_reported = True
                arm_action_enabled = False
                break
            if smolvla_bridge is not None:
                if smolvla_state["toggle_requested"]:
                    smolvla_state["toggle_requested"] = False
                    if smolvla_state["active"]:
                        smolvla_state["active"] = False
                        smolvla_state["next_obs_time"] = None
                        latest_smolvla_action_deg = None
                        smolvla_bridge.receive_action()
                        smolvla_bridge.send_reset()
                        arm_action_enabled = bool(args_cli.gr00t_apply_actions)
                        print(">>> [smolvla] policy control OFF; SO-Arm returns to manual/GR00T target source.", flush=True)
                    else:
                        smolvla_state["active"] = True
                        smolvla_state["next_obs_time"] = None
                        latest_smolvla_action_deg = None
                        smolvla_bridge.receive_action()
                        smolvla_bridge.send_reset()
                        arm_action_enabled = True
                        print(">>> [smolvla] policy control ON; publishing camera/state observations.", flush=True)
                if _smolvla_proc is not None:
                    _smolvla_rc = _smolvla_proc.poll()
                    if _smolvla_rc is not None and not smolvla_state["proc_exit_reported"]:
                        smolvla_state["proc_exit_reported"] = True
                        smolvla_state["active"] = False
                        print(f">>> [smolvla] policy runner exited (code={_smolvla_rc}); check {args_cli.smolvla_log}", flush=True)
                smolvla_action = smolvla_bridge.receive_action()
                if smolvla_action is not None:
                    latest_smolvla_action_deg = clip_gr00t_action(smolvla_action)
                    if (
                        smolvla_state["active"]
                        and args_cli.smolvla_action_log_every > 0
                        and policy_step % args_cli.smolvla_action_log_every == 0
                    ):
                        print(f">>> SmolVLA action received deg: {latest_smolvla_action_deg}", flush=True)
            if leader_action_sub is not None:
                leader_action = leader_action_sub.receive()
                gripper_raw = receive_leader_gripper_raw()
                if gripper_raw is not None:
                    latest_leader_gripper_raw = gripper_raw
                if leader_action is not None:
                    target_deg = clip_leader_action(leader_action.joint_target_deg)
                    latest_leader_action_deg = target_deg.copy()
                    if (
                        arm_action_enabled
                        and not smolvla_state["active"]
                        and leader_apply_allowed
                    ):
                        target_rad = torch.as_tensor(
                            calibrated_leader_action_sim_rad(target_deg),
                            device=sim.device,
                            dtype=joint_targets.dtype,
                        )
                        # The legacy six-axis conversion includes a fixed 50-degree
                        # gripper placeholder. Apply only the six arm values here;
                        # servo ID 7 exclusively owns the gripper target below.
                        joint_targets[:, sim_arm_only_joint_ids] = target_rad[sim_arm_only_mask]
                    if args_cli.leader_action_log_every > 0 and policy_step % args_cli.leader_action_log_every == 0:
                        print(f">>> leader action applied deg: {target_deg}")
                if (
                    gripper_raw is not None
                    and arm_action_enabled
                    and not smolvla_state["active"]
                    and leader_apply_allowed
                ):
                    # Direct, unfiltered mapping from the measured leader range.
                    # Direction remains reversed: leader close -> sim close.
                    grip_deg = float(np.interp(gripper_raw, (1593.0, 2918.0), (100.0, -10.0)))
                    joint_targets[:, gripper_joint_idx] = math.radians(grip_deg)

            if policy_step % policy_decimation == 0 and not args_cli.static_spawn:
                scripted_motion_finished = False
                external_base_command = (
                    active_arm_server.base_command(
                        round(sim.current_time * 1_000_000_000)
                    )
                    if args_cli.active_gap_arm and active_arm_server is not None
                    else None
                )
                if args_cli.active_gap_arm:
                    external_supervisor_command_available = bool(
                        external_base_command is not None
                    )
                    if (
                        external_base_command is not None
                        and np.linalg.norm(external_base_command) > 1.0e-6
                    ):
                        external_supervisor_motion_started = True
                if external_base_command is not None:
                    vel_cmd_b = torch.as_tensor(
                        external_base_command,
                        dtype=torch.float32,
                        device=sim.device,
                    ).view(1, 3)
                elif args_cli.camera_qa:
                    vel_cmd_b = torch.tensor(
                        [[args_cli.camera_qa_forward_vel, 0.0, 0.0]],
                        dtype=torch.float32,
                        device=sim.device,
                    )
                elif args_cli.demo_pan:
                    vel_cmd_b = torch.tensor(
                        [[args_cli.demo_forward_vel, 0.0, 0.0]],
                        dtype=torch.float32,
                        device=sim.device,
                    )
                elif args_cli.demo_fold_walk:
                    vel_cmd_b = torch.tensor(
                        [[args_cli.demo_forward_vel, 0.0, 0.0]],
                        dtype=torch.float32,
                        device=sim.device,
                    )
                elif args_cli.camera_aim_qa:
                    vel_cmd_b = torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
                elif maze_inspector is not None:
                    if policy_step < args_cli.scripted_route_stand_steps:
                        vel_cmd_b = torch.zeros((1,3),dtype=torch.float32,device=sim.device)
                    else:
                        position=robot.data.root_pos_w[0]
                        w,x,y,z=map(float,robot.data.root_quat_w[0].detach().cpu().numpy())
                        yaw=math.atan2(2*(w*z+x*y),1-2*(y*y+z*z))
                        scan_angles,scan_ranges,scan_sequence=maze_scan.sample(sim_cfg.dt*policy_decimation,position,yaw)
                        contact_forces=arm_contact_sensor_runtime.data.net_forces_w[0,foot_contact_body_ids]
                        maze_foot_positions=robot.data.body_pos_w[0,maze_foot_body_ids].detach().cpu().numpy()
                        maze_foot_forces=contact_forces.detach().cpu().numpy()
                        # Collision-only Nav2 consumes no camera observations.
                        # Reading all camera annotators at every actor tick needlessly
                        # transfers RGB/depth; the display reads them at its own rate.
                        wrist_output=(wrist_camera_sensor.data.output
                                      if wrist_camera_sensor is not None
                                      and (args_cli.maze_nav2_inspection or not (args_cli.maze_nav2 or args_cli.maze_locomotion_only)) else {})
                        wrist_rgb=wrist_output.get('rgb')
                        maze_joint_positions=robot.data.joint_pos[0,sim_arm_joint_ids].detach().cpu().numpy()
                        maze_joint_velocities=robot.data.joint_vel[0,sim_arm_joint_ids].detach().cpu().numpy()
                        maze_feedback={
                            'base_height':float(position[2]),
                            'tilt_deg':math.degrees(math.acos(max(-1.,min(1.,1.-2.*(x*x+y*y))))),
                            'body_velocity':[float(robot.data.root_lin_vel_b[0,0]),float(robot.data.root_lin_vel_b[0,1]),float(robot.data.root_ang_vel_b[0,2])],
                            'feet':int(ground_support(maze_foot_forces,maze_foot_positions).sum()),
                            'foot_z':np.round(maze_foot_positions[:,2],3).tolist(),
                            'foot_fz':np.round(maze_foot_forces[:,2],1).tolist(),
                            'angular_speed':float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[0])),
                            'joints':dict(zip(sim_arm_joint_names,map(float,maze_joint_positions))),
                            'joint_velocities':dict(zip(sim_arm_joint_names,map(float,maze_joint_velocities))),
                            'frame':int(wrist_camera_sensor.frame[0].item()) if wrist_camera_sensor is not None else -1,
                            'image_valid':wrist_rgb is not None and wrist_rgb.numel()>0,
                        }
                        if args_cli.maze_observation_navigation and policy_step%20==0:
                            if os.environ.get('MAZE_NBV_AUDIT')=='1' or os.environ.get('MAZE_NBV_TEACHER')=='1':
                                maze_feedback['nbv_geometry']={
                                    'all_joints':dict(zip(robot.joint_names,map(float,robot.data.joint_pos[0].detach().cpu().numpy()))),
                                    'gripper_to_world':camera_pose_matrix(
                                        robot.data.body_pos_w[0,gripper_link_idx].detach().cpu().numpy(),
                                        robot.data.body_quat_w[0,gripper_link_idx].detach().cpu().numpy()),
                                    'base_to_world':camera_pose_matrix(
                                        robot.data.body_pos_w[0,base_link_idx].detach().cpu().numpy(),
                                        robot.data.body_quat_w[0,base_link_idx].detach().cpu().numpy()),
                                    'camera_offset':camera_pose_matrix(
                                        np.asarray(WRIST_CAMERA_LOCAL_POS),np.asarray(WRIST_CAMERA_LOCAL_ROT)),
                                }
                            maze_feedback['camera_visibility']=[]
                            maze_feedback['local_map_masks']={}
                            for visibility_name,visibility_camera in (('front',front_camera_sensor),('wrist',wrist_camera_sensor)):
                                visibility_depth=visibility_camera.data.output.get('distance_to_image_plane')
                                if visibility_depth is not None:
                                    if os.environ.get('MAZE_LOCAL_OCCUPANCY')=='1':
                                        from robot_pixel_mask import robot_pixel_mask
                                        instance_ids=visibility_camera.data.output['instance_id_segmentation_fast']
                                        maze_feedback['local_map_masks'][visibility_name]=robot_pixel_mask(
                                            instance_ids[0].detach().cpu().numpy(),
                                            visibility_camera.data.info[0]['instance_id_segmentation_fast'])
                                    maze_feedback['camera_visibility'].append(dict(
                                        name=visibility_name,frame=int(visibility_camera.frame[0]),
                                        depth=visibility_depth[0].detach().cpu().numpy(),
                                        intrinsic=visibility_camera.data.intrinsic_matrices[0].detach().cpu().numpy(),
                                        camera_to_world=camera_pose_matrix(
                                            visibility_camera.data.pos_w[0].detach().cpu().numpy(),
                                            visibility_camera.data.quat_w_opengl[0].detach().cpu().numpy())))
                                    if os.environ.get('MAZE_DYNAMIC_MISSION')=='1':
                                        visibility_rgb=visibility_camera.data.output.get('rgb')
                                        if visibility_rgb is not None:
                                            maze_feedback.setdefault('hazard_samples',[]).append(dict(
                                                **maze_feedback['camera_visibility'][-1],
                                                rgb=visibility_rgb[0,:,:,:3].detach().cpu().numpy()))
                        from inspection_control import needs_wrist_depth
                        if args_cli.maze_nav2_inspection and needs_wrist_depth(maze_inspector.arm):
                            wrist_depth=wrist_output.get('distance_to_image_plane')
                            if wrist_depth is not None:
                                maze_feedback['wrist_points']=depth_world_points(
                                    wrist_depth[0].detach().cpu().numpy(),
                                    wrist_camera_sensor.data.intrinsic_matrices[0].detach().cpu().numpy(),
                                    camera_pose_matrix(wrist_camera_sensor.data.pos_w[0].detach().cpu().numpy(),
                                                       wrist_camera_sensor.data.quat_w_opengl[0].detach().cpu().numpy()))
                        if (args_cli.maze_observed_navigation and maze_inspector.state=='drive' and policy_step%40==0
                            and float(torch.linalg.vector_norm(robot.data.root_ang_vel_b[0]))<.05
                            and float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[0,:2]))<.05):
                            front_out=front_camera_sensor.data.output
                            front_depth=front_out.get('distance_to_image_plane')
                            front_rgb=front_out.get('rgb')
                            if front_depth is not None and front_rgb is not None:
                                maze_feedback['front_cells']=red_cube_cells(
                                    front_rgb[0].detach().cpu().numpy(),front_depth[0].detach().cpu().numpy(),
                                    front_camera_sensor.data.intrinsic_matrices[0].detach().cpu().numpy(),
                                    camera_pose_matrix(front_camera_sensor.data.pos_w[0].detach().cpu().numpy(),
                                                       front_camera_sensor.data.quat_w_opengl[0].detach().cpu().numpy()))
                        if args_cli.maze_observed_navigation and maze_inspector.arm is not None and maze_inspector.arm.name.startswith('look '):
                            depth=wrist_output.get('distance_to_image_plane')
                            maze_feedback['rgbd_valid']=depth is not None and float(torch.isfinite(depth).float().mean())>.30
                            maze_feedback['observed_cells']=set()
                            if maze_feedback['rgbd_valid'] and maze_feedback['image_valid']:
                                maze_feedback['wrist_points']=depth_world_points(
                                    depth[0].detach().cpu().numpy(),
                                    wrist_camera_sensor.data.intrinsic_matrices[0].detach().cpu().numpy(),
                                    camera_pose_matrix(wrist_camera_sensor.data.pos_w[0].detach().cpu().numpy(),
                                                       wrist_camera_sensor.data.quat_w_opengl[0].detach().cpu().numpy()))
                                maze_feedback['observed_cells']=red_cube_cells(
                                    wrist_rgb[0].detach().cpu().numpy(),depth[0].detach().cpu().numpy(),
                                    wrist_camera_sensor.data.intrinsic_matrices[0].detach().cpu().numpy(),
                                    camera_pose_matrix(wrist_camera_sensor.data.pos_w[0].detach().cpu().numpy(),
                                                       wrist_camera_sensor.data.quat_w_opengl[0].detach().cpu().numpy()))
                        if policy_step%20==0:
                            for sensor_window in camera_viewports:
                                if getattr(sensor_window,'sensor_image_panel',False):sensor_window.update_sensor_image()
                        command=maze_inspector.step(sim_cfg.dt*policy_decimation,
                            (float(position[0]),float(position[1])),yaw,
                            float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[0,:2])),
                            scan_angles,scan_ranges,scan_sequence,maze_feedback)
                        if args_cli.maze_nav2_inspection:
                            # A few diagnostic stills, not a training dataset.
                            # Save at the accepted fresh wrist frame, independent
                            # of which desktop window currently has focus.
                            for observation in maze_inspector.inspection.observations:
                                if 'image_path' not in observation and wrist_rgb is not None:
                                    from PIL import Image
                                    snapshot=Path('/home/iy/Isaac/maze_visual_demo')/(
                                        f'nav2-{maze_inspector.run_id}-inspection-'
                                        f"{observation['inspection']:03d}-{observation['side']}.png")
                                    Image.fromarray(wrist_rgb[0,:,:,:3].detach().cpu().numpy().astype(np.uint8)).save(snapshot)
                                    observation['image_path']=str(snapshot)
                                    observation['camera_position']=wrist_camera_sensor.data.pos_w[0].detach().cpu().tolist()
                                    observation['camera_quat_opengl']=wrist_camera_sensor.data.quat_w_opengl[0].detach().cpu().tolist()
                                    print('NAV2_INSPECTION_IMAGE',snapshot,flush=True)
                        if maze_inspector.failure:
                            print('MAZE_CONTROL_FAILED',maze_inspector.failure,flush=True)
                            if args_cli.maze_nav2:
                                maze_inspector.close()
                            raise RuntimeError(maze_inspector.failure)
                        if (args_cli.maze_locomotion_only or args_cli.maze_nav2) and maze_inspector.done and args_cli.exit_on_route_complete:
                            if args_cli.maze_nav2:
                                maze_inspector.close()
                            break
                        if maze_display_ui is not None and policy_step%40==0:
                            maze_display_ui.motion_status.text=maze_inspector.drift_status(
                                (float(position[0]),float(position[1])))
                        vel_cmd_b=torch.tensor([command],dtype=torch.float32,device=sim.device)
                elif scripted_route_waypoints is not None:
                    if policy_step < args_cli.scripted_route_stand_steps:
                        vel_cmd_b = torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
                    else:
                        # Route navigation must use the articulation root.
                        # A named body link can carry a fixed offset that
                        # rotates around the root and corrupts XY feedback.
                        if args_cli.kinematic_scripted_route:
                            current_x = kinematic_route_x
                            current_y = kinematic_route_y
                            current_yaw = kinematic_route_yaw
                        else:
                            base_position = robot.data.root_pos_w[0]
                            base_quaternion = robot.data.root_quat_w[0]
                            current_x = float(base_position[0].item())
                            current_y = float(base_position[1].item())
                            w, x, y, z = (float(value.item()) for value in base_quaternion)
                            current_yaw = math.atan2(
                                2.0 * (w * z + x * y),
                                1.0 - 2.0 * (y * y + z * z),
                            )

                        while scripted_route_index < len(scripted_route_waypoints):
                            target_x, target_y = scripted_route_waypoints[scripted_route_index]
                            distance = math.hypot(float(target_x) - current_x, float(target_y) - current_y)
                            route_tolerance = (
                                0.03 if args_cli.kinematic_scripted_route
                                else args_cli.scripted_route_tolerance
                            )
                            if distance > route_tolerance:
                                break
                            print(
                                f">>> Scripted route waypoint {scripted_route_index + 1}/"
                                f"{len(scripted_route_waypoints)} reached at "
                                f"({current_x:.2f}, {current_y:.2f})",
                                flush=True,
                            )
                            scripted_route_index += 1

                        if scripted_route_index >= len(scripted_route_waypoints):
                            vel_cmd_b = torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
                            if not scripted_route_complete_reported:
                                completion_action = (
                                    "exiting so the mapper can save the completed map."
                                    if args_cli.exit_on_route_complete
                                    else "holding position for map inspection."
                                )
                                print(f">>> Scripted route COMPLETE; {completion_action}", flush=True)
                                if active_arm_server is not None:
                                    active_arm_server.publish_route_complete(True)
                                scripted_route_complete_reported = True
                            if args_cli.exit_on_route_complete:
                                break
                        else:
                            target_x, target_y = scripted_route_waypoints[scripted_route_index]
                            delta_x = float(target_x) - current_x
                            delta_y = float(target_y) - current_y
                            distance = math.hypot(delta_x, delta_y)
                            target_yaw = math.atan2(delta_y, delta_x)
                            yaw_error = math.atan2(
                                math.sin(target_yaw - current_yaw),
                                math.cos(target_yaw - current_yaw),
                            )
                            if args_cli.kinematic_scripted_route:
                                # Continuous trajectory carrier for mapping:
                                # rotate at <=0.5 rad/s and translate only once
                                # aligned.  Root pose advances by one control
                                # interval, so there are no waypoint teleports.
                                control_dt = sim_cfg.dt * policy_decimation
                                yaw_step = float(np.clip(
                                    yaw_error,
                                    -0.5 * control_dt,
                                    0.5 * control_dt,
                                ))
                                next_yaw = current_yaw + yaw_step
                                travel = 0.0
                                if abs(yaw_error) <= 0.20:
                                    travel = min(
                                        distance,
                                        args_cli.scripted_route_speed * control_dt,
                                    )
                                next_x = current_x + travel * delta_x / distance
                                next_y = current_y + travel * delta_y / distance
                                kinematic_route_x = next_x
                                kinematic_route_y = next_y
                                kinematic_route_yaw = next_yaw
                                root_pose = robot.data.root_state_w[:, :7].clone()
                                root_pose[0, 0] = next_x
                                root_pose[0, 1] = next_y
                                root_pose[0, 2] = kinematic_route_root_z
                                root_pose[0, 3] = math.cos(0.5 * next_yaw)
                                root_pose[0, 4] = 0.0
                                root_pose[0, 5] = 0.0
                                root_pose[0, 6] = math.sin(0.5 * next_yaw)
                                kinematic_route_pose_target = root_pose
                                vel_cmd_b = torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
                            else:
                                # The policy was trained across substantial yaw
                                # commands.  Tiny capped commands cannot correct
                                # its natural heading drift and leave it turning
                                # in place until it falls.  Use a conventional
                                # proportional heading command, never strafe, and
                                # translate only while reasonably aligned.
                                yaw_rate = float(np.clip(
                                    args_cli.scripted_route_yaw_gain * yaw_error,
                                    -args_cli.scripted_route_max_yaw_rate,
                                    args_cli.scripted_route_max_yaw_rate,
                                ))
                                if abs(yaw_error) > args_cli.scripted_route_forward_alignment_rad:
                                    forward_speed = 0.0
                                else:
                                    forward_speed = min(
                                        args_cli.scripted_route_speed,
                                        0.7 * distance,
                                    ) * max(0.25, math.cos(yaw_error))
                                vel_cmd_b = torch.tensor(
                                    [[forward_speed, 0.0, yaw_rate]],
                                    dtype=torch.float32,
                                    device=sim.device,
                                )
                elif args_cli.gait_probe:
                    probe_label, probe_command = gait_probe_command(policy_step * sim_cfg.dt)
                    vel_cmd_b = torch.tensor([probe_command], dtype=torch.float32, device=sim.device)
                elif args_cli.scripted_velocity is not None:
                    scripted_motion_end = args_cli.scripted_stand_steps + args_cli.scripted_velocity_duration_steps
                    scripted_motion_finished = (
                        args_cli.scripted_velocity_duration_steps > 0 and policy_step >= scripted_motion_end
                    )
                    if policy_step < args_cli.scripted_stand_steps or scripted_motion_finished:
                        vel_cmd_b = torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
                    else:
                        vel_cmd_b = torch.tensor(
                            [args_cli.scripted_velocity],
                            dtype=torch.float32,
                            device=sim.device,
                        )
                else:
                    vel_cmd_b = (
                        keyboard.advance().view(1, 3)
                        if keyboard is not None
                        else torch.zeros((1, 3), dtype=torch.float32, device=sim.device)
                    )
                requested_vel_cmd_b = vel_cmd_b
                if args_cli.gait_probe:
                    smoothed_vel_cmd_b.copy_(requested_vel_cmd_b)
                elif maze_inspector is not None:
                    # Maze/Nav2 shares the verified active-gap command contract:
                    # zero and one-axis commands behave like discrete key holds,
                    # while continuous multi-axis MPPI commands are slew-limited
                    # at the policy's native 50 Hz. Passing every correction
                    # directly made the same 12999 gait twist and splay its legs.
                    if sum(
                        abs(float(requested_vel_cmd_b[0, axis].item())) > 1.0e-6
                        for axis in range(3)
                    ) <= 1:
                        smoothed_vel_cmd_b.copy_(requested_vel_cmd_b)
                    else:
                        command_delta_limit = torch.tensor(
                            [[0.01, 0.01, 0.02]],
                            dtype=vel_cmd_b.dtype,
                            device=vel_cmd_b.device,
                        )
                        smoothed_vel_cmd_b.add_(
                            torch.clamp(
                                requested_vel_cmd_b - smoothed_vel_cmd_b,
                                min=-command_delta_limit,
                                max=command_delta_limit,
                            )
                        )
                    vel_cmd_b = smoothed_vel_cmd_b.clone()
                elif scripted_route_waypoints is not None:
                    # Route direction can change abruptly at a waypoint.  A
                    # finite slew limit prevents a command discontinuity while
                    # still allowing heading feedback to overcome policy bias.
                    command_delta_limit = torch.tensor(
                        [[
                            args_cli.scripted_route_linear_slew,
                            0.0,
                            args_cli.scripted_route_yaw_slew,
                        ]],
                        dtype=vel_cmd_b.dtype,
                        device=vel_cmd_b.device,
                    )
                    smoothed_vel_cmd_b.add_(
                        torch.clamp(
                            requested_vel_cmd_b - smoothed_vel_cmd_b,
                            min=-command_delta_limit,
                            max=command_delta_limit,
                        )
                    )
                    vel_cmd_b = smoothed_vel_cmd_b.clone()
                elif args_cli.active_gap_arm:
                    # Autonomous commands arrive from an asynchronous ROS
                    # supervisor and may change discontinuously at a sensor
                    # boundary.  Present the learned gait with a continuous
                    # command contract at its native 50 Hz policy rate.
                    if explicit_base_pause:
                        smoothed_vel_cmd_b.zero_()
                        # Translation remains locked during arm authorization,
                        # but a bounded yaw-only correction lets the physical
                        # gait resist asymmetric arm-load heading drift.
                        smoothed_vel_cmd_b[0, 2] = requested_vel_cmd_b[0, 2]
                    elif sum(
                        abs(float(requested_vel_cmd_b[0, axis].item())) > 1.0e-6
                        for axis in range(3)
                    ) <= 1:
                        # A mutually-exclusive route primitive must behave like
                        # holding or releasing one verified WASD/QE key.  The
                        # former 0.02-rad/s slew kept applying almost the full
                        # old yaw command for 0.35 s after STOP, overshot the
                        # heading, reversed, and eventually tangled the legs.
                        # Direct single-axis/zero commands preserve the policy's
                        # tested teleoperation contract.  Multi-axis corridor
                        # feedback remains slew-limited below.
                        smoothed_vel_cmd_b.copy_(requested_vel_cmd_b)
                    else:
                        command_delta_limit = torch.tensor(
                            [[0.01, 0.01, 0.02]],
                            dtype=vel_cmd_b.dtype,
                            device=vel_cmd_b.device,
                        )
                        smoothed_vel_cmd_b.add_(
                            torch.clamp(
                                requested_vel_cmd_b - smoothed_vel_cmd_b,
                                min=-command_delta_limit,
                                max=command_delta_limit,
                            )
                        )
                    vel_cmd_b = smoothed_vel_cmd_b.clone()
                elif args_cli.idle_stance_fallback:
                    command_delta_limit = torch.tensor(
                        [[0.04, 0.03, 0.04]], dtype=vel_cmd_b.dtype, device=vel_cmd_b.device
                    )
                    smoothed_vel_cmd_b.add_(
                        torch.clamp(
                            requested_vel_cmd_b - smoothed_vel_cmd_b,
                            min=-command_delta_limit,
                            max=command_delta_limit,
                        )
                    )
                    vel_cmd_b = smoothed_vel_cmd_b.clone()
                else:
                    smoothed_vel_cmd_b.copy_(requested_vel_cmd_b)
                _command_active = bool(torch.linalg.vector_norm(requested_vel_cmd_b).item() > 1.0e-6)
                first_motion_command = bool(
                    _command_active and not has_received_motion_command
                )
                if _command_active:
                    has_received_motion_command = True
                if first_motion_command:
                    if args_cli.state_debug_every > 0:
                        print(
                            f">>> [transition] first motion command at step={policy_step}; "
                            "preserving learned balance action history",
                            flush=True,
                        )
                elif _command_active != _prev_command_active and args_cli.state_debug_every > 0:
                    print(
                        f">>> [command_edge] step={policy_step} active={_command_active} "
                        f"requested={requested_vel_cmd_b[0].detach().cpu().tolist()} "
                        f"pause={explicit_base_pause} available={external_supervisor_command_available}",
                        flush=True,
                    )
                _prev_command_active = _command_active
                if args_cli.go2_policy_obs_mode == "flat":
                    policy_obs = torch.cat(
                        (
                            robot.data.root_lin_vel_b,
                            robot.data.root_ang_vel_b,
                            robot.data.projected_gravity_b,
                            vel_cmd_b,
                            robot.data.joint_pos[:, leg_joint_ids] - default_leg_joint_pos,
                            robot.data.joint_vel[:, leg_joint_ids] - default_joint_vel[:, leg_joint_ids],
                            last_action,
                        ),
                        dim=-1,
                    )
                else:
                    rough_joint_pos_obs = (
                        robot.data.joint_pos[:, rough_joint_ids] - default_rough_joint_pos
                    )
                    rough_joint_vel_obs = (
                        robot.data.joint_vel[:, rough_joint_ids]
                        - default_joint_vel[:, rough_joint_ids]
                    )
                    if args_cli.no_arm:
                        # This 247-D export observed six fixed SO-ARM joints
                        # during training even though it only controls 12 leg
                        # actions. Preserve those slots for the arm-free model.
                        missing_arm_obs = torch.zeros(
                            (rough_joint_pos_obs.shape[0], 6),
                            device=sim.device,
                            dtype=rough_joint_pos_obs.dtype,
                        )
                        rough_joint_pos_obs = torch.cat(
                            (rough_joint_pos_obs, missing_arm_obs), dim=-1
                        )
                        rough_joint_vel_obs = torch.cat(
                            (rough_joint_vel_obs, missing_arm_obs), dim=-1
                        )
                    rough_proprio_obs = torch.cat(
                        (
                            robot.data.root_lin_vel_b,
                            robot.data.root_ang_vel_b,
                            robot.data.projected_gravity_b,
                            vel_cmd_b,
                            rough_joint_pos_obs,
                            rough_joint_vel_obs,
                            last_action,
                        ),
                        dim=-1,
                    )
                    if height_scanner is not None:
                        height_scan = height_scanner.data.pos_w[:, 2].unsqueeze(1) - height_scanner.data.ray_hits_w[..., 2] - 0.5
                        height_scan = torch.nan_to_num(height_scan, nan=0.0, posinf=0.0, neginf=0.0)
                    else:
                        # The warehouse floor is flat. Fill only the remaining
                        # height-scan tail after all proprioceptive slots.
                        height_scan_dim = GO2_POLICY_OBS_DIM - rough_proprio_obs.shape[-1]
                        height_scan = torch.zeros(
                            (robot.data.root_lin_vel_b.shape[0], height_scan_dim),
                            device=sim.device,
                        )
                    policy_obs = torch.cat((rough_proprio_obs, height_scan), dim=-1)
                policy_obs_dim = GO2_POLICY_OBS_DIMS[args_cli.go2_policy_obs_mode]
                if policy_obs.shape[-1] != policy_obs_dim:
                    raise RuntimeError(
                        f"Go2 {args_cli.go2_policy_obs_mode} policy observation dimension "
                        f"{policy_obs.shape[-1]} does not match expected {policy_obs_dim}"
                    )
                policy_obs = torch.nan_to_num(policy_obs, nan=0.0, posinf=0.0, neginf=0.0)
                if args_cli.state_debug_every > 0 and policy_step == 0:
                    print(f">>> [policy_obs step=0] {policy_obs[0].detach().cpu().tolist()}", flush=True)
                idle_command = bool(torch.linalg.vector_norm(requested_vel_cmd_b).item() <= 1.0e-6)
                use_startup_stance = (
                    args_cli.idle_stance_fallback and idle_command and not has_received_motion_command
                ) or stationary_base_hold_active
                if use_startup_stance:
                    last_action = torch.zeros_like(last_action)
                else:
                    with torch.inference_mode():
                        last_action = torch.nan_to_num(policy(policy_obs), nan=0.0, posinf=0.0, neginf=0.0)
                        # Once the robot leaves the training distribution an
                        # unconstrained MLP can emit enormous joint commands.
                        # Bound them before applying the 0.25 rad action scale.
                        last_action.clamp_(-GO2_POLICY_ACTION_LIMIT, GO2_POLICY_ACTION_LIMIT)
                # The leg controller must not reset the arm to HOME between
                # VLA packets, rejected packets, or inspection leases.
                previous_arm_targets = (
                    joint_targets[:, external_joint_ids].clone()
                    if hazard_smolvla_bridge is not None else None
                )
                joint_targets = default_joint_pos.clone()
                joint_targets = hold_elbow_rotate_target(joint_targets)
                if previous_arm_targets is not None:
                    joint_targets[:, external_joint_ids] = previous_arm_targets
                leg_policy_scale = 1.0 if ROBOT_MODEL_PROFILE.key == "so101_7motor_reversed" else min(float(policy_step) / float(LEG_POLICY_RAMP_STEPS), 1.0)
                leg_action_scale = GO2_ACTION_SCALE
                joint_targets[:, policy_action_joint_ids] = (
                    default_policy_action_joint_pos + leg_action_scale * leg_policy_scale * last_action
                )
                if args_cli.state_debug_every > 0 and policy_step == 0:
                    print(
                        f">>> [policy_target step=0] default={default_policy_action_joint_pos[0].detach().cpu().tolist()} "
                        f"target={joint_targets[0, policy_action_joint_ids].detach().cpu().tolist()}",
                        flush=True,
                    )
                if gr00t_bridge is not None:
                    action = gr00t_bridge.receive_action()
                    if action is not None:
                        action = clip_gr00t_action(action)
                        latest_gr00t_action_deg = action.copy()
                        print(f">>> GR00T action received ({'applied' if args_cli.gr00t_apply_actions else 'observed'}): {action}")
                leader_target_active = (
                    not smolvla_state["active"]
                    and latest_leader_action_deg is not None
                    and leader_apply_allowed
                )
                if smolvla_state["active"] and latest_smolvla_action_deg is not None:
                    active_arm_target_deg = latest_smolvla_action_deg
                else:
                    active_arm_target_deg = latest_leader_action_deg if leader_target_active else latest_gr00t_action_deg
                if active_arm_target_deg is not None and arm_action_enabled:
                    if leader_target_active:
                        action_rad = torch.as_tensor(
                            calibrated_leader_action_sim_rad(active_arm_target_deg),
                            device=sim.device,
                            dtype=joint_targets.dtype,
                        )
                        joint_targets[:, sim_arm_only_joint_ids] = action_rad[sim_arm_only_mask]
                        if latest_leader_gripper_raw is not None:
                            grip_deg = float(
                                np.interp(
                                    latest_leader_gripper_raw,
                                    (1593.0, 2918.0),
                                    (100.0, -10.0),
                                )
                            )
                            joint_targets[:, gripper_joint_idx] = math.radians(grip_deg)
                    else:
                        action_rad = torch.as_tensor(
                            runtime_external_deg_to_sim_rad(active_arm_target_deg),
                            device=sim.device,
                            dtype=joint_targets.dtype,
                        )
                        joint_targets[:, external_joint_ids] = action_rad
                if args_cli.demo_pan:
                    demo_t = sim.current_time
                    demo_pan_offset = math.radians(args_cli.demo_pan_deg) * math.sin(
                        2.0 * math.pi * demo_t / args_cli.demo_pan_period_s
                    )
                    joint_targets[0, external_joint_ids[0]] = default_joint_pos[0, external_joint_ids[0]] + demo_pan_offset
                    for _local_idx in range(1, len(GR00T_FULL_JOINT_ORDER)):
                        joint_targets[0, external_joint_ids[_local_idx]] = default_joint_pos[0, external_joint_ids[_local_idx]]
                    if policy_step % 100 == 0:
                        print(f">>> [demo_pan] t={demo_t:.2f}s pan_deg={math.degrees(demo_pan_offset):+.1f}", flush=True)
                if args_cli.demo_fold_walk:
                    demo_phase = math.fmod(sim.current_time, args_cli.demo_fold_period_s)
                    normalized_phase = demo_phase / args_cli.demo_fold_period_s
                    home_deg = joint_targets.new_tensor(
                        [0.0, 0.0, 2.0, 0.0, 0.0, 0.0, 100.0]
                    )
                    extended_deg = joint_targets.new_tensor(
                        [0.0, 87.0, 170.0, 0.0, 0.0, 0.0, 100.0]
                    )
                    folded_deg = joint_targets.new_tensor(
                        [0.0, 171.0, 168.0, 2.0, -1.0, 4.0, 100.0]
                    )

                    def _demo_blend(source, destination, segment_phase):
                        weight = 0.5 - 0.5 * math.cos(math.pi * segment_phase)
                        return source + weight * (destination - source)

                    if normalized_phase < 0.125:
                        arm_target_deg = home_deg
                    elif normalized_phase < 0.25:
                        arm_target_deg = _demo_blend(home_deg, extended_deg, (normalized_phase - 0.125) / 0.125)
                    elif normalized_phase < 0.375:
                        arm_target_deg = extended_deg
                    elif normalized_phase < 0.5:
                        arm_target_deg = _demo_blend(extended_deg, folded_deg, (normalized_phase - 0.375) / 0.125)
                    elif normalized_phase < 0.625:
                        arm_target_deg = folded_deg
                    elif normalized_phase < 0.75:
                        arm_target_deg = _demo_blend(folded_deg, extended_deg, (normalized_phase - 0.625) / 0.125)
                    elif normalized_phase < 0.875:
                        arm_target_deg = extended_deg
                    else:
                        arm_target_deg = _demo_blend(extended_deg, home_deg, (normalized_phase - 0.875) / 0.125)
                    arm_target_rad = torch.deg2rad(arm_target_deg)
                    joint_targets[0, sim_arm_joint_ids] = arm_target_rad
                    if policy_step % 100 == 0:
                        print(
                            f">>> [demo_fold_walk] t={sim.current_time:.2f}s "
                            f"phase={normalized_phase:.2f} command=walk",
                            flush=True,
                        )

            if utlidar_state is not None:
                _publish_utlidar_scan(
                    utlidar_state,
                    sim.current_time,
                    policy_step,
                    robot,
                    base_link_idx,
                    wrist_camera_sensor,
                )

            if active_arm_server is not None:
                sim_time_ns = round(sim.current_time * 1_000_000_000)
                forbidden_contact = False
                foot_support_count = 0
                if arm_contact_sensor_runtime is not None:
                    arm_contact_forces = arm_contact_sensor_runtime.data.net_forces_w[
                        0, arm_contact_body_ids
                    ]
                    maximum_arm_contact_force_n = float(
                        torch.linalg.vector_norm(arm_contact_forces, dim=-1).max().item()
                    )
                    if maximum_arm_contact_force_n * sim_cfg.dt > 0.2:
                        arm_forbidden_contact_ticks += 1
                    else:
                        arm_forbidden_contact_ticks = 0
                    forbidden_contact = arm_forbidden_contact_ticks >= 2
                    foot_contact_forces = arm_contact_sensor_runtime.data.net_forces_w[
                        0, foot_contact_body_ids
                    ]
                    foot_support_count = int(
                        (
                            torch.linalg.vector_norm(foot_contact_forces, dim=-1)
                            >= 5.0
                        ).sum().item()
                    )
                measured_sim_rad = (
                    robot.data.joint_pos[0, external_joint_ids]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                measured_external_deg = runtime_sim_rad_to_external_deg(
                    measured_sim_rad,
                )
                measured_external_velocity_rad_s = (
                    robot.data.joint_vel[0, external_joint_ids]
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                gripper_position_base = math_utils.quat_apply_inverse(
                    robot.data.body_quat_w[:, base_link_idx],
                    robot.data.body_pos_w[:, gripper_link_idx]
                    - robot.data.body_pos_w[:, base_link_idx],
                )[0].detach().cpu().numpy()
                authorized_target_deg = active_arm_server.update(
                    sim_time_ns=sim_time_ns,
                    actual_external_deg=measured_external_deg,
                    actual_external_velocity_rad_s=measured_external_velocity_rad_s,
                    forbidden_contact=forbidden_contact,
                    end_effector_position_base=gripper_position_base,
                )
                if authorized_target_deg is not None:
                    joint_targets[:, external_joint_ids] = torch.as_tensor(
                        runtime_external_deg_to_sim_rad(authorized_target_deg),
                        device=sim.device,
                        dtype=joint_targets.dtype,
                    )
                if policy_step % 4 == 0:
                    active_arm_server.publish_joint_state(
                        sim_time_ns,
                        tuple(runtime_external_joint_order),
                        np.deg2rad(measured_external_deg),
                        measured_external_velocity_rad_s,
                    )
                base_position_w = robot.data.body_pos_w[:, base_link_idx]
                base_quaternion_w = robot.data.body_quat_w[:, base_link_idx]
                wrist_position_w, wrist_quaternion_opengl_w = (
                    wrist_camera_sensor._view.get_world_poses()
                )
                wrist_quaternion_ros_w = math_utils.convert_camera_frame_orientation_convention(
                    wrist_quaternion_opengl_w,
                    origin="opengl",
                    target="ros",
                )
                wrist_position_base, wrist_quaternion_base = math_utils.subtract_frame_transforms(
                    base_position_w,
                    base_quaternion_w,
                    wrist_position_w,
                    wrist_quaternion_ros_w,
                )
                if hazard_smolvla_bridge is not None:
                    latest_context = active_arm_server.inspection_context()
                    context_active = bool(
                        latest_context and latest_context.get("active")
                    )
                    next_event_id = (
                        str(latest_context.get("event_id", "")).strip()
                        if context_active
                        else ""
                    )
                    next_target_side = (
                        str(latest_context.get("target_side", "")).strip()
                        if context_active
                        else ""
                    )
                    context_changed = (
                        next_event_id != hazard_smolvla_state["event_id"]
                        or next_target_side != hazard_smolvla_state["target_side"]
                    )
                    if context_active and context_changed:
                        hazard_smolvla_bridge.send_reset()
                        discarded = hazard_smolvla_bridge.drain_actions()
                        hazard_decision_gate.begin(next_event_id, next_target_side)
                        hazard_smolvla_state.update(
                            {
                                "context": dict(latest_context),
                                "event_id": next_event_id,
                                "target_side": next_target_side,
                                "next_obs_time": sim.current_time,
                                "requested_arm_target_deg": None,
                                "applied_arm_target_deg": measured_external_deg.astype(
                                    np.float32
                                ).copy(),
                            }
                        )
                        print(
                            f">>> [smolvla_hazard] lease begin event={next_event_id} "
                            f"side={next_target_side}; discarded_old_outputs={discarded}",
                            flush=True,
                        )
                    elif not context_active and hazard_smolvla_state["event_id"]:
                        hazard_smolvla_bridge.send_reset()
                        discarded = hazard_smolvla_bridge.drain_actions()
                        hazard_decision_gate.clear()
                        hazard_smolvla_state.update(
                            {
                                "context": None,
                                "event_id": None,
                                "target_side": None,
                                "next_obs_time": None,
                                "requested_arm_target_deg": None,
                                "applied_arm_target_deg": None,
                            }
                        )
                        print(
                            f">>> [smolvla_hazard] lease closed; "
                            f"discarded_outputs={discarded}",
                            flush=True,
                        )

                    try:
                        hazard_envelope = hazard_smolvla_bridge.receive_action()
                        if hazard_envelope is not None:
                            if not context_active:
                                raise ValueError(
                                    "policy output arrived without an active inspection lease"
                                )
                            wrist_peek_valid = hazard_wrist_peek_pose_valid(
                                wrist_position_base[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float64),
                                wrist_quaternion_base[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float64),
                                next_target_side,
                                minimum_forward_m=0.22,
                                # This tree has forward-opening T corners. Once
                                # the lens is beyond the entry plane, looking
                                # into the target side is sufficient; teacher
                                # poses put the lens ~8 cm laterally, not 18 cm.
                                minimum_target_lateral_m=0.0,
                            )
                            wrist_entry_clear = hazard_wrist_clears_entry_plane(
                                wrist_position_w[0].detach().cpu().numpy(),
                                latest_context["branch_entry_point_world_m"],
                                latest_context["branch_entry_normal_world"],
                            )
                            wrist_peek_valid = wrist_peek_valid and wrist_entry_clear
                            if policy_step % 200 < 8:
                                print(
                                    f">>> [hazard_trace] event={next_event_id} "
                                    f"peek={wrist_peek_valid} "
                                    f"entry_clear={wrist_entry_clear} "
                                    f"wrist_xyz={wrist_position_base[0].tolist()} "
                                    f"wrist_quat={wrist_quaternion_base[0].tolist()} "
                                    f"measured={np.round(measured_external_deg, 2).tolist()}",
                                    flush=True,
                                )
                            hazard_arm_target_deg, terminal = (
                                hazard_consume_correlated_action(
                                    hazard_envelope,
                                    expected_event_id=next_event_id,
                                    expected_target_side=next_target_side,
                                    gate=hazard_decision_gate,
                                    base_paused=explicit_base_pause,
                                    peek_pose_valid=wrist_peek_valid,
                                )
                            )
                            if not explicit_base_pause:
                                hazard_smolvla_state["requested_arm_target_deg"] = None
                                hazard_smolvla_state["applied_arm_target_deg"] = None
                            else:
                                hazard_smolvla_state["requested_arm_target_deg"] = (
                                    hazard_arm_target_deg
                                )
                            if terminal is not None:
                                active_arm_server.publish_vla_alley_decision(
                                    {
                                        "schema": "binary_alley_vla_decision.v1",
                                        "event_id": terminal.event_id,
                                        "target_side": terminal.target_side,
                                        "signal": terminal.signal,
                                        "raw_decision": terminal.raw_decision,
                                        "stable_samples": terminal.stable_samples,
                                        "peek_validated": True,
                                        "peek_valid_consecutive_frames": (
                                            terminal.peek_valid_consecutive_frames
                                        ),
                                        "base_paused": True,
                                    }
                                )
                                print(
                                    f">>> [smolvla_hazard] terminal event="
                                    f"{terminal.event_id} side={terminal.target_side} "
                                    f"signal={terminal.signal:+d} "
                                    f"raw={terminal.raw_decision:+.3f}",
                                    flush=True,
                                )
                    except (RuntimeError, TypeError, ValueError) as error:
                        hazard_smolvla_state["requested_arm_target_deg"] = None
                        hazard_smolvla_state["rejected_outputs"] += 1
                        rejected_count = hazard_smolvla_state["rejected_outputs"]
                        if rejected_count <= 5 or rejected_count % 25 == 0:
                            print(
                                f">>> [smolvla_hazard] rejected output "
                                f"#{rejected_count}: {error}",
                                flush=True,
                            )

                    requested_hazard_target = hazard_smolvla_state[
                        "requested_arm_target_deg"
                    ]
                    if (
                        context_active
                        and explicit_base_pause
                        and requested_hazard_target is not None
                        and hazard_smolvla_state["event_id"] == next_event_id
                        and hazard_smolvla_state["target_side"] == next_target_side
                    ):
                        previous_hazard_target = hazard_smolvla_state[
                            "applied_arm_target_deg"
                        ]
                        if previous_hazard_target is None:
                            previous_hazard_target = measured_external_deg
                        applied_hazard_target = hazard_slew_arm_target(
                            previous_hazard_target,
                            requested_hazard_target,
                            sim_cfg.dt,
                        )
                        hazard_smolvla_state["applied_arm_target_deg"] = (
                            applied_hazard_target
                        )
                        joint_targets[:, external_joint_ids] = torch.as_tensor(
                            runtime_external_deg_to_sim_rad(
                                applied_hazard_target
                            ),
                            device=sim.device,
                            dtype=joint_targets.dtype,
                        )
                current_root_linear_velocity_w = robot.data.root_lin_vel_w.clone()
                linear_acceleration_w = (
                    current_root_linear_velocity_w - previous_root_linear_velocity_w
                ) / sim_cfg.dt
                previous_root_linear_velocity_w = current_root_linear_velocity_w
                gravity_w = torch.tensor(
                    [[0.0, 0.0, -9.81]],
                    dtype=linear_acceleration_w.dtype,
                    device=linear_acceleration_w.device,
                )
                specific_force_b = math_utils.quat_apply_inverse(
                    robot.data.root_quat_w,
                    linear_acceleration_w - gravity_w,
                )
                active_arm_server.publish_dynamic_truth(
                    sim_time_ns=sim_time_ns,
                    wrist_position_base=wrist_position_base[0].detach().cpu().numpy(),
                    wrist_quaternion_wxyz_base=wrist_quaternion_base[0].detach().cpu().numpy(),
                    base_quaternion_wxyz_world=robot.data.root_quat_w[0].detach().cpu().numpy(),
                    base_angular_velocity_rad_s=robot.data.root_ang_vel_b[0].detach().cpu().numpy(),
                    base_linear_acceleration_m_s2=specific_force_b[0].detach().cpu().numpy(),
                    forbidden_contact=forbidden_contact,
                    foot_support_count=foot_support_count,
                )

            if maze_inspector is not None:
                for name,idx in zip(sim_arm_joint_names,sim_arm_joint_ids):
                    joint_targets[0,idx]=maze_inspector.q[name]
            robot.set_joint_position_target(joint_targets)
            scene.write_data_to_sim()
            # Apply the deterministic carrier after scene.write_data_to_sim(),
            # which otherwise restores the dynamic articulation root.  Repeat
            # on every physics step so contacts cannot accumulate route drift.
            if kinematic_route_pose_target is not None:
                robot.write_root_pose_to_sim(kinematic_route_pose_target)
                robot.write_root_velocity_to_sim(
                    torch.zeros(
                        (1, 6),
                        dtype=kinematic_route_pose_target.dtype,
                        device=sim.device,
                    )
                )
            # This root hold belongs only to the optional scripted-route startup
            # carrier. Explicit NBV pauses are fully physical and must never
            # reach this branch.
            if stationary_base_pose_target is not None:
                robot.write_root_pose_to_sim(stationary_base_pose_target)
                robot.write_root_velocity_to_sim(
                    torch.zeros(
                        (1, 6),
                        dtype=stationary_base_pose_target.dtype,
                        device=sim.device,
                    )
                )
            # SimulationCfg.render_interval must remain 1 in this standalone
            # loop.  Otherwise World.step() advances multiple physics ticks per
            # call and the outer policy_decimation applies a second time (the
            # render_interval=4 failure reduced the 19750 policy to 12.5 Hz).
            sim.step(render=policy_step % args_cli.render_interval == 0)
            scene.update(sim_cfg.dt)
            if maze_display_ui is not None and policy_step % args_cli.render_interval == 0:
                maze_display_ui.update()
            if active_arm_server is not None:
                completion = active_arm_server.pop_completion()
                if completion is not None and completion["result_code"] == 0:
                    wrist_stats = validate_slam_rgbd_frame(
                        "wrist_active_observation",
                        wrist_camera_sensor.data.output,
                        expected_height=WRIST_CAMERA_HEIGHT,
                        expected_width=WRIST_CAMERA_WIDTH,
                    )
                    print(
                        f">>> ACTIVE_GAP_OBSERVATION completion={completion} "
                        f"wrist_depth_valid={wrist_stats[2]}",
                        flush=True,
                    )
                elif completion is not None:
                    print(
                        f">>> ACTIVE_GAP_APPLICATION_FAILED completion={completion}; "
                        "wrist observation not authorized",
                        flush=True,
                    )
            if (
                next_slam_rgbd_validation_step is not None
                and policy_step >= next_slam_rgbd_validation_step
            ):
                front_stats = validate_slam_rgbd_frame(
                    "front",
                    front_camera_sensor.data.output,
                    expected_height=FRONT_CAMERA_HEIGHT,
                    expected_width=FRONT_CAMERA_WIDTH,
                )
                if wrist_camera_sensor is None:
                    print(
                        ">>> SLAM RGB-D validated: front-only "
                        f"depth=[{front_stats[0]:.3f},{front_stats[1]:.3f}]m "
                        f"valid={front_stats[2]}",
                        flush=True,
                    )
                else:
                    wrist_stats = validate_slam_rgbd_frame(
                        "wrist",
                        wrist_camera_sensor.data.output,
                        expected_height=WRIST_CAMERA_HEIGHT,
                        expected_width=WRIST_CAMERA_WIDTH,
                    )
                    print(
                        ">>> SLAM RGB-D validated: "
                        f"wrist_depth=[{wrist_stats[0]:.3f},{wrist_stats[1]:.3f}]m "
                        f"valid={wrist_stats[2]} | "
                        f"front_depth=[{front_stats[0]:.3f},{front_stats[1]:.3f}]m "
                        f"valid={front_stats[2]}",
                        flush=True,
                    )
                next_slam_rgbd_validation_step += SLAM_RGBD_VALIDATION_INTERVAL_STEPS
            if args_cli.state_debug_every > 0 and policy_step == 0:
                print(
                    f">>> [joint_state step=0] pos={robot.data.joint_pos[0, policy_action_joint_ids].detach().cpu().tolist()} "
                    f"vel={robot.data.joint_vel[0, policy_action_joint_ids].detach().cpu().tolist()} "
                    f"torque={robot.data.applied_torque[0, policy_action_joint_ids].detach().cpu().tolist()}",
                    flush=True,
                )
            if args_cli.gait_probe and policy_step % 10 == 0:
                probe_gravity = robot.data.projected_gravity_b[0]
                print('GAIT_PROBE ' + json.dumps({
                    't':round(policy_step * sim_cfg.dt, 3),
                    'phase':gait_probe_command(policy_step * sim_cfg.dt)[0],
                    'command':vel_cmd_b[0].detach().cpu().tolist(),
                    'velocity':robot.data.root_lin_vel_b[0].detach().cpu().tolist(),
                    'angular_velocity':robot.data.root_ang_vel_b[0].detach().cpu().tolist(),
                    'xy':robot.data.root_pos_w[0,:2].detach().cpu().tolist(),
                    'height':float(robot.data.root_pos_w[0,2]),
                    'tilt_deg':math.degrees(math.acos(float(torch.clamp(-probe_gravity[2], -1., 1.)))),
                    'action_max':float(last_action.abs().max()),
                }), flush=True)
            if args_cli.state_debug_every > 0 and policy_step % args_cli.state_debug_every == 0:
                root_pos = robot.data.root_pos_w[0]
                gravity = robot.data.projected_gravity_b[0]
                rtf_wall_elapsed_s = max(time.monotonic() - rtf_wall_start_s, 1.0e-6)
                measured_rtf = max(
                    0.0,
                    (float(sim.current_time) - rtf_sim_start_s) / rtf_wall_elapsed_s,
                )
                print(
                    f">>> [state] step={policy_step} root_z={float(root_pos[2]):.4f} "
                    f"tilt_xy={float(torch.linalg.vector_norm(gravity[:2])):.4f} "
                    f"speed_xy={float(torch.linalg.vector_norm(robot.data.root_lin_vel_b[0, :2])):.4f} "
                    f"command={vel_cmd_b[0].detach().cpu().tolist()} "
                    f"action_max={float(last_action.abs().max()):.4f}"
                    f" vel_b_x={float(robot.data.root_lin_vel_b[0, 0]):.4f}"
                    f" pos_xy=[{float(root_pos[0]):.3f},{float(root_pos[1]):.3f}]"
                    f" rtf={measured_rtf:.3f}"
                    f" supervisor_cmd_ready={external_supervisor_command_available}"
                    f" supervisor_motion_started={external_supervisor_motion_started}"
                    f" base_pause={explicit_base_pause}"
                    f" root_hold={stationary_base_hold_active}"
                    f" quat={robot.data.root_quat_w[0].detach().cpu().tolist()}",
                    flush=True,
                )
            if args_cli.camera_pose_debug_every > 0 and policy_step % args_cli.camera_pose_debug_every == 0:
                print_camera_pose_debug(
                    policy_step,
                    robot,
                    gripper_link_idx,
                    base_link_idx,
                    wrist_camera_local_pos,
                    wrist_camera_local_rot,
                    front_camera_local_pos,
                    front_camera_local_rot,
                )
            # 햅틱 피드백: 그리퍼가 닫으려 하는데 물체에 막혀 있을 때만 파지 토크 전송.
            # 조건 1: target < actual (닫기 방향)
            # 조건 2: 속도 거의 0 (물체에 막힘 = 정지)
            # 두 조건 모두 충족 시 stiffness × 위치오차 = 파지 압력(N·m)
            if haptic_pub is not None:
                _gt = float(joint_targets[0, gripper_joint_idx].detach().cpu().item())
                _ga = float(robot.data.joint_pos[0, gripper_joint_idx].detach().cpu().item())
                _gv = float(robot.data.joint_vel[0, gripper_joint_idx].detach().cpu().item())
                _grip_closing = _gt < _ga
                _grip_slow = abs(_gv) < 0.1
                if _grip_closing and _grip_slow:
                    _grip_torque = min(200.0 * (_ga - _gt), 5.0)
                    if _grip_torque > args_cli.haptic_min_torque:
                        _gforce = min(_grip_torque / args_cli.haptic_max_torque, 1.0)
                    else:
                        _gforce = 0.0
                else:
                    _grip_torque = 0.0
                    _gforce = 0.0
                if args_cli.haptic_debug_every > 0 and policy_step % args_cli.haptic_debug_every == 0:
                    print(
                        f">>> [haptic] force={_gforce:.3f} torque={_grip_torque:.2f}N·m "
                        f"closing={_grip_closing} slow={_grip_slow} "
                        f"tgt={_gt:.3f} act={_ga:.3f} vel={_gv:.4f}",
                        flush=True,
                    )
                try:
                    haptic_pub.send(struct.pack('f', _gforce), flags=zmq.NOBLOCK)
                except zmq.Again:
                    pass
            nonfinite_state = first_nonfinite_state(robot, sim_arm_joint_ids, gripper_link_idx)
            if nonfinite_state is not None:
                if not nonfinite_stop_reported:
                    print(
                        f">>> SAFETY STOP: non-finite robot state detected after sim.step in {nonfinite_state}; "
                        "closing before camera viewport update.",
                        flush=True,
                    )
                    nonfinite_stop_reported = True
                arm_action_enabled = False
                break
            if locomotion_failure_detector is not None:
                maximum_leg_tracking_error_rad = float(
                    torch.max(
                        torch.abs(
                            robot.data.joint_pos[0, leg_joint_ids]
                            - joint_targets[0, leg_joint_ids]
                        )
                    ).item()
                )
                failure_report = locomotion_failure_detector.update(
                    robot.data.body_pos_w[0, base_link_idx].detach().cpu().numpy(),
                    robot.data.body_quat_w[0, base_link_idx].detach().cpu().numpy(),
                    robot.data.body_pos_w[0, _foot_indices].detach().cpu().numpy(),
                    maximum_leg_tracking_error_rad,
                )
                if failure_report is not None:
                    locomotion_failure_reason = failure_report.reason
                    failure_root_pos = robot.data.root_pos_w[0].detach().cpu()
                    failure_root_quat = robot.data.root_quat_w[0].detach().cpu()
                    fw, fx, fy, fz = (float(value.item()) for value in failure_root_quat)
                    failure_root_yaw = math.atan2(
                        2.0 * (fw * fz + fx * fy),
                        1.0 - 2.0 * (fy * fy + fz * fz),
                    )
                    failure_target_waypoint = None
                    failure_distance_to_waypoint_m = None
                    if (
                        scripted_route_waypoints is not None
                        and scripted_route_index < len(scripted_route_waypoints)
                    ):
                        route_target = scripted_route_waypoints[scripted_route_index]
                        failure_target_waypoint = [
                            float(route_target[0]),
                            float(route_target[1]),
                        ]
                        failure_distance_to_waypoint_m = math.hypot(
                            failure_target_waypoint[0] - float(failure_root_pos[0]),
                            failure_target_waypoint[1] - float(failure_root_pos[1]),
                        )
                    print(
                        ">>> LOCOMOTION_FAILURE "
                        + json.dumps(
                            {
                                "reason": failure_report.reason,
                                "metrics": failure_report.metrics,
                                "policy_step": policy_step,
                                "root_position_world_m": [
                                    float(value.item()) for value in failure_root_pos
                                ],
                                "root_yaw_deg": math.degrees(failure_root_yaw),
                                "scripted_route_index": scripted_route_index,
                                "target_waypoint": failure_target_waypoint,
                                "distance_to_waypoint_m": failure_distance_to_waypoint_m,
                                "command_body_mps_radps": [
                                    float(value) for value in vel_cmd_b[0].detach().cpu().tolist()
                                ],
                                "requested_command_body_mps_radps": [
                                    float(value)
                                    for value in requested_vel_cmd_b[0].detach().cpu().tolist()
                                ],
                                "stationary_base_hold_active": stationary_base_hold_active,
                                "last_action_max_abs": float(last_action.abs().max().item()),
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
                    arm_action_enabled = False
                    break
            if gr00t_bridge is not None and args_cli.enable_cameras:
                wrist_out = scene.sensors["wrist_camera"].data.output
                if "rgb" in wrist_out:
                    wrist_rgb = wrist_out["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                    joint_pos_np = sim_rad_to_external_deg(
                        ROBOT_MODEL_PROFILE,
                        robot.data.joint_pos[0, external_joint_ids].detach().cpu().numpy().astype(np.float32),
                    )
                    gr00t_bridge.publish_observation(wrist_rgb, joint_pos_np)
                    if latest_gr00t_action_deg is not None and args_cli.gr00t_apply_actions and policy_step % 20 == 0:
                        joint_err_deg = latest_gr00t_action_deg - joint_pos_np
                        gripper_pos_w = robot.data.body_pos_w[0, gripper_link_idx].detach().cpu().numpy().astype(np.float32)
                        print(f">>> SO-Arm current deg: {joint_pos_np}")
                        print(f">>> SO-Arm target-current deg err: {joint_err_deg}")
                        print(f">>> gripper_link world pos: {gripper_pos_w}")
            if (
                hazard_smolvla_bridge is not None
                and explicit_base_pause
                and hazard_smolvla_state["event_id"]
                and hazard_smolvla_state["context"] is not None
            ):
                sim_time = sim.current_time
                next_obs_time = hazard_smolvla_state["next_obs_time"]
                if next_obs_time is None or sim_time + 1.0e-9 >= next_obs_time:
                    wrist_out = (
                        wrist_camera_sensor.data.output
                        if wrist_camera_sensor is not None
                        else {}
                    )
                    front_out = (
                        front_camera_sensor.data.output
                        if front_camera_sensor is not None
                        else {}
                    )
                    if "rgb" in wrist_out and "rgb" in front_out:
                        wrist_rgb = (
                            wrist_out["rgb"][0]
                            .detach()
                            .cpu()
                            .numpy()[:, :, :3]
                            .astype(np.uint8)
                        )
                        front_rgb = (
                            front_out["rgb"][0]
                            .detach()
                            .cpu()
                            .numpy()[:, :, :3]
                            .astype(np.uint8)
                        )
                        measured_hazard_state_deg = runtime_sim_rad_to_external_deg(
                            robot.data.joint_pos[0, external_joint_ids]
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32)
                        ).astype(np.float32)
                        hazard_smolvla_bridge.publish_observation(
                            front_rgb,
                            wrist_rgb,
                            measured_hazard_state_deg,
                            hazard_task_for_side_runtime(
                                hazard_smolvla_state["target_side"]
                            ),
                            hazard_smolvla_state["target_side"],
                            hazard_smolvla_state["event_id"],
                        )
                        if next_obs_time is None:
                            hazard_smolvla_state["next_obs_time"] = (
                                sim_time + hazard_smolvla_obs_dt
                            )
                        else:
                            hazard_smolvla_state["next_obs_time"] = max(
                                next_obs_time + hazard_smolvla_obs_dt,
                                sim_time + hazard_smolvla_obs_dt,
                            )
            if smolvla_bridge is not None and smolvla_state["active"] and args_cli.enable_cameras:
                sim_time = sim.current_time
                next_obs_time = smolvla_state["next_obs_time"]
                if next_obs_time is None or sim_time + 1.0e-9 >= next_obs_time:
                    wrist_out = wrist_camera_sensor.data.output if wrist_camera_sensor is not None else {}
                    front_out = front_camera_sensor.data.output if front_camera_sensor is not None else {}
                    if "rgb" in wrist_out and "rgb" in front_out:
                        wrist_rgb = wrist_out["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                        front_rgb = front_out["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                        joint_pos_np = sim_rad_to_external_deg(
                            ROBOT_MODEL_PROFILE,
                            robot.data.joint_pos[0, external_joint_ids].detach().cpu().numpy().astype(np.float32),
                        )
                        smolvla_bridge.publish_observation(front_rgb, wrist_rgb, joint_pos_np)
                        if next_obs_time is None:
                            smolvla_state["next_obs_time"] = sim_time + smolvla_obs_dt
                        else:
                            smolvla_state["next_obs_time"] = max(next_obs_time + smolvla_obs_dt, sim_time + smolvla_obs_dt)
            if (
                latest_leader_action_deg is not None
                and args_cli.leader_auto
                and args_cli.leader_action_log_every > 0
                and policy_step % args_cli.leader_action_log_every == 0
            ):
                joint_pos_np = sim_arm_rad_to_leader_action_deg(
                    robot.data.joint_pos[0, sim_arm_joint_ids].detach().cpu().numpy().astype(np.float32)
                )
                joint_err_deg = latest_leader_action_deg - joint_pos_np
                gripper_pos_w = robot.data.body_pos_w[0, gripper_link_idx].detach().cpu().numpy().astype(np.float32)
                print(f">>> SO-Arm current deg: {joint_pos_np}")
                print(f">>> SO-Arm target-current deg err: {joint_err_deg}")
                print(f">>> gripper_link world pos: {gripper_pos_w}")
            if args_cli.camera_qa and camera_qa_out_dir is not None and wrist_camera_sensor is not None:
                final_camera_qa_step = policy_step + 1 >= args_cli.camera_qa_steps
                current_root_x = float(robot.data.root_pos_w[0, 0].detach().cpu().item())
                forward_travel = current_root_x - camera_qa_start_x
                if not camera_qa_travel_reached:
                    if forward_travel >= args_cli.camera_qa_min_travel_m:
                        camera_qa_travel_reached = True
                        print(
                            f">>> [camera_qa] forward travel {forward_travel:.3f}m reached threshold "
                            f"{args_cli.camera_qa_min_travel_m}m, enabling capture",
                            flush=True,
                        )
                    else:
                        if policy_step % 100 == 0:
                            print(f">>> [camera_qa] warmup step={policy_step} travel={forward_travel:.3f}m", flush=True)
                travel_ok = camera_qa_travel_reached
                if travel_ok and (policy_step % 300 == 0 or final_camera_qa_step):
                    from PIL import Image

                    wrist_rgb = wrist_camera_sensor.data.output["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                    Image.fromarray(wrist_rgb).save(camera_qa_out_dir / f"wrist_{policy_step:05d}.png")
                    if front_camera_sensor is not None and "rgb" in front_camera_sensor.data.output:
                        front_rgb = front_camera_sensor.data.output["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                        Image.fromarray(front_rgb).save(camera_qa_out_dir / f"front_{policy_step:05d}.png")

                    axis_forward = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32, device=sim.device)
                    axis_up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float32, device=sim.device)
                    opengl_forward = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=sim.device)
                    opengl_up = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=sim.device)

                    wrist_parent_pos = robot.data.body_state_w[:, gripper_link_idx, :3]
                    wrist_parent_quat = robot.data.body_state_w[:, gripper_link_idx, 3:7]
                    wrist_world_pos, wrist_opengl_quat = math_utils.combine_frame_transforms(
                        wrist_parent_pos,
                        wrist_parent_quat,
                        wrist_camera_local_pos,
                        wrist_camera_local_rot,
                    )
                    wrist_world_quat = math_utils.convert_camera_frame_orientation_convention(
                        wrist_opengl_quat,
                        origin="opengl",
                        target="world",
                    )
                    world_forward_w = math_utils.quat_apply(wrist_world_quat, axis_forward)
                    world_up_w = math_utils.quat_apply(wrist_world_quat, axis_up)
                    optical_forward_w = math_utils.quat_apply(wrist_opengl_quat, opengl_forward)
                    optical_up_w = math_utils.quat_apply(wrist_opengl_quat, opengl_up)
                    forward_dot = torch.sum(
                        torch.nn.functional.normalize(optical_forward_w, dim=-1)
                        * torch.nn.functional.normalize(world_forward_w, dim=-1),
                        dim=-1,
                    )
                    root_pos_w = robot.data.root_pos_w[0].detach().cpu().tolist()
                    qa_report = {
                        "policy_step": policy_step,
                        "forward_velocity_command": args_cli.camera_qa_forward_vel,
                        "root_pos_w": root_pos_w,
                        "gripper_link_pos_w": wrist_parent_pos[0].detach().cpu().tolist(),
                        "wrist_camera_pos_w": wrist_world_pos[0].detach().cpu().tolist(),
                        "world_plus_x_forward_w": world_forward_w[0].detach().cpu().tolist(),
                        "world_convention_plus_x_forward_w": world_forward_w[0].detach().cpu().tolist(),
                        "world_convention_plus_z_up_w": world_up_w[0].detach().cpu().tolist(),
                        "opengl_optical_minus_z_forward_w": optical_forward_w[0].detach().cpu().tolist(),
                        "opengl_plus_y_up_w": optical_up_w[0].detach().cpu().tolist(),
                        "optical_forward_dot_expected": float(forward_dot[0].detach().cpu().item()),
                        "passes_forward_check": bool(forward_dot[0].detach().cpu().item() > 0.99),
                    }
                    with (camera_qa_out_dir / f"report_{policy_step:05d}.json").open("w", encoding="utf-8") as report_file:
                        json.dump(qa_report, report_file, indent=2)
                    print(
                        f">>> [camera_qa] step={policy_step} root_pos={root_pos_w} "
                        f"forward_dot={qa_report['optical_forward_dot_expected']:.6f} "
                        f"pass={qa_report['passes_forward_check']}",
                        flush=True,
                    )
            # 데이터 수집: collect_fps 기준으로 wrist/front RGB + SO-Arm joint deg + leader action deg 기록
            if collector is not None:
                sim_time = sim.current_time
                if collect_state["toggle_requested"]:
                    collect_state["toggle_requested"] = False
                    if collect_state["active"]:
                        if _finish_collect_episode("RIGHT stop"):
                            break
                    elif collect_ep_count < args_cli.max_episodes:
                        if collect_state["needs_reset"]:
                            _reset_collect_scene_for_next_episode()
                            collect_state["needs_reset"] = False
                            collect_frame_count = 0
                            collect_next_time = None
                            collect_episode_start_time = None
                            print(
                                f">>> [collect] RIGHT RESET complete; RIGHT again starts episode {collect_ep_count + 1}/{args_cli.max_episodes}",
                                flush=True,
                            )
                        else:
                            collect_state["active"] = True
                            collect_frame_count = 0
                            collect_next_time = sim_time
                            collect_episode_start_time = sim_time
                            print(
                                f">>> [collect] RIGHT START episode {collect_ep_count + 1}/{args_cli.max_episodes}",
                                flush=True,
                            )
                    else:
                        print(">>> [collect] RIGHT ignored: max_episodes already reached", flush=True)

                if collect_state["active"] and latest_leader_action_deg is not None:
                    if collect_next_time is None:
                        collect_next_time = sim_time
                    if collect_episode_start_time is None:
                        collect_episode_start_time = sim_time
                    collect_due = (
                        policy_step % args_cli.collect_every == 0
                        if args_cli.collect_every > 0
                        else sim_time + 1.0e-9 >= collect_next_time
                    )
                    if collect_due and args_cli.enable_cameras:
                        _w = scene.sensors["wrist_camera"].data.output
                        _f = scene.sensors["front_camera"].data.output
                        if "rgb" in _w:
                            _wrist_rgb = _w["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                            _front_rgb = (
                                _f["rgb"][0].detach().cpu().numpy()[:, :, :3].astype(np.uint8)
                                if "rgb" in _f else None
                            )
                            _joint_deg = sim_arm_rad_to_leader_action_deg(
                                robot.data.joint_pos[0, sim_arm_joint_ids].detach().cpu().numpy().astype(np.float32)
                            )
                            collector.record_frame(
                                _wrist_rgb,
                                _joint_deg,
                                latest_leader_action_deg.copy(),
                                sim_time - collect_episode_start_time,
                                front_rgb=_front_rgb,
                            )
                            collect_frame_count += 1
                            if args_cli.collect_every <= 0:
                                while collect_next_time <= sim_time + 1.0e-9:
                                    collect_next_time += collect_dt
                    if collect_frame_count >= args_cli.episode_len:
                        if _finish_collect_episode("auto length"):
                            break

            # One supervisor-selected candidate alley per episode. Human mode
            # receives H/S from the keyboard; scripted NBV mode receives the
            # correlated wrist-RGB teacher label over ROS.
            if hazard_collector is not None:
                sim_time = sim.current_time
                if args_cli.collect_hazard_nbv_teacher:
                    for teacher_label in active_arm_server.drain_nbv_teacher_labels():
                        hazard_collect_state["pending_teacher_labels"][
                            str(teacher_label["event_id"])
                        ] = teacher_label
                latest_context = active_arm_server.inspection_context()
                # Couple the line guard to the collector's currently open
                # episode, not merely the latest ROS context. The latter can
                # remain latched from the previous stage while the next stage
                # is entering base_pause, which would compare against the
                # wrong branch line.
                body_guard_context = hazard_collect_state["context"]
                # The context can remain latched for a few ROS cycles after
                # the arm has returned HOME.  Guard only the true stopped-arm
                # inspection lease; crossing the line after base_pause=false
                # is normal safe-route travel, not an inspection violation.
                if (
                    hazard_collector.active
                    and
                    explicit_base_pause
                    and body_guard_context
                    and body_guard_context.get("active")
                ):
                    base_position_world_np = (
                        robot.data.body_pos_w[0, base_link_idx]
                        .detach()
                        .cpu()
                        .numpy()
                        .astype(np.float64)
                    )
                    branch_entry_point_world_m = np.asarray(
                        body_guard_context["branch_entry_point_world_m"],
                        dtype=np.float64,
                    )
                    branch_entry_normal_world = np.asarray(
                        body_guard_context["branch_entry_normal_world"],
                        dtype=np.float64,
                    )
                    body_overshoot_limit_m = float(
                        body_guard_context.get(
                            "body_inspection_overshoot_limit_m",
                            BODY_BRANCH_ENTRY_OVERSHOOT_LIMIT_M,
                        )
                    )
                    body_overshoot_m = body_branch_entry_overshoot_m(
                        base_position_world_np,
                        branch_entry_point_world_m,
                        branch_entry_normal_world,
                    )
                    if body_crossed_branch_entry_during_inspection(
                        base_position_world_np,
                        branch_entry_point_world_m,
                        branch_entry_normal_world,
                        limit_m=body_overshoot_limit_m,
                    ):
                        rejected_path = None
                        if hazard_collector.active:
                            rejected_path = hazard_collector.discard_episode(
                                "go2_base_crossed_branch_entry_by_5cm_during_inspection"
                            )
                        locomotion_failure_reason = (
                            "go2_base_crossed_branch_entry_during_inspection"
                        )
                        print(
                            ">>> SAFETY STOP: Go2 base center crossed the active "
                            f"branch-entry line by {body_overshoot_m:.3f} m "
                            f"(limit={body_overshoot_limit_m:.3f} m); "
                            f"arm position is not used; rejected={rejected_path}",
                            flush=True,
                        )
                        break
                # A direct left-to-right scripted scan can publish the next
                # context immediately after the previous terminal label. Keep
                # recording correlated to the active event until that queued
                # label is committed, then pick up the newest context.
                context = (
                    hazard_collect_state["context"]
                    if args_cli.collect_hazard_nbv_teacher
                    and hazard_collector.active
                    and hazard_collect_state["context"] is not None
                    else latest_context
                )
                context_active = bool(context and context.get("active"))
                event_id = str(context.get("event_id", "")) if context_active else ""
                event_completed = event_id in hazard_collect_state["completed_event_ids"]

                if context_active and not event_completed:
                    context_changed = event_id != hazard_collect_state["event_id"]
                    if context_changed and hazard_collector.active:
                        hazard_collector.discard_episode("inspection_context_changed")
                    if context_changed:
                        target_side = str(context["target_side"])
                        task = hazard_task_for_side(target_side)
                        hazard_collect_state.update(
                            {
                                "context": dict(context),
                                "event_id": event_id,
                                "terminal_signal": 0,
                                "terminal_start_time": None,
                                "episode_start_time": None,
                                "next_sample_time": None,
                                "peek_valid_samples": 0,
                                "peek_validated_samples": 0,
                                "latest_peek_valid": False,
                            }
                        )
                        print(
                            f">>> [hazard_collect] READY event={event_id} "
                            f"task={task!r}; "
                            + (
                                "press R to start recording; H/S remains locked until PEEK READY"
                                if args_cli.collect_hazard_vla
                                else "scripted recording starts automatically"
                            ),
                            flush=True,
                        )
                        if args_cli.collect_hazard_nbv_teacher:
                            hazard_collect_state["label_request"] = "start"

                    label_request = hazard_collect_state["label_request"]
                    hazard_collect_state["label_request"] = None
                    if label_request == "start":
                        if hazard_collector.active:
                            print(
                                ">>> [hazard_collect] R ignored: recording is already active",
                                flush=True,
                            )
                        else:
                            active_context = dict(hazard_collect_state["context"])
                            target_side = str(active_context["target_side"])
                            task = hazard_task_for_side(target_side)
                            hazard_collector.start_episode(
                                target_side=target_side,
                                opening_case=str(active_context["opening_case"]),
                                task=task,
                                episode_metadata={
                                    "event_id": event_id,
                                    "stage": int(active_context.get("stage", 0)),
                                    "collection_lap": int(
                                        active_context.get("collection_lap", 0)
                                    ),
                                    "inspection_phase": str(
                                        active_context.get("inspection_phase", "")
                                    ),
                                    "requires_wrist_peek_pose": bool(
                                        active_context.get(
                                            "requires_wrist_peek_pose",
                                            False,
                                        )
                                    ),
                                    "minimum_peek_valid_frames": int(
                                        active_context.get(
                                            "minimum_peek_valid_frames",
                                            PEEK_MINIMUM_CONSECUTIVE_FRAMES,
                                        )
                                    ),
                                },
                            )
                            hazard_collect_state["terminal_signal"] = 0
                            hazard_collect_state["terminal_start_time"] = None
                            hazard_collect_state["episode_start_time"] = sim_time
                            hazard_collect_state["next_sample_time"] = sim_time
                            hazard_collect_state["peek_valid_samples"] = 0
                            hazard_collect_state["peek_validated_samples"] = 0
                            hazard_collect_state["latest_peek_valid"] = False
                            print(
                                f">>> [hazard_collect] START event={event_id}",
                                flush=True,
                            )
                    if label_request == "discard":
                        if hazard_collector.active:
                            rejected_path = hazard_collector.discard_episode(
                                "human_teacher_discarded_bad_view"
                            )
                            print(
                                f">>> [hazard_collect] DISCARDED -> {rejected_path}",
                                flush=True,
                            )
                        hazard_collect_state["terminal_signal"] = 0
                        hazard_collect_state["terminal_start_time"] = None
                        hazard_collect_state["episode_start_time"] = None
                        hazard_collect_state["next_sample_time"] = None
                        hazard_collect_state["peek_valid_samples"] = 0
                        hazard_collect_state["peek_validated_samples"] = 0
                        hazard_collect_state["latest_peek_valid"] = False
                        print(
                            ">>> [hazard_collect] READY after discard; press R to retry",
                            flush=True,
                        )
                    if label_request in (
                        HAZARD_TEACHER_DECISION_HAZARD,
                        HAZARD_TEACHER_DECISION_SAFE,
                    ):
                        if not hazard_collector.active:
                            print(
                                ">>> [hazard_collect] H/S ignored: press R to start first",
                                flush=True,
                            )
                        elif (
                            hazard_collect_state["peek_valid_samples"]
                            < PEEK_MINIMUM_CONSECUTIVE_FRAMES
                        ):
                            print(
                                ">>> [hazard_collect] H/S ignored: wrist camera has not "
                                "reached and faced into the requested alley for "
                                f"{PEEK_MINIMUM_CONSECUTIVE_FRAMES} consecutive frames "
                                f"(current={hazard_collect_state['peek_valid_samples']})",
                                flush=True,
                            )
                        elif hazard_collect_state["terminal_signal"] == 0:
                            hazard_collect_state["terminal_signal"] = int(label_request)
                            hazard_collect_state["terminal_start_time"] = sim_time
                            hazard_collect_state["peek_validated_samples"] = int(
                                hazard_collect_state["peek_valid_samples"]
                            )
                            print(
                                f">>> [hazard_collect] terminal label="
                                f"{'HAZARD' if label_request < 0 else 'SAFE'}; "
                                "hold pose briefly",
                                flush=True,
                            )

                    sample_due = bool(
                        hazard_collector.active
                        and explicit_base_pause
                        and sim_time + 1.0e-9
                        >= hazard_collect_state["next_sample_time"]
                    )
                    if sample_due:
                        wrist_output = scene.sensors["wrist_camera"].data.output
                        front_output = scene.sensors["front_camera"].data.output
                        if "rgb" in wrist_output and "rgb" in front_output:
                            active_context = dict(hazard_collect_state["context"])
                            target_side = str(active_context["target_side"])
                            wrist_position_base_np = (
                                wrist_position_base[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float64)
                            )
                            wrist_quaternion_base_np = (
                                wrist_quaternion_base[0]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float64)
                            )
                            base_position_world_np = (
                                robot.data.body_pos_w[0, base_link_idx]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float64)
                            )
                            branch_entry_point_world_m = np.asarray(
                                active_context["branch_entry_point_world_m"],
                                dtype=np.float64,
                            )
                            branch_entry_normal_world = np.asarray(
                                active_context["branch_entry_normal_world"],
                                dtype=np.float64,
                            )
                            body_overshoot_m = body_branch_entry_overshoot_m(
                                base_position_world_np,
                                branch_entry_point_world_m,
                                branch_entry_normal_world,
                            )
                            peek_pose_valid = wrist_peek_pose_valid(
                                wrist_position_base_np,
                                wrist_quaternion_base_np,
                                target_side,
                                minimum_forward_m=float(
                                    active_context.get(
                                        "minimum_wrist_forward_m",
                                        0.22,
                                    )
                                ),
                            )
                            hazard_collect_state["latest_peek_valid"] = bool(
                                peek_pose_valid
                            )
                            previous_peek_valid_samples = int(
                                hazard_collect_state["peek_valid_samples"]
                            )
                            if peek_pose_valid:
                                hazard_collect_state["peek_valid_samples"] += 1
                            else:
                                hazard_collect_state["peek_valid_samples"] = 0
                            if (
                                previous_peek_valid_samples
                                < PEEK_MINIMUM_CONSECUTIVE_FRAMES
                                <= hazard_collect_state["peek_valid_samples"]
                            ):
                                print(
                                    f">>> [hazard_collect] PEEK READY "
                                    f"event={event_id} side={target_side}; "
                                    "H/S label is now unlocked",
                                    flush=True,
                                )
                            wrist_optical_forward_base_np = wrist_optical_forward_base(
                                wrist_quaternion_base_np
                            )
                            wrist_rgb = (
                                wrist_output["rgb"][0]
                                .detach()
                                .cpu()
                                .numpy()[:, :, :3]
                                .astype(np.uint8)
                            )
                            front_rgb = (
                                front_output["rgb"][0]
                                .detach()
                                .cpu()
                                .numpy()[:, :, :3]
                                .astype(np.uint8)
                            )
                            measured_sim_rad = (
                                robot.data.joint_pos[0, external_joint_ids]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                            applied_target_sim_rad = (
                                joint_targets[0, external_joint_ids]
                                .detach()
                                .cpu()
                                .numpy()
                                .astype(np.float32)
                            )
                            measured_external_deg = runtime_sim_rad_to_external_deg(
                                measured_sim_rad
                            ).astype(np.float32)
                            applied_external_deg = runtime_sim_rad_to_external_deg(
                                applied_target_sim_rad
                            ).astype(np.float32)
                            hazard_collector.record_frame(
                                wrist_rgb=wrist_rgb,
                                front_rgb=front_rgb,
                                joint_pos_deg=measured_external_deg,
                                applied_arm_target_deg=applied_external_deg,
                                decision=hazard_collect_state["terminal_signal"],
                                sim_time=(
                                    sim_time
                                    - hazard_collect_state["episode_start_time"]
                                ),
                                frame_metadata={
                                    "event_id": event_id,
                                    "base_paused": True,
                                    "go2_base_position_world_m": (
                                        base_position_world_np.tolist()
                                    ),
                                    "branch_entry_point_world_m": (
                                        branch_entry_point_world_m.tolist()
                                    ),
                                    "branch_entry_normal_world": (
                                        branch_entry_normal_world.tolist()
                                    ),
                                    "go2_base_branch_overshoot_m": float(
                                        body_overshoot_m
                                    ),
                                    "go2_base_branch_overshoot_limit_m": float(
                                        active_context.get(
                                            "body_inspection_overshoot_limit_m",
                                            BODY_BRANCH_ENTRY_OVERSHOOT_LIMIT_M,
                                        )
                                    ),
                                    "peek_pose_valid": bool(peek_pose_valid),
                                    "peek_valid_consecutive_frames": int(
                                        hazard_collect_state["peek_valid_samples"]
                                    ),
                                    "wrist_position_base_m": (
                                        wrist_position_base_np.tolist()
                                    ),
                                    "wrist_optical_forward_base": (
                                        wrist_optical_forward_base_np.tolist()
                                    ),
                                },
                            )
                            while (
                                hazard_collect_state["next_sample_time"]
                                <= sim_time + 1.0e-9
                            ):
                                hazard_collect_state["next_sample_time"] += hazard_collect_dt

                    if args_cli.collect_hazard_nbv_teacher and hazard_collector.active:
                        teacher_label = hazard_collect_state["pending_teacher_labels"].pop(
                            str(hazard_collect_state["event_id"]),
                            None,
                        )
                        if teacher_label is not None:
                            active_event_id = str(hazard_collect_state["event_id"])
                            signal = int(teacher_label["signal"])
                            peek_validated_samples = int(
                                hazard_collect_state["peek_valid_samples"]
                            )
                            joint_limit_error = hazard_collector.joint_limit_error
                            requires_peek_proof = bool(
                                hazard_collect_state["context"].get(
                                    "requires_wrist_peek_pose",
                                    False,
                                )
                            )
                            if (
                                requires_peek_proof
                                and
                                peek_validated_samples
                                < PEEK_MINIMUM_CONSECUTIVE_FRAMES
                            ):
                                rejected_path = hazard_collector.discard_episode(
                                    "scripted_teacher_missing_wrist_peek_pose_attestation"
                                )
                                print(
                                    f">>> [hazard_collect] REJECTED scripted event="
                                    f"{active_event_id} missing peek proof -> {rejected_path}",
                                    flush=True,
                                )
                            elif joint_limit_error is not None:
                                rejected_path = hazard_collector.discard_episode(
                                    f"joint_limit_contract_violation: {joint_limit_error}"
                                )
                                print(
                                    f">>> [hazard_collect] REJECTED scripted event="
                                    f"{active_event_id} joint limit -> {rejected_path}",
                                    flush=True,
                                )
                            else:
                                hazard_collector.label_last_frame(
                                    signal,
                                    frame_metadata={
                                        "teacher_schema": teacher_label["schema"],
                                        "teacher_signal_source": teacher_label.get(
                                            "signal_source"
                                        ),
                                        "teacher_red_ratio": teacher_label.get("red_ratio"),
                                        "teacher_samples": teacher_label.get("samples"),
                                        "scripted_scan_completed": teacher_label.get(
                                            "scripted_scan_completed"
                                        ),
                                    },
                                )
                                accepted_path = hazard_collector.finish_episode()
                                hazard_collect_state["completed_event_ids"].add(
                                    active_event_id
                                )
                                print(
                                    f">>> [hazard_collect] ACCEPTED scripted "
                                    f"{'HAZARD' if signal < 0 else 'SAFE'} "
                                    f"event={active_event_id} -> {accepted_path}",
                                    flush=True,
                                )
                            hazard_collect_state["context"] = None
                            hazard_collect_state["event_id"] = None
                            hazard_collect_state["terminal_signal"] = 0
                            hazard_collect_state["terminal_start_time"] = None
                            hazard_collect_state["episode_start_time"] = None
                            hazard_collect_state["next_sample_time"] = None
                            hazard_collect_state["peek_valid_samples"] = 0
                            hazard_collect_state["peek_validated_samples"] = 0
                            hazard_collect_state["latest_peek_valid"] = False

                    terminal_start = hazard_collect_state["terminal_start_time"]
                    if (
                        args_cli.collect_hazard_vla
                        and
                        terminal_start is not None
                        and sim_time - terminal_start
                        >= args_cli.hazard_terminal_hold_s
                        and hazard_collector.active
                    ):
                        joint_limit_error = hazard_collector.joint_limit_error
                        peek_validated_samples = int(
                            hazard_collect_state["peek_validated_samples"]
                        )
                        if peek_validated_samples < PEEK_MINIMUM_CONSECUTIVE_FRAMES:
                            rejected_path = hazard_collector.discard_episode(
                                "missing_wrist_peek_pose_attestation"
                            )
                            print(
                                f">>> [hazard_collect] REJECTED missing peek proof "
                                f"-> {rejected_path}",
                                flush=True,
                            )
                        elif joint_limit_error is not None:
                            rejected_path = hazard_collector.discard_episode(
                                f"joint_limit_contract_violation: {joint_limit_error}"
                            )
                            print(
                                f">>> [hazard_collect] REJECTED joint-limit episode "
                                f"-> {rejected_path}; return the arm inside its "
                                "allowed range and press R to retry",
                                flush=True,
                            )
                        else:
                            accepted_path = hazard_collector.finish_episode()
                            signal = int(hazard_collect_state["terminal_signal"])
                            hazard_collect_state["completed_event_ids"].add(event_id)
                            active_context = dict(hazard_collect_state["context"])
                            active_arm_server.publish_human_alley_label(
                                {
                                    "schema": "binary_alley_human_label.v1",
                                    "event_id": event_id,
                                    "target_side": active_context["target_side"],
                                    "opening_case": active_context["opening_case"],
                                    "stage": active_context.get("stage"),
                                    "signal": signal,
                                    "peek_validated": True,
                                    "peek_valid_consecutive_frames": (
                                        peek_validated_samples
                                    ),
                                }
                            )
                            print(
                                f">>> [hazard_collect] ACCEPTED "
                                f"{'HAZARD' if signal < 0 else 'SAFE'} -> {accepted_path}",
                                flush=True,
                            )
                        hazard_collect_state["context"] = None
                        hazard_collect_state["event_id"] = None
                        hazard_collect_state["terminal_signal"] = 0
                        hazard_collect_state["terminal_start_time"] = None
                        hazard_collect_state["episode_start_time"] = None
                        hazard_collect_state["next_sample_time"] = None
                        hazard_collect_state["peek_valid_samples"] = 0
                        hazard_collect_state["peek_validated_samples"] = 0
                        hazard_collect_state["latest_peek_valid"] = False
                else:
                    hazard_collect_state["label_request"] = None
                    if hazard_collector.active:
                        hazard_collector.discard_episode(
                            "supervisor_ended_context_without_terminal_label"
                        )
                    hazard_collect_state["context"] = None
                    hazard_collect_state["event_id"] = None
                    hazard_collect_state["terminal_signal"] = 0
                    hazard_collect_state["terminal_start_time"] = None
                    hazard_collect_state["episode_start_time"] = None
                    hazard_collect_state["next_sample_time"] = None
                    hazard_collect_state["peek_valid_samples"] = 0
                    hazard_collect_state["peek_validated_samples"] = 0
                    hazard_collect_state["latest_peek_valid"] = False
            policy_step += 1
            if args_cli.camera_qa and policy_step >= args_cli.camera_qa_steps:
                print(f">>> [camera_qa] reached {args_cli.camera_qa_steps} steps, closing.")
                break
            if args_cli.max_steps > 0 and policy_step >= args_cli.max_steps:
                print(f">>> [max_steps] reached {args_cli.max_steps} steps, closing.")
                print_robot_pose_summary(robot, _foot_names, _foot_indices, "max_steps final")
                break
            if args_cli.camera_aim_qa and wrist_camera_sensor is not None and policy_step >= STARTUP_SETTLE_STEPS and policy_step % 100 == 0:
                with torch.no_grad():
                    wp_pos = robot.data.body_state_w[:, gripper_link_idx, :3]
                    wp_quat = robot.data.body_state_w[:, gripper_link_idx, 3:7]
                    cam_pos, cam_quat_opengl = math_utils.combine_frame_transforms(
                        wp_pos,
                        wp_quat,
                        wrist_camera_local_pos,
                        wrist_camera_local_rot,
                    )
                    opengl_fwd_axis = torch.tensor([[0.0, 0.0, -1.0]], dtype=torch.float32, device=sim.device)
                    opengl_up_axis = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float32, device=sim.device)
                    ray_dir = math_utils.quat_apply(cam_quat_opengl, opengl_fwd_axis)
                    ray_up = math_utils.quat_apply(cam_quat_opengl, opengl_up_axis)
                    ray_dir_np = ray_dir[0].detach().cpu().numpy().astype(np.float32)
                    cam_pos_np = cam_pos[0].detach().cpu().numpy().astype(np.float32)
                    ray_up_np = ray_up[0].detach().cpu().numpy().astype(np.float32)
                    ray_points = []
                    for dist in (0.05, 0.1, 0.2, 0.3, 0.5):
                        pt = cam_pos_np + ray_dir_np * dist
                        ray_points.append((dist, [round(float(v), 4) for v in pt]))
                    z_dir = float(ray_dir_np[2])
                    x_dir = float(ray_dir_np[0])
                    if z_dir > 0.3:
                        aim = "UP (천장)"
                    elif z_dir < -0.3:
                        aim = "DOWN (바닥/책상)"
                    elif x_dir > 0.5:
                        aim = "FORWARD (전방 수평)"
                    elif x_dir < -0.5:
                        aim = "BACKWARD (후방)"
                    else:
                        aim = "SIDE"
                    print(f">>> [camera_aim] step={policy_step} aim={aim}", flush=True)
                    print(f"    cam_pos_w={[round(float(v),4) for v in cam_pos_np]}", flush=True)
                    print(f"    ray_dir_w={[round(float(v),4) for v in ray_dir_np]} (z>0=위, z<0=아래)", flush=True)
                    print(f"    ray_up_w={[round(float(v),4) for v in ray_up_np]}", flush=True)
                    for dist, pt in ray_points:
                        print(f"    +{dist}m -> {[round(float(v),4) for v in pt]} (z={round(float(pt[2]),4)})", flush=True)
        else:
            # 일시정지 상태일 때는 물리 엔진 강제 업데이트(write_data)를 하지 않고 UI만 갱신
            simulation_app.update()

    if collector is not None:
        collector.close()
    if hazard_collector is not None:
        hazard_collector.close()
    stop_leader_subprocess()
    stop_smolvla_subprocess()
    if maze_inspector is not None and args_cli.maze_nav2:
        maze_inspector.close()
    if active_arm_server is not None:
        active_arm_server.close()
    # Camera/finite smoke-test runs can leave Kit/Replicator cleanup jobs pending on shutdown.
    # Do not wait for Replicator; use immediate cleanup for bounded QA runs so CLI verification exits.
    _skip_app_cleanup = bool(
        args_cli.max_steps > 0
        or args_cli.exit_on_route_complete
        or args_cli.camera_qa
        or args_cli.camera_aim_qa
        or locomotion_failure_reason is not None
    )
    simulation_close_exit = None
    try:
        simulation_app.close(wait_for_replicator=False, skip_cleanup=_skip_app_cleanup)
    except SystemExit as exc:
        # Some Kit builds raise SystemExit(0) from close(). Preserve cleanup,
        # then restore the simulator's real safety-failure exit code below.
        simulation_close_exit = exc
    if gr00t_bridge is not None:
        gr00t_bridge.close()
    if smolvla_bridge is not None:
        smolvla_bridge.close()
    if hazard_smolvla_bridge is not None:
        hazard_smolvla_bridge.close()
    if leader_action_sub is not None:
        leader_action_sub.close()
    del camera_viewports
    if locomotion_failure_reason is not None:
        raise SystemExit(42)
    if simulation_close_exit is not None:
        raise simulation_close_exit


if __name__ == "__main__":
    try:
        main()
    except Exception as simulation_error:
        # Also clean up GPU resources when setup/UI construction fails.
        # Print BEFORE Kit teardown, which can otherwise obscure the original
        # exception by entering its stop/render loop while the timeline pauses.
        import traceback
        traceback.print_exc()
        sys.stderr.flush()
        if 'nav2_client' in sys.modules:
            sys.modules['nav2_client'].close_active(f'{type(simulation_error).__name__}: {simulation_error}')
        try:
            simulation_app.close(wait_for_replicator=False,skip_cleanup=True)
        except SystemExit:
            # Kit's successful close must not hide the original runtime failure.
            pass
        raise

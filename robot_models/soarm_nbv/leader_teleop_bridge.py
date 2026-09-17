"""Bridge a physical SO leader arm to the Isaac Sim SO-Arm ZMQ action port."""

from __future__ import annotations

import argparse
import json
import struct
import sys
import time
import zmq
from pathlib import Path

import numpy as np

THIS_DIR = Path(__file__).resolve().parent
ROBOT_MODELS_ROOT = THIS_DIR.parent
if str(ROBOT_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_MODELS_ROOT))

from soarm_nbv.safety import SOARM_JOINT_ORDER, clamp_joint_targets_deg
from soarm_nbv.zmq_bridge import ActionPublisher, SoArmAction, ZmqEndpointConfig


LEROBOT_SRC = Path("/home/iy/Isaac/lerobot/src")
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.teleoperators.so_leader import SO100Leader, SO101Leader  # noqa: E402
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig  # noqa: E402

LEADER_GRIPPER_MIN_DEG = 0.0
LEADER_GRIPPER_MAX_DEG = 100.0
SIM_GRIPPER_MIN_DEG = -20.0
SIM_GRIPPER_MAX_DEG = 100.0


class RuntimeOffsetConfig:
    def __init__(self, path: str):
        self.path = Path(path).expanduser()
        self._last_mtime_ns: int | None = None
        self._offsets_deg = np.zeros(len(SOARM_JOINT_ORDER), dtype=np.float32)

    @property
    def offsets_deg(self) -> np.ndarray:
        if not self.path:
            return self._offsets_deg
        try:
            stat = self.path.stat()
        except FileNotFoundError:
            if self._last_mtime_ns is not None:
                self._last_mtime_ns = None
                self._offsets_deg = np.zeros(len(SOARM_JOINT_ORDER), dtype=np.float32)
                print(f">>> Runtime offsets cleared (missing file): {self.path}")
            return self._offsets_deg

        if stat.st_mtime_ns == self._last_mtime_ns:
            return self._offsets_deg

        payload = json.loads(self.path.read_text())
        new_offsets = np.zeros(len(SOARM_JOINT_ORDER), dtype=np.float32)
        if isinstance(payload, dict):
            for idx, joint_name in enumerate(SOARM_JOINT_ORDER):
                if joint_name in payload:
                    new_offsets[idx] = float(payload[joint_name])
        else:
            raise ValueError(f"Runtime offset file must be a JSON object: {self.path}")

        self._last_mtime_ns = stat.st_mtime_ns
        self._offsets_deg = new_offsets
        print(f">>> Runtime offsets loaded from {self.path}: {self._offsets_deg}")
        return self._offsets_deg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SO leader to Isaac Sim ZMQ teleoperation bridge.")
    parser.add_argument("--leader-port", required=True, help="Serial port of the physical SO leader arm.")
    parser.add_argument(
        "--leader-type",
        choices=("so100_leader", "so101_leader"),
        default="so101_leader",
        help="LeRobot teleoperator type.",
    )
    parser.add_argument("--leader-id", default="sim_leader", help="Calibration id used by LeRobot.")
    parser.add_argument(
        "--calibration-dir",
        default="",
        help="Optional LeRobot calibration directory override.",
    )
    parser.add_argument(
        "--action-port",
        type=int,
        default=5556,
        help="ZMQ action port consumed by nbv_v5.",
    )
    parser.add_argument("--fps", type=float, default=30.0, help="Target publish rate.")
    parser.add_argument(
        "--no-calibrate",
        action="store_true",
        help="Skip automatic calibration on connect and use the saved calibration as-is.",
    )
    parser.add_argument(
        "--runtime-offset-json",
        default="",
        help="Optional JSON file with per-joint live trim offsets in degrees.",
    )
    parser.add_argument(
        "--feedback-port",
        type=int,
        default=5557,
        help="ZMQ port for haptic grip-force feedback from sim.",
    )
    parser.add_argument(
        "--enable-haptic-feedback",
        action="store_true",
        help="Experimental: write grip-force resistance back to the physical leader gripper.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Print every N published actions. 0 disables per-action logging.",
    )
    return parser.parse_args()


def make_leader(args: argparse.Namespace):
    calibration_dir = Path(args.calibration_dir) if args.calibration_dir else None
    config = SOLeaderTeleopConfig(
        port=args.leader_port,
        id=args.leader_id,
        use_degrees=True,
        calibration_dir=calibration_dir,
    )
    leader_cls = SO100Leader if args.leader_type == "so100_leader" else SO101Leader
    return leader_cls(config)


def action_dict_to_array(action_dict: dict[str, float]) -> np.ndarray:
    target = np.asarray([action_dict[f"{joint}.pos"] for joint in SOARM_JOINT_ORDER], dtype=np.float32)
    target[-1] = np.interp(
        target[-1],
        (LEADER_GRIPPER_MIN_DEG, LEADER_GRIPPER_MAX_DEG),
        (SIM_GRIPPER_MIN_DEG, SIM_GRIPPER_MAX_DEG),
    )
    return clamp_joint_targets_deg(target)


def main() -> int:
    args = parse_args()
    period_s = 1.0 / max(args.fps, 1.0)

    leader = make_leader(args)
    publisher = ActionPublisher(ZmqEndpointConfig(action_port=args.action_port))
    runtime_offsets = RuntimeOffsetConfig(args.runtime_offset_json) if args.runtime_offset_json else None
    # 햅틱 피드백은 리더 그리퍼 모터 레지스터(Goal_Position/Torque_Limit/Torque_Enable)를
    # 직접 쓰므로 기본값은 OFF다. 기본 텔레옵은 리더를 읽기 전용 센서처럼만 사용한다.
    feedback_sub = None
    HAPTIC_TORQUE_MAX = 900
    HAPTIC_THRESHOLD = 0.01
    HAPTIC_TIMEOUT_S = 0.25
    HAPTIC_TORQUE_STEP = 5
    HAPTIC_OPENING_DELTA = 0.03
    HAPTIC_WRITE_COOLDOWN_S = 0.033  # 루프 주기(30Hz=33ms)에 맞춤: 사이클당 최대 1회 쓰기
    HAPTIC_ERROR_COOLDOWN_S = 0.5

    last_grip_force = 0.0
    last_feedback_time = 0.0
    _haptic_active = False
    _haptic_hold_pos = None
    _haptic_torque_limit = 0
    last_leader_gripper_pos = None
    last_read_error_log_time = 0.0
    last_haptic_write_time = 0.0
    haptic_error_until = 0.0
    last_haptic_write_error_log_time = 0.0
    if args.enable_haptic_feedback:
        feedback_ctx = zmq.Context.instance()
        feedback_sub = feedback_ctx.socket(zmq.SUB)
        feedback_sub.setsockopt(zmq.RCVHWM, 1)
        feedback_sub.setsockopt(zmq.CONFLATE, 1)
        feedback_sub.connect(f"tcp://127.0.0.1:{args.feedback_port}")
        feedback_sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f">>> Haptic feedback SUB: port {args.feedback_port}")
    else:
        print(">>> Haptic feedback disabled (default); leader gripper motor is read-only")

    print(f">>> SO leader bridge starting on {args.leader_port}")
    print(f">>> Leader type: {args.leader_type}")
    print(f">>> Action PUB tcp://*:{args.action_port}")
    print(f">>> Joint order: {', '.join(SOARM_JOINT_ORDER)}")
    if runtime_offsets is not None:
        print(f">>> Runtime offset JSON: {runtime_offsets.path}")

    leader.connect(calibrate=not args.no_calibrate)
    step_count = 0
    try:
        while True:
            loop_start = time.perf_counter()
            now = time.monotonic()
            try:
                raw_action = leader.get_action()
            except Exception as exc:
                if now - last_read_error_log_time >= 1.0:
                    print(f">>> WARNING: leader read failed; keeping bridge alive: {exc}", flush=True)
                    last_read_error_log_time = now
                if args.enable_haptic_feedback and _haptic_active:
                    try:
                        leader.bus.write("Torque_Enable", "gripper", 0)
                    except Exception:
                        pass
                    _haptic_active = False
                    _haptic_hold_pos = None
                    _haptic_torque_limit = 0
                time.sleep(period_s)
                continue

            target_deg = action_dict_to_array(raw_action)
            leader_gripper_pos = raw_action["gripper.pos"]
            if args.enable_haptic_feedback:
                # 리더 그리퍼를 여는 중이면 sim feedback과 무관하게 즉시 저항 해제.
                # ZMQ 통신 지연(~33ms) 동안 열기 저항이 남아 있는 것을 막는다.
                if last_leader_gripper_pos is not None:
                    _gripper_delta = float(leader_gripper_pos) - last_leader_gripper_pos
                    if _gripper_delta > HAPTIC_OPENING_DELTA:
                        last_grip_force = 0.0
                last_leader_gripper_pos = float(leader_gripper_pos)
            if runtime_offsets is not None:
                target_deg = clamp_joint_targets_deg(target_deg + runtime_offsets.offsets_deg)

            if args.enable_haptic_feedback and feedback_sub is not None:
                # 햅틱 피드백 수신 (non-blocking). 끊기거나 깨진 패킷이면 0으로 떨어뜨린다.
                try:
                    raw_fb = feedback_sub.recv(flags=zmq.NOBLOCK)
                    if len(raw_fb) == 4:
                        feedback_value = struct.unpack("f", raw_fb)[0]
                        if np.isfinite(feedback_value):
                            last_grip_force = float(np.clip(feedback_value, 0.0, 1.0))
                            last_feedback_time = now
                    else:
                        last_grip_force = 0.0
                except zmq.Again:
                    pass
                if last_feedback_time == 0.0 or now - last_feedback_time > HAPTIC_TIMEOUT_S:
                    last_grip_force = 0.0

                # 리더 그리퍼 모터에 파지력 비례 저항 부여.
                # 직렬 버스 안정성: 쓰기는 최소 100ms 간격, 에러 발생 시 500ms 백오프.
                # 단 저항 해제(Torque_Enable=0)는 즉시 허용해 파지 해제가 늦어지지 않게 한다.
                haptic_torque_limit = int(last_grip_force * HAPTIC_TORQUE_MAX)
                _can_write = now >= haptic_error_until and (now - last_haptic_write_time) >= HAPTIC_WRITE_COOLDOWN_S
                try:
                    if last_grip_force > HAPTIC_THRESHOLD and haptic_torque_limit > 0:
                        if not _haptic_active and _can_write:
                            _haptic_hold_pos = int(leader_gripper_pos)
                            leader.bus.write("Goal_Position", "gripper", _haptic_hold_pos)
                            leader.bus.write("Torque_Limit", "gripper", haptic_torque_limit)
                            leader.bus.write("Torque_Enable", "gripper", 1)
                            _haptic_torque_limit = haptic_torque_limit
                            _haptic_active = True
                            last_haptic_write_time = now
                        elif _haptic_active and abs(haptic_torque_limit - _haptic_torque_limit) >= HAPTIC_TORQUE_STEP and _can_write:
                            leader.bus.write("Torque_Limit", "gripper", haptic_torque_limit)
                            _haptic_torque_limit = haptic_torque_limit
                            last_haptic_write_time = now
                    elif _haptic_active:
                        leader.bus.write("Torque_Enable", "gripper", 0)
                        _haptic_active = False
                        _haptic_hold_pos = None
                        _haptic_torque_limit = 0
                        last_haptic_write_time = now
                except Exception as exc:
                    if now - last_haptic_write_error_log_time >= 1.0:
                        print(f">>> WARNING: haptic write failed; disabling haptic this cycle: {exc}", flush=True)
                        last_haptic_write_error_log_time = now
                    _haptic_active = False
                    _haptic_hold_pos = None
                    _haptic_torque_limit = 0
                    haptic_error_until = now + HAPTIC_ERROR_COOLDOWN_S

            publisher.publish(SoArmAction(joint_target_deg=target_deg))
            if args.log_every > 0 and step_count % args.log_every == 0:
                print(f">>> leader target deg: {target_deg} | haptic: {last_grip_force:.2f}")
            elapsed = time.perf_counter() - loop_start
            step_count += 1
            time.sleep(max(period_s - elapsed, 0.0))
    except KeyboardInterrupt:
        print(">>> SO leader bridge stopped")
        return 0
    finally:
        if args.enable_haptic_feedback and _haptic_active:
            try:
                leader.bus.write("Torque_Enable", "gripper", 0)
            except Exception:
                pass
        try:
            leader.disconnect()
        finally:
            publisher.close()
            if feedback_sub is not None:
                feedback_sub.close()


if __name__ == "__main__":
    raise SystemExit(main())

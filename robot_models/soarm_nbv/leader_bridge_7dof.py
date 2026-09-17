#!/usr/bin/env python3
"""
7DOF SO-Arm Leader → Isaac Sim ZMQ Bridge.

The main 6-value payload keeps the validated order:
shoulder_pan, shoulder_lift, elbow_flex, elbow_rotate, wrist_flex, wrist_roll.
Servo IDs 5 and 6 intentionally cross the physical joint order.

The extra leader motor (servo ID 7) is published separately on the gripper
channel so the legacy six-float action payload stays policy-compatible.
"""
from __future__ import annotations

import argparse
import struct
import sys
import time
import zlib

import zmq


# scservo_sdk for direct motor reads
from scservo_sdk import PortHandler, PacketHandler

# ZMQ bridge protocol
_ROBOT_MODELS = "/home/iy/Isaac/Robotics/robot_models"
if _ROBOT_MODELS not in sys.path:
    sys.path.insert(0, _ROBOT_MODELS)
from soarm_nbv.zmq_bridge import ActionPublisher, SoArmAction, ZmqEndpointConfig
from soarm_nbv.robot_model_profile import (
    CUSTOM_LEADER_CONTROL_ORDER,
    map_custom_leader_readings_to_commands,
)

BAUD = 1_000_000
GRIPPER_PORT = 5558


def parse_args():
    """go2_soarm.py --leader_auto 호환 CLI"""
    parser = argparse.ArgumentParser(description="7DOF SO-Arm Leader Bridge")
    parser.add_argument("--leader-port", default="/dev/ttyACM0")
    parser.add_argument("--leader-type", default="so101_leader")
    parser.add_argument("--leader-id", default="sim_leader")
    parser.add_argument("--calibration-dir", default="")
    parser.add_argument("--action-port", type=int, default=5556)
    parser.add_argument("--fps", type=float, default=30.0)
    parser.add_argument("--no-calibrate", action="store_true")
    parser.add_argument("--runtime-offset-json", default="")
    parser.add_argument("--feedback-port", type=int, default=5557)
    parser.add_argument("--enable-haptic-feedback", action="store_true")
    parser.add_argument("--log-every", type=int, default=0)
    args, _unknown = parser.parse_known_args()
    return args

# Validated physical Servo ID order. IDs 5 and 6 are crossed on the arm.
MOTOR_IDS = [1, 2, 3, 4, 5, 6]
GRIPPER_MOTOR_ID = 7


def raw_to_deg(raw_pos: int) -> float:
    """Convert a Feetech position step to the calibrated raw-degree convention."""
    return ((float(raw_pos) - 2048.0) / 2048.0) * 100.0


def publish_gripper_raw(socket, raw_pos: int, seq: int) -> None:
    """Send one little-endian uint32 sequence + uint32 raw gripper position with a CRC32 checksum."""
    payload = struct.pack("<II", seq & 0xFFFFFFFF, raw_pos)
    socket.send(payload + struct.pack("<I", zlib.crc32(payload) & 0xFFFFFFFF))




def main():
    args = parse_args()
    print("=" * 60)
    print("  7DOF SO-Arm Leader Bridge (validated six-motor mapping)")
    print("=" * 60)

    ph = PortHandler(args.leader_port)
    pk = PacketHandler(0)
    if not ph.openPort():
        raise RuntimeError(f"Failed to open leader port: {args.leader_port}")
    if not ph.setBaudRate(BAUD):
        raise RuntimeError(f"Failed to configure leader baud rate: {BAUD}")

    # Verify motors
    print("모터 확인:")
    for mid in MOTOR_IDS + [GRIPPER_MOTOR_ID]:
        try:
            pos, res, _ = pk.read2ByteTxRx(ph, mid, 56)
            if res == 0:
                print(f"  ID {mid}: OK (pos={pos})")
            else:
                print(f"  ID {mid}: 응답 없음!")
        except:
            print(f"  ID {mid}: 읽기 실패!")
        time.sleep(0.05)

    # Gripper channel (separate from the legacy six-float action payload)
    # PUB connects to the simulator's bound SUB endpoint so either process can
    # restart independently without fighting over port 5558.
    gripper_ctx = zmq.Context()
    gripper_pub = gripper_ctx.socket(zmq.PUB)
    gripper_pub.setsockopt(zmq.SNDHWM, 1)
    gripper_pub.setsockopt(zmq.CONFLATE, 1)
    gripper_pub.connect(f"tcp://localhost:{GRIPPER_PORT}")

    # ZMQ publisher
    publisher = ActionPublisher(ZmqEndpointConfig(action_port=args.action_port))
    print(f"\nZMQ Action Publisher: port {args.action_port}")
    print(f"Gripper Publisher: port {GRIPPER_PORT}")
    print(f"Payload order: {', '.join(CUSTOM_LEADER_CONTROL_ORDER)}")

    period = 1.0 / max(args.fps, 1.0)
    step = 0
    print(f"\n텔레오퍼레이션 시작 ({args.fps}fps)")
    print("gripper = 시뮬레이션에서 50도로 고정")
    print("Ctrl+C로 종료\n")

    try:
        while True:
            t0 = time.perf_counter()

            readings = {}
            gripper_raw_pos = None
            for mid in MOTOR_IDS:
                try:
                    pos, res, _ = pk.read2ByteTxRx(ph, mid, 56)
                    if res == 0:
                        readings[mid] = raw_to_deg(pos)
                except Exception:
                    pass
                time.sleep(0.005)
            try:
                pos, res, _ = pk.read2ByteTxRx(ph, GRIPPER_MOTOR_ID, 56)
                if res == 0:
                    gripper_raw_pos = pos
            except Exception:
                pass

            try:
                action_deg = map_custom_leader_readings_to_commands(readings).astype("float32")
            except ValueError as error:
                if step % 30 == 0:
                    print(f">>> WARNING: {error}", flush=True)
                time.sleep(period)
                step += 1
                continue

            action = SoArmAction(joint_target_deg=action_deg)
            publisher.publish(action)
            if gripper_raw_pos is not None:
                publish_gripper_raw(gripper_pub, gripper_raw_pos, step)

            step += 1
            if args.log_every > 0 and step % args.log_every == 0:
                values = " ".join(
                    f"{name}={action_deg[index]:+.1f}°"
                    for index, name in enumerate(CUSTOM_LEADER_CONTROL_ORDER)
                )
                print(f"  [{step}] {values}", flush=True)

            # Maintain FPS
            elapsed = time.perf_counter() - t0
            sleep_time = period - elapsed
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n텔레오퍼레이션 종료.")
    finally:
        ph.closePort()


if __name__ == "__main__":
    main()

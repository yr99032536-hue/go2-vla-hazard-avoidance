#!/usr/bin/env python3
"""Read raw motor position for wrist_flex to manually fix calibration."""
import sys
sys.path.insert(0, "/home/iy/Isaac/Robotics/lerobot/src")

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus
import json, time

CALIB_FILE = "/home/iy/lerobot/calibration/teleoperators/so_leader/sim_leader.json"
with open(CALIB_FILE) as f:
    cal = json.load(f)

bus = FeetechMotorsBus(
    port="/dev/ttyACM0",
    motors={
        "shoulder_pan": Motor(1, "sts3215", MotorNormMode.DEGREES),
        "shoulder_lift": Motor(2, "sts3215", MotorNormMode.DEGREES),
        "elbow_flex": Motor(3, "sts3215", MotorNormMode.DEGREES),
        "wrist_flex": Motor(4, "sts3215", MotorNormMode.DEGREES),
        "wrist_roll": Motor(5, "sts3215", MotorNormMode.DEGREES),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
    calibration={k: MotorCalibration(**v) for k, v in cal.items()},
)
bus.connect()

print("=== wrist_flex 원시 모터 위치 읽기 ===")
print("리더암 wrist_flex를 시뮬 0도 자세에 맞추세요.\n")

for i in range(10):
    raw = bus.read("Present_Position", "wrist_flex", normalize=False)
    deg = bus.read("Present_Position", "wrist_flex", normalize=True)
    print(f"  raw={raw:6.0f}  deg={deg:8.2f}")
    time.sleep(0.3)

print("\n=== 최소/최대 range 구하기 ===")
print("wrist_flex를 양 끝까지 천천히 움직이세요... (10초)")
positions = []
start = time.time()
while time.time() - start < 10:
    try:
        raw = bus.read("Present_Position", "wrist_flex", normalize=False)
        positions.append(raw)
    except Exception:
        pass
    time.sleep(0.05)

if positions:
    lo, hi = min(positions), max(positions)
    mid = (lo + hi) / 2
    print(f"  range_min = {lo:.0f}")
    print(f"  range_max = {hi:.0f}")
    print(f"  midpoint  = {mid:.0f}")
    print(f"\n현재 wrist_flex homing_offset = {cal['wrist_flex']['homing_offset']}")

bus.disconnect()

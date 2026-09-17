#!/usr/bin/env python3
"""Re-calibrate only wrist_flex, wrist_roll, gripper for sim_leader."""

import sys
sys.path.insert(0, "/home/iy/Isaac/Robotics/lerobot/src")

from lerobot.motors import Motor, MotorCalibration, MotorNormMode
from lerobot.motors.feetech import FeetechMotorsBus, OperatingMode
from pathlib import Path

CALIB_DIR = Path("/home/iy/lerobot/calibration/teleoperators/so_leader")
CALIB_FILE = CALIB_DIR / "sim_leader.json"
LEADER_ID = "sim_leader"

# Load existing calibration
import json
with open(CALIB_FILE) as f:
    old_cal = json.load(f)

print(f"기존 캘리브레이션 로드: {CALIB_FILE}")
for joint in ["wrist_flex", "wrist_roll", "gripper"]:
    print(f"  {joint}: {old_cal[joint]}")

# Target joints to re-calibrate
TARGET_JOINTS = ["wrist_flex"]
ALL_JOINTS = ["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll", "gripper"]

norm_mode_body = MotorNormMode.DEGREES
bus = FeetechMotorsBus(
    port="/dev/ttyACM0",
    motors={
        "shoulder_pan": Motor(1, "sts3215", norm_mode_body),
        "shoulder_lift": Motor(2, "sts3215", norm_mode_body),
        "elbow_flex": Motor(3, "sts3215", norm_mode_body),
        "wrist_flex": Motor(4, "sts3215", norm_mode_body),
        "wrist_roll": Motor(5, "sts3215", norm_mode_body),
        "gripper": Motor(6, "sts3215", MotorNormMode.RANGE_0_100),
    },
    calibration={k: MotorCalibration(**v) for k, v in old_cal.items()},
)

print("\n모터 버스 연결 중...")
bus.connect()

# Step 1: Move all joints to middle
input("\n[1/2] 모든 관절을 중간(홈) 위치로 이동시키고 Enter를 누르세요...")

# Only re-record homing for target joints
bus.disable_torque()
for motor in TARGET_JOINTS:
    bus.write("Operating_Mode", motor, OperatingMode.POSITION.value)

# Read current positions for homing offset
from lerobot.motors.feetech.tables import MODEL_RESOLUTION as FEETECH_MODEL_RESOLUTION
homing_offsets = {}
for motor in TARGET_JOINTS:
    present_pos = bus.read("Present_Position", motor, normalize=False)
    model = bus._id_to_model(bus.motors[motor].id)
    max_res = FEETECH_MODEL_RESOLUTION[model] - 1
    homing_offsets[motor] = int(-((present_pos - max_res // 2) % max_res))
    print(f"  {motor}: pos={present_pos}, homing_offset={homing_offsets[motor]}")

# Step 2: Move target joints through full range
input("\n[2/2] wrist_flex, wrist_roll, gripper를 각각 최대 범위로 천천히 움직이세요. 끝나면 Enter...")

# Record ranges
range_mins = {}
range_maxes = {}

# Start recording
positions = {motor: [] for motor in TARGET_JOINTS}
print("범위 기록 중... 관절을 계속 움직이세요...")
import time
start = time.time()
while time.time() - start < 8:
    for motor in TARGET_JOINTS:
        try:
            pos = bus.read("Present_Position", motor, normalize=False)
            positions[motor].append(pos)
        except Exception:
            pass
    time.sleep(0.05)

for motor in TARGET_JOINTS:
    if positions[motor]:
        range_mins[motor] = min(positions[motor])
        range_maxes[motor] = max(positions[motor])
    else:
        range_mins[motor] = old_cal[motor]["range_min"]
        range_maxes[motor] = old_cal[motor]["range_max"]
    print(f"  {motor}: range [{range_mins[motor]}, {range_maxes[motor]}]")

# Build new calibration
new_cal = dict(old_cal)
for motor in TARGET_JOINTS:
    new_cal[motor] = {
        "id": old_cal[motor]["id"],
        "drive_mode": old_cal[motor]["drive_mode"],
        "homing_offset": homing_offsets[motor],
        "range_min": range_mins[motor],
        "range_max": range_maxes[motor],
    }
    print(f"\n{motor}:")
    print(f"  기존: {old_cal[motor]}")
    print(f"  변경: {new_cal[motor]}")

# Save
confirm = input("\n이 캘리브레이션을 저장할까요? (y/n): ")
if confirm.strip().lower() == "y":
    # Write to motor
    cal_to_write = {k: MotorCalibration(**v) for k, v in new_cal.items()}
    for motor in TARGET_JOINTS:
        offset = new_cal[motor]["homing_offset"]
        # Feetetch homing_offset register is ±2047
        if abs(offset) > 2047:
            offset = ((offset + 2048) % 4096) - 2048
            print(f"  {motor}: homing_offset clamped to {offset}")
            new_cal[motor]["homing_offset"] = offset
        bus.write("Homing_Offset", motor, offset, normalize=False)
        bus.write("Min_Position_Limit", motor, new_cal[motor]["range_min"], normalize=False)
        bus.write("Max_Position_Limit", motor, new_cal[motor]["range_max"], normalize=False)

    # Save file
    with open(CALIB_FILE, "w") as f:
        json.dump(new_cal, f, indent=4)
    print(f"저장 완료: {CALIB_FILE}")
else:
    print("취소됨")

bus.disconnect()
print("완료")

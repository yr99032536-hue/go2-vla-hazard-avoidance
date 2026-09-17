"""STS3215 그리퍼 모터 상태 진단."""
import sys
import time
from pathlib import Path

ROBOT_MODELS_ROOT = Path(__file__).resolve().parents[1]
if str(ROBOT_MODELS_ROOT) not in sys.path:
    sys.path.insert(0, str(ROBOT_MODELS_ROOT))

LEROBOT_SRC = Path("/home/iy/Isaac/lerobot/src")
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.teleoperators.so_leader import SO101Leader
from lerobot.teleoperators.so_leader.config_so_leader import SOLeaderTeleopConfig

PORT = "/dev/ttyACM0"
LEADER_ID = "teleop_leader_v1"
CALIB_DIR = "/home/iy/lerobot/calibration/teleoperators/so_leader"


def safe_read(bus, name):
    try:
        val = bus.read(name, "gripper")
        if isinstance(val, (tuple, list)):
            return val[0]
        return val
    except Exception as e:
        return f"ERR:{e}"


def main():
    config = SOLeaderTeleopConfig(port=PORT, id=LEADER_ID, use_degrees=True, calibration_dir=Path(CALIB_DIR))
    leader = SO101Leader(config)
    print(">>> Connecting...")
    leader.connect(calibrate=False)
    bus = leader.bus

    motor_id = bus.motors["gripper"].id
    print(f">>> Gripper motor id={motor_id}")
    print(">>> Reading for 5s (touch/move gripper to see changes)...\n")

    for i in range(5):
        pos = safe_read(bus, "Present_Position")
        load = safe_read(bus, "Present_Load")
        curr = safe_read(bus, "Present_Current")
        speed = safe_read(bus, "Present_Speed")
        volt = safe_read(bus, "Present_Voltage")
        temp = safe_read(bus, "Present_Temperature")
        torque_en = safe_read(bus, "Torque_Enable")
        print(f"[{i+1}s] pos={pos} load={load} current={curr} speed={speed} voltage={volt} temp={temp} torque_en={torque_en}")
        time.sleep(1.0)

    print("\n>>> Done.")
    leader.disconnect()


if __name__ == "__main__":
    main()

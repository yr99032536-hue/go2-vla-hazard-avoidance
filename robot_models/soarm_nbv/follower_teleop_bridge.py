"""Bridge a ZMQ action port to a physical SO follower arm using official LeRobot API."""

import argparse
import sys
import time
from pathlib import Path
import numpy as np
import zmq

# Standalone deployment for Jetson
SOARM_JOINT_ORDER = (
    "shoulder_pan",
    "shoulder_lift",
    "elbow_flex",
    "wrist_flex",
    "wrist_roll",
    "gripper",
)

# 젯슨 내부의 르로봇 경로 (Jetson specific)
LEROBOT_SRC = Path("/home/unitree/lerobot/src")
if str(LEROBOT_SRC) not in sys.path:
    sys.path.insert(0, str(LEROBOT_SRC))

from lerobot.robots.so_follower import SO100Follower, SO101Follower
from lerobot.robots.so_follower.config_so_follower import SOFollowerRobotConfig

def parse_args():
    parser = argparse.ArgumentParser(description="ZMQ to LeRobot SO Follower bridge.")
    parser.add_argument("--follower-port", default="/dev/ttyACM0", help="Serial port of the follower.")
    parser.add_argument("--follower-type", choices=("so100_follower", "so101_follower"), default="so101_follower")
    parser.add_argument("--follower-id", default="sim_follower", help="Calibration id.")
    parser.add_argument("--action-port", type=int, default=5556, help="ZMQ port to listen on.")
    parser.add_argument("--leader-ip", default="192.168.123.99", help="IP address of the PC running the leader.")
    parser.add_argument("--no-calibrate", action="store_true", help="Skip calibration on connect.")
    return parser.parse_args()

def make_follower(args):
    config = SOFollowerRobotConfig(
        port=args.follower_port,
        id=args.follower_id,
        use_degrees=True,
    )
    follower_cls = SO100Follower if args.follower_type == "so100_follower" else SO101Follower
    return follower_cls(config)

def main():
    args = parse_args()
    
    follower = make_follower(args)
    print(f">>> SO follower bridge starting on {args.follower_port}")
    
    # 르로봇 공식 연결 (캘리브레이션 툴이 켜질 수 있음)
    follower.connect(calibrate=not args.no_calibrate)
    
    # ZMQ Setup
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    if hasattr(zmq, "TCP_NODELAY"):
        socket.setsockopt(zmq.TCP_NODELAY, 1)
        
    endpoint = f"tcp://{args.leader_ip}:{args.action_port}"
    socket.connect(endpoint)
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    print(f">>> Listening for actions on {endpoint}...")

    try:
        while True:
            try:
                raw = socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
                
            target_deg = np.frombuffer(raw, dtype=np.float32)
            if len(target_deg) != 6:
                continue
                
            # Create LeRobot action dict
            action_dict = {}
            for i, joint_name in enumerate(SOARM_JOINT_ORDER):
                action_dict[f"{joint_name}.pos"] = float(target_deg[i])
                
            # Send to official API
            follower.send_action(action_dict)
            
    except KeyboardInterrupt:
        print("\n>>> Follower bridge stopped")
    finally:
        follower.disconnect()
        socket.close()

if __name__ == "__main__":
    main()

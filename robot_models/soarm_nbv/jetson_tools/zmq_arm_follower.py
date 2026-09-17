import sys
import zmq
import struct
import time
import sys
import zmq
import struct
import time
import numpy as np
import scservo_sdk as scs

# ============================================================================
# [VERY IMPORTANT] 
# You MUST replace these values with the exact calibration data of the FOLLOWER ARM.
# Do NOT use the Leader Arm's values!
# Unplug the Follower Arm, plug it into the PC, run `lerobot_calibrate`, 
# and copy the values from the resulting JSON file here.
# ============================================================================
FOLLOWER_CALIB = {
    1: {"range_min": 730,  "range_max": 3464},
    2: {"range_min": 946,  "range_max": 3232},
    3: {"range_min": 816,  "range_max": 3053},
    4: {"range_min": 936,  "range_max": 3335},
    5: {"range_min": 0,    "range_max": 4095},
    6: {"range_min": 1633, "range_max": 3176},
}

def deg_to_tick(joint_id, deg):
    cal = FOLLOWER_CALIB[joint_id]
    rmin, rmax = cal["range_min"], cal["range_max"]

    if joint_id == 6:
        # 그리퍼: 리더에서 0~100 범위로 전송됨 (LeRobot RANGE_0_100 모드)
        # 0~100 → range_min~range_max 선형 매핑
        pct = max(0.0, min(100.0, deg))
        tick = rmin + (pct / 100.0) * (rmax - rmin)
    else:
        # 일반 관절: degree → tick (LeRobot 공식)
        mid = (rmin + rmax) / 2.0
        tick = (deg * 4095.0 / 360.0) + mid

    tick = int(max(rmin, min(rmax, tick)))
    return tick

def main():
    port = "/dev/ttyACM0"
    baudrate = 1000000
    
    port_handler = scs.PortHandler(port)
    packet_handler = scs.PacketHandler(0)
    
    if not port_handler.openPort():
        print(f"Failed to open port {port}")
        # fallback to ttyUSB0 just in case
        port = "/dev/ttyUSB0"
        port_handler = scs.PortHandler(port)
        if not port_handler.openPort():
            print(f"Failed to open port {port} too.")
            return

    if not port_handler.setBaudRate(baudrate):
        print("Failed to set baudrate")
        return
        
    print(f"Successfully connected to Feetech bus on {port}")
        
    # Enable torque and set properties for all 6 motors
    for i in range(1, 7):
        packet_handler.write1ByteTxRx(port_handler, i, 40, 1) # Torque_Enable
        packet_handler.write1ByteTxRx(port_handler, i, 41, 254) # Acceleration
        
    # ZMQ Sub
    context = zmq.Context()
    socket = context.socket(zmq.SUB)
    socket.setsockopt(zmq.RCVHWM, 1)
    socket.setsockopt(zmq.CONFLATE, 1)
    if hasattr(zmq, "TCP_NODELAY"):
        socket.setsockopt(zmq.TCP_NODELAY, 1)
        
    pc_ip = "192.168.123.99"
    action_port = 5556
    socket.connect(f"tcp://{pc_ip}:{action_port}")
    socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    print(f"Listening for actions on tcp://{pc_ip}:{action_port}...")
    
    # 42 is Goal_Position address, length is 2 bytes
    group_write = scs.GroupSyncWrite(port_handler, packet_handler, 42, 2)
    
    try:
        while True:
            try:
                raw = socket.recv(flags=zmq.NOBLOCK)
            except zmq.Again:
                time.sleep(0.001)
                continue
                
            target_deg = np.frombuffer(raw, dtype=np.float32)
            if len(target_deg) != 6:
                print(f"Expected 6 floats, got {len(target_deg)}")
                continue
            
            group_write.clearParam()
            for i in range(6):
                joint_id = i + 1
                tick = deg_to_tick(joint_id, target_deg[i])
                param = [scs.SCS_LOBYTE(tick), scs.SCS_HIBYTE(tick)]
                group_write.addParam(joint_id, param)
                
            group_write.txPacket()
            
    except KeyboardInterrupt:
        print("\nStopping follower...")
    finally:
        for i in range(1, 7):
            packet_handler.write1ByteTxRx(port_handler, i, 40, 0) # Torque_Disable
        port_handler.closePort()
        print("Port closed.")
        
if __name__ == "__main__":
    main()

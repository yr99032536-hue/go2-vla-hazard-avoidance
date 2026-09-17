import sys
import time
import scservo_sdk as scs

def main():
    port = "/dev/ttyACM0"
    baudrate = 1000000
    
    port_handler = scs.PortHandler(port)
    packet_handler = scs.PacketHandler(0)
    
    if not port_handler.openPort():
        print(f"Failed to open port {port}")
        return
    if not port_handler.setBaudRate(baudrate):
        print("Failed to set baudrate")
        return
        
    print("="*50)
    print("   SO-ARM FOLLOWER CALIBRATION TOOL V2 (JETSON)   ")
    print("="*50)
    print("\n>>> Disabling motor torque...")
    
    for i in range(1, 7):
        packet_handler.write1ByteTxRx(port_handler, i, 40, 0) # Torque Disable
        
    print(">>> Motors are now loose!")
    print("\n[INSTRUCTION]")
    print("1. Please manually move the Follower Arm to its exact ZERO (neutral) position.")
    print("2. Ensure it physically perfectly matches the zero pose of the Leader Arm.")
    print("3. Hold it steadily in that position.")
    input("\nPress ENTER when ready to capture the new Homing Offsets... ")
    
    print("\n>>> Capturing motor positions...")
    
    new_offsets = {}
    for i in range(1, 7):
        pos, comm, error = packet_handler.read2ByteTxRx(port_handler, i, 56) # Present_Position
        if comm == scs.COMM_SUCCESS and error == 0:
            # FIX: The calculation formula in zmq_arm_follower.py adds 2048 to the homing offset.
            # So the homing value saved to the file MUST be (pos - 2048).
            corrected_homing = pos - 2048
            new_offsets[i] = corrected_homing
            print(f"  Joint {i} Ticks at Zero: {pos} -> Homing Offset stored: {corrected_homing}")
        else:
            print(f"  Joint {i} Error: Comm={comm}, Err={error}")
            
    if len(new_offsets) == 6:
        print("\n>>> Updating zmq_arm_follower.py with corrected offsets...")
        
        script_path = "/home/unitree/zmq_arm_follower.py"
        try:
            with open(script_path, "r") as f:
                lines = f.readlines()
                
            in_follower_block = False
            for idx, line in enumerate(lines):
                if "FOLLOWER_HOMING = {" in line:
                    in_follower_block = True
                    continue
                if in_follower_block and "}" in line:
                    in_follower_block = False
                    continue
                    
                if in_follower_block:
                    for j in range(1, 7):
                        if line.strip().startswith(f"{j}:"):
                            lines[idx] = f'    {j}: {new_offsets[j]},\n'
                            break
                        
            with open(script_path, "w") as f:
                f.writelines(lines)
            print(">>> Successfully patched /home/unitree/zmq_arm_follower.py!")
            print(">>> Calibration Complete. You can now re-run the follower script.")
        except Exception as e:
            print(f"Failed to auto-patch the file: {e}")
            print("Please update the homing offsets manually.")
    else:
        print("\n>>> Failed to read all 6 joints. Calibration aborted.")

    port_handler.closePort()

if __name__ == "__main__":
    main()

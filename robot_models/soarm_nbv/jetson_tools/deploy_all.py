import paramiko
import os

jetson_ip = "192.168.123.18"
jetson_user = "unitree"
jetson_pass = "123"

files_to_upload = [
    ("/home/iy/robot_models/soarm_nbv/jetson_tools/zmq_arm_follower.py", "/home/unitree/zmq_arm_follower.py"),
    ("/home/iy/robot_models/soarm_nbv/jetson_tools/calibrate_follower_v2.py", "/home/unitree/calibrate_follower_v2.py")
]

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print("Connecting to Jetson...")
    ssh.connect(jetson_ip, username=jetson_user, password=jetson_pass)
    sftp = ssh.open_sftp()
    
    for local_path, remote_path in files_to_upload:
        print(f"Uploading {os.path.basename(local_path)}...")
        sftp.put(local_path, remote_path)
        # Make executable
        sftp.chmod(remote_path, 0o755)
        print(f"Successfully uploaded to {remote_path}")
        
    sftp.close()
    
except Exception as e:
    print(f"Error: {e}")
finally:
    ssh.close()

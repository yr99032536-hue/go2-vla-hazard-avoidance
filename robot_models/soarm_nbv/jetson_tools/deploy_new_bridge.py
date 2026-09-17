import paramiko
import os

jetson_ip = "192.168.123.18"
jetson_user = "unitree"
jetson_pass = "123"

local_file = "/home/iy/robot_models/soarm_nbv/follower_teleop_bridge.py"
remote_file = "/home/unitree/follower_teleop_bridge.py"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print("Connecting to Jetson...")
    ssh.connect(jetson_ip, username=jetson_user, password=jetson_pass)
    sftp = ssh.open_sftp()
    
    print(f"Uploading {os.path.basename(local_file)}...")
    sftp.put(local_file, remote_file)
    sftp.chmod(remote_file, 0o755)
    print(f"Successfully uploaded to {remote_file}")
    
    sftp.close()
    
except Exception as e:
    print(f"Error: {e}")
finally:
    ssh.close()

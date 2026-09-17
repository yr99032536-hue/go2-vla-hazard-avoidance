import paramiko
import os

jetson_ip = "192.168.123.18"
jetson_user = "unitree"
jetson_pass = "123"

local_file = "/home/iy/robot_models/soarm_nbv/jetson_tools/zmq_arm_follower.py"
remote_file = "/home/unitree/zmq_arm_follower.py"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    ssh.connect(jetson_ip, username=jetson_user, password=jetson_pass)
    sftp = ssh.open_sftp()
    sftp.put(local_file, remote_file)
    sftp.chmod(remote_file, 0o755)
    sftp.close()
    print("Deployed zmq_arm_follower.py successfully")
except Exception as e:
    print(f"Error: {e}")
finally:
    ssh.close()

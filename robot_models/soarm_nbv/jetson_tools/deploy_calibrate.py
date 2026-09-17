import paramiko
import os

local_path = "/home/iy/.gemini/antigravity/brain/c61d9736-1ad8-4f87-b997-8e9454979fdc/scratch/calibrate_follower.py"
remote_path = "/home/unitree/calibrate_follower.py"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print("Connecting to Jetson...")
    ssh.connect('192.168.123.18', username='unitree', password='123', timeout=5)
    
    print("Uploading calibrate_follower.py...")
    sftp = ssh.open_sftp()
    sftp.put(local_path, remote_path)
    sftp.close()
    
    ssh.exec_command(f"chmod +x {remote_path}")
    print("Upload and chmod completed.")
    
except Exception as e:
    print("Deploy Failed:", str(e))
finally:
    ssh.close()

import os
import tarfile
import paramiko

local_sdk_dir = "/home/iy/miniconda3/envs/lerobot/lib/python3.12/site-packages/scservo_sdk"
local_tar_path = "/tmp/scservo_sdk.tar.gz"

print("Compressing scservo_sdk folder...")
try:
    with tarfile.open(local_tar_path, "w:gz") as tar:
        tar.add(local_sdk_dir, arcname="scservo_sdk")
    print(f"Compressed into {local_tar_path} successfully.")
except Exception as e:
    print("Compression Failed:", str(e))
    exit(1)

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print("Connecting to Go2 Jetson (192.168.123.18)...")
    ssh.connect('192.168.123.18', username='unitree', password='123', timeout=5)
    
    print("Uploading scservo_sdk tarball to Jetson via SFTP...")
    sftp = ssh.open_sftp()
    sftp.put(local_tar_path, '/home/unitree/scservo_sdk.tar.gz')
    sftp.close()
    print("Upload completed.")
    
    print("Extracting scservo_sdk on Jetson...")
    ssh.exec_command("mkdir -p /home/unitree/.local/lib/python3.8/site-packages")
    stdin, stdout, stderr = ssh.exec_command(
        "tar -xzf /home/unitree/scservo_sdk.tar.gz -C /home/unitree/.local/lib/python3.8/site-packages && rm /home/unitree/scservo_sdk.tar.gz"
    )
    print(stdout.read().decode().strip())
    print(stderr.read().decode().strip())
    
    stdin, stdout, stderr = ssh.exec_command("ls -la /home/unitree/.local/lib/python3.8/site-packages/scservo_sdk")
    ls_out = stdout.read().decode().strip()
    print("Verification /home/unitree/.local/lib/python3.8/site-packages/scservo_sdk content:")
    print(ls_out)
    print("scservo_sdk deployment completed successfully!")
    
except Exception as e:
    print("Jetson SSH Deploy Failed:", str(e))
finally:
    ssh.close()
    if os.path.exists(local_tar_path):
        os.remove(local_tar_path)

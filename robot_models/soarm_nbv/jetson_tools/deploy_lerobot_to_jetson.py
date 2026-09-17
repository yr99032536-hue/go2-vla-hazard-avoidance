import os
import tarfile
import paramiko

# 1. PC 본체에서 LeRobot src 폴더를 tar.gz로 압축
local_src_dir = "/home/iy/Isaac/Robotics/lerobot/lerobot/src"
local_tar_path = "/tmp/lerobot_src.tar.gz"

print(">>> Compressing PC's LeRobot src folder...")
try:
    with tarfile.open(local_tar_path, "w:gz") as tar:
        tar.add(local_src_dir, arcname="src")
    print(f">>> Compressed into {local_tar_path} successfully.")
except Exception as e:
    print("Compression Failed:", str(e))
    exit(1)

# 2. SFTP를 통해 Go2 Jetson으로 압축파일 전송
ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print(">>> Connecting to Go2 Jetson (192.168.123.18)...")
    ssh.connect('192.168.123.18', username='unitree', password='123', timeout=5)
    
    print(">>> Uploading LeRobot src tarball to Jetson via SFTP...")
    sftp = ssh.open_sftp()
    sftp.put(local_tar_path, '/home/unitree/lerobot_src.tar.gz')
    sftp.close()
    print(">>> Upload completed.")
    
    # 3. Jetson에서 압축 풀기 및 정리
    print(">>> Extracting LeRobot src on Jetson...")
    ssh.exec_command("mkdir -p /home/unitree/lerobot")
    stdin, stdout, stderr = ssh.exec_command(
        "tar -xzf /home/unitree/lerobot_src.tar.gz -C /home/unitree/lerobot && rm /home/unitree/lerobot_src.tar.gz"
    )
    print(stdout.read().decode().strip())
    print(stderr.read().decode().strip())
    
    # 4. 검증: Jetson에 /home/unitree/lerobot/src 가 잘 들어왔는지 확인
    stdin, stdout, stderr = ssh.exec_command("ls -la /home/unitree/lerobot/src")
    ls_out = stdout.read().decode().strip()
    print(">>> Verification /home/unitree/lerobot/src content:")
    print(ls_out)
    print(">>> LeRobot deployment to Go2 Jetson completed successfully!")
    
except Exception as e:
    print("Jetson SSH Deploy Failed:", str(e))
finally:
    ssh.close()
    if os.path.exists(local_tar_path):
        os.remove(local_tar_path)

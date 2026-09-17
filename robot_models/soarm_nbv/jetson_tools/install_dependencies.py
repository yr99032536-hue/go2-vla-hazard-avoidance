import paramiko

jetson_ip = "192.168.123.18"
jetson_user = "unitree"
jetson_pass = "123"

ssh = paramiko.SSHClient()
ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())

try:
    print("Connecting to Jetson to install dependencies...")
    ssh.connect(jetson_ip, username=jetson_user, password=jetson_pass)
    
    # 르로봇을 완전히 설치하면 종속성(draccus, einops 등)이 모두 해결됩니다.
    command = "pip3 install draccus"
    print(f"Executing: {command}")
    
    stdin, stdout, stderr = ssh.exec_command(command)
    
    # 실시간 로그 출력
    for line in iter(stdout.readline, ""):
        print(line, end="")
        
    err = stderr.read().decode()
    if err:
        print(f"Errors:\n{err}")
        
except Exception as e:
    print(f"Error: {e}")
finally:
    ssh.close()

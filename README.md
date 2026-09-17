# Go2 + SO-Arm VLA 능동 위험 회피 실험

몸체가 골목에 들어가기 전에 로봇팔 끝 카메라로 가려진 골목을 확인하고, 학습된 SmolVLA의 판단에 따라 안전한 길을 선택하는 연구다. Go2 사족보행 로봇 위에 SO-Arm 7자유도 로봇팔을 실고, 3단으로 이어진 T자 골목에서 빨간 위험 큐브를 피해 목적지까지 도달한다.

## 핵심 방법

### 1. SmolVLA 8차원 액션 공간

이 실험의 SmolVLA는 총 **8개의 액션 공간**을 가진다.

- `action[0:7]`: SO-Arm 7개 모터의 목표 각도 (단위: 도)
- `action[7]`: 골목 판정 신호 (연속 실수 → 위험/안전/확인 중)

```
[모터1, 모터2, 모터3, 모터4, 모터5, 모터6, 모터7, 판정]
 각도    각도    각도    각도    각도    각도    각도    값
```

### 2. 위험 신호 학습

마지막 액션 `action[7]`은 실제 모터를 움직이지 않는다. 대신 **주행기에 전달할 골목 판정 신호**로 사용하도록 학습시켰다.

| 출력값 | 의미 | 실행기 해석 |
|---|---|---|
| 약 -1 | 해당 골목에 빨간 위험 큐브 존재 | 위험 (-1) |
| 약 0 | 아직 확인 중 | 출발 승인 안 함 (0) |
| 약 +1 | 위험 큐브 없음 | 안전 (+1) |

학습 데이터에는 손목 카메라 영상과 함께 정답 판정이 포함됐다. 비전 교사가 손목 RGB 영상에서 빨간색 큐브의 유무를 확인해 위험(-1) 또는 안전(+1) 라벨을 만들고, SmolVLA가 이를 `action[7]`로 예측하도록 파인튜닝했다. 실행 시에는 별도 색상 검출 코드가 개입하지 않고, 모델이 영상을 보고 직접 이 값을 출력한다.

### 3. 폐루프 주행 구조

```
골목 앞 정지 → 팔 전개·좌우 관측 → 3연속 안정 판정
    → 안전 골목 선택 → Go2 보행 정책으로 이동 → 3단 반복
```

- **주행 감독기**: 골목 앞에서 정지시키고, 양쪽 관측이 끝나면 안전 후보를 선택한다.
- **판정 게이트**: 카메라가 골목을 실제로 보고 있는지 확인하고, 같은 판정이 3번 연속 나와야 최종 신호로 받아들인다.
- **Go2 보행 정책**: 12999 체크포인트가 속도 명령을 받아 다리를 움직인다. VLA는 다리를 직접 제어하지 않는다.

## 결과

무작위 큐브 배치 GUI 평가에서 3회 연속 완주 및 신호·경로 감사 통과했다.

| 회차 | 배치 (1~3단계 큐브 위치) | 결과 |
|---|---|---|
| 1 | RRL | 완주, 신호·경로 일치 |
| 2 | LLL | 완주, 신호·경로 일치 |
| 3 | RLR | 완주, 신호·경로 일치 |

각 회차마다 SmolVLA가 6개 판정(3골목 × 좌우)을 출력했고, 모두 실제 큐브 위치와 일치했다.

## 실행

```bash
cd /home/iy/Isaac/Go2_Intelligence_Framework
./scripts/run_binary_tree_vision_demo.sh
```

관련 코드는 다음 위치에 있다.

- `src/go2_active_slam/go2_active_slam/binary_tree_hazard_supervisor.py` — 주행 감독기
- `/home/iy/Isaac/Robotics/robot_models/soarm_nbv/` — SmolVLA 실행기·데이터 계약
- `/home/iy/Isaac/Robotics/robot_models/src/sim/go2_soarm.py` — Isaac Sim 로봇 실행

## 저장소 구조

```
go2-vla-hazard-avoidance/
├── README.md
├── framework/                  # Go2_Intelligence_Framework 실험 코드
│   ├── scripts/                # 실행 스크립트
│   │   ├── run_binary_tree_vision_demo.sh      # VLA 평가 실행
│   │   ├── run_binary_tree_hazard_demo.sh      # 교사 시연
│   │   ├── run_binary_tree_human_collect.sh    # 사람 교사 수집
│   │   ├── run_binary_tree_nbv_collect.sh      # NBV 교사 수집
│   │   └── ...
│   └── src/
│       ├── go2_active_slam/            # 주행 감독기 ROS 2 패키지
│       └── go2_active_slam_interfaces/ # 메시지·서비스 정의
└── robot_models/               # Robotics/robot_models 실험 코드
    ├── soarm_nbv/              # SmolVLA 실행기·데이터 계약·수집기
    │   ├── smolvla_hazard_policy_runner.py     # SmolVLA 추론 실행기
    │   ├── hazard_vla_contract.py              # 8차원 액션 계약
    │   ├── hazard_vla_runtime.py               # 판정 게이트·전송
    │   └── ...
    └── src/
        ├── go2_soarm.py                        # Isaac Sim 로봇 실행
        ├── generate_binary_tree_hazard_map.py  # 3단 골목 맵 생성
        ├── run_active_slam_binary_tree_ros2.sh # Isaac Sim 런처
        └── verify_go2_policy_abi.py            # 보행 정책 검증
```

## 사전 준비

이 코드는 다음 환경을 가정한다.

- Isaac Sim 5.1 + Isaac Lab
- ROS 2 Humble
- LeRobot (SmolVLA)
- Go2 보행 정책 체크포인트 (12999)
- SmolVLA 파인튜닝 체크포인트 (7-state / 8-action)

체크포인트와 대용량 데이터는 저장소에 포함하지 않았다. 로컬 경로는 다음과 같다.

- 보행 정책: `/home/iy/Isaac/IsaacLab/logs/rsl_rl/unitree_go2_so101_7motor_reversed_flat/2026-09-02_05-03-06_home_extend_fold_walk_1024env_headless_lr1e4_v7/exported/policy.pt`
- SmolVLA: `/home/iy/Isaac/Robotics/data/smolvla_runs/binary_tree_hazard_102_v2/checkpoints/020000/pretrained_model`

## 실행 방법

```bash
# 1. ROS 2 워크스페이스 빌드 (framework/src 기준)
cd framework
colcon build --packages-select go2_active_slam_interfaces go2_active_slam

# 2. 실행 스크립트에 robot_models 경로 전달
export ROBOT_MODELS_ROOT=$(pwd)/../robot_models
./scripts/run_binary_tree_vision_demo.sh
```

스크립트는 Isaac Sim, ROS 2, SmolVLA 실행기를 함께 띄우고 골목 위험 회피를 실행한다.

## 참고 자료

- [SmolVLA: A Vision-Language-Action Model for Affordable and Efficient Robotics](https://arxiv.org/abs/2506.01844)
- 본 실험의 8번째 액션 설계(위험 판정 신호)는 위 논문의 기본 구조를 확장한 것이다.

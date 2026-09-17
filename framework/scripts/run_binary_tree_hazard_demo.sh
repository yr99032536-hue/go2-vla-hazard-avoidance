#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/install/setup.bash"
set -euo pipefail

FRAMEWORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_MODELS_ROOT="${ROBOT_MODELS_ROOT:-/home/iy/Isaac/Robotics/robot_models}"
GENERATOR="$ROBOT_MODELS_ROOT/src/sim/generate_binary_tree_hazard_map.py"
BINARY_TREE_USD="${BINARY_TREE_USD:-$ROBOT_MODELS_ROOT/assets/usd/binary_tree_hazard/binary_tree_hazard.usda}"
BINARY_TREE_LAYOUT="${BINARY_TREE_LAYOUT:-$ROBOT_MODELS_ROOT/assets/usd/binary_tree_hazard/binary_tree_hazard_layout.json}"
BINARY_TREE_SEED="${BINARY_TREE_SEED:-47}"
RUN_STAMP="${BINARY_TREE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${BINARY_TREE_RUN_ROOT:-/home/iy/Isaac/Robotics/data/binary_tree_hazard/seed_$(printf '%03d' "$BINARY_TREE_SEED")/$RUN_STAMP}"
RUN_LOG="$RUN_ROOT/simulator.log"
SLAM_DB="$RUN_ROOT/rtabmap.db"
ACTIVE_SLAM_LEDGER="$RUN_ROOT/decision_ledger.jsonl"
SECRET_PATH="$(mktemp /tmp/binary_tree_hazard_secret.XXXXXX)"

if [[ ! -f "$GENERATOR" ]]; then
  printf 'Binary-tree map generator is missing: %s\n' "$GENERATOR" >&2
  exit 2
fi

lingering_pattern='/go2_soarm.py|run_active_slam_[^ ]*_ros2.sh|run_warehouse_rtabmap.sh|run_binary_tree_hazard_demo.sh|ros2 launch .*go2_rtabmap.launch.py|/rtabmap .*--ros-args|rviz2 .*go2_sim|binary_tree_hazard_supervisor'
pgrep -af "$lingering_pattern" | awk -v self="$$" '$1 != self' >/tmp/binary_tree_lingering.txt || true
if [[ -s /tmp/binary_tree_lingering.txt ]]; then
  printf 'Refusing to start: an Isaac/SLAM process is already active.\n' >&2
  head -n 20 /tmp/binary_tree_lingering.txt >&2
  rm -f /tmp/binary_tree_lingering.txt "$SECRET_PATH"
  exit 2
fi
rm -f /tmp/binary_tree_lingering.txt

printf 'GPU preflight:\n'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
printf 'System-memory preflight:\n'
free -h | sed -n '1,2p'

generator_args=(
  --output "$BINARY_TREE_USD"
  --layout-output "$BINARY_TREE_LAYOUT"
  --seed "$BINARY_TREE_SEED"
)
if [[ -n "${BINARY_TREE_HAZARD_PATTERN:-}" ]]; then
  generator_args+=(--hazard-pattern "$BINARY_TREE_HAZARD_PATTERN")
fi
python3 "$GENERATOR" "${generator_args[@]}"

mkdir -p "$RUN_ROOT"
umask 077
/usr/bin/python3 -c 'import os, pathlib, sys; pathlib.Path(sys.argv[1]).write_bytes(os.urandom(32))' "$SECRET_PATH"

export ROBOT_MODELS_ROOT
export BINARY_TREE_USD
export BINARY_TREE_LAYOUT
export ACTIVE_SLAM_TRANSACTION_CONTRACT="${ACTIVE_SLAM_TRANSACTION_CONTRACT:-$ROBOT_MODELS_ROOT/soarm_nbv/contracts/transaction_v2.json}"
export ACTIVE_SLAM_HMAC_SECRET="$SECRET_PATH"
export ACTIVE_SLAM_LEDGER
export ACTIVE_GAP_ARM=1
export ACTIVE_ARM_TRAJECTORY_DURATION_S="${ACTIVE_ARM_TRAJECTORY_DURATION_S:-2.5}"
export ACTIVE_GAP_MIN_MAP_REVISION="${ACTIVE_GAP_MIN_MAP_REVISION:-1}"
export SLAM_SCENE=binary_tree
export SLAM_SENSOR=rgbd
export SLAM_USE_WRIST=0
export SLAM_DB
export HEADLESS=0
export RVIZ="${RVIZ:-0}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

sim_extra_args=()
if [[ "${BINARY_TREE_SHOW_CAMERA_VIEWPORT:-0}" == "1" ]]; then
  sim_extra_args+=(--show_camera_viewport)
fi

runner_pid=""
cleanup() {
  set +e
  if [[ -n "$runner_pid" ]] && ps -p "$runner_pid" -o pid= >/dev/null; then
    kill -s TERM "$runner_pid"
    for _ in $(seq 1 15); do
      if ! ps -p "$runner_pid" -o pid= >/dev/null; then break; fi
      sleep 1
    done
    if ps -p "$runner_pid" -o pid= >/dev/null; then
      kill -s KILL "$runner_pid"
    fi
  fi
  rm -f "$SECRET_PATH"
}
trap cleanup EXIT INT TERM

printf 'Starting GUI binary-tree hazard experiment. Artifacts: %s\n' "$RUN_ROOT"
"$FRAMEWORK_ROOT/scripts/run_warehouse_rtabmap.sh" "${sim_extra_args[@]}" >"$RUN_LOG" 2>&1 &
runner_pid=$!

printf 'Waiting for front/wrist RGB-D, map, odometry, and arm action server...\n'
ready=0
for _ in $(seq 1 240); do
  topics="$(ros2 topic list --no-daemon --spin-time 1 || true)"
  actions="$(ros2 action list || true)"
  if printf '%s\n' "$topics" | grep -Fx /joint_states >/dev/null \
      && printf '%s\n' "$topics" | grep -Fx /odom >/dev/null \
      && printf '%s\n' "$topics" | grep -Fx /map >/dev/null \
      && printf '%s\n' "$topics" | grep -Fx /camera/depth/image_rect_raw >/dev/null \
      && printf '%s\n' "$topics" | grep -Fx /wrist_camera/color/image_raw >/dev/null \
      && printf '%s\n' "$actions" | grep -Fx /active_slam/apply_arm_trajectory >/dev/null; then
    ready=1
    break
  fi
  if ! ps -p "$runner_pid" -o pid= >/dev/null; then
    printf 'Simulation exited before binary-tree prerequisites became ready. See %s\n' "$RUN_LOG" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  printf 'Timed out waiting for binary-tree prerequisites. See %s\n' "$RUN_LOG" >&2
  exit 1
fi

PYTHONUNBUFFERED=1 ros2 run go2_active_slam binary_tree_hazard_supervisor
printf 'Binary-tree experiment completed. DB: %s Ledger: %s\n' "$SLAM_DB" "$ACTIVE_SLAM_LEDGER"

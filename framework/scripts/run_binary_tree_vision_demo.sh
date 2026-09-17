#!/usr/bin/env bash
source /opt/ros/humble/setup.bash
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)/install/setup.bash"
set -euo pipefail

FRAMEWORK_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ROBOT_MODELS_ROOT="${ROBOT_MODELS_ROOT:-/home/iy/Isaac/Robotics/robot_models}"
GENERATOR="$ROBOT_MODELS_ROOT/src/sim/generate_binary_tree_hazard_map.py"
SIM_LAUNCHER="$ROBOT_MODELS_ROOT/src/sim/run_active_slam_binary_tree_ros2.sh"
POLICY_VERIFIER="$ROBOT_MODELS_ROOT/src/sim/verify_go2_policy_abi.py"
BINARY_TREE_USD="${BINARY_TREE_USD:-$ROBOT_MODELS_ROOT/assets/usd/binary_tree_hazard/binary_tree_hazard.usda}"
BINARY_TREE_LAYOUT="${BINARY_TREE_LAYOUT:-$ROBOT_MODELS_ROOT/assets/usd/binary_tree_hazard/binary_tree_hazard_layout.json}"
BINARY_TREE_SEED="${BINARY_TREE_SEED:-47}"
RUN_STAMP="${BINARY_TREE_RUN_STAMP:-$(date +%Y%m%d_%H%M%S)}"
RUN_ROOT="${BINARY_TREE_RUN_ROOT:-/home/iy/Isaac/Robotics/data/binary_tree_vision/seed_$(printf '%03d' "$BINARY_TREE_SEED")/$RUN_STAMP}"
HUMAN_DATA_ROOT="${BINARY_TREE_HUMAN_DATA_ROOT:-$RUN_ROOT/human_demos}"
NBV_TEACHER_DATA_ROOT="${BINARY_TREE_NBV_TEACHER_DATA_ROOT:-$RUN_ROOT/nbv_teacher_demos}"
RUN_LOG="$RUN_ROOT/simulator.log"
ACTIVE_SLAM_LEDGER="$RUN_ROOT/decision_ledger.jsonl"
RESET_STATUS_FILE="$RUN_ROOT/episode_reset_status.json"
SECRET_PATH="$(mktemp /tmp/binary_tree_vision_secret.XXXXXX)"

export BINARY_TREE_VLA_ARM_POLICY="${BINARY_TREE_VLA_ARM_POLICY:-0}"
export BINARY_TREE_COLLECT_NBV_TEACHER="${BINARY_TREE_COLLECT_NBV_TEACHER:-0}"
BINARY_TREE_NBV_COLLECTION_LAPS="${BINARY_TREE_NBV_COLLECTION_LAPS:-1}"
BINARY_TREE_NBV_COLLECTION_START_LAP="${BINARY_TREE_NBV_COLLECTION_START_LAP:-1}"
if ! [[ "$BINARY_TREE_NBV_COLLECTION_LAPS" =~ ^[1-9][0-9]*$ ]]; then
  printf 'BINARY_TREE_NBV_COLLECTION_LAPS must be a positive integer: %s\n' \
    "$BINARY_TREE_NBV_COLLECTION_LAPS" >&2
  rm -f "$SECRET_PATH"
  exit 2
fi
if ! [[ "$BINARY_TREE_NBV_COLLECTION_START_LAP" =~ ^[1-9][0-9]*$ ]] \
    || (( BINARY_TREE_NBV_COLLECTION_START_LAP > BINARY_TREE_NBV_COLLECTION_LAPS )); then
  printf 'BINARY_TREE_NBV_COLLECTION_START_LAP must be within 1..%s: %s\n' \
    "$BINARY_TREE_NBV_COLLECTION_LAPS" "$BINARY_TREE_NBV_COLLECTION_START_LAP" >&2
  rm -f "$SECRET_PATH"
  exit 2
fi
if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" != "1" && "$BINARY_TREE_NBV_COLLECTION_LAPS" != "1" ]]; then
  printf 'Multiple laps are supported only for scripted NBV teacher collection.\n' >&2
  rm -f "$SECRET_PATH"
  exit 2
fi
if [[ -z "${BINARY_TREE_MANUAL_ARM_TELEOP+x}" ]]; then
  if [[ "$BINARY_TREE_VLA_ARM_POLICY" == "1" || "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" ]]; then
    export BINARY_TREE_MANUAL_ARM_TELEOP=0
  else
    export BINARY_TREE_MANUAL_ARM_TELEOP=1
  fi
else
  export BINARY_TREE_MANUAL_ARM_TELEOP
fi
export BINARY_TREE_VLA_DECISION_TIMEOUT_S="${BINARY_TREE_VLA_DECISION_TIMEOUT_S:-45}"
if [[ "$BINARY_TREE_VLA_ARM_POLICY" == "1" && "$BINARY_TREE_MANUAL_ARM_TELEOP" == "1" ]]; then
  printf 'VLA arm policy and manual arm teleoperation are mutually exclusive.\n' >&2
  rm -f "$SECRET_PATH"
  exit 2
fi
if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" && "$BINARY_TREE_MANUAL_ARM_TELEOP" == "1" ]]; then
  printf 'Scripted NBV teacher collection requires BINARY_TREE_MANUAL_ARM_TELEOP=0.\n' >&2
  rm -f "$SECRET_PATH"
  exit 2
fi
if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" && "${BINARY_TREE_COLLECT_HUMAN:-0}" == "1" ]]; then
  printf 'Scripted NBV and human collection modes are mutually exclusive.\n' >&2
  rm -f "$SECRET_PATH"
  exit 2
fi
if [[ "$BINARY_TREE_VLA_ARM_POLICY" == "1" ]]; then
  BINARY_TREE_SMOLVLA_POLICY_PATH="${BINARY_TREE_SMOLVLA_POLICY_PATH:-}"
  if [[ -z "$BINARY_TREE_SMOLVLA_POLICY_PATH" || ! -d "$BINARY_TREE_SMOLVLA_POLICY_PATH" ]]; then
    printf 'VLA mode requires a trained 7-state/8-action checkpoint directory via BINARY_TREE_SMOLVLA_POLICY_PATH: %s\n' \
      "$BINARY_TREE_SMOLVLA_POLICY_PATH" >&2
    rm -f "$SECRET_PATH"
    exit 2
  fi
fi

lingering_pattern='/go2_soarm.py|run_active_slam_[^ ]*_ros2.sh|run_binary_tree_(hazard|vision)_demo.sh|/rtabmap .*--ros-args|rviz2 |binary_tree_hazard_supervisor|leader_(bridge_7dof|teleop_bridge).py|smolvla_hazard_policy_runner.py'
pgrep -af "$lingering_pattern" | awk -v self="$$" '$1 != self' > /tmp/binary_tree_vision_lingering.txt || true
if [[ -s /tmp/binary_tree_vision_lingering.txt ]]; then
  printf 'Refusing to start: an Isaac/SLAM process is already active.\n' >&2
  head -n 20 /tmp/binary_tree_vision_lingering.txt >&2
  rm -f /tmp/binary_tree_vision_lingering.txt "$SECRET_PATH"
  exit 2
fi
rm -f /tmp/binary_tree_vision_lingering.txt

BASELINE_VRAM_MIB="$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | head -n 1 | tr -d ' ')"
printf 'GPU preflight:\n'
nvidia-smi --query-gpu=name,memory.used,memory.total --format=csv,noheader
nvidia-smi --query-compute-apps=pid,used_memory,name --format=csv,noheader || true
printf 'System-memory preflight:\n'
free -h | sed -n '1,2p'

generator_args=(--output "$BINARY_TREE_USD" --layout-output "$BINARY_TREE_LAYOUT" --seed "$BINARY_TREE_SEED")
if [[ -n "${BINARY_TREE_HAZARD_PATTERN:-}" ]]; then
  generator_args+=(--hazard-pattern "$BINARY_TREE_HAZARD_PATTERN")
fi
if [[ -n "${BINARY_TREE_CUBE_PATTERN:-}" ]]; then
  if [[ "$BINARY_TREE_VLA_ARM_POLICY" != "1" || "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" || "${BINARY_TREE_COLLECT_HUMAN:-0}" == "1" ]]; then
    printf 'Cube-only intervention is for VLA evaluation, not teacher collection.\n' >&2
    rm -f "$SECRET_PATH"
    exit 2
  fi
  generator_args+=(--cube-pattern "$BINARY_TREE_CUBE_PATTERN")
fi
if [[ "${BINARY_TREE_OPEN_RETURN_CONNECTORS:-0}" == "1" ]]; then
  if [[ "$BINARY_TREE_VLA_ARM_POLICY" != "1" || "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" || "${BINARY_TREE_COLLECT_HUMAN:-0}" == "1" ]]; then
    printf 'Open return connectors are restricted to VLA evaluation.\n' >&2
    rm -f "$SECRET_PATH"
    exit 2
  fi
  generator_args+=(--open-return-connectors)
fi
python3 "$GENERATOR" "${generator_args[@]}"

mkdir -p "$RUN_ROOT"
rm -f "$RESET_STATUS_FILE" "$RESET_STATUS_FILE.tmp"
umask 077
/usr/bin/python3 -c 'import os, pathlib, sys; pathlib.Path(sys.argv[1]).write_bytes(os.urandom(32))' "$SECRET_PATH"

export BINARY_TREE_USD BINARY_TREE_LAYOUT ACTIVE_SLAM_LEDGER
export ACTIVE_SLAM_TRANSACTION_CONTRACT="${ACTIVE_SLAM_TRANSACTION_CONTRACT:-$ROBOT_MODELS_ROOT/soarm_nbv/contracts/transaction_v2.json}"
export ACTIVE_SLAM_HMAC_SECRET="$SECRET_PATH"
export ACTIVE_GAP_ARM=1
export ACTIVE_ARM_TRAJECTORY_DURATION_S="${ACTIVE_ARM_TRAJECTORY_DURATION_S:-2.5}"
export ACTIVE_GAP_MIN_MAP_REVISION=0
export TMAZE_REQUIRE_RTAB_HEALTH=0
export TMAZE_MIN_FRONT_CLEARANCE_M="${TMAZE_MIN_FRONT_CLEARANCE_M:-0.25}"
export BINARY_TREE_LIDAR_APPROACH=1
export BINARY_TREE_BASE_SPEED="${BINARY_TREE_BASE_SPEED:-0.80}"
export BINARY_TREE_RED_RATIO_THRESHOLD="${BINARY_TREE_RED_RATIO_THRESHOLD:-0.001}"
export ACTIVE_ARM_ENFORCE_RUNTIME_VELOCITY_LIMITS=0
export BINARY_TREE_WRIST_DEBUG_DIR="${BINARY_TREE_WRIST_DEBUG_DIR:-$RUN_ROOT/wrist_debug}"
export BINARY_TREE_BALANCE_WARMUP_S="${BINARY_TREE_BALANCE_WARMUP_S:-0.10}"
export BINARY_TREE_ROUTE_SEED="${BINARY_TREE_ROUTE_SEED:-$BINARY_TREE_SEED}"
export GO2_POLICY_PATH="${GO2_POLICY_PATH:-/home/iy/Isaac/IsaacLab/logs/rsl_rl/unitree_go2_so101_7motor_reversed_flat/2026-09-02_05-03-06_home_extend_fold_walk_1024env_headless_lr1e4_v7/exported/policy.pt}"
export GO2_POLICY_SOURCE_CHECKPOINT="${GO2_POLICY_SOURCE_CHECKPOINT:-/home/iy/Isaac/IsaacLab/logs/rsl_rl/unitree_go2_so101_7motor_reversed_flat/2026-09-02_05-03-06_home_extend_fold_walk_1024env_headless_lr1e4_v7/model_12999.pt}"
export GO2_POLICY_OBS_MODE="${GO2_POLICY_OBS_MODE:-flat}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

"/home/iy/miniconda3/envs/isaacsim-5.1/bin/python" "$POLICY_VERIFIER" \
  --policy "$GO2_POLICY_PATH" \
  --source-checkpoint "$GO2_POLICY_SOURCE_CHECKPOINT" \
  --observation-mode "$GO2_POLICY_OBS_MODE"

sim_pid=""
supervisor_pid=""
cleanup() {
  set +e
  if [[ -n "$supervisor_pid" ]] && ps -p "$supervisor_pid" -o pid= >/dev/null; then
    kill -s TERM "$supervisor_pid"
    for _ in $(seq 1 10); do
      if ! ps -p "$supervisor_pid" -o pid= >/dev/null; then break; fi
      sleep 1
    done
    if ps -p "$supervisor_pid" -o pid= >/dev/null; then kill -s KILL "$supervisor_pid"; fi
  fi
  if [[ -n "$sim_pid" ]] && ps -p "$sim_pid" -o pid= >/dev/null; then
    # Let Python unwind the simulation loop and call simulation_app.close().
    # SIGTERM can arrive inside a PhysX event callback and trigger a recursive
    # carb mutex assertion during teardown.
    kill -s INT "$sim_pid"
    for _ in $(seq 1 30); do
      if ! ps -p "$sim_pid" -o pid= >/dev/null; then break; fi
      sleep 1
    done
    if ps -p "$sim_pid" -o pid= >/dev/null; then
      kill -s TERM "$sim_pid"
      for _ in $(seq 1 15); do
        if ! ps -p "$sim_pid" -o pid= >/dev/null; then break; fi
        sleep 1
      done
    fi
    if ps -p "$sim_pid" -o pid= >/dev/null; then kill -s KILL "$sim_pid"; fi
  fi
  hazard_runner_pids="$(pgrep -f "$ROBOT_MODELS_ROOT/soarm_nbv/smolvla_hazard_policy_runner.py" || true)"
  if [[ -n "$hazard_runner_pids" ]]; then
    kill -s TERM $hazard_runner_pids 2>/dev/null || true
    for _ in $(seq 1 10); do
      if ! pgrep -f "$ROBOT_MODELS_ROOT/soarm_nbv/smolvla_hazard_policy_runner.py" >/dev/null; then break; fi
      sleep 1
    done
  fi
  if [[ "${BINARY_TREE_COLLECT_HUMAN:-0}" == "1" && -d "$HUMAN_DATA_ROOT" ]]; then
    PYTHONPATH="$ROBOT_MODELS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      /usr/bin/python3 "$ROBOT_MODELS_ROOT/soarm_nbv/recover_hazard_attempts.py" \
      "$HUMAN_DATA_ROOT" || true
  fi
  if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" && -d "$NBV_TEACHER_DATA_ROOT" ]]; then
    PYTHONPATH="$ROBOT_MODELS_ROOT${PYTHONPATH:+:$PYTHONPATH}" \
      /usr/bin/python3 "$ROBOT_MODELS_ROOT/soarm_nbv/recover_hazard_attempts.py" \
      "$NBV_TEACHER_DATA_ROOT" || true
  fi
  rm -f "$SECRET_PATH"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

sim_args=(
  --active_gap_arm
  --lidar_slam
  --render_interval 4
  --abort_on_locomotion_failure
  --idle_stance_fallback
  --state_debug_every 200
  --go2_policy_path "$GO2_POLICY_PATH"
  --go2_policy_obs_mode "$GO2_POLICY_OBS_MODE"
)
if [[ "$BINARY_TREE_MANUAL_ARM_TELEOP" == "1" ]]; then
  LEADER_PORT_DEV="${LEADER_PORT_DEV:-/dev/serial/by-id/usb-1a86_USB_Single_Serial_5A7C123117-if00}"
  if [[ ! -e "$LEADER_PORT_DEV" ]]; then
    printf 'Manual arm teleoperation requested but leader port is missing: %s\n' "$LEADER_PORT_DEV" >&2
    exit 2
  fi
  sim_args+=(
    --leader_auto
    --leader_apply_only_when_base_paused
    --leader_port_dev "$LEADER_PORT_DEV"
    --leader_action_log_every 50
  )
fi
if [[ "$BINARY_TREE_VLA_ARM_POLICY" == "1" ]]; then
  sim_args+=(
    --enable_hazard_smolvla_policy
    --hazard_smolvla_policy_path "$BINARY_TREE_SMOLVLA_POLICY_PATH"
    --hazard_smolvla_obs_port "${BINARY_TREE_SMOLVLA_OBS_PORT:-5585}"
    --hazard_smolvla_action_port "${BINARY_TREE_SMOLVLA_ACTION_PORT:-5586}"
    --hazard_smolvla_fps "${BINARY_TREE_SMOLVLA_FPS:-30}"
    --hazard_smolvla_device "${BINARY_TREE_SMOLVLA_DEVICE:-cuda}"
  )
  if [[ "${BINARY_TREE_SMOLVLA_NO_AMP:-0}" == "1" ]]; then
    sim_args+=(--hazard_smolvla_no_amp)
  fi
fi
if [[ "${BINARY_TREE_COLLECT_HUMAN:-0}" == "1" ]]; then
  if [[ "$BINARY_TREE_VLA_ARM_POLICY" == "1" ]]; then
    printf 'Human collection and trained VLA inference cannot run together.\n' >&2
    exit 2
  fi
  if [[ "$BINARY_TREE_MANUAL_ARM_TELEOP" != "1" ]]; then
    printf 'Human hazard collection requires BINARY_TREE_MANUAL_ARM_TELEOP=1.\n' >&2
    exit 2
  fi
  sim_args+=(
    --collect_hazard_vla
    --hazard_collect_out_dir "$HUMAN_DATA_ROOT"
    --hazard_collect_fps "${BINARY_TREE_HUMAN_COLLECT_FPS:-30}"
    --hazard_terminal_hold_s "${BINARY_TREE_HUMAN_TERMINAL_HOLD_S:-0.35}"
  )
  printf 'Human teacher keys: R=start, H=HAZARD, S=SAFE, X=discard.\n'
fi
if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" ]]; then
  if [[ "$BINARY_TREE_VLA_ARM_POLICY" == "1" ]]; then
    printf 'Scripted NBV collection and trained VLA inference cannot run together.\n' >&2
    exit 2
  fi
  sim_args+=(
    --collect_hazard_nbv_teacher
    --hazard_collect_out_dir "$NBV_TEACHER_DATA_ROOT"
    --hazard_collect_fps "${BINARY_TREE_NBV_COLLECT_FPS:-30}"
  )
  if (( BINARY_TREE_NBV_COLLECTION_LAPS > 1 )); then
    sim_args+=(
      --binary_tree_repeat_reset_status_file "$RESET_STATUS_FILE"
    )
  fi
  printf 'Scripted NBV teacher collection: automatic per-alley episodes -> %s\n' \
    "$NBV_TEACHER_DATA_ROOT"
  printf 'Persistent-GUI batch: %s laps x 6 alley episodes = %s episodes.\n' \
    "$BINARY_TREE_NBV_COLLECTION_LAPS" "$((BINARY_TREE_NBV_COLLECTION_LAPS * 6))"
  if (( BINARY_TREE_NBV_COLLECTION_START_LAP > 1 )); then
    printf 'Resuming persistent-GUI target at lap %s/%s.\n' \
      "$BINARY_TREE_NBV_COLLECTION_START_LAP" "$BINARY_TREE_NBV_COLLECTION_LAPS"
  fi
fi
if [[ "${BINARY_TREE_SHOW_CAMERA_VIEWPORT:-1}" == "1" ]]; then
  sim_args+=(--show_camera_viewport)
fi
printf 'Starting GUI vision-only binary-tree task. No RTAB-Map, TSDF, or RViz. Artifacts: %s\n' "$RUN_ROOT"
"$SIM_LAUNCHER" "${sim_args[@]}" > "$RUN_LOG" 2>&1 &
sim_pid=$!

base_route_seed="$BINARY_TREE_ROUTE_SEED"
start_supervisor() {
  local lap="$1"
  local lap_route_seed="$((base_route_seed + lap - 1))"
  # Start before the first clock tick on lap 1. Later supervisors are started
  # only after the simulator atomically acknowledges its in-place reset.
  if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" ]]; then
    BINARY_TREE_COLLECTION_LAP="$lap" \
    BINARY_TREE_ROUTE_SEED="$lap_route_seed" \
    PYTHONUNBUFFERED=1 \
      ros2 run go2_active_slam binary_tree_hazard_supervisor &
  else
    PYTHONUNBUFFERED=1 ros2 run go2_active_slam binary_tree_hazard_supervisor &
  fi
  supervisor_pid=$!
  printf 'Started binary-tree supervisor lap %s/%s (route seed %s).\n' \
    "$lap" "$BINARY_TREE_NBV_COLLECTION_LAPS" "$lap_route_seed"
}

if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" ]]; then
  accepted_before="$(find "$NBV_TEACHER_DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'env_*' 2>/dev/null | wc -l)"
  rejected_before="$(find "$NBV_TEACHER_DATA_ROOT" -mindepth 3 -maxdepth 3 -type d -path '*/rejected/*' 2>/dev/null | wc -l)"
  staging_before="$(find "$NBV_TEACHER_DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name '.attempt_*' 2>/dev/null | wc -l)"
  expected_before="$(((BINARY_TREE_NBV_COLLECTION_START_LAP - 1) * 6))"
  if [[ "$accepted_before" -ne "$expected_before" || "$rejected_before" -ne 0 || "$staging_before" -ne 0 ]]; then
    printf 'Collection resume preflight failed: accepted=%s expected=%s rejected=%s staging=%s\n' \
      "$accepted_before" "$expected_before" "$rejected_before" "$staging_before" >&2
    exit 1
  fi
fi

start_supervisor "$BINARY_TREE_NBV_COLLECTION_START_LAP"

printf 'Waiting for single-scan LiDAR, RGB-D safety input, wrist RGB, joints, internal route odometry, and arm action server...\n'
ready=0
for _ in $(seq 1 "${ISAAC_STARTUP_TIMEOUT_S:-240}"); do
  topics="$(ros2 topic list --no-daemon --spin-time 1 || true)"
  actions="$(ros2 action list || true)"
  teleop_ready=1
  if [[ "$BINARY_TREE_MANUAL_ARM_TELEOP" == "1" ]] \
      && ! pgrep -f "$ROBOT_MODELS_ROOT/soarm_nbv/leader_bridge_7dof.py" >/dev/null; then
    teleop_ready=0
  fi
  vla_ready=1
  if [[ "$BINARY_TREE_VLA_ARM_POLICY" == "1" ]] \
      && ! pgrep -f "$ROBOT_MODELS_ROOT/soarm_nbv/smolvla_hazard_policy_runner.py" >/dev/null; then
    vla_ready=0
  fi
  if grep -Fxq /camera/depth/image_rect_raw <<< "$topics" \
      && grep -Fxq /wrist_camera/color/image_raw <<< "$topics" \
      && grep -Fxq /utlidar/scan <<< "$topics" \
      && grep -Fxq /joint_states <<< "$topics" \
      && grep -Fxq /odom <<< "$topics" \
      && grep -Fxq /active_slam/apply_arm_trajectory <<< "$actions" \
      && grep -Fxq /active_slam/vla_alley_decision <<< "$topics" \
      && [[ "$teleop_ready" -eq 1 ]] \
      && [[ "$vla_ready" -eq 1 ]]; then
    ready=1
    break
  fi
  if ! ps -p "$sim_pid" -o pid= >/dev/null; then
    printf 'Simulator exited before vision-task prerequisites became ready. See %s\n' "$RUN_LOG" >&2
    exit 1
  fi
  if ! ps -p "$supervisor_pid" -o pid= >/dev/null; then
    printf 'Supervisor exited before vision-task prerequisites became ready. See %s\n' "$RUN_LOG" >&2
    exit 1
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  printf 'Timed out waiting for vision-task prerequisites. See %s\n' "$RUN_LOG" >&2
  exit 1
fi

for lap in $(seq "$BINARY_TREE_NBV_COLLECTION_START_LAP" "$BINARY_TREE_NBV_COLLECTION_LAPS"); do
  while ps -p "$supervisor_pid" -o pid= >/dev/null \
      && ps -p "$sim_pid" -o pid= >/dev/null; do
    sleep 1
  done

  if ! ps -p "$sim_pid" -o pid= >/dev/null; then
    set +e
    wait "$sim_pid"
    simulator_rc=$?
    set -e
    sim_pid=""
    if ps -p "$supervisor_pid" -o pid= >/dev/null; then
      kill -s TERM "$supervisor_pid"
      wait "$supervisor_pid" 2>/dev/null || true
    fi
    supervisor_pid=""
    printf 'Simulator exited during lap %s/%s (code %s). See %s\n' \
      "$lap" "$BINARY_TREE_NBV_COLLECTION_LAPS" "$simulator_rc" "$RUN_LOG" >&2
    if [[ "$simulator_rc" -eq 0 ]]; then exit 1; fi
    exit "$simulator_rc"
  fi

  set +e
  wait "$supervisor_pid"
  supervisor_rc=$?
  set -e
  supervisor_pid=""
  if [[ "$supervisor_rc" -ne 0 ]]; then
    printf 'Vision supervisor failed on lap %s/%s with code %s. Ledger: %s\n' \
      "$lap" "$BINARY_TREE_NBV_COLLECTION_LAPS" "$supervisor_rc" "$ACTIVE_SLAM_LEDGER" >&2
    exit "$supervisor_rc"
  fi

  if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" ]]; then
    accepted_count="$(find "$NBV_TEACHER_DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name 'env_*' 2>/dev/null | wc -l)"
    rejected_count="$(find "$NBV_TEACHER_DATA_ROOT" -mindepth 3 -maxdepth 3 -type d -path '*/rejected/*' 2>/dev/null | wc -l)"
    staging_count="$(find "$NBV_TEACHER_DATA_ROOT" -mindepth 2 -maxdepth 2 -type d -name '.attempt_*' 2>/dev/null | wc -l)"
    expected_count="$((lap * 6))"
    if [[ "$accepted_count" -ne "$expected_count" || "$rejected_count" -ne 0 || "$staging_count" -ne 0 ]]; then
      printf 'Collection audit failed after lap %s: accepted=%s expected=%s rejected=%s staging=%s\n' \
        "$lap" "$accepted_count" "$expected_count" "$rejected_count" "$staging_count" >&2
      exit 1
    fi
    printf 'Lap %s/%s accepted and audited: %s/%s episodes.\n' \
      "$lap" "$BINARY_TREE_NBV_COLLECTION_LAPS" "$accepted_count" \
      "$((BINARY_TREE_NBV_COLLECTION_LAPS * 6))"
  fi

  if (( lap == BINARY_TREE_NBV_COLLECTION_LAPS )); then
    break
  fi

  next_lap="$((lap + 1))"
  rm -f "$RESET_STATUS_FILE" "$RESET_STATUS_FILE.tmp"
  if ! timeout 20s ros2 topic pub --once /active_slam/episode_reset_request \
      std_msgs/msg/UInt32 "{data: $next_lap}" >/dev/null; then
    printf 'Failed to publish reset request for lap %s.\n' "$next_lap" >&2
    exit 1
  fi
  reset_ready=0
  for _ in $(seq 1 "${BINARY_TREE_RESET_TIMEOUT_S:-240}"); do
    if ! ps -p "$sim_pid" -o pid= >/dev/null; then
      printf 'Simulator exited while resetting for lap %s. See %s\n' "$next_lap" "$RUN_LOG" >&2
      exit 1
    fi
    if [[ -s "$RESET_STATUS_FILE" ]] \
        && grep -Fq '"status": "ready"' "$RESET_STATUS_FILE" \
        && grep -Fq "\"ready_lap\": $next_lap" "$RESET_STATUS_FILE"; then
      reset_ready=1
      break
    fi
    sleep 1
  done
  if [[ "$reset_ready" -ne 1 ]]; then
    printf 'Timed out waiting for in-place simulator reset for lap %s. See %s\n' \
      "$next_lap" "$RUN_LOG" >&2
    exit 1
  fi
  start_supervisor "$next_lap"
done

if [[ "$BINARY_TREE_COLLECT_NBV_TEACHER" == "1" ]]; then
  printf 'Vision-only binary-tree task completed: %s lap(s), %s episode(s). Ledger: %s\n' \
    "$BINARY_TREE_NBV_COLLECTION_LAPS" "$((BINARY_TREE_NBV_COLLECTION_LAPS * 6))" \
    "$ACTIVE_SLAM_LEDGER"
else
  printf 'Vision-only binary-tree task completed. Ledger: %s\n' "$ACTIVE_SLAM_LEDGER"
fi

#!/usr/bin/env bash
set -eo pipefail

source /opt/ros/humble/setup.bash
framework_root="/home/iy/Isaac/Go2_Intelligence_Framework"
robot_models_root="/home/iy/Isaac/Robotics/robot_models"
source "$framework_root/install/setup.bash"
set -u

existing_sim_processes="$(
  pgrep -af '[g]o2_soarm.py|[i]saac-sim|[r]tabmap|[r]viz2' || true
)"
if [[ -n "$existing_sim_processes" ]]; then
  printf 'Refusing to start a second Isaac/SLAM visualization stack:\n%s\n' \
    "$existing_sim_processes" >&2
  exit 2
fi
printf 'Preflight GPU state:\n'
nvidia-smi --query-compute-apps=pid,process_name,used_memory --format=csv,noheader || true
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu \
  --format=csv,noheader
printf 'Preflight system memory:\n'
free -h

generator="$robot_models_root/src/sim/generate_cluttered_warehouse.py"
warehouse_usd="${WAREHOUSE_USD:-$robot_models_root/assets/usd/warehouse_cluttered.usda}"
warehouse_seed="${WAREHOUSE_SEED:-47}"
export WAREHOUSE_SEED="$warehouse_seed"
lap_index="${NBV_TEACHER_LAP_INDEX:-0}"
experiment_root="${NBV_EXPERIMENT_ROOT:-/home/iy/Isaac/Robotics/data/active_mapping_ab/seed_$(printf '%03d' "$warehouse_seed")/arm_front_wrist_nbv/lap_$(printf '%03d' "$lap_index")}"
mkdir -p "$experiment_root"
secret_path="$(mktemp /tmp/nbv_teacher_secret.XXXXXX)"
run_log="${NBV_TEACHER_RUN_LOG:-$experiment_root/simulator.log}"
tsdf_log="${NBV_TSDF_LOG:-$experiment_root/tsdf.log}"
slam_db="${NBV_TEACHER_SLAM_DB:-$experiment_root/rtabmap.db}"
tsdf_ply="${NBV_TSDF_OUTPUT_PLY:-$experiment_root/map_tsdf.ply}"
tsdf_metadata="${NBV_TSDF_OUTPUT_METADATA_JSON:-$experiment_root/metrics.json}"
save_response_log="${NBV_TSDF_SAVE_RESPONSE_LOG:-$experiment_root/save_tsdf_response.log}"
allow_overwrite="${NBV_ALLOW_OVERWRITE:-0}"
for output_path in "$slam_db" "$tsdf_ply" "$tsdf_metadata"; do
  if [[ -e "$output_path" && "$allow_overwrite" != "1" ]]; then
    printf 'Refusing to overwrite existing NBV experiment artifact: %s\n' "$output_path" >&2
    exit 2
  fi
done

export ACTIVE_SLAM_TRANSACTION_CONTRACT="${ACTIVE_SLAM_TRANSACTION_CONTRACT:-$robot_models_root/soarm_nbv/contracts/transaction_v2.json}"
export ACTIVE_SLAM_HMAC_SECRET="$secret_path"
export ACTIVE_SLAM_LEDGER="${ACTIVE_SLAM_LEDGER:-$experiment_root/active_slam_ledger.jsonl}"
export ACTIVE_GAP_ARM=1
export ACTIVE_GAP_MIN_MAP_REVISION=1
export SLAM_SCENE=warehouse
export SLAM_SENSOR=rgbd
export SLAM_USE_WRIST=0
export SLAM_DB="$slam_db"
export WAREHOUSE_USD="$warehouse_usd"
# Collection runs are always visible so the operator can catch falls, tangles,
# bad spawn poses, and sensor/scene problems that numeric checks may miss.
export HEADLESS=0
export RVIZ="${RVIZ:-0}"
export EXIT_ON_ROUTE_COMPLETE=0
export GO2_POLICY_PATH="${GO2_POLICY_PATH:-/home/iy/Isaac/IsaacLab/logs/rsl_rl/unitree_go2_so101_7motor_reversed_rough_walking/resume_12000_to_20000/exported/policy.pt}"
export GO2_POLICY_SOURCE_CHECKPOINT="${GO2_POLICY_SOURCE_CHECKPOINT:-/home/iy/Isaac/IsaacLab/logs/rsl_rl/unitree_go2_so101_7motor_reversed_rough_walking/resume_12000_to_20000/model_19750.pt}"
export GO2_POLICY_OBS_MODE="${GO2_POLICY_OBS_MODE:-flat}"
export NBV_TEACHER_OUT="${NBV_TEACHER_OUT:-$experiment_root/teacher_dataset}"
export NBV_TEACHER_SAFETY_MAX_EVENTS="${NBV_TEACHER_SAFETY_MAX_EVENTS:-${NBV_TEACHER_EPISODES:-12}}"
export NBV_TEACHER_COMPLETE_ON_ROUTE="${NBV_TEACHER_COMPLETE_ON_ROUTE:-1}"
export NBV_TEACHER_LAP_INDEX="$lap_index"
export NBV_TEACHER_CANDIDATE_LIMIT="${NBV_TEACHER_CANDIDATE_LIMIT:-9}"
export NBV_TEACHER_REARM_TRAVEL_M="${NBV_TEACHER_REARM_TRAVEL_M:-0.35}"
export NBV_TEACHER_GAP_EXCLUSION_RADIUS_M="${NBV_TEACHER_GAP_EXCLUSION_RADIUS_M:-0.60}"
export NBV_TEACHER_GAP_CHECK_INTERVAL_S="${NBV_TEACHER_GAP_CHECK_INTERVAL_S:-0.50}"
export NBV_TEACHER_HOME_SETTLE_S="${NBV_TEACHER_HOME_SETTLE_S:-1.50}"
export NBV_TEACHER_WARMUP_S="${NBV_TEACHER_WARMUP_S:-6.0}"
export NBV_TEACHER_FPS="${NBV_TEACHER_FPS:-10.0}"
# Simulation time advances while CPU-heavy candidate scoring and ROS action
# dispatch run.  This simulation-only oracle lane uses a longer transport
# lease; deployment keeps the supervisor's default 5-second maximum.  No
# trajectory, collision, or runtime joint safety check is relaxed.
export ACTIVE_ARM_AUTHORIZATION_TTL_S="${ACTIVE_ARM_AUTHORIZATION_TTL_S:-10.0}"
export ACTIVE_ARM_AUTHORIZATION_TTL_MAX_S="${ACTIVE_ARM_AUTHORIZATION_TTL_MAX_S:-10.0}"
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp

"/home/iy/miniconda3/envs/isaacsim-5.1/bin/python" \
  "$robot_models_root/src/sim/verify_go2_policy_abi.py" \
  --policy "$GO2_POLICY_PATH" \
  --source-checkpoint "$GO2_POLICY_SOURCE_CHECKPOINT" \
  --observation-mode "$GO2_POLICY_OBS_MODE"

umask 077
head -c 32 /dev/urandom > "$secret_path"
python3 "$generator" --output "$warehouse_usd" --seed "$warehouse_seed" --layout outer_loop

warehouse_route=(
  -5.20 -4.45  -4.75 -4.45  -4.20 -4.45  -3.50 -4.55  -2.80 -4.60
  -2.20 -4.50  -1.00 -4.40  -0.80 -3.70   0.80 -3.70   0.80 -4.80
   1.50 -4.80   3.50 -4.45   5.30 -4.00   5.50 -2.80   5.60 -0.40
   4.70  0.30   4.70  2.80   5.00  3.80   3.70  3.80   1.00  3.80
   0.40  4.50  -1.00  4.50  -3.60  4.50  -5.10  4.00  -5.40  3.00
  -5.30  0.00  -4.70 -0.50  -4.70 -2.80  -5.20 -4.45
)
if [[ -n "${NBV_TEACHER_ROUTE_XY:-}" ]]; then
  read -r -a warehouse_route <<< "$NBV_TEACHER_ROUTE_XY"
  if [[ "${#warehouse_route[@]}" -lt 4 || $(( ${#warehouse_route[@]} % 2 )) -ne 0 ]]; then
    printf 'NBV_TEACHER_ROUTE_XY must contain at least two XY pairs, got %s values.\n' \
      "${#warehouse_route[@]}" >&2
    exit 2
  fi
fi

route_motion_args=()
if [[ "${NBV_TEACHER_KINEMATIC_ROUTE:-0}" == "1" ]]; then
  route_motion_args+=(--kinematic_scripted_route)
  export NBV_TEACHER_ROUTE_MOTION_MODE=kinematic_carrier
else
  export NBV_TEACHER_ROUTE_MOTION_MODE=physical_rl_policy
fi

runner_pid=""
tsdf_pid=""
teacher_pid=""
cleanup() {
  set +e
  if [[ -n "$teacher_pid" ]] && ps -p "$teacher_pid" -o pid= >/dev/null; then
    kill -s TERM "$teacher_pid"
    for _ in $(seq 1 10); do
      if ! ps -p "$teacher_pid" -o pid= >/dev/null; then break; fi
      sleep 1
    done
    if ps -p "$teacher_pid" -o pid= >/dev/null; then kill -s KILL "$teacher_pid"; fi
  fi
  if [[ -n "$tsdf_pid" ]] && ps -p "$tsdf_pid" -o pid= >/dev/null; then
    kill -s TERM "$tsdf_pid"
    for _ in $(seq 1 10); do
      if ! ps -p "$tsdf_pid" -o pid= >/dev/null; then break; fi
      sleep 1
    done
    if ps -p "$tsdf_pid" -o pid= >/dev/null; then kill -s KILL "$tsdf_pid"; fi
  fi
  if [[ -n "$runner_pid" ]] && ps -p "$runner_pid" -o pid= >/dev/null; then
    kill -s TERM "$runner_pid"
    for _ in $(seq 1 10); do
      if ! ps -p "$runner_pid" -o pid= >/dev/null; then break; fi
      sleep 1
    done
    if ps -p "$runner_pid" -o pid= >/dev/null; then kill -s KILL "$runner_pid"; fi
  fi
  rm -f "$secret_path"
}
trap cleanup EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

"$framework_root/scripts/run_warehouse_rtabmap.sh" \
  --robot_spawn_xy -5.20 -4.45 \
  --robot_spawn_yaw_deg 0 \
  --viewport_topdown \
  --scripted_route "${warehouse_route[@]}" \
  --scripted_route_stand_steps "${NBV_TEACHER_ROUTE_STAND_STEPS:-100}" \
  --scripted_route_speed "${ROUTE_SPEED:-1.00}" \
  --scripted_route_tolerance 0.35 \
  --scripted_route_yaw_gain "${ROUTE_YAW_GAIN:-1.00}" \
  --scripted_route_max_yaw_rate "${ROUTE_MAX_YAW_RATE:-0.50}" \
  --scripted_route_forward_alignment_rad "${ROUTE_FORWARD_ALIGNMENT_RAD:-0.50}" \
  --scripted_route_linear_slew "${ROUTE_LINEAR_SLEW:-0.02}" \
  --scripted_route_yaw_slew "${ROUTE_YAW_SLEW:-0.02}" \
  --show_scripted_route \
  "${route_motion_args[@]}" \
  --abort_on_locomotion_failure \
  --state_debug_every "${STATE_DEBUG_EVERY:-25}" \
  --render_interval "${RENDER_INTERVAL:-4}" > "$run_log" 2>&1 &
runner_pid=$!

printf 'Waiting for NBV teacher prerequisites. Simulator log: %s\n' "$run_log"
ready=0
for _ in $(seq 1 "${ISAAC_STARTUP_TIMEOUT_S:-240}"); do
  topics="$(ros2 topic list --no-daemon --spin-time 1 || true)"
  actions="$(ros2 action list || true)"
  required_topics=(
    /camera/color/image_raw /camera/depth/image_rect_raw /camera/camera_info
    /wrist_camera/color/image_raw /wrist_camera/depth/image_rect_raw /wrist_camera/camera_info
    /joint_states /odom /map /tf
    /active_slam/route_complete
  )
  missing=0
  for topic in "${required_topics[@]}"; do
    if ! grep -Fxq "$topic" <<< "$topics"; then missing=1; break; fi
  done
  if [[ "$missing" -eq 0 ]] && grep -Fxq /active_slam/apply_arm_trajectory <<< "$actions"; then
    ready=1
    break
  fi
  if ! ps -p "$runner_pid" -o pid= >/dev/null; then
    printf 'Simulation exited before teacher prerequisites became ready.\n' >&2
    tail -80 "$run_log" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$ready" -ne 1 ]]; then
  printf 'Timed out waiting for NBV teacher prerequisites.\n' >&2
  exit 1
fi

env \
  NBV_TSDF_ENABLE_FRONT=1 \
  NBV_TSDF_ENABLE_WRIST=1 \
  NBV_TSDF_WRIST_REQUIRES_GATE=1 \
  NBV_TSDF_PUBLISH_SURFACE="${NBV_TSDF_PUBLISH_SURFACE:-1}" \
  NBV_TSDF_PUBLISH_GUIDANCE="${NBV_TSDF_PUBLISH_GUIDANCE:-0}" \
  NBV_DEPTH_STRIDE="${NBV_DEPTH_STRIDE:-16}" \
  NBV_TSDF_SAVE_ON_SHUTDOWN=0 \
  NBV_TSDF_OUTPUT_PLY="$tsdf_ply" \
  NBV_TSDF_OUTPUT_METADATA_JSON="$tsdf_metadata" \
  NBV_TSDF_EXPERIMENT_CONDITION=arm_front_wrist_nbv \
  NBV_TSDF_FUSION_FRAME=odom \
  WAREHOUSE_SEED="$warehouse_seed" \
  "$framework_root/scripts/run_nbv_tsdf.sh" > "$tsdf_log" 2>&1 &
tsdf_pid=$!

tsdf_ready=0
for _ in $(seq 1 "${NBV_TSDF_STARTUP_TIMEOUT_S:-60}"); do
  topics="$(ros2 topic list --no-daemon --spin-time 1 || true)"
  services="$(ros2 service list || true)"
  if grep -Fxq /active_slam/tsdf_status <<< "$topics" \
    && grep -Fxq /active_slam/save_tsdf <<< "$services"; then
    tsdf_ready=1
    break
  fi
  if ! ps -p "$tsdf_pid" -o pid= >/dev/null; then
    printf 'TSDF fusion exited before becoming ready.\n' >&2
    tail -80 "$tsdf_log" >&2 || true
    exit 1
  fi
  sleep 1
done
if [[ "$tsdf_ready" -ne 1 ]]; then
  printf 'Timed out waiting for TSDF fusion.\n' >&2
  exit 1
fi

if [[ "$NBV_TEACHER_COMPLETE_ON_ROUTE" == "1" ]]; then
  printf 'Collecting map-gap-driven lap %s (seed=%s, safety_max_events=%s) into %s\n' \
    "$NBV_TEACHER_LAP_INDEX" "$warehouse_seed" "$NBV_TEACHER_SAFETY_MAX_EVENTS" "$NBV_TEACHER_OUT"
else
  printf 'Collecting map-gap-driven NBV teacher episodes (safety_max_events=%s) into %s\n' \
    "$NBV_TEACHER_SAFETY_MAX_EVENTS" "$NBV_TEACHER_OUT"
fi
PYTHONUNBUFFERED=1 ros2 run go2_active_slam nbv_teacher_collector &
teacher_pid=$!
while true; do
  teacher_state="$(ps -p "$teacher_pid" -o stat= 2>/dev/null | tr -d '[:space:]')"
  if [[ -z "$teacher_state" || "$teacher_state" == Z* ]]; then
    break
  fi
  if ! ps -p "$runner_pid" -o pid= >/dev/null; then
    set +e
    wait "$runner_pid"
    runner_status=$?
    set -e
    runner_pid=""
    printf 'Simulator/RTAB-Map runner exited during NBV collection (status=%s).\n' "$runner_status" >&2
    tail -80 "$run_log" >&2 || true
    if [[ "$runner_status" -eq 0 ]]; then runner_status=1; fi
    exit "$runner_status"
  fi
  if ! ps -p "$tsdf_pid" -o pid= >/dev/null; then
    set +e
    wait "$tsdf_pid"
    tsdf_status=$?
    set -e
    tsdf_pid=""
    printf 'TSDF fusion exited during NBV collection (status=%s).\n' "$tsdf_status" >&2
    tail -80 "$tsdf_log" >&2 || true
    if [[ "$tsdf_status" -eq 0 ]]; then tsdf_status=1; fi
    exit "$tsdf_status"
  fi
  sleep 1
done
set +e
wait "$teacher_pid"
teacher_status=$?
set -e
teacher_pid=""
if [[ "$teacher_status" -ne 0 ]]; then
  printf 'NBV teacher collector failed with status %s.\n' "$teacher_status" >&2
  exit "$teacher_status"
fi
ros2 service call /active_slam/save_tsdf std_srvs/srv/Trigger '{}' > "$save_response_log"
if ! grep -Eq 'success[=:][[:space:]]*(true|True)' "$save_response_log"; then
  printf 'TSDF save service did not report success:\n' >&2
  sed -n '1,80p' "$save_response_log" >&2
  exit 1
fi
for artifact in "$tsdf_ply" "$tsdf_metadata" "$slam_db"; do
  if [[ ! -s "$artifact" ]]; then
    printf 'NBV experiment artifact is missing or empty: %s\n' "$artifact" >&2
    exit 1
  fi
done
image_count="$(find "$NBV_TEACHER_OUT" -type f -name '*.png' | wc -l)"
if [[ "$image_count" -gt 0 && "$image_count" -lt 3 ]]; then
  printf 'NBV teacher dataset has too few images: %s\n' "$image_count" >&2
  exit 1
fi
if [[ "$image_count" -eq 0 ]]; then
  printf 'No qualifying new map gap was detected; zero arm episodes were forced.\n'
fi
printf 'NBV teacher collection complete: %s\n' "$NBV_TEACHER_OUT"
printf 'TSDF map saved: %s (%s teacher PNG files)\n' "$tsdf_ply" "$image_count"

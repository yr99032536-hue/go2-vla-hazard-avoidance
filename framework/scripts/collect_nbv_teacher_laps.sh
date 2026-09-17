#!/usr/bin/env bash
set -euo pipefail
trap 'exit 130' INT
trap 'exit 143' TERM

framework_root="/home/iy/Isaac/Go2_Intelligence_Framework"
single_lap_runner="$framework_root/scripts/run_nbv_teacher_collect.sh"
laps="${NBV_TEACHER_LAPS:-3}"
base_seed="${NBV_TEACHER_BASE_SEED:-47}"
seed_stride="${NBV_TEACHER_SEED_STRIDE:-1}"
output="${NBV_TEACHER_OUT:-/home/iy/Isaac/Robotics/data/nbv_teacher_collect}"
gpu_index="${NBV_GPU_INDEX:-0}"
gpu_recovery_timeout_s="${NBV_GPU_RECOVERY_TIMEOUT_S:-180}"
gpu_recovery_margin_mib="${NBV_GPU_RECOVERY_MARGIN_MIB:-768}"
lingering_pattern='/go2_soarm.py|run_active_slam_[^ ]*_ros2.sh|run_warehouse_rtabmap.sh|ros2 launch .*go2_rtabmap.launch.py|/rtabmap .*--ros-args|rviz2 .*go2_sim'

for value_name in laps base_seed seed_stride; do
  value="${!value_name}"
  if [[ ! "$value" =~ ^[0-9]+$ ]]; then
    printf '%s must be a non-negative integer, got: %s\n' "$value_name" "$value" >&2
    exit 2
  fi
done
if [[ "$laps" -lt 1 ]]; then
  printf 'NBV_TEACHER_LAPS must be at least one.\n' >&2
  exit 2
fi
if ! command -v nvidia-smi >/dev/null; then
  printf 'nvidia-smi is required for the Isaac GPU recovery safety gate.\n' >&2
  exit 2
fi

gpu_value_mib() {
  local field="$1"
  nvidia-smi --id="$gpu_index" --query-gpu="$field" --format=csv,noheader,nounits \
    | head -1 | tr -d '[:space:]'
}

assert_no_lingering_simulation() {
  local matches
  matches="$(pgrep -af "$lingering_pattern" || true)"
  if [[ -n "$matches" ]]; then
    printf 'Refusing to start another Isaac lap; related processes are still alive:\n%s\n' "$matches" >&2
    return 1
  fi
}

wait_for_gpu_recovery() {
  local deadline stable_checks used_mib
  deadline=$((SECONDS + gpu_recovery_timeout_s))
  stable_checks=0
  while (( SECONDS <= deadline )); do
    if ! assert_no_lingering_simulation; then
      stable_checks=0
    else
      used_mib="$(gpu_value_mib memory.used)"
      if [[ "$used_mib" =~ ^[0-9]+$ ]] && (( used_mib <= baseline_gpu_used_mib + gpu_recovery_margin_mib )); then
        stable_checks=$((stable_checks + 1))
        if (( stable_checks >= 2 )); then
          printf 'GPU recovery confirmed: used=%s MiB baseline=%s MiB.\n' \
            "$used_mib" "$baseline_gpu_used_mib"
          return 0
        fi
      else
        stable_checks=0
      fi
    fi
    sleep 5
  done
  printf 'GPU/Vulkan resources did not recover within %s seconds; aborting batch.\n' \
    "$gpu_recovery_timeout_s" >&2
  return 1
}

mkdir -p "$output"
assert_no_lingering_simulation
baseline_gpu_used_mib="$(gpu_value_mib memory.used)"
if [[ ! "$baseline_gpu_used_mib" =~ ^[0-9]+$ ]]; then
  printf 'Could not read baseline GPU memory usage.\n' >&2
  exit 2
fi
if [[ "${NBV_SAFETY_PREFLIGHT_ONLY:-0}" == "1" ]]; then
  printf 'Isaac batch safety preflight passed: GPU %s used=%s MiB, no related process remains.\n' \
    "$gpu_index" "$baseline_gpu_used_mib"
  exit 0
fi
printf 'Starting %s independent NBV collection laps. Output: %s\n' "$laps" "$output"
for ((lap_index = 0; lap_index < laps; lap_index++)); do
  assert_no_lingering_simulation
  warehouse_seed=$((base_seed + lap_index * seed_stride))
  printf '\n[%s/%s] Starting lap_index=%s warehouse_seed=%s\n' \
    "$((lap_index + 1))" "$laps" "$lap_index" "$warehouse_seed"
  env \
    WAREHOUSE_SEED="$warehouse_seed" \
    NBV_TEACHER_LAP_INDEX="$lap_index" \
    NBV_TEACHER_COMPLETE_ON_ROUTE=1 \
    NBV_TEACHER_OUT="$output" \
    NBV_TEACHER_RUN_LOG="${NBV_TEACHER_RUN_LOG_PREFIX:-/tmp/nbv_teacher_lap}_${lap_index}.log" \
    "$single_lap_runner"
  printf '[%s/%s] Completed lap_index=%s warehouse_seed=%s\n' \
    "$((lap_index + 1))" "$laps" "$lap_index" "$warehouse_seed"
  wait_for_gpu_recovery
done

printf 'All %s NBV collection laps completed: %s\n' "$laps" "$output"

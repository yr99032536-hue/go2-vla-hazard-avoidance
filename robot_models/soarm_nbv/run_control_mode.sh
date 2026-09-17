#!/usr/bin/env bash

set -euo pipefail

MODE="${1:-}"
LEADER_PORT="${LEADER_PORT:-/dev/ttyACM0}"
LEADER_TYPE="${LEADER_TYPE:-so101_leader}"
LEADER_ID="${LEADER_ID:-sim_leader}"
ACTION_PORT="${ACTION_PORT:-5556}"
FPS="${FPS:-30}"
ALPHA="${ALPHA:-0.35}"
RUNTIME_OFFSET_JSON="${RUNTIME_OFFSET_JSON:-/tmp/soarm_runtime_offset.json}"
GROOT_INTERVAL="${GROOT_INTERVAL:-2.0}"
GROOT_DENOISING_STEPS="${GROOT_DENOISING_STEPS:-1}"

ROBOT_MODELS_ROOT="/home/iy/Isaac/Robotics/robot_models"
LEROBOT_PY="/home/iy/miniconda3/envs/lerobot/bin/python"
OPENVLA_PY="/home/iy/miniconda3/envs/openvla/bin/python"
LEADER_SCRIPT="${ROBOT_MODELS_ROOT}/soarm_nbv/leader_teleop_bridge.py"
LEADER_LOG="${LEADER_LOG:-/tmp/soarm_leader_teleop.log}"
GROOT_LOG="${GROOT_LOG:-/tmp/soarm_nbv_policy_nbv4.log}"

usage() {
    cat <<EOF
Usage:
  $(basename "$0") leader
  $(basename "$0") groot

Environment overrides:
  LEADER_PORT=/dev/ttyACM0
  LEADER_TYPE=so101_leader
  LEADER_ID=sim_leader
  ACTION_PORT=5556
  FPS=30
  ALPHA=0.35
  RUNTIME_OFFSET_JSON=/tmp/soarm_runtime_offset.json
  LEADER_LOG=/tmp/soarm_leader_teleop.log
  GROOT_LOG=/tmp/soarm_nbv_policy_nbv4.log
EOF
}

stop_existing_sources() {
    pkill -f "${LEADER_SCRIPT}" 2>/dev/null || true
    pkill -f "soarm_nbv.gr00t_policy_node" 2>/dev/null || true
}

start_leader() {
    stop_existing_sources
    cd "${ROBOT_MODELS_ROOT}"
    setsid nohup "${LEROBOT_PY}" -u "${LEADER_SCRIPT}" \
        --leader-port "${LEADER_PORT}" \
        --leader-type "${LEADER_TYPE}" \
        --leader-id "${LEADER_ID}" \
        --action-port "${ACTION_PORT}" \
        --fps "${FPS}" \
        --alpha "${ALPHA}" \
        --runtime-offset-json "${RUNTIME_OFFSET_JSON}" \
        > "${LEADER_LOG}" 2>&1 < /dev/null &
    echo "leader mode started"
    echo "pid: $!"
    echo "log: ${LEADER_LOG}"
}

start_groot() {
    stop_existing_sources
    cd "${ROBOT_MODELS_ROOT}"
    setsid nohup env \
        CUDA_HOME=/home/iy/miniconda3/envs/openvla \
        PATH=/home/iy/miniconda3/envs/openvla/bin:${PATH} \
        LD_LIBRARY_PATH=/home/iy/miniconda3/envs/openvla/lib:${LD_LIBRARY_PATH:-} \
        HF_MODULES_CACHE=/tmp/hf_modules \
        PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
        PYTHONPATH="${ROBOT_MODELS_ROOT}" \
        "${OPENVLA_PY}" -u -m soarm_nbv.gr00t_policy_node \
        --interval "${GROOT_INTERVAL}" \
        --denoising-steps "${GROOT_DENOISING_STEPS}" \
        > "${GROOT_LOG}" 2>&1 < /dev/null &
    echo "groot mode started"
    echo "pid: $!"
    echo "log: ${GROOT_LOG}"
}

case "${MODE}" in
    leader)
        start_leader
        ;;
    groot)
        start_groot
        ;;
    *)
        usage
        exit 1
        ;;
esac

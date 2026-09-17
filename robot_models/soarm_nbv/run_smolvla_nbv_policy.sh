#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/nbv/checkpoint [runner arguments...]" >&2
  exit 2
fi

policy_path="$1"
shift
if [[ ! -d "$policy_path" ]]; then
  echo "Checkpoint directory not found: $policy_path" >&2
  exit 1
fi

export PYTHONPATH="/home/iy/Isaac/Robotics/robot_models${PYTHONPATH:+:$PYTHONPATH}"
exec /home/iy/miniconda3/envs/lerobot/bin/python \
  /home/iy/Isaac/Robotics/robot_models/soarm_nbv/smolvla_nbv_policy_runner.py \
  --policy-path "$policy_path" "$@"

#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 3 ]]; then
  echo "Usage: $0 COLLECTED_SESSION_DIR DATASET_REPO_ID OUTPUT_DIR" >&2
  exit 2
fi

lerobot_env="/home/iy/miniconda3/envs/lerobot"
export LD_LIBRARY_PATH="$lerobot_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/home/iy/Isaac/Robotics/robot_models${PYTHONPATH:+:$PYTHONPATH}"

exec "$lerobot_env/bin/python" \
  /home/iy/Isaac/Robotics/robot_models/soarm_nbv/convert_to_lerobot.py \
  --src "$1" \
  --repo-id "$2" \
  --out-dir "$3"

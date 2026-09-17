#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  echo "Usage: $0 DATASET_REPO_ID OUTPUT_DIR [lerobot-train overrides...]" >&2
  exit 2
fi

dataset_repo_id="$1"
output_dir="$2"
shift 2
lerobot_env="/home/iy/miniconda3/envs/lerobot"
script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# scipy/transformers requires the newer C++ runtime installed in this env.
export LD_LIBRARY_PATH="$lerobot_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

verify_args=(--repo-id "$dataset_repo_id")
dataset_root_args=()
if [[ -n "${NBV_DATASET_ROOT:-}" ]]; then
  verify_args+=(--root "$NBV_DATASET_ROOT")
  dataset_root_args+=(--dataset.root="$NBV_DATASET_ROOT")
fi
"$lerobot_env/bin/python" "$script_dir/verify_nbv_lerobot_dataset.py" "${verify_args[@]}"

exec "$lerobot_env/bin/lerobot-train" \
  --policy.path=lerobot/smolvla_base \
  --dataset.repo_id="$dataset_repo_id" \
  "${dataset_root_args[@]}" \
  --batch_size="${NBV_BATCH_SIZE:-4}" \
  --steps="${NBV_TRAIN_STEPS:-20000}" \
  --save_freq="${NBV_SAVE_FREQ:-2000}" \
  --output_dir="$output_dir" \
  --policy.dtype=bfloat16 \
  "$@"

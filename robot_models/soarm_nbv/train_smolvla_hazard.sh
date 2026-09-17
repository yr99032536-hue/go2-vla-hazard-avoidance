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
base_model="${HAZARD_BASE_MODEL:-/home/iy/Isaac/Robotics/robot_models/checkpoints/smolvla}"
tensorboard_log_dir="${HAZARD_TENSORBOARD_LOGDIR:-/home/iy/Isaac/Robotics/data/tensorboard/binary_tree_hazard_102_v1/train}"
export LD_LIBRARY_PATH="$lerobot_env/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export PYTHONPATH="/home/iy/Isaac/Robotics/robot_models${PYTHONPATH:+:$PYTHONPATH}"

verify_args=(--repo-id "$dataset_repo_id")
dataset_root_args=()
if [[ -n "${HAZARD_DATASET_ROOT:-}" ]]; then
  verify_args+=(--root "$HAZARD_DATASET_ROOT")
  dataset_root_args+=(--dataset.root="$HAZARD_DATASET_ROOT")
fi
"$lerobot_env/bin/python" "$script_dir/verify_hazard_lerobot_dataset.py" "${verify_args[@]}"

exec "$lerobot_env/bin/lerobot-train" \
  --policy.path="$base_model" \
  --policy.input_features=null \
  --policy.output_features=null \
  --dataset.repo_id="$dataset_repo_id" \
  "${dataset_root_args[@]}" \
  --batch_size="${HAZARD_BATCH_SIZE:-2}" \
  --steps="${HAZARD_TRAIN_STEPS:-20000}" \
  --save_freq="${HAZARD_SAVE_FREQ:-1000}" \
  --log_freq="${HAZARD_LOG_FREQ:-50}" \
  --eval_freq=0 \
  --seed="${HAZARD_TRAIN_SEED:-47}" \
  --num_workers="${HAZARD_NUM_WORKERS:-2}" \
  --tensorboard_log_dir="$tensorboard_log_dir" \
  --tensorboard_flush_secs="${HAZARD_TENSORBOARD_FLUSH_SECS:-30}" \
  --output_dir="$output_dir" \
  --save_checkpoint=true \
  --wandb.enable="${HAZARD_WANDB_ENABLE:-false}" \
  --wandb.project="${HAZARD_WANDB_PROJECT:-lerobot}" \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.push_to_hub=false \
  --policy.chunk_size=50 \
  --policy.n_action_steps=10 \
  --policy.freeze_vision_encoder=true \
  --policy.train_expert_only=true \
  --policy.train_state_proj=true \
  "$@"

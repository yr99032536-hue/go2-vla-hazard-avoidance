#!/usr/bin/env bash
set -euo pipefail

tensorboard_bin="/home/iy/miniconda3/envs/isaacsim-5.1/bin/tensorboard"
log_dir="${HAZARD_TENSORBOARD_LOGDIR:-/home/iy/Isaac/Robotics/data/tensorboard/binary_tree_hazard_102_v1}"
host="${HAZARD_TENSORBOARD_HOST:-127.0.0.1}"
port="${HAZARD_TENSORBOARD_PORT:-6006}"
server_log="$log_dir/tensorboard_server.log"
pid_file="$log_dir/tensorboard_server.pid"
url="http://$host:$port"

mkdir -p "$log_dir"

if curl --silent --fail --max-time 1 "$url/" >/dev/null 2>&1; then
  echo "TensorBoard is already running: $url"
else
  nohup "$tensorboard_bin" \
    --logdir "$log_dir" \
    --host "$host" \
    --port "$port" \
    >>"$server_log" 2>&1 &
  server_pid=$!
  printf '%s\n' "$server_pid" >"$pid_file"

  for _ in {1..30}; do
    if curl --silent --fail --max-time 1 "$url/" >/dev/null 2>&1; then
      echo "TensorBoard started: $url"
      break
    fi
    sleep 0.2
  done

  if ! curl --silent --fail --max-time 1 "$url/" >/dev/null 2>&1; then
    echo "TensorBoard did not start. Check: $server_log" >&2
    exit 1
  fi
fi

if [[ -n "${DISPLAY:-}" ]]; then
  xdg-open "$url" >/dev/null 2>&1 || true
fi

echo "Persistent logs: $log_dir"

#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ISAAC_ENV="${ISAAC_ENV:-/home/iy/miniconda3/envs/isaacsim-5.1}"
ROS_BRIDGE_ROOT="$ISAAC_ENV/lib/python3.11/site-packages/isaacsim/exts/isaacsim.ros2.bridge/humble"
ISAAC_INTERFACE_ROOT="${ISAAC_INTERFACE_ROOT:-/home/iy/Isaac/Go2_Intelligence_Framework/install_isaac311/go2_active_slam_interfaces}"
BINARY_TREE_USD="${BINARY_TREE_USD:-$ROOT/assets/usd/binary_tree_hazard/binary_tree_hazard.usda}"

if [[ ! -d "$ROS_BRIDGE_ROOT/lib" ]]; then
  printf 'Isaac bundled ROS 2 bridge is missing: %s\n' "$ROS_BRIDGE_ROOT" >&2
  exit 2
fi
if [[ ! -f "$BINARY_TREE_USD" ]]; then
  printf 'Binary-tree hazard scene is missing: %s\n' "$BINARY_TREE_USD" >&2
  exit 2
fi

export ROS_DISTRO=humble
export RMW_IMPLEMENTATION=rmw_fastrtps_cpp
export PYTHONPATH="$ROS_BRIDGE_ROOT/rclpy:$ISAAC_INTERFACE_ROOT/lib/python3.11/site-packages${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROS_BRIDGE_ROOT/lib:$ISAAC_INTERFACE_ROOT/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export AMENT_PREFIX_PATH="$ISAAC_INTERFACE_ROOT${AMENT_PREFIX_PATH:+:$AMENT_PREFIX_PATH}"

exec "$ISAAC_ENV/bin/python" "$ROOT/src/sim/go2_soarm.py" \
  --environment_usd "$BINARY_TREE_USD" \
  --slam_rgbd \
  --slam_ros2 \
  --viewport_topdown \
  "$@"

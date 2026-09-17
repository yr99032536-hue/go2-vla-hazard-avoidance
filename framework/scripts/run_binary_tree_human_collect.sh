#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Data collection is intentionally GUI-only.  Do not pass HEADLESS through to
# the simulator even if it exists in the caller environment.
unset HEADLESS
export BINARY_TREE_COLLECT_HUMAN=1
export BINARY_TREE_MANUAL_ARM_TELEOP=1
export BINARY_TREE_SHOW_CAMERA_VIEWPORT=1

exec "$SCRIPT_DIR/run_binary_tree_vision_demo.sh" "$@"

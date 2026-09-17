#!/usr/bin/env bash
set -euo pipefail

# Data collection is always visible by workspace rule.  Remove inherited
# headless settings before delegating to the single-process task wrapper.
unset HEADLESS
export BINARY_TREE_COLLECT_NBV_TEACHER=1
export BINARY_TREE_MANUAL_ARM_TELEOP=0
export BINARY_TREE_VLA_ARM_POLICY=0
export BINARY_TREE_NBV_COLLECTION_LAPS="${BINARY_TREE_NBV_COLLECTION_LAPS:-17}"

exec "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/run_binary_tree_vision_demo.sh" "$@"

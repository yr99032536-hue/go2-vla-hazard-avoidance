# SO-Arm NBV Modules

This folder contains the clean, modular SO-Arm NBV integration path.

## Layout

```text
soarm_nbv/
  __init__.py
  zmq_bridge.py           # Transport-only ZMQ observation/action channels
  adapter.py              # Isaac Sim <-> GR00T observation/action conversion
  safety.py               # Joint limits, clipping, and smoothing helpers
  gr00t_policy_node.py    # Policy process: observation in, action out
  nbv_soarm_sim.py        # Minimal SO-Arm camera/ZMQ Isaac Sim test process
```

## Process Split

```text
Isaac Sim SO-Arm process
  -> publishes camera image + joint state
  <- receives arm action

GR00T policy process
  <- receives camera image + joint state
  -> publishes arm action
```

The transport layer does not import Isaac Sim or GR00T. This keeps dependency
boundaries clean and makes it possible to test ZMQ without loading the model.

## Ports

```text
observation: tcp://localhost:5555
action:      tcp://localhost:5556
```

## Current Model Note

`SO_ARM_Starter_Gr00t` is a GR00T N1.5 checkpoint with `model_type:
gr00t_n1_5`. The main local Isaac-GR00T checkout is N1.6-oriented, so the
N1.5-compatible worktree is used for this checkpoint:

```text
/home/iy/Isaac/Robotics/Isaac-GR00T-n1.5
```

The SO_ARM checkpoint and the local `n1.5-release` tag differ by one action-head
parameter (`future_tokens`). The worktree has a local compatibility patch that
removes that extra token path; after the patch the model produces finite actions.

On the RTX 4070 SUPER 12GB, Isaac Sim + GR00T N1.5 only fits with a low-memory
policy setting:

```text
--denoising-steps 1
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
```

The process is currently for observation/action verification. Keep
`--apply-actions` off until action ranges and task behavior are reviewed.

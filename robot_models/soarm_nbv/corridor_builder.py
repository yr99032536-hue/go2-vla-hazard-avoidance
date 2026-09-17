"""Procedural corridor/junction builder for Isaac Sim.

Spawns USD box primitives to form 7 junction types:
    FLR  = Front + Left + Right (T-shape from approach)
    FL   = Front + Left
    FR   = Front + Right
    LR   = Left + Right (T perpendicular)
    F    = Front only (straight)
    L    = Left only
    R    = Right only

Coordinate convention (robot faces +X):
    +X = forward
    +Y = left
    -Y = right
    +Z = up

Junction center is at the world position passed in.
Approach corridor extends in the -X direction.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# Junction type constants
JUNCTION_TYPES = ["FLR", "FL", "FR", "LR", "L", "R"]

# Keep env_0 on the reference single-turn corridor shape.
ENV0_FIXED_JUNCTION_TYPE = "L"

# Corridor parameter ranges for randomization
CORRIDOR_WIDTHS  = [1.2, 1.8, 2.4, 3.0]       # meters
CORRIDOR_HEIGHTS = [1.6, 1.9, 2.2, 2.5, 2.8, 3.0]  # meters
BRANCH_LENGTH    = 3.0   # meters (each open branch)
APPROACH_LENGTH  = 2.5   # meters (behind junction, robot stands here)
WALL_THICKNESS   = 0.20  # meters
ENV0_WALL_000_FIXED_WIDTH = 1.2
ENV0_WALL_000_FIXED_CX = -(ENV0_WALL_000_FIXED_WIDTH / 2 + APPROACH_LENGTH / 2)
ENV0_WALL_000_FIXED_CY = ENV0_WALL_000_FIXED_WIDTH / 2 + WALL_THICKNESS / 2


@dataclass
class WallSpec:
    """A single wall rectangle (axis-aligned box)."""
    cx: float   # center X
    cy: float   # center Y
    cz: float   # center Z
    sx: float   # size X
    sy: float   # size Y
    sz: float   # size Z
    label: str = ""


@dataclass
class CorridorConfig:
    junction_type: str
    corridor_width: float
    corridor_height: float
    branch_length: float
    approach_length: float
    wall_color: tuple[float, float, float]
    wall_roughness: float
    floor_color: tuple[float, float, float]
    lighting_intensity: float
    robot_lateral_offset: float   # y offset of robot start position
    fixed_wall_000_anchor: bool = False


def random_corridor_config(env_idx: int = -1) -> CorridorConfig:
    """Sample a random corridor configuration."""
    wall_hue = random.uniform(0.0, 1.0)
    wall_color = _hsv_to_rgb(wall_hue, random.uniform(0.1, 0.4), random.uniform(0.5, 0.9))
    floor_color = _hsv_to_rgb(random.uniform(0.0, 1.0), 0.1, random.uniform(0.3, 0.6))
    w = random.choice(CORRIDOR_WIDTHS)
    lateral_offsets = np.linspace(-w * 0.25, w * 0.25, 6).tolist()
    
    if env_idx == 0:
        junction_type = ENV0_FIXED_JUNCTION_TYPE
    elif env_idx >= 0 and env_idx < 3:
        j_types = ["FLR", "FL", "FR"]  # Front is open
    elif env_idx >= 3:
        j_types = ["LR", "L", "R"]     # Front is closed
    else:
        j_types = JUNCTION_TYPES

    return CorridorConfig(
        junction_type=junction_type if env_idx == 0 else random.choice(j_types),
        corridor_width=w,
        corridor_height=random.choice(CORRIDOR_HEIGHTS),
        branch_length=BRANCH_LENGTH,
        approach_length=APPROACH_LENGTH,
        wall_color=wall_color,
        wall_roughness=random.uniform(0.1, 0.9),
        floor_color=floor_color,
        lighting_intensity=random.uniform(300.0, 5000.0),
        robot_lateral_offset=random.choice(lateral_offsets),
        fixed_wall_000_anchor=(env_idx == 0),
    )


def compute_walls(cfg: CorridorConfig) -> list[WallSpec]:
    """Compute all wall boxes for a junction configuration.

    The junction center is at world origin (0, 0, 0) in local env space.
    Caller must add env_origin offset.
    """
    W  = cfg.corridor_width
    H  = cfg.corridor_height
    L  = cfg.branch_length
    A  = cfg.approach_length
    T  = WALL_THICKNESS
    HZ = H / 2.0   # vertical center of walls
    jt = cfg.junction_type

    has_F = "F" in jt
    has_L = "L" in jt
    has_R = "R" in jt

    center_x = 0.0
    center_y = 0.0
    if cfg.fixed_wall_000_anchor:
        fixed_inner_x = ENV0_WALL_000_FIXED_CX + A / 2
        fixed_left_inner_y = ENV0_WALL_000_FIXED_CY - T / 2
        center_x = fixed_inner_x + W / 2
        center_y = fixed_left_inner_y - W / 2

    walls: list[WallSpec] = []

    # ------------------------------------------------------------------ #
    # APPROACH CORRIDOR  (x: -W/2-A .. -W/2)  (robot enters from -X)
    # ------------------------------------------------------------------ #
    # Left wall of approach
    walls.append(WallSpec(
        cx=center_x - (W / 2 + A / 2), cy=center_y + W / 2 + T / 2, cz=HZ,
        sx=A, sy=T, sz=H, label="approach_left"
    ))
    # Right wall of approach
    walls.append(WallSpec(
        cx=center_x - (W / 2 + A / 2), cy=center_y - (W / 2 + T / 2), cz=HZ,
        sx=A, sy=T, sz=H, label="approach_right"
    ))
    # Back wall of approach
    walls.append(WallSpec(
        cx=center_x - (W / 2 + A + T / 2), cy=center_y, cz=HZ,
        sx=T, sy=W + 2 * T, sz=H, label="approach_back"
    ))

    # ------------------------------------------------------------------ #
    # JUNCTION SQUARE CORNERS (fill closed corners with pillar-like pieces)
    # The junction square is x: -W/2..W/2, y: -W/2..W/2
    # Closed corners get a T×T pillar so walls meet cleanly.
    # ------------------------------------------------------------------ #
    # Back-left corner: always solid (approach meets left side)
    # Back-right corner: always solid (approach meets right side)
    # Front-left corner: solid if neither F nor L
    # Front-right corner: solid if neither F nor R

    # ------------------------------------------------------------------ #
    # SIDE WALLS WITHIN JUNCTION SQUARE (x: -W/2..W/2)
    # At y=+W/2: present between approach and left branch (if L open,
    #   this segment is absent since the left branch opens there).
    # At y=-W/2: same logic for right.
    # At x=+W/2: present if F is closed.
    # ------------------------------------------------------------------ #

    # Junction left boundary (y=W/2): only needed if left is CLOSED
    if not has_L:
        walls.append(WallSpec(
            cx=center_x, cy=center_y + W / 2 + T / 2, cz=HZ,
            sx=W, sy=T, sz=H, label="junction_left_closed"
        ))
    # Junction right boundary (y=-W/2): only needed if right is CLOSED
    if not has_R:
        walls.append(WallSpec(
            cx=center_x, cy=center_y - (W / 2 + T / 2), cz=HZ,
            sx=W, sy=T, sz=H, label="junction_right_closed"
        ))
    # Junction front boundary (x=W/2): only needed if front is CLOSED
    if not has_F:
        walls.append(WallSpec(
            cx=center_x + W / 2 + T / 2, cy=center_y, cz=HZ,
            sx=T, sy=W + 2 * T, sz=H, label="junction_front_closed"
        ))

    # ------------------------------------------------------------------ #
    # BRANCH WALLS (one branch per open direction)
    # ------------------------------------------------------------------ #

    if has_F:
        # Front branch (x: W/2 .. W/2+L)
        walls.append(WallSpec(
            cx=center_x + W / 2 + L / 2, cy=center_y + W / 2 + T / 2, cz=HZ,
            sx=L, sy=T, sz=H, label="front_branch_left"
        ))
        walls.append(WallSpec(
            cx=center_x + W / 2 + L / 2, cy=center_y - (W / 2 + T / 2), cz=HZ,
            sx=L, sy=T, sz=H, label="front_branch_right"
        ))
        walls.append(WallSpec(
            cx=center_x + W / 2 + L + T / 2, cy=center_y, cz=HZ,
            sx=T, sy=W + 2 * T, sz=H, label="front_branch_end"
        ))

    if has_L:
        # Left branch (y: W/2 .. W/2+L)
        walls.append(WallSpec(
            cx=center_x - (W / 2 + T / 2), cy=center_y + W / 2 + L / 2, cz=HZ,
            sx=T, sy=L, sz=H, label="left_branch_back"
        ))
        walls.append(WallSpec(
            cx=center_x + W / 2 + T / 2, cy=center_y + W / 2 + L / 2, cz=HZ,
            sx=T, sy=L, sz=H, label="left_branch_front"
        ))
        walls.append(WallSpec(
            cx=center_x, cy=center_y + W / 2 + L + T / 2, cz=HZ,
            sx=W + 2 * T, sy=T, sz=H, label="left_branch_end"
        ))

    if has_R:
        # Right branch (y: -W/2 .. -W/2-L)
        walls.append(WallSpec(
            cx=center_x - (W / 2 + T / 2), cy=center_y - (W / 2 + L / 2), cz=HZ,
            sx=T, sy=L, sz=H, label="right_branch_back"
        ))
        walls.append(WallSpec(
            cx=center_x + W / 2 + T / 2, cy=center_y - (W / 2 + L / 2), cz=HZ,
            sx=T, sy=L, sz=H, label="right_branch_front"
        ))
        walls.append(WallSpec(
            cx=center_x, cy=center_y - (W / 2 + L + T / 2), cz=HZ,
            sx=W + 2 * T, sy=T, sz=H, label="right_branch_end"
        ))

    # ------------------------------------------------------------------ #
    # CORNER FILLER PILLARS at junction square corners
    # (prevents diagonal gaps where walls meet at 90 degrees)
    # ------------------------------------------------------------------ #
    # Back-left corner (x=-W/2, y=W/2): approach left meets left branch back (if L open)
    # or junction_left_closed wall already covers it.
    # These T×T pillars fill any remaining gaps.
    corner_configs = [
        # (cx, cy, present_always)
        (center_x - (W / 2 + T / 2), center_y + W / 2 + T / 2,  True,  "corner_back_left"),
        (center_x - (W / 2 + T / 2), center_y - (W / 2 + T / 2), True, "corner_back_right"),
        (center_x + W / 2 + T / 2,   center_y + W / 2 + T / 2,  not (has_F and has_L), "corner_front_left"),
        (center_x + W / 2 + T / 2,   center_y - (W / 2 + T / 2), not (has_F and has_R), "corner_front_right"),
    ]
    for cx, cy, present, label in corner_configs:
        if present:
            walls.append(WallSpec(
                cx=cx, cy=cy, cz=HZ,
                sx=T, sy=T, sz=H, label=label
            ))

    return walls


def robot_start_position(cfg: CorridorConfig) -> tuple[float, float, float]:
    """Local position where the robot (GO2) should be placed in env space.

    Spawn beside wall_000 (the approach-left wall), staying just inside the corridor.
    """
    wall_clearance = 0.15
    x = -(cfg.corridor_width / 2 + cfg.approach_length / 2)
    y = cfg.corridor_width / 2 - wall_clearance
    return (x, y, 0.0)


def wall_000_position(cfg: CorridorConfig) -> tuple[float, float, float]:
    """Local center position of wall_000_approach_left in env space."""
    if cfg.fixed_wall_000_anchor:
        return (ENV0_WALL_000_FIXED_CX, ENV0_WALL_000_FIXED_CY, cfg.corridor_height / 2)
    W = cfg.corridor_width
    H = cfg.corridor_height
    A = cfg.approach_length
    T = WALL_THICKNESS
    return (-(W / 2 + A / 2), W / 2 + T / 2, H / 2)


def robot_start_position_from_wall(
    cfg: CorridorConfig,
    wall_to_robot_offset: tuple[float, float, float] | None,
) -> tuple[float, float, float]:
    """Place robot using a stored offset from wall_000 center when available."""
    if wall_to_robot_offset is None:
        return robot_start_position(cfg)
    wall_x, wall_y, _ = wall_000_position(cfg)
    dx, dy, dz = wall_to_robot_offset
    return (wall_x + dx, wall_y + dy, dz)


# ------------------------------------------------------------------ #
# Isaac Sim USD spawning (call after sim.reset())
# ------------------------------------------------------------------ #

def spawn_corridor(
    stage,
    env_origin: tuple[float, float, float],
    cfg: CorridorConfig,
    env_idx: int,
) -> None:
    """Spawn all corridor wall prims into the USD stage.

    Must be called after sim.reset() so prim paths are writable.
    """
    try:
        from pxr import UsdGeom, UsdPhysics, UsdShade, Gf, Sdf
    except ImportError as e:
        raise RuntimeError("pxr (USD) not available. Run inside Isaac Sim env.") from e

    walls = compute_walls(cfg)
    base_path = f"/World/envs/env_{env_idx}/corridor"

    # Hide existing corridor prims if re-spawning (since RemovePrim might not work fully across layers)
    existing = stage.GetPrimAtPath(base_path)
    if existing.IsValid():
        for child in existing.GetChildren():
            if "wall" in child.GetName():
                UsdGeom.Imageable(child).MakeInvisible()

    # Parent xform for this corridor
    UsdGeom.Xform.Define(stage, base_path)

    r, g, b = cfg.wall_color
    roughness = cfg.wall_roughness

    for i, wall in enumerate(walls):
        prim_path = f"{base_path}/wall_{i:03d}_{wall.label}"
        _spawn_box(
            stage, prim_path,
            # corridor lives under /World/envs/env_i, so positions here must stay
            # in env-local coordinates. Adding env_origin again double-offsets it.
            world_pos=(wall.cx, wall.cy, wall.cz),
            size=(wall.sx, wall.sy, wall.sz),
            color=(r, g, b),
            roughness=roughness,
        )

    # Floor (large fixed-size slab under entire env to cover all branches)
    W = cfg.corridor_width
    L = cfg.branch_length
    A = cfg.approach_length
    T = WALL_THICKNESS
    floor_ext = 15.0  # fixed size to prevent gaps and allow tighter packing
    _spawn_box(
        stage, f"{base_path}/floor",
        world_pos=(0.0, 0.0, -T / 2),
        size=(floor_ext, floor_ext, T),
        color=cfg.floor_color,
        roughness=0.9,
    )


def _spawn_box(
    stage,
    prim_path: str,
    world_pos: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    roughness: float,
) -> None:
    """Spawn a single USD Cube (box) at world_pos with given size and material."""
    from pxr import UsdGeom, UsdPhysics, Gf

    # Hide existing prim to avoid ghost walls if re-spawning fails
    if stage.GetPrimAtPath(prim_path).IsValid():
        UsdGeom.Imageable(stage.GetPrimAtPath(prim_path)).MakeInvisible()

    xform = UsdGeom.Xform.Define(stage, prim_path)
    xform.ClearXformOpOrder()
    UsdGeom.Imageable(xform).MakeVisible()

    # Explicitly cast to Python float for C++ double compatibility (numpy types cause errors)
    px, py, pz = float(world_pos[0]), float(world_pos[1]), float(world_pos[2])
    sx, sy, sz = float(size[0]) / 2, float(size[1]) / 2, float(size[2]) / 2

    xform.AddTranslateOp().Set(Gf.Vec3d(px, py, pz))
    # UsdGeom.Cube default extent is [-1,1], so scale = size/2
    xform.AddScaleOp().Set(Gf.Vec3d(sx, sy, sz))

    cube_path = prim_path + "/cube"
    cube = UsdGeom.Cube.Define(stage, cube_path)
    cube_prim = stage.GetPrimAtPath(cube_path)

    # Color (display color)
    r, g, b = float(color[0]), float(color[1]), float(color[2])
    cube.GetDisplayColorAttr().Set([Gf.Vec3f(r, g, b)])

    # Physics collider
    UsdPhysics.CollisionAPI.Apply(cube_prim)


# ------------------------------------------------------------------ #
# Helpers
# ------------------------------------------------------------------ #

def _hsv_to_rgb(h: float, s: float, v: float) -> tuple[float, float, float]:
    import colorsys
    return colorsys.hsv_to_rgb(h, s, v)


def describe(cfg: CorridorConfig) -> str:
    direction_map = {
        "FLR": "Front+Left+Right",
        "FL":  "Front+Left",
        "FR":  "Front+Right",
        "LR":  "Left+Right",
        "F":   "Front only",
        "L":   "Left only",
        "R":   "Right only",
    }
    return (
        f"[{cfg.junction_type}] {direction_map.get(cfg.junction_type, '?')} | "
        f"W={cfg.corridor_width}m H={cfg.corridor_height}m | "
        f"offset_y={cfg.robot_lateral_offset:+.2f}m"
    )


if __name__ == "__main__":
    # Dry-run: print wall specs for each junction type
    print("Wall counts per junction type (W=1.8m, H=2.2m):\n")
    for jt in JUNCTION_TYPES:
        cfg = CorridorConfig(
            junction_type=jt,
            corridor_width=1.8,
            corridor_height=2.2,
            branch_length=3.0,
            approach_length=2.5,
            wall_color=(0.8, 0.8, 0.8),
            wall_roughness=0.5,
            floor_color=(0.4, 0.4, 0.4),
            lighting_intensity=3000.0,
            robot_lateral_offset=0.0,
        )
        walls = compute_walls(cfg)
        print(f"  {jt:4s} ({describe(cfg).split('|')[1].strip()}): {len(walls)} walls")
        for w in walls:
            print(f"        {w.label:30s} pos=({w.cx:+.2f},{w.cy:+.2f},{w.cz:+.2f}) "
                  f"size=({w.sx:.2f},{w.sy:.2f},{w.sz:.2f})")
        print()

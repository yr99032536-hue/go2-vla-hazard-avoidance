#!/usr/bin/env python3
"""Generate a deterministic three-stage binary-branch hazard experiment map.

At every stage the robot approaches from west to east.  A T junction exposes
one north/left branch and one south/right branch.  A near-corner wall lip hides
the red cube from the body camera while keeping it reachable by a wrist-camera
peek from outside the branch-entry line.  The unsafe branch remains a leaf and
the safe child continues to the next decision.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import math
import random
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "assets/usd/binary_tree_hazard/binary_tree_hazard.usda"
DEFAULT_LAYOUT_OUTPUT = (
    ROOT / "assets/usd/binary_tree_hazard/binary_tree_hazard_layout.json"
)
STAGE_JUNCTION_X = (3, 9, 15)
STAGE_COUNT = len(STAGE_JUNCTION_X)
DEFAULT_CELL_SIZE_M = 1.6
DEFAULT_SEED = 47
WALL_THICKNESS_M = 0.18
WALL_HEIGHT_M = 1.6
SCAN_STANDOFF_CELLS = 0.90
REALIGN_APPROACH_LEAD_M = 0.14
TURN_CLEARANCE_INSET_M = 0.15
BLIND_POCKET_DEPTH_CELLS = 4
# Make the first decision read as a visibly deep T junction.  Later stages stay
# compact so the complete three-stage task still fits in the existing scene.
STAGE_BRANCH_LATERAL_DEPTH_CELLS = (2, 2, 2)
HAZARD_CUBE_SIZE_XY_M = 0.72
HAZARD_END_CLEARANCE_M = 0.10
HAZARD_FORWARD_OFFSET_M = -0.05
FINISH_GOAL_INSET_M = 0.20


@dataclass(frozen=True)
class StageSpec:
    index: int
    junction_x: int
    hazard_side: str
    safe_side: str
    branch_lateral_depth_cells: int
    scan_pose: tuple[float, float, float]
    hazard_cube: tuple[float, float, float]
    safe_route: tuple[tuple[float, float, float], ...]
    branch_routes: dict[str, dict[str, tuple[tuple[float, float, float], ...]]]


def _fmt(value: float) -> str:
    return f"{value:.5f}".rstrip("0").rstrip(".")


def _cube(
    name: str,
    position: tuple[float, float, float],
    size: tuple[float, float, float],
    color: tuple[float, float, float],
    *,
    collision: bool = True,
) -> str:
    api = ' (\n        prepend apiSchemas = ["PhysicsCollisionAPI"]\n    )' if collision else ""
    return f'''    def Cube "{name}"{api}
    {{
        double size = 1
        color3f[] primvars:displayColor = [({_fmt(color[0])}, {_fmt(color[1])}, {_fmt(color[2])})]
        double3 xformOp:scale = ({_fmt(size[0])}, {_fmt(size[1])}, {_fmt(size[2])})
        double3 xformOp:translate = ({_fmt(position[0])}, {_fmt(position[1])}, {_fmt(position[2])})
        uniform token[] xformOpOrder = ["xformOp:translate", "xformOp:scale"]
    }}
'''


def hazard_pattern(seed: int, explicit: str | None = None) -> tuple[str, ...]:
    if explicit is not None:
        normalized = explicit.strip().upper()
        if len(normalized) != STAGE_COUNT or any(side not in "LR" for side in normalized):
            raise ValueError(f"hazard pattern must contain exactly {STAGE_COUNT} L/R characters")
        return tuple("left" if side == "L" else "right" for side in normalized)
    rng = random.Random(seed)
    return tuple(rng.choice(("left", "right")) for _ in range(STAGE_COUNT))


def _route_goal(
    grid_x: float,
    grid_y: float,
    yaw: float,
    cell_size_m: float,
) -> tuple[float, float, float]:
    return (grid_x * cell_size_m, grid_y * cell_size_m, yaw)


def build_layout(
    seed: int = DEFAULT_SEED,
    *,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    explicit_hazard_pattern: str | None = None,
) -> tuple[set[tuple[int, int]], tuple[StageSpec, ...], dict[str, object]]:
    if not 1.3 <= cell_size_m <= 2.2:
        raise ValueError("cell_size_m must be within 1.3..2.2 for Go2 clearance")
    hazards = hazard_pattern(seed, explicit_hazard_pattern)
    open_cells: set[tuple[int, int]] = {(x, 0) for x in range(-1, 4)}
    blocked_edges: list[tuple[tuple[int, int], tuple[int, int]]] = []
    stages: list[StageSpec] = []

    for stage_index, (junction_x, hazard_side) in enumerate(
        zip(STAGE_JUNCTION_X, hazards, strict=True),
        start=1,
    ):
        safe_side = "right" if hazard_side == "left" else "left"
        safe_sign = 1 if safe_side == "left" else -1
        hazard_sign = -safe_sign
        branch_lateral_depth_cells = STAGE_BRANCH_LATERAL_DEPTH_CELLS[
            stage_index - 1
        ]

        # The route remains a mirrored L-shaped tree, but the hazard itself is
        # placed in the first side pocket.  A short near-corner wall lip hides
        # it from the body camera.  The SO-Arm can move its wrist optical center
        # past that lip while the Go2 body remains west of the branch-entry
        # line, so visual evidence is available before any base entry.
        pocket_end_x = junction_x + BLIND_POCKET_DEPTH_CELLS
        # Each branch first runs laterally away from the approach corridor,
        # then turns east.  Stage 1 uses two lateral cells, producing a deep T
        # before either arm turns toward the next connector.  At the east end,
        # both arms return toward the centerline; the hazard arm's final edge
        # is blocked below so it remains a leaf in the binary tree.
        for sign in (-1, 1):
            open_cells.update(
                (junction_x, sign * lateral_cell)
                for lateral_cell in range(1, branch_lateral_depth_cells + 1)
            )
            open_cells.update(
                (x, sign * branch_lateral_depth_cells)
                for x in range(junction_x, pocket_end_x + 1)
            )
            open_cells.update(
                (pocket_end_x, sign * lateral_cell)
                for lateral_cell in range(1, branch_lateral_depth_cells)
            )
        open_cells.add((pocket_end_x, 0))
        blocked_edges.append(
            ((pocket_end_x, hazard_sign), (pocket_end_x, 0))
        )

        is_last = stage_index == STAGE_COUNT
        if is_last:
            centerline_end_x = junction_x + 5
        else:
            centerline_end_x = STAGE_JUNCTION_X[stage_index]
        open_cells.update((x, 0) for x in range(pocket_end_x, centerline_end_x + 1))

        junction_world_x = junction_x * cell_size_m
        scan_pose = (
            junction_world_x - SCAN_STANDOFF_CELLS * cell_size_m,
            0.0,
            0.0,
        )
        hazard_lateral_depth_m = (
            branch_lateral_depth_cells * cell_size_m
            + 0.5 * cell_size_m
            - 0.5 * WALL_THICKNESS_M
            - 0.5 * HAZARD_CUBE_SIZE_XY_M
            - HAZARD_END_CLEARANCE_M
        )
        hazard_cube = (
            junction_world_x + HAZARD_FORWARD_OFFSET_M,
            hazard_sign * hazard_lateral_depth_m,
            0.55,
        )

        continuation_goal = (
            (
                centerline_end_x * cell_size_m - FINISH_GOAL_INSET_M,
                0.0,
                0.0,
            )
            if is_last
            else (
                (
                    STAGE_JUNCTION_X[stage_index] - SCAN_STANDOFF_CELLS
                )
                * cell_size_m
                - REALIGN_APPROACH_LEAD_M,
                0.0,
                0.0,
            )
        )
        branch_routes: dict[
            str,
            dict[str, tuple[tuple[float, float, float], ...]],
        ] = {}
        for side, sign in (("left", 1), ("right", -1)):
            side_yaw = math.pi / 2 if side == "left" else -math.pi / 2
            probe_route = (
                (
                    junction_x * cell_size_m - TURN_CLEARANCE_INSET_M,
                    0.0,
                    side_yaw,
                ),
                (
                    junction_x * cell_size_m - TURN_CLEARANCE_INSET_M,
                    sign * branch_lateral_depth_cells * cell_size_m,
                    0.0,
                ),
            )
            continue_route = (
                (
                    pocket_end_x * cell_size_m - TURN_CLEARANCE_INSET_M,
                    sign * branch_lateral_depth_cells * cell_size_m,
                    -side_yaw,
                ),
                (
                    pocket_end_x * cell_size_m - TURN_CLEARANCE_INSET_M,
                    0.0,
                    0.0,
                ),
                continuation_goal,
            )
            backtrack_route = (
                (
                    junction_x * cell_size_m - TURN_CLEARANCE_INSET_M,
                    sign * branch_lateral_depth_cells * cell_size_m,
                    -side_yaw,
                ),
                (
                    junction_x * cell_size_m - TURN_CLEARANCE_INSET_M,
                    0.0,
                    0.0,
                ),
            )
            branch_routes[side] = {
                "probe_route": probe_route,
                "continue_route": continue_route,
                "backtrack_route": backtrack_route,
            }

        safe_route = (
            branch_routes[safe_side]["probe_route"]
            + branch_routes[safe_side]["continue_route"]
        )

        stages.append(
            StageSpec(
                index=stage_index,
                junction_x=junction_x,
                hazard_side=hazard_side,
                safe_side=safe_side,
                branch_lateral_depth_cells=branch_lateral_depth_cells,
                scan_pose=scan_pose,
                hazard_cube=hazard_cube,
                safe_route=safe_route,
                branch_routes=branch_routes,
            )
        )

    manifest = {
        "schema": "binary_tree_hazard_layout.v1",
        "task_id": "inspect_hidden_alley_with_wrist_camera",
        "language_instruction": (
            "look behind walls and inspect hidden alley space with the wrist camera"
        ),
        "seed": seed,
        "stage_count": STAGE_COUNT,
        "cell_size_m": cell_size_m,
        "corridor_width_m": cell_size_m,
        "blind_pocket_depth_m": BLIND_POCKET_DEPTH_CELLS * cell_size_m,
        "stage_branch_lateral_depth_cells": list(
            STAGE_BRANCH_LATERAL_DEPTH_CELLS
        ),
        "inspection_geometry": "deep_side_alleys_without_entry_occluders",
        "branch_entry_occluders": False,
        "hazard_lateral_depth_m": abs(stages[0].hazard_cube[1]),
        "hazard_lateral_depth_m_by_stage": [
            abs(stage.hazard_cube[1]) for stage in stages
        ],
        "peek_gate_base_frame": {
            "minimum_forward_m": 0.22,
            "minimum_target_lateral_m": 0.18,
            "minimum_target_axis_component": 0.55,
            "minimum_consecutive_frames": 5,
        },
        "spawn_pose": [0.0, 0.0, 0.0],
        "hazard_pattern": [stage.hazard_side for stage in stages],
        "stages": [
            {
                "index": stage.index,
                "junction_center": [stage.junction_x * cell_size_m, 0.0],
                "branch_entry_x": stage.junction_x * cell_size_m - 0.5 * cell_size_m,
                "base_clearance_before_branch_entry_m": (
                    SCAN_STANDOFF_CELLS - 0.5
                ) * cell_size_m,
                "branch_lateral_depth_cells": stage.branch_lateral_depth_cells,
                "branch_lateral_depth_m": (
                    stage.branch_lateral_depth_cells * cell_size_m
                ),
                "scan_pose": list(stage.scan_pose),
                "hazard_side": stage.hazard_side,
                "safe_side": stage.safe_side,
                "hazard_cube": list(stage.hazard_cube),
                "hazard_lateral_depth_m": abs(stage.hazard_cube[1]),
                "safe_route": [list(goal) for goal in stage.safe_route],
                "branch_routes": {
                    side: {
                        route_name: [list(goal) for goal in route]
                        for route_name, route in routes.items()
                    }
                    for side, routes in stage.branch_routes.items()
                },
            }
            for stage in stages
        ],
        "finish_pose": list(stages[-1].safe_route[-1]),
        "blocked_edges_grid": [
            [list(first), list(second)] for first, second in blocked_edges
        ],
    }
    return open_cells, tuple(stages), manifest


def _merged_boundary_walls(
    open_cells: set[tuple[int, int]],
    cell_size_m: float,
    blocked_edges: set[frozenset[tuple[int, int]]] | None = None,
) -> list[tuple[str, tuple[float, float, float], tuple[float, float, float]]]:
    blocked = blocked_edges or set()

    def is_open_edge(first: tuple[int, int], second: tuple[int, int]) -> bool:
        return second in open_cells and frozenset((first, second)) not in blocked

    horizontal: dict[int, set[int]] = {}
    vertical: dict[int, set[int]] = {}
    for x, y in open_cells:
        if not is_open_edge((x, y), (x, y + 1)):
            horizontal.setdefault(2 * y + 1, set()).add(x)
        if not is_open_edge((x, y), (x, y - 1)):
            horizontal.setdefault(2 * y - 1, set()).add(x)
        if not is_open_edge((x, y), (x + 1, y)):
            vertical.setdefault(2 * x + 1, set()).add(y)
        if not is_open_edge((x, y), (x - 1, y)):
            vertical.setdefault(2 * x - 1, set()).add(y)

    def runs(values: set[int]) -> list[tuple[int, int]]:
        ordered = sorted(values)
        result: list[tuple[int, int]] = []
        if not ordered:
            return result
        start = previous = ordered[0]
        for value in ordered[1:]:
            if value != previous + 1:
                result.append((start, previous))
                start = value
            previous = value
        result.append((start, previous))
        return result

    walls = []
    wall_index = 0
    for doubled_y, x_values in sorted(horizontal.items()):
        for start, end in runs(x_values):
            length = (end - start + 1) * cell_size_m + WALL_THICKNESS_M
            center_x = (start + end) * 0.5 * cell_size_m
            center_y = doubled_y * 0.5 * cell_size_m
            walls.append(
                (
                    f"WallH_{wall_index:03d}",
                    (center_x, center_y, WALL_HEIGHT_M / 2),
                    (length, WALL_THICKNESS_M, WALL_HEIGHT_M),
                )
            )
            wall_index += 1
    for doubled_x, y_values in sorted(vertical.items()):
        for start, end in runs(y_values):
            length = (end - start + 1) * cell_size_m + WALL_THICKNESS_M
            center_x = doubled_x * 0.5 * cell_size_m
            center_y = (start + end) * 0.5 * cell_size_m
            walls.append(
                (
                    f"WallV_{wall_index:03d}",
                    (center_x, center_y, WALL_HEIGHT_M / 2),
                    (WALL_THICKNESS_M, length, WALL_HEIGHT_M),
                )
            )
            wall_index += 1
    return walls


def render_usda(
    open_cells: set[tuple[int, int]],
    stages: tuple[StageSpec, ...],
    *,
    seed: int,
    cell_size_m: float,
    cube_sides: tuple[str, ...] | None = None,
    open_return_connectors: bool = False,
) -> str:
    hazards = "".join("L" if side == "left" else "R" for side in (
        cube_sides or tuple(stage.hazard_side for stage in stages)
    ))
    min_x = (min(x for x, _ in open_cells) - 1.0) * cell_size_m
    max_x = (max(x for x, _ in open_cells) + 1.0) * cell_size_m
    min_y = (min(y for _, y in open_cells) - 1.0) * cell_size_m
    max_y = (max(y for _, y in open_cells) + 1.0) * cell_size_m
    floor_center = ((min_x + max_x) / 2, (min_y + max_y) / 2, -0.05)
    floor_size = (max_x - min_x, max_y - min_y, 0.10)

    lines = [
        "#usda 1.0\n",
        "(\n",
        '    defaultPrim = "BinaryTreeHazardMap"\n',
        '    upAxis = "Z"\n',
        "    metersPerUnit = 1\n",
        "    customLayerData = {\n",
        '        string generator = "generate_binary_tree_hazard_map.py"\n',
        f"        int seed = {seed}\n",
        f'        string hazardPattern = "{hazards}"\n',
        f"        int stageCount = {STAGE_COUNT}\n",
        "    }\n",
        ")\n\n",
        'def Xform "BinaryTreeHazardMap"\n{\n',
        '''    def Camera "TopDownCamera"
    {
        float2 clippingRange = (0.1, 100)
        float horizontalAperture = 360
        token projection = "orthographic"
        float verticalAperture = 105
        double3 xformOp:translate = (15.2, 0, 25)
        uniform token[] xformOpOrder = ["xformOp:translate"]
    }
''',
        _cube("Ground", floor_center, floor_size, (0.22, 0.24, 0.27)),
    ]
    wall_color = (0.38, 0.43, 0.50)
    blocked_edges = {
        frozenset(
            (
                (stage.junction_x + BLIND_POCKET_DEPTH_CELLS, 1 if stage.hazard_side == "left" else -1),
                (stage.junction_x + BLIND_POCKET_DEPTH_CELLS, 0),
            )
        )
        for stage in stages
    }
    if open_return_connectors:
        blocked_edges = set()
    for name, position, size in _merged_boundary_walls(
        open_cells,
        cell_size_m,
        blocked_edges,
    ):
        lines.append(_cube(name, position, size, wall_color))

    for cube_index, stage in enumerate(stages):
        cube_position = stage.hazard_cube
        if cube_sides is not None:
            cube_position = (cube_position[0], abs(cube_position[1]) * (
                1 if cube_sides[cube_index] == "left" else -1
            ), cube_position[2])
        lines.append(
            _cube(
                f"HazardRedCubeStage{stage.index}",
                cube_position,
                (HAZARD_CUBE_SIZE_XY_M, HAZARD_CUBE_SIZE_XY_M, 1.10),
                (0.95, 0.025, 0.02),
            )
        )
    finish = stages[-1].safe_route[-1]
    lines.append(
        _cube(
            "FinishBlueMarker",
            (finish[0], finish[1], 0.006),
            (0.42, 0.42, 0.012),
            (0.04, 0.48, 0.95),
            collision=False,
        )
    )
    lines.append("}\n")
    return "".join(lines)


def generate(
    output: Path,
    layout_output: Path,
    *,
    seed: int = DEFAULT_SEED,
    cell_size_m: float = DEFAULT_CELL_SIZE_M,
    explicit_hazard_pattern: str | None = None,
    explicit_cube_pattern: str | None = None,
    open_return_connectors: bool = False,
) -> dict[str, object]:
    open_cells, stages, manifest = build_layout(
        seed,
        cell_size_m=cell_size_m,
        explicit_hazard_pattern=explicit_hazard_pattern,
    )
    cube_sides = hazard_pattern(seed, explicit_cube_pattern) if explicit_cube_pattern else None
    if cube_sides is not None:
        manifest["topology_hazard_pattern"] = manifest["hazard_pattern"]
        manifest["hazard_pattern"] = list(cube_sides)
        manifest["evaluation_note"] = "Cubes moved only; walls and branch routes unchanged. Safe branches may be dead ends."
        for stage, side in zip(manifest["stages"], cube_sides, strict=True):
            stage["hazard_side"] = side
            stage["safe_side"] = "right" if side == "left" else "left"
            stage["hazard_cube"][1] = abs(stage["hazard_cube"][1]) * (1 if side == "left" else -1)
            routes = stage["branch_routes"][stage["safe_side"]]
            stage["safe_route"] = routes["probe_route"] + routes["continue_route"]
    if open_return_connectors:
        manifest["blocked_edges_grid"] = []
        manifest["open_return_connectors"] = True
        manifest["evaluation_note"] = "Both branches reconnect to the next stage; topology is fixed independently of cube positions."
    output.parent.mkdir(parents=True, exist_ok=True)
    layout_output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_usda(open_cells, stages, seed=seed, cell_size_m=cell_size_m,
                    cube_sides=cube_sides, open_return_connectors=open_return_connectors),
        encoding="utf-8",
    )
    layout_output.write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--layout-output", type=Path, default=DEFAULT_LAYOUT_OUTPUT)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--cell-size", type=float, default=DEFAULT_CELL_SIZE_M)
    parser.add_argument(
        "--hazard-pattern",
        help="Optional three-character L/R pattern. Seed 47 defaults to RLR.",
    )
    parser.add_argument(
        "--cube-pattern",
        help="Evaluation-only L/R cube positions; preserve original walls and routes.",
    )
    parser.add_argument(
        "--open-return-connectors", action="store_true",
        help="Evaluation topology: reconnect both branches regardless of cube placement.",
    )
    args = parser.parse_args()
    manifest = generate(
        args.output.expanduser().resolve(),
        args.layout_output.expanduser().resolve(),
        seed=args.seed,
        cell_size_m=args.cell_size,
        explicit_hazard_pattern=args.hazard_pattern,
        explicit_cube_pattern=args.cube_pattern,
        open_return_connectors=args.open_return_connectors,
    )
    print(
        f"Generated {args.output} and {args.layout_output}: "
        f"seed={manifest['seed']} hazards={manifest['hazard_pattern']}"
    )


if __name__ == "__main__":
    main()

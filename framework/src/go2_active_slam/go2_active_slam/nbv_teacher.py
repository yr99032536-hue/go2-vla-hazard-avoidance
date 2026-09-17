"""Deterministic next-best-view teacher over sparse mapping evidence.

Kinematics and collision checking stay outside this module.  The caller passes
only candidates that the arm planner can reach, plus an explicit feasibility
flag.  This module then scores the mapping value of each candidate.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Iterable

import numpy as np

from .sparse_tsdf import SparseTsdfVolume


Voxel = tuple[int, int, int]


def candidate_joint_grid(gripper_deg: float) -> tuple[tuple[str, np.ndarray], ...]:
    """Nine balanced seven-motor SO-Arm poses for teacher look-ahead.

    Pan, lift, and elbow rotation each cover three conservative levels.  A
    Latin-square layout gives every actuator useful variation without turning
    each NBV event into a 27-candidate Cartesian sweep.
    """
    candidates = []
    for row, lift_deg in enumerate((22.0, 30.0, 38.0)):
        pan_values = (-42.0, 0.0, 42.0) if row % 2 == 0 else (42.0, 0.0, -42.0)
        elbow_rotate_rows = (
            (-30.0, 0.0, 30.0),
            (0.0, 30.0, -30.0),
            (30.0, -30.0, 0.0),
        )
        for pan_deg, elbow_rotate_deg in zip(
            pan_values,
            elbow_rotate_rows[row],
            strict=True,
        ):
            candidate_id = (
                f"pan{pan_deg:+.0f}_lift{lift_deg:.0f}_"
                f"elrot{elbow_rotate_deg:+.0f}"
            )
            candidates.append(
                (
                    candidate_id,
                    np.asarray(
                        (
                            pan_deg,
                            lift_deg,
                            -60.0,
                            elbow_rotate_deg,
                            8.0,
                            0.0,
                            gripper_deg,
                        ),
                        dtype=np.float64,
                    ),
                )
            )
    return tuple(candidates)


@dataclass(frozen=True)
class ViewCandidate:
    candidate_id: str
    joint_target_deg: tuple[float, ...]
    camera_position_map_m: tuple[float, float, float]
    camera_forward_map: tuple[float, float, float]
    feasible: bool = True
    clearance_m: float = math.inf


@dataclass(frozen=True)
class CandidateScore:
    candidate: ViewCandidate
    visible_gap_voxels: tuple[Voxel, ...]
    information_gain: int
    motion_cost: float
    clearance_cost: float
    alignment: float
    total: float


@dataclass(frozen=True)
class TeacherDecision:
    selected: CandidateScore
    ranked: tuple[CandidateScore, ...]


@dataclass(frozen=True)
class LookaheadRgbd:
    """Privileged simulation observation used only to create a teacher label."""

    candidate: ViewCandidate
    depth_m: np.ndarray
    rgb: np.ndarray
    intrinsics: tuple[float, float, float, float]
    transform_map_camera: np.ndarray


@dataclass(frozen=True)
class LookaheadScore:
    candidate: ViewCandidate
    target_gap_revealed: bool
    new_surface_voxels: int
    new_known_voxels: int
    motion_cost: float
    total: float


@dataclass(frozen=True)
class LookaheadDecision:
    selected: LookaheadScore
    ranked: tuple[LookaheadScore, ...]


def _unit(vector: np.ndarray, name: str) -> np.ndarray:
    if vector.shape != (3,) or not np.all(np.isfinite(vector)):
        raise ValueError(f"{name} must contain three finite values")
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-9:
        raise ValueError(f"{name} must be non-zero")
    return vector / norm


def _line_is_clear(
    origin: np.ndarray,
    target: np.ndarray,
    occupied: set[Voxel],
    resolution_m: float,
) -> bool:
    ray = target - origin
    distance = float(np.linalg.norm(ray))
    if distance <= resolution_m:
        return True
    direction = ray / distance
    # Stop before the target voxel: the gap itself may border an occupied cell.
    for sample_distance in np.arange(resolution_m, max(resolution_m, distance - resolution_m), resolution_m * 0.5):
        voxel = tuple(np.floor((origin + direction * sample_distance) / resolution_m).astype(np.int64).tolist())
        if voxel in occupied:
            return False
    return True


def score_candidates(
    candidates: Iterable[ViewCandidate],
    *,
    occluded_voxels: Iterable[Voxel],
    occupied_voxels: Iterable[Voxel],
    resolution_m: float,
    current_joint_deg: Iterable[float],
    maximum_range_m: float = 3.0,
    horizontal_fov_deg: float = 70.0,
    motion_weight: float = 0.02,
    clearance_weight: float = 2.0,
    minimum_clearance_m: float = 0.05,
) -> TeacherDecision:
    """Select the candidate expected to reveal the most occluded voxels.

    Total score = visible voxel count + average view alignment - normalized arm
    motion cost - low-clearance cost.  Ties are broken by candidate_id.
    """
    if resolution_m <= 0.0 or maximum_range_m <= 0.0:
        raise ValueError("resolution and maximum range must be positive")
    if not 0.0 < horizontal_fov_deg < 180.0:
        raise ValueError("horizontal_fov_deg must be between 0 and 180")
    current = np.asarray(tuple(current_joint_deg), dtype=np.float64)
    if current.shape != (7,) or not np.all(np.isfinite(current)):
        raise ValueError("current_joint_deg must contain seven finite values")
    gaps = sorted(set(occluded_voxels))
    occupied = set(occupied_voxels)
    cosine_limit = math.cos(math.radians(horizontal_fov_deg * 0.5))
    scores: list[CandidateScore] = []

    for candidate in candidates:
        if not candidate.feasible:
            continue
        joints = np.asarray(candidate.joint_target_deg, dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            continue
        origin = np.asarray(candidate.camera_position_map_m, dtype=np.float64)
        if origin.shape != (3,) or not np.all(np.isfinite(origin)):
            continue
        try:
            forward = _unit(np.asarray(candidate.camera_forward_map, dtype=np.float64), "camera_forward_map")
        except ValueError:
            continue

        visible: list[Voxel] = []
        alignments: list[float] = []
        for voxel in gaps:
            target = (np.asarray(voxel, dtype=np.float64) + 0.5) * resolution_m
            ray = target - origin
            distance = float(np.linalg.norm(ray))
            if not resolution_m <= distance <= maximum_range_m:
                continue
            alignment = float(np.dot(forward, ray / distance))
            if alignment < cosine_limit:
                continue
            if not _line_is_clear(origin, target, occupied, resolution_m):
                continue
            visible.append(voxel)
            alignments.append(alignment)

        information_gain = len(visible)
        alignment_score = float(np.mean(alignments)) if alignments else 0.0
        # 180 degrees per joint is one unit; this keeps gain dominant and stable.
        motion_cost = float(np.linalg.norm((joints - current) / 180.0))
        clearance = float(candidate.clearance_m)
        clearance_cost = 0.0 if not math.isfinite(clearance) else max(0.0, minimum_clearance_m - clearance) / minimum_clearance_m
        total = information_gain + alignment_score - motion_weight * motion_cost - clearance_weight * clearance_cost
        scores.append(
            CandidateScore(
                candidate=candidate,
                visible_gap_voxels=tuple(visible),
                information_gain=information_gain,
                motion_cost=motion_cost,
                clearance_cost=clearance_cost,
                alignment=alignment_score,
                total=total,
            )
        )

    if not scores:
        raise ValueError("no feasible NBV teacher candidates")
    scores.sort(key=lambda item: (-item.total, -item.information_gain, item.motion_cost, item.candidate.candidate_id))
    return TeacherDecision(selected=scores[0], ranked=tuple(scores))


def score_simulation_lookahead(
    baseline: SparseTsdfVolume,
    observations: Iterable[LookaheadRgbd],
    *,
    current_joint_deg: Iterable[float],
    stride: int = 8,
    known_gain_weight: float = 0.1,
    motion_weight: float = 0.02,
    target_point_map_m: Iterable[float] | None = None,
) -> LookaheadDecision:
    """Choose a label using candidate wrist RGB-D rendered in simulation.

    Every candidate is fused into an independent clone of the same baseline.
    Thus evaluation order cannot contaminate later candidates. If a triggering
    map gap is supplied, candidates that actually see through that voxel are
    ranked ahead of unrelated global gain. Candidate RGB-D never enters
    SmolVLA; only the selected seven-motor target becomes the label.
    """
    current = np.asarray(tuple(current_joint_deg), dtype=np.float64)
    if current.shape != (7,) or not np.all(np.isfinite(current)):
        raise ValueError("current_joint_deg must contain seven finite values")
    baseline_surface = baseline.surface_voxels()
    baseline_known = baseline_surface | baseline.free_voxels()
    target_point = None
    if target_point_map_m is not None:
        target_point = np.asarray(tuple(target_point_map_m), dtype=np.float64)
        if target_point.shape != (3,) or not np.all(np.isfinite(target_point)):
            raise ValueError("target_point_map_m must contain three finite values")
    scores: list[LookaheadScore] = []
    for observation in observations:
        candidate = observation.candidate
        if not candidate.feasible:
            continue
        joints = np.asarray(candidate.joint_target_deg, dtype=np.float64)
        if joints.shape != (7,) or not np.all(np.isfinite(joints)):
            continue
        trial = baseline.clone()
        trial.integrate(
            observation.depth_m,
            observation.rgb,
            observation.intrinsics,
            observation.transform_map_camera,
            source=f"teacher:{candidate.candidate_id}",
            stamp_ns=1,
            stride=stride,
        )
        new_surface = len(trial.surface_voxels() - baseline_surface)
        new_known = len((trial.surface_voxels() | trial.free_voxels()) - baseline_known)
        target_gap_revealed = False
        if target_point is not None:
            transform_camera_map = np.linalg.inv(observation.transform_map_camera)
            target_optical = transform_camera_map @ np.append(target_point, 1.0)
            target_depth = float(target_optical[2])
            if target_depth > 0.15:
                fx, fy, cx, cy = observation.intrinsics
                pixel_u = int(round(fx * target_optical[0] / target_depth + cx))
                pixel_v = int(round(fy * target_optical[1] / target_depth + cy))
                height, width = observation.depth_m.shape
                if 0 <= pixel_u < width and 0 <= pixel_v < height:
                    observed_depth = float(observation.depth_m[pixel_v, pixel_u])
                    target_gap_revealed = math.isfinite(observed_depth) and observed_depth >= target_depth - 0.05
        motion_cost = float(np.linalg.norm((joints - current) / 180.0))
        total = float(new_surface + known_gain_weight * new_known - motion_weight * motion_cost)
        scores.append(LookaheadScore(candidate, target_gap_revealed, new_surface, new_known, motion_cost, total))
    if not scores:
        raise ValueError("no feasible simulation look-ahead candidates")
    scores.sort(
        key=lambda item: (
            -int(item.target_gap_revealed),
            -item.total,
            -item.new_surface_voxels,
            -item.new_known_voxels,
            item.motion_cost,
            item.candidate.candidate_id,
        )
    )
    return LookaheadDecision(selected=scores[0], ranked=tuple(scores))

#!/usr/bin/env python3
"""Read-only integrity audit for binary-tree NBV teacher demonstrations."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image


EVENT_RE = re.compile(r"^lap-(\d{3})-stage-(\d+)-(left|right)-(\d+)$")
EXPECTED_HAZARD_SIDE = {1: "right", 2: "left", 3: "right"}


def audit(root: Path, *, expected_episodes: int, expected_laps: int) -> dict:
    episodes = sorted(root.glob("session_*/env_*"))
    errors: list[str] = []
    events: list[str] = []
    labels: Counter[int] = Counter()
    laps: defaultdict[int, list[tuple[int, str, int, int]]] = defaultdict(list)
    resolutions: Counter[tuple[int, int]] = Counter()
    modes: Counter[str] = Counter()
    total_frames = 0
    decoded_pngs = 0
    max_overshoot_m = 0.0

    def fail(message: str) -> None:
        if len(errors) < 50:
            errors.append(message)

    for index, episode in enumerate(episodes, start=1):
        try:
            meta = json.loads((episode / "meta.json").read_text(encoding="utf-8"))
            arrays = {
                name: np.load(episode / f"{name}.npy", mmap_mode="r")
                for name in ("states", "arm_actions", "actions", "decisions", "timestamps")
            }
            frame_count = int(meta.get("num_frames", -1))
            total_frames += frame_count
            expected_shapes = {
                "states": (frame_count, 7),
                "arm_actions": (frame_count, 7),
                "actions": (frame_count, 8),
                "decisions": (frame_count, 1),
                "timestamps": (frame_count,),
            }
            for name, shape in expected_shapes.items():
                if arrays[name].shape != shape:
                    fail(f"{episode}: {name} shape {arrays[name].shape} != {shape}")
                if not np.isfinite(arrays[name]).all():
                    fail(f"{episode}: {name} contains non-finite values")
            if not np.allclose(arrays["actions"][:, :7], arrays["arm_actions"], rtol=0, atol=1e-6):
                fail(f"{episode}: action arm prefix mismatch")
            if not np.allclose(arrays["actions"][:, 7], arrays["decisions"][:, 0], rtol=0, atol=1e-6):
                fail(f"{episode}: action decision suffix mismatch")
            if frame_count > 1 and not np.all(np.diff(arrays["timestamps"]) > 0):
                fail(f"{episode}: timestamps are not strictly increasing")
            nonzero = np.flatnonzero(np.abs(arrays["decisions"][:, 0]) > 1e-6)
            if len(nonzero) != 1 or int(nonzero[0]) != frame_count - 1:
                fail(f"{episode}: terminal decision indices {nonzero.tolist()}")
            terminal = int(round(float(arrays["decisions"][-1, 0])))
            if terminal not in (-1, 1):
                fail(f"{episode}: invalid terminal decision {terminal}")
            if meta.get("terminal_decision") != terminal:
                fail(f"{episode}: metadata terminal decision mismatch")
            if meta.get("disposition") != "accepted":
                fail(f"{episode}: disposition is {meta.get('disposition')!r}")
            if meta.get("action_dim") != 8 or meta.get("state_dim") != 7:
                fail(f"{episode}: 7D-state/8D-action contract mismatch")

            rows = [
                json.loads(line)
                for line in (episode / "frame_metadata.jsonl").read_text(encoding="utf-8").splitlines()
                if line.strip()
            ]
            if len(rows) != frame_count:
                fail(f"{episode}: metadata rows {len(rows)} != {frame_count}")
            if not rows:
                continue

            event = rows[0].get("event_id")
            events.append(event)
            match = EVENT_RE.match(event or "")
            if not match:
                fail(f"{episode}: invalid event id {event!r}")
                lap = stage = serial = -1
                side = ""
            else:
                lap = int(match.group(1))
                stage = int(match.group(2))
                side = match.group(3)
                serial = int(match.group(4))
                laps[lap].append((stage, side, serial, terminal))
            if side != meta.get("target_side"):
                fail(f"{episode}: target-side mismatch")
            expected_terminal = -1 if side == EXPECTED_HAZARD_SIDE.get(stage) else 1
            if terminal != expected_terminal:
                fail(f"{episode}: label {terminal} != scene truth {expected_terminal}")
            labels[terminal] += 1
            if any(row.get("event_id") != event for row in rows):
                fail(f"{episode}: event id drift within episode")
            if any(row.get("target_side") != side for row in rows):
                fail(f"{episode}: target side drift within episode")
            if any(row.get("base_paused") is not True for row in rows):
                fail(f"{episode}: recorded a frame while the base was moving")
            for row in rows:
                overshoot = float(row.get("go2_base_branch_overshoot_m", float("inf")))
                limit = float(row.get("go2_base_branch_overshoot_limit_m", 0.05))
                if np.isfinite(overshoot):
                    max_overshoot_m = max(max_overshoot_m, overshoot)
                if not np.isfinite(overshoot) or overshoot >= limit + 1e-6:
                    fail(f"{episode}: base overshoot {overshoot} >= {limit}")
                    break
            terminal_rows = [i for i, row in enumerate(rows) if row.get("terminal_label_frame") is True]
            if terminal_rows != [frame_count - 1]:
                fail(f"{episode}: terminal metadata rows {terminal_rows[-5:]}")
            final = rows[-1]
            if final.get("decision") != terminal or final.get("scripted_scan_completed") is not True:
                fail(f"{episode}: final metadata contract mismatch")
            if int(final.get("teacher_samples", 0)) < 1:
                fail(f"{episode}: no teacher samples")
            if terminal == -1 and float(final.get("teacher_red_ratio", -1)) <= 0:
                fail(f"{episode}: hazard label has no red pixels")
            if any(int(row.get("decision", 0)) != 0 for row in rows[:-1]):
                fail(f"{episode}: decision label appears before terminal frame")

            for camera in ("front", "wrist"):
                images = sorted((episode / "frames").glob(f"{camera}_*.png"))
                if len(images) != frame_count:
                    fail(f"{episode}: {camera} image count {len(images)} != {frame_count}")
                for image_index, image_path in enumerate(images):
                    expected_name = f"{camera}_{image_index:06d}.png"
                    if image_path.name != expected_name:
                        fail(f"{episode}: non-contiguous {camera} image {image_path.name}")
                        break
                    try:
                        with Image.open(image_path) as image:
                            image.load()
                            resolutions[image.size] += 1
                            modes[image.mode] += 1
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        fail(f"{image_path}: decode failed: {exc}")
                    decoded_pngs += 1
        except Exception as exc:  # pragma: no cover - diagnostic path
            fail(f"{episode}: {type(exc).__name__}: {exc}")

        if index % 20 == 0 or index == len(episodes):
            print(f"audit progress {index}/{len(episodes)} episodes", flush=True)

    if len(episodes) != expected_episodes:
        fail(f"episode count {len(episodes)} != {expected_episodes}")
    if len(set(events)) != expected_episodes:
        fail(f"unique event count {len(set(events))} != {expected_episodes}")
    expected_per_label = expected_episodes // 2
    if labels != Counter({-1: expected_per_label, 1: expected_per_label}):
        fail(f"label balance {dict(labels)}")
    if set(laps) != set(range(1, expected_laps + 1)):
        fail(f"lap ids {sorted(laps)}")
    expected_pairs = Counter({(stage, side): 1 for stage in (1, 2, 3) for side in ("left", "right")})
    for lap in range(1, expected_laps + 1):
        entries = laps.get(lap, [])
        pairs = Counter((stage, side) for stage, side, _, _ in entries)
        serials = sorted(serial for _, _, serial, _ in entries)
        if len(entries) != 6 or pairs != expected_pairs or serials != list(range(1, 7)):
            fail(f"lap {lap}: incomplete event topology {entries}")

    rejected_files = sum(
        1 for path in root.rglob("*") if path.is_file() and "rejected" in path.parts
    )
    staging_artifacts = [
        str(path)
        for path in root.rglob("*")
        if path.name.lower() in ("staging", ".staging") or path.name.endswith(".partial")
    ]
    if rejected_files:
        fail(f"rejected files {rejected_files}")
    if staging_artifacts:
        fail(f"staging artifacts {staging_artifacts[:5]}")

    total_bytes = sum(path.stat().st_size for path in root.rglob("*") if path.is_file())
    return {
        "status": "PASS" if not errors else "FAIL",
        "episodes": len(episodes),
        "unique_events": len(set(events)),
        "laps": len(laps),
        "labels": {str(key): value for key, value in sorted(labels.items())},
        "frames": total_frames,
        "decoded_pngs": decoded_pngs,
        "image_resolutions": {f"{key[0]}x{key[1]}": value for key, value in resolutions.items()},
        "image_modes": dict(modes),
        "dataset_bytes": total_bytes,
        "max_base_overshoot_m": max_overshoot_m,
        "rejected_files": rejected_files,
        "staging_artifacts": len(staging_artifacts),
        "errors": errors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("root", type=Path)
    parser.add_argument("--expected-episodes", type=int, default=102)
    parser.add_argument("--expected-laps", type=int, default=17)
    args = parser.parse_args()
    summary = audit(
        args.root.expanduser().resolve(),
        expected_episodes=args.expected_episodes,
        expected_laps=args.expected_laps,
    )
    print(json.dumps(summary, indent=2, sort_keys=True), flush=True)
    return 0 if summary["status"] == "PASS" else 1


if __name__ == "__main__":
    sys.exit(main())

"""Read-only live status, or save a post-run audit of randomized VLA trials."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import re


def audit(root: Path) -> dict:
    layout = json.loads((root / "layout.json").read_text())
    rows = []
    for line in (root / "decision_ledger.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass  # An in-progress writer can leave one incomplete final line.
    labels = [r for r in rows if r.get("state") == "TREE_VLA8_SIGNAL"]
    selections = [r for r in rows if r.get("state") == "TREE_SAFE_BRANCH_SELECTED"]
    stages = []
    for stage in layout["stages"]:
        number = stage["index"]
        by_side = {}
        for row in labels:
            if re.search(rf"(?:^|-)stage-{number}-", row.get("event_id", "")):
                by_side[row["inspected_side"]] = row
        chosen = next((r for r in selections if r.get("stage") == number), None)
        expected = {s: (-1 if s == stage["hazard_side"] else 1) for s in ("left", "right")}
        labels_correct = len(by_side) == 2 and all(
            r.get("vla8_signal") == expected[s]
            and r.get("signal_source") == "trained_smolvla_action_7"
            for s, r in by_side.items()
        )
        before_departure = chosen is not None and len(by_side) == 2 and all(
            r["stamp_ns"] < chosen["stamp_ns"] for r in by_side.values()
        )
        safe_choice = chosen is not None and chosen.get("selected_safe") == stage["safe_side"]
        no_truth_veto = chosen is not None and chosen.get("simulator_truth_used_for_control") is False
        stages.append({
            "stage": number, "actual_cube": stage["hazard_side"],
            "raw": {s: r.get("raw_decision") for s, r in by_side.items()},
            "selected": chosen.get("selected_safe") if chosen else None,
            "labels_correct": labels_correct, "before_departure": before_departure,
            "safe_choice": safe_choice, "truth_veto_reported_off": no_truth_veto,
        })
    complete = next((r for r in reversed(rows) if r.get("state") == "COMPLETE"), None)
    holds = [r.get("reason") for r in rows if r.get("state") == "HOLD"]
    passed = bool(complete and complete.get("stages_completed") == 3 and not holds
                  and len(labels) == 6 and len(selections) == 3
                  and all(s["labels_correct"] and s["before_departure"]
                          and s["safe_choice"] and s["truth_veto_reported_off"] for s in stages))
    return {
        "trial": root.name, "last_state": rows[-1].get("state") if rows else None,
        "last_sim_s": rows[-1].get("stamp_ns", 0) / 1e9 if rows else 0,
        "stage_results": stages, "complete": complete is not None,
        "holds": holds, "signal_count": len(labels), "selection_count": len(selections),
        "signal_and_route_audit_pass": passed,
        "scope": "Post-run label/selection/completion audit; not pixel-level collision or physical pose verification.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path)
    parser.add_argument("--save", action="store_true")
    args = parser.parse_args()
    result = audit(args.root)
    encoded = json.dumps(result, indent=2, ensure_ascii=False) + "\n"
    if args.save:
        (args.root / "audit.json").write_text(encoded)
    print(encoded)

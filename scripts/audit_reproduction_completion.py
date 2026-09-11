#!/usr/bin/env python3
"""Fail-closed completion audit for the paper reproduction."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any] | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def check(name: str, achieved: bool, evidence: str, detail: str) -> dict[str, object]:
    return {"requirement": name, "achieved": achieved, "evidence": evidence, "detail": detail}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--comparison", type=Path, default=Path("outputs/comparison/paper_vs_local.csv"))
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison/completion_audit.json"))
    args = parser.parse_args()
    rows = list(csv.DictReader(args.comparison.open(encoding="utf-8"))) if args.comparison.exists() else []
    ubfc = load_json(Path("outputs/data_checks/ubfc.json"))
    mmpd = load_json(Path("outputs/data_checks/mmpd.json")) or load_json(
        Path("outputs/data_checks/mmpd-mini.json")
    )
    light_optimization = load_json(Path("outputs/optimization/light_rppgvit.json"))
    accuracy = [row for row in rows if row["table"] == "accuracy"]
    parameters = [row for row in rows if row["metric"] == "Params_M"]
    flops = [row for row in rows if row["metric"] == "FLOPs_G"]
    latency = [row for row in rows if row["metric"] == "Latency_ms"]
    requirements = [
        check(
            "UBFC-rPPG raw data complete",
            bool(ubfc and ubfc.get("ready")),
            "outputs/data_checks/ubfc.json",
            f"videos={ubfc.get('videos', 0) if ubfc else 0}/42; labels={ubfc.get('labels', 0) if ubfc else 0}/42",
        ),
        check(
            "MMPD authorized raw data complete",
            bool(mmpd and mmpd.get("ready")),
            "outputs/data_checks/mmpd[-mini].json",
            f"mat_files={mmpd.get('mat_files', 0) if mmpd else 0}/660; mini/full variant must remain explicit",
        ),
        check(
            "Tables 1-2 accuracy reproduced",
            len(accuracy) == 56 and all(row["status"] == "measured" for row in accuracy),
            str(args.comparison),
            f"measured={sum(row['status'] == 'measured' for row in accuracy)}/{len(accuracy) or 56}; variant-mismatch results do not prove the paper table",
        ),
        check(
            "Table 3 parameter counts measured",
            len(parameters) == 7 and all(row["status"] == "measured" for row in parameters),
            str(args.comparison),
            f"measured={sum(row['status'] == 'measured' for row in parameters)}/7",
        ),
        check(
            "Table 3 FLOPs measured",
            len(flops) == 7 and all(row["status"] == "measured" for row in flops),
            str(args.comparison),
            f"measured={sum(row['status'] == 'measured' for row in flops)}/7; unsupported operators remain disclosed",
        ),
        check(
            "Light-rPPGViT deployment optimization reproduced",
            bool(
                light_optimization
                and light_optimization.get("status") == "measured"
                and light_optimization.get("reparameterization", {}).get("verified") is True
                and light_optimization.get("qat", {}).get("evaluated") is True
                and light_optimization.get("structured_pruning", {}).get("evaluated") is True
            ),
            "outputs/optimization/light_rppgvit.json",
            (
                "Requires measured re-parameterization equivalence plus explicitly configured and "
                "evaluated QAT and structured-pruning trials; the paper omits bit width, observer, "
                "backend, pruning rate, criterion, and fine-tuning schedule"
            ),
        ),
        check(
            "Table 3 Snapdragon XR2 latency reproduced",
            len(latency) == 7 and all(row["status"] == "measured" for row in latency),
            str(args.comparison),
            f"XR2-context measured={sum(row['status'] == 'measured' for row in latency)}/7",
        ),
        check(
            "XR2 power experiment reproduced",
            Path("outputs/xr2/power.json").exists(),
            "outputs/xr2/power.json",
            "Requires Trepn Profiler and the target XR2 AR device",
        ),
    ]
    achieved = sum(bool(item["achieved"]) for item in requirements)
    report = {
        "complete": achieved == len(requirements),
        "requirements_achieved": achieved,
        "requirements_total": len(requirements),
        "requirements": requirements,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["complete"] else 2)


if __name__ == "__main__":
    main()

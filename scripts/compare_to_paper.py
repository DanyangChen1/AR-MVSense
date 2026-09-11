#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


ACCURACY_PATHS = {
    ("UBFC", "TSCAN"): (Path("outputs/toolbox-results/ubfc/tscan/metrics.json"), "measured"),
    ("UBFC", "PhysNet"): (Path("outputs/toolbox-results/ubfc/physnet/metrics.json"), "measured"),
    ("UBFC", "DeepPhys"): (Path("outputs/toolbox-results/ubfc/deepphys/metrics.json"), "measured"),
    ("UBFC", "PhysFormer"): (Path("outputs/toolbox-results/ubfc/physformer/metrics.json"), "measured"),
    ("UBFC", "EfficientPhys"): (Path("outputs/toolbox-results/ubfc/efficientphys/metrics.json"), "measured"),
    ("UBFC", "rPPGViT"): (
        Path("outputs/ubfc/rppgvit/trials/window64_cosine_physical_batch4/metrics.json"),
        "measured",
    ),
    ("UBFC", "Light-rPPGViT"): (
        Path("outputs/ubfc/light_rppgvit/trials/physical_batch4/metrics.json"),
        "measured",
    ),
    ("MMPD", "TSCAN"): (Path("outputs/toolbox-results/mmpd-mini/tscan/metrics.json"), "dataset_variant_mismatch"),
    ("MMPD", "PhysNet"): (Path("outputs/toolbox-results/mmpd-mini/physnet/metrics.json"), "dataset_variant_mismatch"),
    ("MMPD", "DeepPhys"): (Path("outputs/toolbox-results/mmpd-mini/deepphys/metrics.json"), "dataset_variant_mismatch"),
    ("MMPD", "PhysFormer"): (Path("outputs/toolbox-results/mmpd-mini/physformer/metrics.json"), "dataset_variant_mismatch"),
    ("MMPD", "EfficientPhys"): (Path("outputs/toolbox-results/mmpd-mini/efficientphys/metrics.json"), "dataset_variant_mismatch"),
    ("MMPD", "rPPGViT"): (Path("outputs/mmpd-mini/rppgvit/metrics.json"), "dataset_variant_mismatch"),
    ("MMPD", "Light-rPPGViT"): (Path("outputs/mmpd-mini/light_rppgvit/metrics.json"), "dataset_variant_mismatch"),
}


def load_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def accuracy_rows(targets: dict[str, Any]) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for dataset, models in targets["accuracy"].items():
        for model, expected in models.items():
            result_spec = ACCURACY_PATHS.get((dataset, model))
            result_path = result_spec[0] if result_spec else None
            measured_status = result_spec[1] if result_spec else "pending"
            actual = load_json(result_path) if result_path else None
            for metric, target in expected.items():
                local = actual.get(metric) if actual else None
                rows.append(
                    {
                        "table": "accuracy",
                        "dataset": dataset,
                        "model": model,
                        "metric": metric,
                        "paper": target,
                        "local": local,
                        "delta_local_minus_paper": None if local is None else float(local) - target,
                        "status": measured_status if local is not None else "pending",
                        "evidence": str(result_path) if result_path else "baseline runner not yet mapped",
                        "note": (
                            "mini-MMPD result; paper does not identify whether Table 2 used mini or full data"
                            if local is not None and measured_status == "dataset_variant_mismatch"
                            else ""
                        ),
                    }
                )
    return rows


def efficiency_rows(
    targets: dict[str, Any], benchmark_path: Path, baseline_benchmark_path: Path
) -> list[dict[str, object]]:
    benchmark = load_json(benchmark_path)
    models = benchmark.get("models", {}) if benchmark else {}
    local_map: dict[str, dict[str, float]] = {}
    if "rPPGViT" in models:
        item = models["rPPGViT"]
        local_map["rPPGViT"] = {
            "Params_M": item["parameters"] / 1e6,
            "FLOPs_G": item.get("paper_comparable_known_fma_ops", item["fvcore_known_fma_ops"]) / 1e9,
            "Latency_ms": item["per_frame_ms_median"],
        }
    if "Light-rPPGViT-deploy" in models:
        item = models["Light-rPPGViT-deploy"]
        local_map["Light-rPPGViT"] = {
            "Params_M": item["parameters"] / 1e6,
            "FLOPs_G": item.get("paper_comparable_known_fma_ops", item["fvcore_known_fma_ops"]) / 1e9,
            "Latency_ms": item["per_frame_ms_median"],
        }
    baseline_benchmark = load_json(baseline_benchmark_path)
    for name, item in (baseline_benchmark.get("models", {}) if baseline_benchmark else {}).items():
        local_map[name] = {
            "Params_M": item["parameters"] / 1e6,
            "FLOPs_G": item.get(
                "paper_comparable_known_fma_ops_normalized_128_frames",
                item["fvcore_known_fma_ops_normalized_128_frames"],
            ) / 1e9,
            "Latency_ms": item["per_output_frame_ms_median"],
        }
    rows: list[dict[str, object]] = []
    for model, expected in targets["efficiency"].items():
        actual = local_map.get(model)
        for metric, target in expected.items():
            local = actual.get(metric) if actual else None
            status = "pending"
            note = ""
            if local is not None:
                status = "context_mismatch" if metric == "Latency_ms" else "measured"
                if metric == "Latency_ms":
                    note = "Apple M2 CPU throughput per input frame; paper uses Snapdragon XR2 per-frame latency"
                elif metric == "FLOPs_G":
                    note = "paper-compatible one-FMA convention; lower bound because FVCore lists unsupported operators"
            rows.append(
                {
                    "table": "efficiency",
                    "dataset": "",
                    "model": model,
                    "metric": metric,
                    "paper": target,
                    "local": local,
                    "delta_local_minus_paper": None if local is None else float(local) - target,
                    "status": status,
                    "evidence": (
                        str(benchmark_path)
                        if model in {"rPPGViT", "Light-rPPGViT"}
                        else str(baseline_benchmark_path)
                    ) if local is not None else "not measured locally",
                    "note": note,
                }
            )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--targets", type=Path, default=Path("configs/paper_targets.json"))
    parser.add_argument(
        "--benchmark", type=Path, default=Path("outputs/benchmarks/apple_m2_cpu.json")
    )
    parser.add_argument(
        "--baseline-benchmark",
        type=Path,
        default=Path("outputs/benchmarks/toolbox_baselines_apple_m2_cpu.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/comparison/paper_vs_local.csv"))
    args = parser.parse_args()
    targets = json.loads(args.targets.read_text(encoding="utf-8"))
    rows = accuracy_rows(targets) + efficiency_rows(
        targets, args.benchmark, args.baseline_benchmark
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "table", "dataset", "model", "metric", "paper", "local",
        "delta_local_minus_paper", "status", "evidence", "note",
    ]
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    counts: dict[str, int] = {}
    for row in rows:
        counts[str(row["status"])] = counts.get(str(row["status"]), 0) + 1
    print(json.dumps({"output": str(args.output), "rows": len(rows), "status": counts}, indent=2))


if __name__ == "__main__":
    main()

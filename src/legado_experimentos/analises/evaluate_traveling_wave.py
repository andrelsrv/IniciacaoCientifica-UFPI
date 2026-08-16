"""Calibra em treino e avalia o localizador físico somente na validação."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from manifest import read_manifest, resolve_pl4_path
from signal_io import read_canonical_pl4
from traveling_wave_localizer import TravelingWaveConfig, locate


def _read_feature_cache(path: Path) -> dict[str, dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return {row["run_id"]: row for row in csv.DictReader(stream)}


def calibrate_velocity(manifest_path: Path, cache: dict[str, dict[str, str]]) -> float:
    estimates = []
    for row in read_manifest(manifest_path):
        if row.split != "train" or abs(row.remote_length_km - row.distance_km) < 50:
            continue
        item = cache[row.run_id]
        delay_us = (
            float(item["bea_arrival_s"]) - float(item["pdt_arrival_s"])
        ) * 1e6
        if abs(delay_us) >= 50:
            estimates.append((row.remote_length_km - row.distance_km) / delay_us)
    velocity = float(np.median(estimates))
    if not 0.2 <= velocity <= 0.35:
        raise ValueError(f"Velocidade calibrada implausível: {velocity} km/us")
    return velocity


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cache = _read_feature_cache(args.feature_cache)
    velocity = calibrate_velocity(args.manifest, cache)
    config = TravelingWaveConfig(velocity_km_per_us=velocity)
    results = []
    for row in read_manifest(args.manifest):
        if row.split != "validation":
            continue
        signals = read_canonical_pl4(resolve_pl4_path(row, args.manifest))
        found = locate(signals, config)
        results.append({
            "run_id": row.run_id,
            "fault_class": row.fault_class,
            "true_distance_km": row.distance_km,
            "estimated_distance_km": found.distance_km,
            "absolute_error_km": abs(found.distance_km - row.distance_km),
            "estimated_remote_km": found.remote_distance_km,
            "consistency_error_us": found.consistency_error_us,
            "correlation_pdt": found.correlation_pdt,
            "correlation_bea": found.correlation_bea,
            "candidate_margin": found.candidate_margin,
            "conclusive": found.conclusive,
            "inconclusive_reason": found.inconclusive_reason or "",
        })
    errors = np.asarray([float(row["absolute_error_km"]) for row in results])
    conclusive_results = [row for row in results if bool(row["conclusive"])]
    conclusive_errors = np.asarray(
        [float(row["absolute_error_km"]) for row in conclusive_results]
    )
    by_class: dict[str, list[float]] = defaultdict(list)
    conclusive_by_class: dict[str, list[float]] = defaultdict(list)
    for row in results:
        by_class[str(row["fault_class"])].append(float(row["absolute_error_km"]))
        if bool(row["conclusive"]):
            conclusive_by_class[str(row["fault_class"])].append(
                float(row["absolute_error_km"])
            )
    report = {
        "method": "directional_reflection_pair_v1",
        "test_unseen_opened": False,
        "calibrated_velocity_km_per_us": velocity,
        "validation_runs": len(results),
        "metrics_km": {
            "mae": float(np.mean(errors)),
            "median": float(np.median(errors)),
            "p95": float(np.percentile(errors, 95)),
            "maximum": float(np.max(errors)),
        },
        "conclusive_coverage": len(conclusive_results) / len(results),
        "inconclusive_runs": [
            row["run_id"] for row in results if not bool(row["conclusive"])
        ],
        "conclusive_metrics_km": {
            "mae": float(np.mean(conclusive_errors)),
            "median": float(np.median(conclusive_errors)),
            "p95": float(np.percentile(conclusive_errors, 95)),
            "maximum": float(np.max(conclusive_errors)),
        },
        "mae_by_class_km": {
            label: float(np.mean(class_errors))
            for label, class_errors in sorted(by_class.items())
        },
        "conclusive_mae_by_class_km": {
            label: float(np.mean(class_errors))
            for label, class_errors in sorted(conclusive_by_class.items())
        },
        "config": config.__dict__,
        "results": results,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({key: value for key, value in report.items() if key != "results"}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

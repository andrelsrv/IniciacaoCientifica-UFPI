"""Executa uma única avaliação cega com o pipeline previamente congelado."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from adaptive_localizer import locate_adaptive
from feature_extraction import extract_features
from manifest import read_manifest, resolve_pl4_path
from robustness_evaluation import CONDITIONS, perturb
from signal_io import read_canonical_pl4
from traveling_wave_localizer import TravelingWaveConfig


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def _location_metrics(errors: list[float]) -> dict[str, float | None]:
    if not errors:
        return {"mae": None, "median": None, "p95": None, "maximum": None}
    values = np.asarray(errors)
    return {
        "mae": float(np.mean(values)),
        "median": float(np.median(values)),
        "p95": float(np.percentile(values, 95)),
        "maximum": float(np.max(values)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    freeze = json.loads(args.freeze.read_text(encoding="utf-8"))
    expected = freeze["classifier"]["sha256"]
    actual = _sha256(args.classifier)
    if actual != expected:
        raise ValueError(f"Classificador difere do congelamento: {actual} != {expected}")
    artifact = joblib.load(args.classifier)
    classifier = artifact["classifier"]
    feature_names = tuple(artifact["feature_names"])
    velocity = float(freeze["traveling_wave_velocity_km_per_us"])
    config = TravelingWaveConfig(velocity_km_per_us=velocity)
    test_rows = [row for row in read_manifest(args.manifest) if row.split == "test_unseen"]
    classes = tuple(sorted({row.fault_class for row in test_rows}))
    buckets = {
        condition.name: {"truth": [], "prediction": [], "errors": [], "rejected": []}
        for condition in CONDITIONS
    }
    for index, row in enumerate(test_rows, start=1):
        original = read_canonical_pl4(resolve_pl4_path(row, args.manifest))
        for condition in CONDITIONS:
            signals = perturb(original, row.run_id, condition)
            features = extract_features(signals)
            if features.names != feature_names:
                raise RuntimeError("Atributos do teste diferem do artefato congelado.")
            prediction = str(classifier.predict(features.values.reshape(1, -1))[0])
            location = locate_adaptive(signals, config)
            bucket = buckets[condition.name]
            bucket["truth"].append(row.fault_class)
            bucket["prediction"].append(prediction)
            if location.conclusive and location.distance_km is not None:
                bucket["errors"].append(abs(location.distance_km - row.distance_km))
            else:
                bucket["rejected"].append(row.run_id)
        if index % 10 == 0:
            print(f"[{index}/{len(test_rows)}] casos cegos avaliados", flush=True)
    conditions = {}
    for condition in CONDITIONS:
        bucket = buckets[condition.name]
        truth = np.asarray(bucket["truth"])
        prediction = np.asarray(bucket["prediction"])
        conditions[condition.name] = {
            "parameters": condition.__dict__,
            "classification_accuracy": float(accuracy_score(truth, prediction)),
            "classification_macro_f1": float(f1_score(truth, prediction, average="macro")),
            "classification_confusion_matrix": confusion_matrix(
                truth, prediction, labels=classes
            ).tolist(),
            "localization_conclusive_coverage": len(bucket["errors"]) / len(test_rows),
            "localization_conclusive_metrics_km": _location_metrics(bucket["errors"]),
            "inconclusive_count": len(bucket["rejected"]),
            "inconclusive_runs": bucket["rejected"],
        }
    report = {
        "evaluation": "FINAL_TEST_UNSEEN",
        "pipeline_frozen_before_test": True,
        "freeze_file_sha256": _sha256(args.freeze),
        "classifier_sha256": actual,
        "test_base_runs": len(test_rows),
        "conditions": conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({**report, "conditions": {
        name: {key: value for key, value in item.items() if key not in {"classification_confusion_matrix", "inconclusive_runs"}}
        for name, item in conditions.items()
    }}, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

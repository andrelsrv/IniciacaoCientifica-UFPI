"""Avalia classificação e localização sob perturbações de medição."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from adaptive_localizer import locate_adaptive
from feature_extraction import extract_features
from manifest import read_manifest, resolve_pl4_path
from multiscale_localizer import locate_multiscale
from signal_io import SignalData, read_canonical_pl4
from traveling_wave_localizer import TravelingWaveConfig, locate


@dataclass(frozen=True)
class Perturbation:
    name: str
    snr_db: float | None = None
    gain_error_pct: float = 0.0
    sync_error_us: float = 0.0


CONDITIONS = (
    Perturbation("ideal"),
    Perturbation("snr_60db", snr_db=60),
    Perturbation("snr_40db", snr_db=40),
    Perturbation("snr_30db", snr_db=30),
    Perturbation("gain_1pct", gain_error_pct=1),
    Perturbation("sync_1us", sync_error_us=1),
    Perturbation(
        "combined_snr30_gain1_sync1",
        snr_db=30,
        gain_error_pct=1,
        sync_error_us=1,
    ),
)


def _rng(run_id: str, condition: str) -> np.random.Generator:
    digest = hashlib.sha256(f"{run_id}|{condition}|20260802".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "little"))


def perturb(signals: SignalData, run_id: str, condition: Perturbation) -> SignalData:
    rng = _rng(run_id, condition.name)
    values = signals.values.copy()
    if condition.gain_error_pct:
        gains = 1 + rng.uniform(
            -condition.gain_error_pct / 100,
            condition.gain_error_pct / 100,
            size=values.shape[1],
        )
        values *= gains
    if condition.snr_db is not None:
        rms = np.sqrt(np.mean(values**2, axis=0))
        noise_rms = rms / (10 ** (condition.snr_db / 20))
        values += rng.normal(size=values.shape) * noise_rms
    if condition.sync_error_us:
        dt_us = float(np.median(np.diff(signals.time_s))) * 1e6
        magnitude = max(1, int(round(condition.sync_error_us / dt_us)))
        shift = magnitude if rng.integers(0, 2) else -magnitude
        bea = values[:, [3, 4, 5, 9, 10, 11]].copy()
        if shift > 0:
            bea[shift:] = bea[:-shift]
            bea[:shift] = bea[shift]
        else:
            amount = -shift
            bea[:-amount] = bea[amount:]
            bea[-amount:] = bea[-amount - 1]
        values[:, [3, 4, 5, 9, 10, 11]] = bea
    return SignalData(signals.time_s, values, signals.channel_names)


def _load_cache(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def _train_classifier(cache: list[dict[str, str]], artifact_path: Path):
    artifact = joblib.load(artifact_path)
    names = tuple(artifact["feature_names"])
    train = [row for row in cache if row["split"] == "train"]
    x = np.asarray([[float(row[name]) for name in names] for row in train])
    y = np.asarray([row["fault_class"] for row in train])
    model = ExtraTreesClassifier(
        n_estimators=400,
        min_samples_leaf=1,
        max_features="sqrt",
        class_weight="balanced",
        random_state=20260802,
        n_jobs=-1,
    ).fit(x, y)
    return model, names


def _metrics(errors: list[float]) -> dict[str, float | None]:
    if not errors:
        return {"mae": None, "median": None, "p95": None, "maximum": None}
    array = np.asarray(errors)
    return {
        "mae": float(np.mean(array)),
        "median": float(np.median(array)),
        "p95": float(np.percentile(array, 95)),
        "maximum": float(np.max(array)),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--feature-cache", type=Path, required=True)
    parser.add_argument("--baseline-artifact", type=Path, required=True)
    parser.add_argument("--traveling-wave-report", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--robust-classifier-artifact", type=Path)
    parser.add_argument("--multiscale-localizer", action="store_true")
    parser.add_argument("--adaptive-localizer", action="store_true")
    args = parser.parse_args()
    cache = _load_cache(args.feature_cache)
    if args.robust_classifier_artifact:
        robust_artifact = joblib.load(args.robust_classifier_artifact)
        classifier = robust_artifact["classifier"]
        feature_names = tuple(robust_artifact["feature_names"])
        classifier_training = "augmented_train_only"
    else:
        classifier, feature_names = _train_classifier(cache, args.baseline_artifact)
        classifier_training = "ideal_train_only"
    tw_report = json.loads(args.traveling_wave_report.read_text(encoding="utf-8"))
    config = TravelingWaveConfig(**tw_report["config"])
    validation = [row for row in read_manifest(args.manifest) if row.split == "validation"]
    collected: dict[str, dict[str, object]] = {
        condition.name: {
            "truth": [], "prediction": [], "errors": [], "inconclusive": [],
        }
        for condition in CONDITIONS
    }
    for index, row in enumerate(validation, start=1):
        original = read_canonical_pl4(resolve_pl4_path(row, args.manifest))
        for condition in CONDITIONS:
            signals = perturb(original, row.run_id, condition)
            features = extract_features(signals)
            prediction = str(classifier.predict(features.values.reshape(1, -1))[0])
            bucket = collected[condition.name]
            bucket["truth"].append(row.fault_class)
            bucket["prediction"].append(prediction)
            try:
                if args.adaptive_localizer:
                    location = locate_adaptive(signals, config)
                elif args.multiscale_localizer:
                    location = locate_multiscale(signals, config)
                else:
                    location = locate(signals, config)
                if location.conclusive and location.distance_km is not None:
                    bucket["errors"].append(abs(location.distance_km - row.distance_km))
                else:
                    bucket["inconclusive"].append(row.run_id)
            except ValueError:
                bucket["inconclusive"].append(row.run_id)
        if index % 10 == 0:
            print(f"[{index}/{len(validation)}] casos-base avaliados", flush=True)

    report_conditions = {}
    for condition in CONDITIONS:
        bucket = collected[condition.name]
        truth = np.asarray(bucket["truth"])
        prediction = np.asarray(bucket["prediction"])
        errors = list(bucket["errors"])
        report_conditions[condition.name] = {
            "parameters": condition.__dict__,
            "classification_accuracy": float(accuracy_score(truth, prediction)),
            "classification_macro_f1": float(f1_score(truth, prediction, average="macro")),
            "localization_conclusive_coverage": len(errors) / len(validation),
            "localization_conclusive_metrics_km": _metrics(errors),
            "inconclusive_count": len(bucket["inconclusive"]),
            "inconclusive_runs": bucket["inconclusive"],
        }
    report = {
        "test_unseen_opened": False,
        "classifier_trained_runs": 350,
        "classifier_training": classifier_training,
        "localizer": (
            "adaptive_snr" if args.adaptive_localizer else
            "multiscale_consensus" if args.multiscale_localizer else "single_scale"
        ),
        "validation_base_runs": len(validation),
        "deterministic_perturbation_seed": 20260802,
        "conditions": report_conditions,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

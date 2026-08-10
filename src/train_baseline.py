"""Extrai atributos e treina baseline sem abrir o conjunto de teste cego."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, ExtraTreesRegressor
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, mean_absolute_error

from feature_extraction import FEATURE_VERSION, extract_features
from manifest import ManifestRow, read_manifest, resolve_pl4_path
from signal_io import read_canonical_pl4


ALLOWED_DEVELOPMENT_SPLITS = frozenset({"train", "validation"})


def extract_development_rows(manifest_path: Path) -> tuple[list[dict[str, object]], tuple[str, ...]]:
    extracted: list[dict[str, object]] = []
    feature_names: tuple[str, ...] | None = None
    for row in read_manifest(manifest_path):
        if row.split not in ALLOWED_DEVELOPMENT_SPLITS:
            continue  # O PL4 cego não é resolvido nem aberto.
        signals = read_canonical_pl4(resolve_pl4_path(row, manifest_path))
        result = extract_features(signals)
        if feature_names is None:
            feature_names = result.names
        elif feature_names != result.names:
            raise RuntimeError("Ordem de atributos inconsistente.")
        record: dict[str, object] = {
            "run_id": row.run_id,
            "split": row.split,
            "fault_class": row.fault_class,
            "distance_km": row.distance_km,
            "event_time_s": result.event_time_s,
            "pdt_arrival_s": result.pdt_arrival_s,
            "bea_arrival_s": result.bea_arrival_s,
        }
        record.update(zip(result.names, result.values, strict=True))
        extracted.append(record)
    if feature_names is None:
        raise ValueError("Manifesto não contém treino/validação.")
    return extracted, feature_names


def _matrix(rows: list[dict[str, object]], names: tuple[str, ...]) -> np.ndarray:
    # Esta é a fronteira do modelo: nenhum caminho, run_id, split, rótulo ou
    # parâmetro físico entra em X.
    return np.asarray([[float(row[name]) for name in names] for row in rows])


def train(rows: list[dict[str, object]], names: tuple[str, ...]) -> tuple[dict[str, object], dict[str, object]]:
    train_rows = [row for row in rows if row["split"] == "train"]
    validation_rows = [row for row in rows if row["split"] == "validation"]
    x_train, x_val = _matrix(train_rows, names), _matrix(validation_rows, names)
    y_train = np.asarray([str(row["fault_class"]) for row in train_rows])
    y_val = np.asarray([str(row["fault_class"]) for row in validation_rows])
    d_train = np.asarray([float(row["distance_km"]) for row in train_rows])
    d_val = np.asarray([float(row["distance_km"]) for row in validation_rows])

    classifier_candidates = []
    for leaf in (1, 2, 4):
        model = ExtraTreesClassifier(
            n_estimators=400, min_samples_leaf=leaf, max_features="sqrt",
            class_weight="balanced", random_state=20260802, n_jobs=-1,
        ).fit(x_train, y_train)
        prediction = model.predict(x_val)
        classifier_candidates.append((f1_score(y_val, prediction, average="macro"), leaf, model))
    classifier_f1, classifier_leaf, classifier_dev = max(classifier_candidates, key=lambda item: item[0])
    class_prediction = classifier_dev.predict(x_val)

    regressor_candidates = []
    for leaf in (1, 2, 4):
        for max_features in (0.7, 1.0):
            model = ExtraTreesRegressor(
                n_estimators=500, min_samples_leaf=leaf, max_features=max_features,
                random_state=20260802, n_jobs=-1,
            ).fit(x_train, d_train)
            prediction = model.predict(x_val)
            regressor_candidates.append((mean_absolute_error(d_val, prediction), leaf, max_features, model))
    regressor_mae, regressor_leaf, regressor_features, regressor_dev = min(
        regressor_candidates, key=lambda item: item[0]
    )
    distance_prediction = regressor_dev.predict(x_val)
    distance_error = np.abs(distance_prediction - d_val)

    classes = tuple(sorted(set(y_train)))
    report = {
        "feature_version": FEATURE_VERSION,
        "test_unseen_opened": False,
        "train_runs": len(train_rows),
        "validation_runs": len(validation_rows),
        "feature_count": len(names),
        "selected": {
            "classifier_min_samples_leaf": classifier_leaf,
            "regressor_min_samples_leaf": regressor_leaf,
            "regressor_max_features": regressor_features,
        },
        "validation_classification": {
            "accuracy": float(accuracy_score(y_val, class_prediction)),
            "macro_f1": float(classifier_f1),
            "classes": classes,
            "confusion_matrix": confusion_matrix(y_val, class_prediction, labels=classes).tolist(),
            "recall_by_class": {
                label: float(np.mean(class_prediction[y_val == label] == label)) for label in classes
            },
        },
        "validation_localization_km": {
            "mae": float(regressor_mae),
            "median_absolute_error": float(np.median(distance_error)),
            "p95_absolute_error": float(np.percentile(distance_error, 95)),
            "max_absolute_error": float(np.max(distance_error)),
        },
        "detected_event_time_s": {
            "min": float(min(float(row["event_time_s"]) for row in rows)),
            "max": float(max(float(row["event_time_s"]) for row in rows)),
        },
    }

    x_all = _matrix(rows, names)
    classifier_final = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=classifier_leaf, max_features="sqrt",
        class_weight="balanced", random_state=20260802, n_jobs=-1,
    ).fit(x_all, np.asarray([str(row["fault_class"]) for row in rows]))
    regressor_final = ExtraTreesRegressor(
        n_estimators=500, min_samples_leaf=regressor_leaf, max_features=regressor_features,
        random_state=20260802, n_jobs=-1,
    ).fit(x_all, np.asarray([float(row["distance_km"]) for row in rows]))
    artifact = {
        "feature_version": FEATURE_VERSION,
        "feature_names": names,
        "classifier": classifier_final,
        "localizer": regressor_final,
        "class_counts": dict(Counter(str(row["fault_class"]) for row in rows)),
        "trained_splits": ("train", "validation"),
        "test_unseen_used": False,
    }
    return artifact, report


def write_feature_cache(rows: list[dict[str, object]], output: Path) -> None:
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    rows, names = extract_development_rows(args.manifest)
    write_feature_cache(rows, args.output_dir / "development_features.csv")
    artifact, report = train(rows, names)
    joblib.dump(artifact, args.output_dir / "baseline.joblib")
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

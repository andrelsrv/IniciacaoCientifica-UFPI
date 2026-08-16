"""Retreina o classificador combinando o manifesto v9 (com augmentacao de
ruido completa, via pl4) com a malha densa de alta impedancia v10 (features
ja pre-extraidas, sem pl4 -- os arquivos brutos foram apagados apos a
extracao para economizar espaco em disco, entao essas amostras entram so
na condicao 'ideal', sem as 6 variantes de ruido)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from feature_extraction import FEATURE_VERSION, extract_features
from manifest import read_manifest, resolve_pl4_path
from robustness_evaluation import CONDITIONS, perturb
from signal_io import read_canonical_pl4


def _candidate_models():
    for leaf in (1, 2, 4):
        yield f"extra_trees_leaf_{leaf}", ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=leaf, max_features="sqrt",
            class_weight="balanced", random_state=20260802, n_jobs=-1,
        )
    for leaf in (1, 2, 4):
        yield f"random_forest_leaf_{leaf}", RandomForestClassifier(
            n_estimators=600, min_samples_leaf=leaf, max_features="sqrt",
            class_weight="balanced_subsample", random_state=20260802, n_jobs=-1,
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dense-grid-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x: list[np.ndarray] = []
    train_y: list[str] = []
    validation: dict[str, list[tuple[np.ndarray, str]]] = {c.name: [] for c in CONDITIONS}
    feature_names = None

    rows = read_manifest(args.manifest)
    development = [row for row in rows if row.split in {"train", "validation"}]
    for index, row in enumerate(development, start=1):
        original = read_canonical_pl4(resolve_pl4_path(row, args.manifest))
        for condition in CONDITIONS:
            result = extract_features(perturb(original, row.run_id, condition))
            if feature_names is None:
                feature_names = result.names
            elif result.names != feature_names:
                raise RuntimeError("Ordem de atributos inconsistente (manifesto v9).")
            if row.split == "train":
                train_x.append(result.values)
                train_y.append(row.fault_class)
            else:
                validation[condition.name].append((result.values, row.fault_class))
        if index % 100 == 0:
            print(f"[v9] {index}/{len(development)} casos-base extraidos", flush=True)

    assert feature_names is not None
    dense_grid_count = 0
    for npz_path in sorted(args.dense_grid_dir.glob("*.npz")):
        data = np.load(npz_path, allow_pickle=True)
        names = tuple(data["names"])
        if names != tuple(feature_names):
            raise RuntimeError(f"Ordem de atributos inconsistente ({npz_path.name}).")
        x = data["X"]
        meta = data["meta"]
        y = meta[:, 1]
        train_x.append(x)
        train_y.extend(y.tolist())
        dense_grid_count += len(y)
        print(f"[v10] {npz_path.stem}: +{len(y)} casos (condicao ideal apenas)")

    x_train = np.vstack(train_x)
    y_train = np.asarray(train_y)
    print(f"\nTotal treino: {len(y_train)} amostras ({len(y_train) - dense_grid_count} com augmentacao "
          f"de ruido do manifesto v9 + {dense_grid_count} da malha densa v10, condicao ideal)")

    candidate_reports = {}
    fitted = {}
    for name, model in _candidate_models():
        model.fit(x_train, y_train)
        scores = {}
        for condition in CONDITIONS:
            items = validation[condition.name]
            x = np.vstack([item[0] for item in items])
            y = np.asarray([item[1] for item in items])
            prediction = model.predict(x)
            scores[condition.name] = {
                "accuracy": float(accuracy_score(y, prediction)),
                "macro_f1": float(f1_score(y, prediction, average="macro")),
            }
        robust_names = ("snr_40db", "snr_30db", "combined_snr30_gain1_sync1")
        worst_robust_f1 = min(scores[key]["macro_f1"] for key in robust_names)
        average_robust_f1 = float(np.mean([scores[key]["macro_f1"] for key in robust_names]))
        candidate_reports[name] = {
            "worst_robust_macro_f1": worst_robust_f1,
            "average_robust_macro_f1": average_robust_f1,
            "conditions": scores,
        }
        fitted[name] = model
        print(f"  {name}: worst_robust_macro_f1={worst_robust_f1:.4f}")

    selected_name = max(
        candidate_reports,
        key=lambda name: (
            candidate_reports[name]["worst_robust_macro_f1"],
            candidate_reports[name]["average_robust_macro_f1"],
        ),
    )
    artifact = {
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "classifier": fitted[selected_name],
        "selected_model": selected_name,
        "training_base_runs": len(development) + dense_grid_count,
        "training_augmented_runs": len(y_train),
        "training_conditions": tuple(c.name for c in CONDITIONS),
        "trained_splits": ("train",),
        "test_unseen_used": False,
        "dense_grid_high_z_runs": dense_grid_count,
    }
    joblib.dump(artifact, args.output_dir / "robust_classifier.joblib")
    report = {
        "feature_version": FEATURE_VERSION,
        "training_base_runs": len(development) + dense_grid_count,
        "training_augmented_runs": len(y_train),
        "dense_grid_high_z_runs": dense_grid_count,
        "selected_model": selected_name,
        "selected_validation": candidate_reports[selected_name],
        "candidate_summary": {
            name: {
                "worst_robust_macro_f1": item["worst_robust_macro_f1"],
                "average_robust_macro_f1": item["average_robust_macro_f1"],
            }
            for name, item in candidate_reports.items()
        },
    }
    (args.output_dir / "validation_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Retreina o classificador (mesma base de dados do v15: manifesto original
+ malhas densas de alta impedancia) testando diferentes valores de
max_features no RandomForest, para verificar se aumentar o numero de
atributos considerados por divisao evita que a floresta abandone features
fisicamente diagnosticas (ex: sequence_ratio de sequencia zero) em favor de
features genericas quando o volume de dados de alta impedancia cresce."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from feature_extraction import FEATURE_VERSION, extract_features
from manifest import read_manifest, resolve_pl4_path
from robustness_evaluation import CONDITIONS, perturb
from signal_io import read_canonical_pl4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dense-grid-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-features", required=True,
                         help="'sqrt', 'log2', 'all', ou uma fracao ex 0.3, 0.5")
    parser.add_argument("--min-samples-leaf", type=int, default=1)
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_features = args.max_features
    if max_features == "all":
        max_features = None
    elif max_features not in ("sqrt", "log2"):
        max_features = float(max_features)

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
                raise RuntimeError("Ordem de atributos inconsistente (manifesto).")
            if row.split == "train":
                train_x.append(result.values)
                train_y.append(row.fault_class)
            else:
                validation[condition.name].append((result.values, row.fault_class))
        if index % 200 == 0:
            print(f"[manifesto] {index}/{len(development)} casos-base extraidos", flush=True)

    assert feature_names is not None
    dense_grid_count = 0
    for grid_dir in args.dense_grid_dir:
        for npz_path in sorted(grid_dir.glob("*.npz")):
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

    x_train = np.vstack(train_x)
    y_train = np.asarray(train_y)
    print(f"Total treino: {len(y_train)} amostras ({dense_grid_count} da malha densa)")
    print(f"max_features={args.max_features} min_samples_leaf={args.min_samples_leaf}")

    model = RandomForestClassifier(
        n_estimators=600, min_samples_leaf=args.min_samples_leaf, max_features=max_features,
        class_weight="balanced_subsample", random_state=20260802, n_jobs=-1,
    )
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
    print(f"worst_robust_macro_f1={worst_robust_f1:.4f}")

    names_arr = list(feature_names)
    imp = model.feature_importances_
    order = np.argsort(imp)[::-1][:12]
    print("Top12 feature importances:")
    for i in order:
        print(f"  {names_arr[i]:45s} {imp[i]:.4f}")
    zero_seq_rank = sorted(range(len(imp)), key=lambda i: -imp[i])
    for target in ("sequence_ratio__BEA_V_zero", "sequence_ratio__PDT_V_zero", "sequence_ratio__BEA_I_zero"):
        idx = names_arr.index(target)
        rank = zero_seq_rank.index(idx) + 1
        print(f"  rank de {target}: #{rank} (importancia {imp[idx]:.4f})")

    artifact = {
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "classifier": model,
        "selected_model": f"random_forest_maxfeat_{args.max_features}_leaf{args.min_samples_leaf}",
        "training_augmented_runs": len(y_train),
        "dense_grid_high_z_runs": dense_grid_count,
        "trained_splits": ("train",),
        "test_unseen_used": False,
    }
    joblib.dump(artifact, args.output_dir / "robust_classifier.joblib")
    report = {"worst_robust_macro_f1": worst_robust_f1, "conditions": scores,
              "max_features": args.max_features, "min_samples_leaf": args.min_samples_leaf}
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

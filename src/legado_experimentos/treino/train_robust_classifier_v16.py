"""Retreina o classificador combinando o manifesto original (augmentacao de
ruido completa, via pl4) com malhas de features pre-extraidas de alta
impedancia (v10, v11, ...). Diferente do v14/v15, aplica pesos por amostra
para equilibrar a contribuicao de cada fonte de dados POR CLASSE: sem isso,
o volume muito maior de amostras da malha densa (soh condicao ideal) acaba
dominando o treino e degrada classes que ja funcionavam bem em baixa
impedancia (ex: AB, BC caíram de ~100% para ~65-68% em Rfault<=100ohm)."""

from __future__ import annotations

import argparse
import json
from collections import Counter
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
    parser.add_argument("--dense-grid-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dense-grid-weight-fraction", type=float, default=0.5,
                         help="Fracao do peso total (por classe) atribuida a malha densa "
                              "combinada; o restante vai para o manifesto original. "
                              "0.5 = peso igual entre as duas fontes.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_x: list[np.ndarray] = []
    train_y: list[str] = []
    train_source: list[str] = []
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
                train_source.append("manifest")
            else:
                validation[condition.name].append((result.values, row.fault_class))
        if index % 100 == 0:
            print(f"[manifesto] {index}/{len(development)} casos-base extraidos", flush=True)

    assert feature_names is not None
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
            train_source.extend(["dense_grid"] * len(y))
            print(f"[{grid_dir.name}] {npz_path.stem}: +{len(y)} casos (condicao ideal apenas)")

    x_train = np.vstack(train_x)
    y_train = np.asarray(train_y)
    source_train = np.asarray(train_source)

    # Pesos: para cada classe, a soma dos pesos das amostras "manifest" deve
    # igualar (1 - fracao) do total, e a soma das "dense_grid" deve igualar
    # a fracao configurada -- independente de quantas amostras cada fonte
    # realmente tem. Isso impede que o volume maior da malha densa domine o
    # treino e apague o que o manifesto original ja ensinava bem.
    frac_dense = args.dense_grid_weight_fraction
    frac_manifest = 1.0 - frac_dense
    sample_weight = np.ones(len(y_train), dtype=float)
    for cls in sorted(set(y_train)):
        cls_mask = y_train == cls
        for source, frac in (("manifest", frac_manifest), ("dense_grid", frac_dense)):
            mask = cls_mask & (source_train == source)
            n = int(mask.sum())
            if n > 0:
                sample_weight[mask] = frac / n

    print(f"\nTotal treino: {len(y_train)} amostras "
          f"({int((source_train == 'manifest').sum())} do manifesto original (com ruido), "
          f"{int((source_train == 'dense_grid').sum())} das malhas densas (ideal apenas))")
    print(f"Pesos por amostra ajustados para {frac_manifest:.0%} manifesto / {frac_dense:.0%} malha densa, por classe.")
    print("\nDistribuicao bruta por classe/fonte:")
    for cls in sorted(set(y_train)):
        counts = Counter(source_train[y_train == cls])
        print(f"  {cls}: {dict(counts)}")

    candidate_reports = {}
    fitted = {}
    for name, model in _candidate_models():
        model.fit(x_train, y_train, sample_weight=sample_weight)
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
        "training_base_runs": len(y_train),
        "training_augmented_runs": len(y_train),
        "training_conditions": tuple(c.name for c in CONDITIONS),
        "trained_splits": ("train",),
        "test_unseen_used": False,
        "dense_grid_weight_fraction": frac_dense,
        "dense_grid_dirs": [str(d) for d in args.dense_grid_dir],
    }
    joblib.dump(artifact, args.output_dir / "robust_classifier.joblib")
    report = {
        "feature_version": FEATURE_VERSION,
        "training_augmented_runs": len(y_train),
        "dense_grid_weight_fraction": frac_dense,
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

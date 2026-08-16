"""Treina um RandomForest em duas etapas via warm_start: a primeira metade
das arvores ve SOMENTE o manifesto original (baixa impedancia, com
augmentacao de ruido completa) e fica congelada; a segunda metade e
adicionada depois vendo o conjunto combinado (manifesto + malhas densas de
alta impedancia). O objetivo e que a floresta final tenha "arvores
especialistas" em baixa-Z (que preservam o comportamento do v12, que
tinha 100% nesses cenarios) somadas a arvores que generalizam para alta-Z,
sem que as arvores especialistas sejam apagadas/sobrescritas pelo volume
maior de dados novos -- o que aconteceu nas tentativas anteriores (v15,
v16) porque todas as arvores viam sempre o dataset combinado inteiro."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from cached_manifest import load_manifest_cache
from feature_extraction import FEATURE_VERSION, extract_features
from manifest import read_manifest, resolve_pl4_path
from robustness_evaluation import CONDITIONS, perturb
from signal_io import read_canonical_pl4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dense-grid-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-specialist-trees", type=int, default=300)
    parser.add_argument("--n-general-trees", type=int, default=300)
    parser.add_argument("--max-features", default="0.3")
    parser.add_argument("--manifest-cache", type=Path, default=None,
                         help="Cache .npz de precompute_manifest_features.py (evita reextrair).")
    parser.add_argument("--boost-classes", default="",
                         help="Classes separadas por virgula que recebem peso extra na etapa 2 "
                              "(ex: AB,BC), sem alterar o numero de arvores gerais.")
    parser.add_argument("--boost-weight", type=float, default=3.0,
                         help="Multiplicador de peso aplicado as amostras do manifesto original "
                              "(baixa-Z) das classes em --boost-classes, na etapa 2.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_features = args.max_features
    if max_features == "all":
        max_features = None
    elif max_features not in ("sqrt", "log2"):
        max_features = float(max_features)

    if args.manifest_cache is not None and args.manifest_cache.is_file():
        print(f"Carregando features do cache: {args.manifest_cache}")
        x_manifest, manifest_y, validation, feature_names = load_manifest_cache(args.manifest_cache)
        y_manifest = np.asarray(manifest_y)
    else:
        manifest_x: list[np.ndarray] = []
        manifest_y = []
        validation = {c.name: [] for c in CONDITIONS}
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
                    manifest_x.append(result.values)
                    manifest_y.append(row.fault_class)
                else:
                    validation[condition.name].append((result.values, row.fault_class))
            if index % 200 == 0:
                print(f"[manifesto] {index}/{len(development)} casos-base extraidos", flush=True)

        assert feature_names is not None
        x_manifest = np.vstack(manifest_x)
        y_manifest = np.asarray(manifest_y)

    dense_x = [x_manifest]
    dense_y = [y_manifest]
    dense_grid_count = 0
    for grid_dir in args.dense_grid_dir:
        for npz_path in sorted(grid_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            names = tuple(data["names"])
            if names != tuple(feature_names):
                raise RuntimeError(f"Ordem de atributos inconsistente ({npz_path.name}).")
            dense_x.append(data["X"])
            y = data["meta"][:, 1]
            dense_y.append(y)
            dense_grid_count += len(y)
    x_combined = np.vstack(dense_x)
    y_combined = np.concatenate(dense_y)

    print(f"Manifesto (baixa-Z): {len(y_manifest)} amostras")
    print(f"Combinado (baixa+alta-Z): {len(y_combined)} amostras ({dense_grid_count} da malha densa)")
    print(f"max_features={args.max_features}, {args.n_specialist_trees} arvores especialistas + "
          f"{args.n_general_trees} arvores gerais")

    model = RandomForestClassifier(
        n_estimators=args.n_specialist_trees, max_features=max_features,
        class_weight="balanced_subsample", random_state=20260802, n_jobs=-1,
        warm_start=True,
    )
    print("Etapa 1: treinando arvores especialistas em baixa-Z (manifesto original)...")
    model.fit(x_manifest, y_manifest)

    model.n_estimators = args.n_specialist_trees + args.n_general_trees
    print("Etapa 2: adicionando arvores gerais (dataset combinado)...")
    boost_classes = {c.strip() for c in args.boost_classes.split(",") if c.strip()}
    if boost_classes:
        n_manifest = len(y_manifest)
        sample_weight = np.ones(len(y_combined), dtype=float)
        is_manifest_row = np.zeros(len(y_combined), dtype=bool)
        is_manifest_row[:n_manifest] = True
        for cls in boost_classes:
            mask = is_manifest_row & (y_combined == cls)
            sample_weight[mask] = args.boost_weight
            print(f"  peso {args.boost_weight}x para {mask.sum()} amostras de {cls} (manifesto, baixa-Z)")
        model.fit(x_combined, y_combined, sample_weight=sample_weight)
    else:
        model.fit(x_combined, y_combined)

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

    artifact = {
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "classifier": model,
        "selected_model": f"random_forest_specialist{args.n_specialist_trees}_general{args.n_general_trees}",
        "training_augmented_runs": len(y_manifest) + len(y_combined),
        "dense_grid_high_z_runs": dense_grid_count,
        "trained_splits": ("train",),
        "test_unseen_used": False,
    }
    joblib.dump(artifact, args.output_dir / "robust_classifier.joblib")
    report = {"worst_robust_macro_f1": worst_robust_f1, "conditions": scores}
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

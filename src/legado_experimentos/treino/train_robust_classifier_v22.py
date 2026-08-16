"""Igual ao train_robust_classifier_v18.py (duas etapas via warm_start,
arvores especialistas em baixa-Z + arvores gerais no dataset combinado),
mas adiciona a feature phase_asymmetry__{PDT,BEA}_I (Fase 3 do plano de
fechar a lacuna ABG/BCG em alta impedancia) a TODOS os datasets antes do
treino -- manifesto, malhas densas v10/v11, fronteira v19/v21 -- via
feature_extraction.add_phase_asymmetry(), que deriva a nova feature dos
rms_ratio ja armazenados, sem precisar resimular nenhum caso no ATP."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from cached_manifest import load_manifest_cache
from feature_extraction import FEATURE_VERSION, add_phase_asymmetry, extract_features
from manifest import read_manifest, resolve_pl4_path
from robustness_evaluation import CONDITIONS, perturb
from signal_io import read_canonical_pl4


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--dense-grid-dir", type=Path, action="append", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-specialist-trees", type=int, default=400)
    parser.add_argument("--n-general-trees", type=int, default=300)
    parser.add_argument("--max-features", default="0.3")
    parser.add_argument("--manifest-cache", type=Path, default=None,
                         help="Cache .npz de precompute_manifest_features.py (evita reextrair).")
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

    original_feature_names = tuple(feature_names)
    feature_names = original_feature_names
    x_manifest, feature_names = add_phase_asymmetry(x_manifest, feature_names)
    # validation vem do cache com o feature_names ANTIGO (61 atributos) se
    # veio de load_manifest_cache, ou ja com phase_asymmetry se veio de
    # extract_features ao vivo (ramo sem --manifest-cache) -- usa sempre
    # original_feature_names (nunca o ja-aumentado) para localizar as
    # colunas de origem, senao a fatia fica errada quando a insercao nao
    # e' no fim da lista.
    fixed_validation = {}
    for name, items in validation.items():
        fixed_items = []
        for values, cls in items:
            if len(values) != len(feature_names):
                values, _ = add_phase_asymmetry(values, original_feature_names)
            fixed_items.append((values, cls))
        fixed_validation[name] = fixed_items
    validation = fixed_validation

    dense_x = [x_manifest]
    dense_y = [y_manifest]
    dense_grid_count = 0
    for grid_dir in args.dense_grid_dir:
        for npz_path in sorted(grid_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            names = tuple(data["names"])
            x, names = add_phase_asymmetry(data["X"], names)
            if names != feature_names:
                raise RuntimeError(f"Ordem de atributos inconsistente ({npz_path.name}).")
            dense_x.append(x)
            y = data["meta"][:, 1]
            dense_y.append(y)
            dense_grid_count += len(y)
    x_combined = np.vstack(dense_x)
    y_combined = np.concatenate(dense_y)

    print(f"Manifesto (baixa-Z): {len(y_manifest)} amostras")
    print(f"Combinado (baixa+alta-Z): {len(y_combined)} amostras ({dense_grid_count} da malha densa)")
    print(f"Atributos: {len(feature_names)} (inclui phase_asymmetry__PDT_I e __BEA_I)")
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
        "selected_model": f"random_forest_specialist{args.n_specialist_trees}_general{args.n_general_trees}_phaseasym",
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

"""Igual ao train_robust_classifier_v22.py (duas etapas via warm_start +
feature phase_asymmetry), mas expurga da base de treino todo o dado de
faltas puramente fase-fase (AB/BC/CA/ABC) com Rfault > 0.5 ohm.

Confirmado com o orientador (2026-08-14) que Rfault so tem caminho fisico
ate faltas com terra -- faltas fase-fase sao sempre francas na pratica.
O gerador Python tinha uma topologia errada (resistor em serie entre as
fases) que nao corresponde ao template de referencia do ATPDraw (chave
liga as fases diretamente, sem passar pelo Rfault). Isso significa que
1.314-1.490 casos por classe nas malhas densas anteriores (v10/v11/v19/v21/v23)
representam um cenario que nao ocorre na pratica e precisam ser excluidos
do treino -- ver fault_case_generator.py (topologia corrigida) e
gen_phase_phase_bolted_v24.py (malha nova, Rfault=0 fixo, so
distancia x angulo x remoto como variaveis livres)."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

from fault_case_generator import PHASE_PHASE_FAULTS
from feature_extraction import FEATURE_VERSION, add_phase_asymmetry, extract_features
from manifest import read_manifest, resolve_pl4_path
from robustness_evaluation import CONDITIONS, perturb
from signal_io import read_canonical_pl4

RFAULT_FRANCA_TOLERANCE_OHM = 0.5


def _load_manifest_cache_with_rfault(cache_path: Path, manifest_path: Path):
    data = np.load(cache_path, allow_pickle=True)
    x_all, y_all = data["X"], data["y"]
    split_all, condition_all, run_id_all = data["split"], data["condition"], data["run_id"]
    names = tuple(data["names"])

    rfault_by_run_id = {row.run_id: row.rfault_ohm for row in read_manifest(manifest_path)}

    def is_contaminated(run_id: str, fault_class: str) -> bool:
        return fault_class in PHASE_PHASE_FAULTS and rfault_by_run_id[run_id] > RFAULT_FRANCA_TOLERANCE_OHM

    train_mask = split_all == "train"
    train_keep = np.array([
        not is_contaminated(rid, cls) for rid, cls in zip(run_id_all[train_mask], y_all[train_mask])
    ])
    train_x = x_all[train_mask][train_keep]
    train_y = y_all[train_mask][train_keep].tolist()
    n_dropped_train = int((~train_keep).sum())

    validation: dict[str, list[tuple[np.ndarray, str]]] = {}
    val_mask = split_all == "validation"
    n_dropped_val = 0
    for cond in sorted(set(condition_all[val_mask].tolist())):
        cond_mask = val_mask & (condition_all == cond)
        keep = np.array([
            not is_contaminated(rid, cls) for rid, cls in zip(run_id_all[cond_mask], y_all[cond_mask])
        ])
        n_dropped_val += int((~keep).sum())
        validation[cond] = list(zip(list(x_all[cond_mask][keep]), y_all[cond_mask][keep].tolist()))

    print(f"Expurgo fase-fase (Rfault>{RFAULT_FRANCA_TOLERANCE_OHM}ohm): "
          f"{n_dropped_train} de treino, {n_dropped_val} de validacao (por condicao) removidos.")
    return train_x, train_y, validation, names


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
    parser.add_argument("--boost-classes", default="",
                         help="Classes separadas por virgula que recebem peso extra na "
                              "etapa 2 (ex: AB,BC), sem alterar o numero de arvores gerais.")
    parser.add_argument("--boost-weight", type=float, default=3.0,
                         help="Multiplicador de peso para as amostras das classes em "
                              "--boost-classes, na etapa 2.")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    max_features = args.max_features
    if max_features == "all":
        max_features = None
    elif max_features not in ("sqrt", "log2"):
        max_features = float(max_features)

    if args.manifest_cache is not None and args.manifest_cache.is_file():
        print(f"Carregando features do cache: {args.manifest_cache}")
        x_manifest, manifest_y, validation, feature_names = _load_manifest_cache_with_rfault(
            args.manifest_cache, args.manifest
        )
        y_manifest = np.asarray(manifest_y)
    else:
        manifest_x: list[np.ndarray] = []
        manifest_y = []
        validation = {c.name: [] for c in CONDITIONS}
        feature_names = None

        rows = read_manifest(args.manifest)
        development = [
            row for row in rows
            if row.split in {"train", "validation"}
            and not (row.fault_class in PHASE_PHASE_FAULTS and row.rfault_ohm > RFAULT_FRANCA_TOLERANCE_OHM)
        ]
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
    dense_grid_dropped = 0
    for grid_dir in args.dense_grid_dir:
        for npz_path in sorted(grid_dir.glob("*.npz")):
            data = np.load(npz_path, allow_pickle=True)
            names = tuple(data["names"])
            x, names = add_phase_asymmetry(data["X"], names)
            if names != feature_names:
                raise RuntimeError(f"Ordem de atributos inconsistente ({npz_path.name}).")
            meta = data["meta"]
            classes = meta[:, 1]
            rfaults = meta[:, 3].astype(float)
            contaminated = np.array([
                cls in PHASE_PHASE_FAULTS and rf > RFAULT_FRANCA_TOLERANCE_OHM
                for cls, rf in zip(classes, rfaults)
            ])
            if contaminated.any():
                dense_grid_dropped += int(contaminated.sum())
                print(f"  {npz_path.name}: descartando {int(contaminated.sum())}/{len(classes)} "
                      f"casos fase-fase com Rfault>{RFAULT_FRANCA_TOLERANCE_OHM}ohm")
            keep = ~contaminated
            dense_x.append(x[keep])
            dense_y.append(classes[keep])
            dense_grid_count += int(keep.sum())
    x_combined = np.vstack(dense_x)
    y_combined = np.concatenate(dense_y)

    print(f"Manifesto (baixa-Z): {len(y_manifest)} amostras")
    print(f"Combinado (baixa+alta-Z): {len(y_combined)} amostras ({dense_grid_count} da malha densa, "
          f"{dense_grid_dropped} descartados por contaminacao fase-fase)")
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
    boost_classes = {c.strip() for c in args.boost_classes.split(",") if c.strip()}
    if boost_classes:
        sample_weight = np.ones(len(y_combined), dtype=float)
        for cls in boost_classes:
            mask = y_combined == cls
            sample_weight[mask] = args.boost_weight
            print(f"  peso {args.boost_weight}x para {mask.sum()} amostras de {cls}")
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

    # Especialistas binarios para os 3 pares francas-vs-aterradas que o
    # modelo geral (10 classes) confunde: um pequeno classificador
    # focado SO nesse par consegue muito mais acuracia do que o modelo
    # geral no mesmo par (79-98% vs 56-89%), porque nao precisa dividir
    # capacidade com as outras 8 classes. Usado como segunda opiniao em
    # infer_fault.py: quando o modelo geral prevê uma das 6 classes
    # envolvidas, o especialista correspondente arbitra a decisao final.
    print("\nTreinando especialistas binarios (franca vs aterrada)...")
    specialist_pairs = (("AB", "ABG"), ("BC", "BCG"), ("CA", "CAG"))
    specialists = {}
    for cls_a, cls_b in specialist_pairs:
        mask = (y_combined == cls_a) | (y_combined == cls_b)
        x_pair = x_combined[mask]
        y_pair = y_combined[mask]
        spec = RandomForestClassifier(
            n_estimators=300, max_features=0.3, class_weight="balanced_subsample",
            random_state=20260814, n_jobs=-1,
        )
        spec.fit(x_pair, y_pair)
        specialists[f"{cls_a}_{cls_b}"] = spec
        print(f"  especialista {cls_a}/{cls_b}: {mask.sum()} amostras de treino")

    artifact = {
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "classifier": model,
        "specialists": specialists,
        "selected_model": f"random_forest_specialist{args.n_specialist_trees}_general{args.n_general_trees}_phaseasym_bolted",
        "training_augmented_runs": len(y_manifest) + len(y_combined),
        "dense_grid_high_z_runs": dense_grid_count,
        "phase_phase_contamination_dropped": dense_grid_dropped,
        "trained_splits": ("train",),
        "test_unseen_used": False,
    }
    joblib.dump(artifact, args.output_dir / "robust_classifier.joblib")
    report = {"worst_robust_macro_f1": worst_robust_f1, "conditions": scores}
    (args.output_dir / "validation_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

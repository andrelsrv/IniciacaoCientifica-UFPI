"""Classificador hierarquico: (1) binario terra/sem-terra, depois (2a) qual
fase(s) dentro de "sem terra" (AB/BC/CA/ABC) e (2b) qual fase(s) dentro de
"com terra" (AG/BG/CG/ABG/BCG/CAG). Objetivo: isolar a decisao "tem terra",
que tem sinal fisico forte (corrente de sequencia zero), da decisao de fase,
reduzindo a confusao AB<->ABG etc. observada no classificador plano (v3).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score, f1_score

from feature_extraction import FEATURE_VERSION, extract_features
from manifest import read_manifest, resolve_pl4_path
from robustness_evaluation import CONDITIONS, perturb
from signal_io import read_canonical_pl4

GROUNDED = {"AG", "BG", "CG", "ABG", "BCG", "CAG"}
UNGROUNDED = {"AB", "BC", "CA", "ABC"}

MANIFEST = Path(r"C:\RESULTPESQUISA\campaign_v4_600km\manifest_combined_600km.csv")
OUTPUT_DIR = Path(r"C:\RESULTPESQUISA\campaign_v4_600km\classifier_hierarchical_v1")


def _make_model() -> ExtraTreesClassifier:
    return ExtraTreesClassifier(
        n_estimators=600, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced", random_state=20260808, n_jobs=-1,
    )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(MANIFEST)
    development = [row for row in rows if row.split in {"train", "validation"}]

    train_x, train_ground, train_class = [], [], []
    val_by_condition: dict[str, list[tuple[np.ndarray, str]]] = {c.name: [] for c in CONDITIONS}
    feature_names = None

    for index, row in enumerate(development, start=1):
        original = read_canonical_pl4(resolve_pl4_path(row, MANIFEST))
        for condition in CONDITIONS:
            result = extract_features(perturb(original, row.run_id, condition))
            if feature_names is None:
                feature_names = result.names
            if row.split == "train":
                train_x.append(result.values)
                train_ground.append(row.fault_class in GROUNDED)
                train_class.append(row.fault_class)
            else:
                val_by_condition[condition.name].append((result.values, row.fault_class))
        if index % 50 == 0:
            print(f"[{index}/{len(development)}] casos-base extraidos", flush=True)

    x_train = np.vstack(train_x)
    ground_train = np.asarray(train_ground)
    class_train = np.asarray(train_class)

    ground_model = _make_model()
    ground_model.fit(x_train, ground_train)

    grounded_model = _make_model()
    grounded_model.fit(x_train[ground_train], class_train[ground_train])

    ungrounded_model = _make_model()
    ungrounded_model.fit(x_train[~ground_train], class_train[~ground_train])

    def predict(x: np.ndarray) -> str:
        is_grounded = ground_model.predict(x.reshape(1, -1))[0]
        sub_model = grounded_model if is_grounded else ungrounded_model
        return str(sub_model.predict(x.reshape(1, -1))[0])

    scores = {}
    for condition in CONDITIONS:
        items = val_by_condition[condition.name]
        y_true = [label for _, label in items]
        y_pred = [predict(x) for x, _ in items]
        scores[condition.name] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "macro_f1": float(f1_score(y_true, y_pred, average="macro")),
        }
        ground_true = [label in GROUNDED for label in y_true]
        ground_pred = [ground_model.predict(x.reshape(1, -1))[0] for x, _ in items]
        scores[condition.name]["ground_binary_accuracy"] = float(accuracy_score(ground_true, ground_pred))

    artifact = {
        "feature_version": FEATURE_VERSION,
        "feature_names": feature_names,
        "ground_model": ground_model,
        "grounded_model": grounded_model,
        "ungrounded_model": ungrounded_model,
        "training_base_runs": len({r.run_id for r in development if r.split == "train"}),
    }
    joblib.dump(artifact, OUTPUT_DIR / "hierarchical_classifier.joblib")
    (OUTPUT_DIR / "validation_report.json").write_text(
        json.dumps(scores, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(json.dumps(scores, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()

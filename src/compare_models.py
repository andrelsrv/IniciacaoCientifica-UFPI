"""Compara ExtraTrees/RandomForest (ja usados) com GradientBoosting e
HistGradientBoosting (scikit-learn, sem dependencia externa) no mesmo
conjunto de treino (1146 casos, campanha v6) e no mesmo conjunto de teste
independente (300 casos, 5 lotes)."""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import (
    ExtraTreesClassifier, GradientBoostingClassifier,
    HistGradientBoostingClassifier, RandomForestClassifier,
)
from sklearn.metrics import accuracy_score, f1_score

from feature_extraction import extract_features
from manifest import read_manifest, resolve_pl4_path
from signal_io import read_canonical_pl4

MANIFEST = Path(r"C:\RESULTPESQUISA\campaign_v6_ab_focus\manifest_combined_v6.csv")
TEST_BATCHES = [
    (Path(r"C:\RESULTPESQUISA\quick_check"), "quick_check_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v2"), "quick_check_v2_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v3_600km"), "quick_check_v3_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v4_final"), "quick_check_v4_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v5_final"), "quick_check_v5_report.json"),
]


def load_independent_test_set() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for root, fname in TEST_BATCHES:
        report = json.loads((root / fname).read_text(encoding="utf-8"))
        for r in report["results"]:
            run_id = r["run_id"]
            pl4_path = root / run_id / f"{run_id}.pl4"
            if not pl4_path.is_file():
                continue
            signals = read_canonical_pl4(pl4_path)
            f = extract_features(signals)
            xs.append(f.values)
            ys.append(r["true_class"])
    return np.vstack(xs), np.asarray(ys)


def main() -> None:
    rows = read_manifest(MANIFEST)
    train_rows = [r for r in rows if r.split == "train"]
    print("Extraindo atributos de", len(train_rows), "casos de treino (ideal)...")
    xs, ys = [], []
    for index, row in enumerate(train_rows, start=1):
        signals = read_canonical_pl4(resolve_pl4_path(row, MANIFEST))
        f = extract_features(signals)
        xs.append(f.values)
        ys.append(row.fault_class)
        if index % 100 == 0:
            print(f"  [{index}/{len(train_rows)}]", flush=True)
    x_train = np.vstack(xs)
    y_train = np.asarray(ys)

    print("Carregando conjunto de teste independente...")
    x_test, y_test = load_independent_test_set()
    print("Teste:", len(y_test), "casos\n")

    candidates = {
        "ExtraTrees (atual)": ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced", random_state=20260802, n_jobs=-1,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=600, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced_subsample", random_state=20260802, n_jobs=-1,
        ),
        "GradientBoosting": GradientBoostingClassifier(
            n_estimators=300, max_depth=3, learning_rate=0.1, random_state=20260802,
        ),
        "HistGradientBoosting": HistGradientBoostingClassifier(
            max_iter=300, random_state=20260802,
        ),
    }

    results = []
    for name, model in candidates.items():
        t0 = time.time()
        model.fit(x_train, y_train)
        train_time = time.time() - t0
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average="macro")
        results.append({"model": name, "accuracy": acc, "macro_f1": f1, "train_time_s": train_time})
        print(f"{name:22s} acuracia={acc:.1%}  macro-F1={f1:.3f}  tempo_treino={train_time:.1f}s")

    Path("model_comparison_result.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

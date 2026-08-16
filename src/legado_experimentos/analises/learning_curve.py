"""Curva de aprendizado: treina o classificador com quantidades crescentes de
casos de treino (subamostragem estratificada por classe) e mede a acurácia
nos 160 casos independentes já gerados (quick_check + quick_check_v2 +
quick_check_v3_600km). Objetivo: estimar quantos casos seriam necessários
para chegar perto de 100%, ou mostrar que o ganho ja estagnou.
"""

from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score

from feature_extraction import extract_features
from manifest import read_manifest, resolve_pl4_path
from signal_io import read_canonical_pl4

MANIFEST = Path(r"C:\RESULTPESQUISA\campaign_v4_600km\manifest_combined_600km.csv")
SIZES = [60, 120, 200, 300, 446]
SEED = 20260808900

TEST_BATCHES = [
    (Path(r"C:\RESULTPESQUISA\quick_check"), "quick_check_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v2"), "quick_check_v2_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v3_600km"), "quick_check_v3_report.json"),
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

    print("Extraindo atributos (ideal) para todos os", len(train_rows), "casos de treino...")
    by_class: dict[str, list[tuple[np.ndarray, str, str]]] = defaultdict(list)
    for index, row in enumerate(train_rows, start=1):
        signals = read_canonical_pl4(resolve_pl4_path(row, MANIFEST))
        f = extract_features(signals)
        by_class[row.fault_class].append((f.values, row.fault_class, row.run_id))
        if index % 50 == 0:
            print(f"  [{index}/{len(train_rows)}]", flush=True)

    print("Carregando conjunto de teste independente (160 casos)...")
    x_test, y_test = load_independent_test_set()
    print("Teste independente:", len(y_test), "casos")

    rng = random.Random(SEED)
    results = []
    for size in SIZES:
        per_class = max(1, size // len(by_class))
        subset = []
        for cls, items in by_class.items():
            picked = rng.sample(items, min(per_class, len(items)))
            subset.extend(picked)
        x_train = np.vstack([v for v, _, _ in subset])
        y_train = np.asarray([c for _, c, _ in subset])

        model = ExtraTreesClassifier(
            n_estimators=600, min_samples_leaf=1, max_features="sqrt",
            class_weight="balanced", random_state=20260808, n_jobs=-1,
        )
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = float(accuracy_score(y_test, y_pred))
        results.append({"train_size": len(subset), "accuracy": acc})
        print(f"treino={len(subset):4d} casos  ->  acuracia no teste independente = {acc:.1%}")

    Path("learning_curve_result.json").write_text(
        json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8"
    )


if __name__ == "__main__":
    main()

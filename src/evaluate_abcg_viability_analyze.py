"""Reaproveita os 60 .pl4 ja gerados por evaluate_abcg_viability_full.py e
faz a validacao cruzada (sem re-rodar o ATP)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from signal_io import read_canonical_pl4
from feature_extraction import extract_features

OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\abcg_feasibility_full")
N_PAIRS = 30
SEED = 20260810


def main() -> None:
    x_all, y_all, names = [], [], None
    for index in range(1, N_PAIRS + 1):
        run_id = f"run_1{index:05d}"
        run_dir = OUTPUT_ROOT / run_id
        abc_pl4 = run_dir / f"{run_id}.pl4"
        abcg_pl4 = run_dir / f"{run_id}_configured_abcg.pl4"
        if not abc_pl4.is_file() or not abcg_pl4.is_file():
            print("faltando:", run_id)
            continue
        f_abc = extract_features(read_canonical_pl4(abc_pl4))
        f_abcg = extract_features(read_canonical_pl4(abcg_pl4))
        names = f_abc.names
        x_all.append(f_abc.values)
        y_all.append("ABC")
        x_all.append(f_abcg.values)
        y_all.append("ABCG")

    x = np.vstack(x_all)
    y = np.asarray(y_all)
    print("Amostras:", len(y))

    model = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")
    print(f"Acuracia por fold: {[f'{s:.1%}' for s in scores]}")
    print(f"Acuracia media: {scores.mean():.1%}  desvio: {scores.std():.1%}")

    model.fit(x, y)
    importances = sorted(zip(names, model.feature_importances_), key=lambda t: -t[1])[:10]
    print("\nAtributos mais importantes para distinguir ABC de ABC-G:")
    for name, imp in importances:
        print(f"  {name:40s} {imp:.4f}")


if __name__ == "__main__":
    main()

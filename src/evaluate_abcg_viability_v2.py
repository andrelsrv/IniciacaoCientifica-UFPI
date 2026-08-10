"""Teste de viabilidade CORRIGIDO: ABC vs ABC-G, cada simulacao gerada em
diretorio proprio via generate_simulation (sem reaproveitar pasta/arquivos
temporarios entre execucoes, ao contrario da tentativa anterior que estava
contaminada)."""

from __future__ import annotations

import random
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from simulation_generator import SimulationParameters, generate_simulation
from signal_io import read_canonical_pl4
from feature_extraction import extract_features

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\abcg_feasibility_v2")
N_PAIRS = 15
SEED = 20260810


def main() -> None:
    rng = random.Random(SEED)
    x_all, y_all, names = [], [], None

    for index in range(1, N_PAIRS + 1):
        distance_km = round(rng.uniform(15.0, 600.0), 2)
        remote_km = round(rng.uniform(max(1.0, 100.0 - distance_km), 400.0), 2)
        rfault_ohm = round(rng.uniform(0.01, 100.0), 3)
        angle_deg = round(rng.uniform(0.0, 359.9), 1)
        print(f"[{index}/{N_PAIRS}] dist={distance_km:.1f}km rfault={rfault_ohm:.2f} ang={angle_deg:.1f}", flush=True)

        for fault_class, prefix in (("ABC", "9"), ("ABCG", "8")):
            run_id = f"run_{prefix}{index:05d}"
            params = SimulationParameters(
                run_id=run_id, fault_class=fault_class, distance_km=distance_km,
                remote_length_km=remote_km, rfault_ohm=rfault_ohm,
                incidence_angle_deg=angle_deg, split="train",
            )
            result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
            f = extract_features(read_canonical_pl4(Path(result["file_path"])))
            names = f.names
            x_all.append(f.values)
            y_all.append(fault_class)

    x = np.vstack(x_all)
    y = np.asarray(y_all)
    print("\nAmostras:", len(y))

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
    print("\nAtributos mais importantes:")
    for name, imp in importances:
        print(f"  {name:40s} {imp:.4f}")


if __name__ == "__main__":
    main()

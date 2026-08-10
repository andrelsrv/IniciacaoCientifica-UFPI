"""Varredura de Rfault crescente (100 a 3000 ohm) para achar onde a
classificacao/deteccao realmente degrada, em vez de manter 100 ohm como
limite arbitrario. Usa o classificador v6 (calibrado) ja treinado apenas ate
100 ohm, para ver como ele generaliza (ou nao) alem do que foi treinado."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from simulation_generator import SimulationParameters, generate_simulation
from signal_io import read_canonical_pl4
from feature_extraction import extract_features

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\high_impedance_sweep_500km")
CLASSIFIER = Path(r"C:\RESULTPESQUISA\campaign_v6_ab_focus\classifier_calibrated_v5\calibrated_classifier.joblib")

RFAULT_VALUES = [50, 100, 150, 200, 300, 500, 750, 1000, 1500, 2000, 3000]
FAULT_CLASS = "AG"
DISTANCE_KM = 500.0
REMOTE_KM = 200.0
ANGLE_DEG = 30.0


def main() -> None:
    artifact = joblib.load(CLASSIFIER)
    print(f"{'Rfault(ohm)':>12s} {'classe prevista':>16s} {'confianca':>10s} {'I0_PDT':>8s} {'I0_BEA':>8s}")
    for index, rfault in enumerate(RFAULT_VALUES, start=1):
        run_id = f"run_{index:06d}"
        params = SimulationParameters(
            run_id=run_id, fault_class=FAULT_CLASS, distance_km=DISTANCE_KM,
            remote_length_km=REMOTE_KM, rfault_ohm=float(rfault),
            incidence_angle_deg=ANGLE_DEG, split="train",
        )
        result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
        signals = read_canonical_pl4(Path(result["file_path"]))
        f = extract_features(signals)
        probs = artifact["classifier"].predict_proba(f.values.reshape(1, -1))[0]
        pred = str(artifact["classifier"].classes_[np.argmax(probs)])
        conf = float(np.max(probs))
        i0_pdt = f.values[f.names.index("sequence_ratio__PDT_I_zero")]
        i0_bea = f.values[f.names.index("sequence_ratio__BEA_I_zero")]
        hit = "OK" if pred == FAULT_CLASS else "ERRO"
        print(f"{rfault:12d} {pred:>10s} ({hit})  {conf*100:8.1f}%  {i0_pdt:8.2f} {i0_bea:8.2f}")


if __name__ == "__main__":
    main()

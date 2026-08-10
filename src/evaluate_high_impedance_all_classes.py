"""Confirma a faixa de Rfault ampliada (ate 3000 ohm) para todas as 10
classes, nao so AG, antes de oficializar a faixa no projeto."""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np

from fault_case_generator import FAULT_SWITCHES
from simulation_generator import SimulationParameters, generate_simulation
from signal_io import read_canonical_pl4
from feature_extraction import extract_features

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\high_impedance_all_classes")
CLASSIFIER = Path(r"C:\RESULTPESQUISA\campaign_v6_ab_focus\classifier_calibrated_v5\calibrated_classifier.joblib")

RFAULT_VALUES = [1000.0, 2000.0, 3000.0]
DISTANCE_KM = 300.0
REMOTE_KM = 200.0
ANGLE_DEG = 60.0


def main() -> None:
    artifact = joblib.load(CLASSIFIER)
    print(f"{'classe':6s} {'Rfault':>8s} {'previsto':>10s} {'confianca':>10s}  status")
    index = 0
    hits = 0
    total = 0
    for fault_class in sorted(FAULT_SWITCHES):
        for rfault in RFAULT_VALUES:
            index += 1
            run_id = f"run_{index:06d}"
            params = SimulationParameters(
                run_id=run_id, fault_class=fault_class, distance_km=DISTANCE_KM,
                remote_length_km=REMOTE_KM, rfault_ohm=rfault,
                incidence_angle_deg=ANGLE_DEG, split="train",
            )
            result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
            signals = read_canonical_pl4(Path(result["file_path"]))
            f = extract_features(signals)
            probs = artifact["classifier"].predict_proba(f.values.reshape(1, -1))[0]
            pred = str(artifact["classifier"].classes_[np.argmax(probs)])
            conf = float(np.max(probs))
            hit = pred == fault_class
            hits += hit
            total += 1
            status = "OK" if hit else "ERRO"
            print(f"{fault_class:6s} {rfault:8.0f} {pred:>10s} {conf*100:9.1f}%  {status}")

    print(f"\nAcuracia geral (Rfault 1000-3000 ohm, todas as classes): {hits}/{total} = {hits/total:.1%}")


if __name__ == "__main__":
    main()

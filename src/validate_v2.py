"""Lote de validacao informal maior para o classificador v2, com semente nova
(nao reutiliza parametros do treino original, do augment nem do quick_check
anterior). Nao substitui uma campanha cega formal."""

from __future__ import annotations

import json
import random
from pathlib import Path

from fault_case_generator import FAULT_SWITCHES
from simulation_generator import SimulationParameters, generate_simulation
from feature_extraction import extract_features
from signal_io import read_canonical_pl4
import joblib
import numpy as np

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\quick_check_v2")
CLASSIFIER_V2 = Path(r"C:\RESULTPESQUISA\campaign_v4_augment\classifier_robust_v2\robust_classifier.joblib")

CASES_PER_CLASS = 6
SEED = 20260807099


def _build_cases() -> list[tuple[str, float, float, float, float]]:
    rng = random.Random(SEED)
    cases: list[tuple[str, float, float, float, float]] = []
    for fault_class in sorted(FAULT_SWITCHES):
        for _ in range(CASES_PER_CLASS):
            distance_km = round(rng.uniform(15.0, 450.0), 2)
            remote_length_km = round(rng.uniform(max(1.0, 100.0 - distance_km), 500.0), 2)
            rfault_ohm = round(rng.uniform(0.01, 100.0), 3)
            angle_deg = round(rng.uniform(0.0, 359.9), 1)
            cases.append((fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg))
    return cases


def main() -> None:
    artifact = joblib.load(CLASSIFIER_V2)
    cases = _build_cases()
    results = []
    confusion: dict[str, dict[str, int]] = {c: {} for c in sorted(FAULT_SWITCHES)}

    for index, (fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg) in enumerate(cases, start=1):
        run_id = f"run_7{index:05d}"
        params = SimulationParameters(
            run_id=run_id, fault_class=fault_class, distance_km=distance_km,
            remote_length_km=remote_length_km, rfault_ohm=rfault_ohm,
            incidence_angle_deg=angle_deg, split="test_unseen",
        )
        print(f"[{index}/{len(cases)}] Gerando {run_id} ({fault_class}, {distance_km}km)...")
        sim_result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
        pl4_path = Path(sim_result["file_path"])

        signals = read_canonical_pl4(pl4_path)
        features = extract_features(signals)
        probs = artifact["classifier"].predict_proba(features.values.reshape(1, -1))[0]
        best = int(np.argmax(probs))
        predicted_class = str(artifact["classifier"].classes_[best])
        vote = float(probs[best])
        hit = predicted_class == fault_class

        confusion[fault_class][predicted_class] = confusion[fault_class].get(predicted_class, 0) + 1
        results.append({
            "run_id": run_id, "true_class": fault_class, "predicted_class": predicted_class,
            "hit": hit, "vote_fraction": vote, "distance_km": distance_km,
        })
        print(f"    -> previsto={predicted_class} ({vote:.1%}) real={fault_class} [{'OK' if hit else 'ERRO'}]")

    accuracy = sum(r["hit"] for r in results) / len(results)
    print(f"\nAcuracia v2 (lote novo, {len(results)} casos): {accuracy:.1%}")
    print("\nMatriz de confusao (real -> previsto: contagem):")
    for true_class, preds in confusion.items():
        print(f"  {true_class}: {preds}")

    report_path = OUTPUT_ROOT / "quick_check_v2_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({"accuracy": accuracy, "confusion": confusion, "results": results}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()

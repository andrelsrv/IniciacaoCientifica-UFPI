"""Lote de validacao informal para o classificador+localizador v3, cobrindo a
faixa estendida 15-600km (antes: 15-450km). Semente nova, nao reutiliza
parametros do treino nem dos lotes anteriores. Testa classificacao e
localizacao (maximum_distance_km=600)."""

from __future__ import annotations

import json
import random
from pathlib import Path

import joblib
import numpy as np

from fault_case_generator import FAULT_SWITCHES
from simulation_generator import SimulationParameters, generate_simulation
from feature_extraction import extract_features
from signal_io import read_canonical_pl4
from traveling_wave_localizer import TravelingWaveConfig, locate

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\quick_check_v3_600km")
CLASSIFIER_V3 = Path(r"C:\RESULTPESQUISA\campaign_v4_600km\classifier_robust_v3\robust_classifier.joblib")
VELOCITY_KM_PER_US = 0.29927465071442727

CASES_PER_CLASS = 5
SEED = 20260808600


def _build_cases() -> list[tuple[str, float, float, float, float]]:
    rng = random.Random(SEED)
    cases: list[tuple[str, float, float, float, float]] = []
    for fault_class in sorted(FAULT_SWITCHES):
        for _ in range(CASES_PER_CLASS):
            distance_km = round(rng.uniform(15.0, 600.0), 2)
            remote_length_km = round(rng.uniform(max(1.0, 100.0 - distance_km), 400.0), 2)
            rfault_ohm = round(rng.uniform(0.01, 100.0), 3)
            angle_deg = round(rng.uniform(0.0, 359.9), 1)
            cases.append((fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg))
    return cases


def main() -> None:
    artifact = joblib.load(CLASSIFIER_V3)
    localizer_config = TravelingWaveConfig(
        velocity_km_per_us=VELOCITY_KM_PER_US, maximum_distance_km=600.0
    )
    cases = _build_cases()
    results = []

    for index, (fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg) in enumerate(cases, start=1):
        run_id = f"run_5{index:05d}"
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
        class_hit = predicted_class == fault_class

        try:
            location = locate(signals, localizer_config)
            loc_distance = location.distance_km
            loc_conclusive = location.conclusive
            loc_error_km = abs(loc_distance - distance_km) if loc_conclusive else None
        except ValueError as exc:
            loc_distance = None
            loc_conclusive = False
            loc_error_km = None

        results.append({
            "run_id": run_id, "true_class": fault_class, "predicted_class": predicted_class,
            "class_hit": class_hit, "vote_fraction": vote, "distance_km": distance_km,
            "predicted_distance_km": loc_distance, "location_conclusive": loc_conclusive,
            "location_error_km": loc_error_km,
        })
        loc_str = f"dist_prevista={loc_distance:.2f}km erro={loc_error_km:.2f}km" if loc_conclusive and loc_distance is not None else "localizacao inconclusiva"
        print(f"    -> classe={predicted_class} ({vote:.1%}) real={fault_class} [{'OK' if class_hit else 'ERRO'}]  {loc_str}")

    class_accuracy = sum(r["class_hit"] for r in results) / len(results)
    conclusive_results = [r for r in results if r["location_conclusive"]]
    loc_coverage = len(conclusive_results) / len(results)
    loc_errors = [r["location_error_km"] for r in conclusive_results]
    mae = float(np.mean(loc_errors)) if loc_errors else None

    print(f"\nAcuracia de classificacao (v3, 15-600km, {len(results)} casos): {class_accuracy:.1%}")
    print(f"Cobertura de localizacao conclusiva: {loc_coverage:.1%}")
    if mae is not None:
        print(f"MAE de localizacao (casos conclusivos): {mae:.3f} km")

    report_path = OUTPUT_ROOT / "quick_check_v3_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps({
            "class_accuracy": class_accuracy, "location_coverage": loc_coverage,
            "location_mae_km": mae, "results": results,
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()

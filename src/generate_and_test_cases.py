"""Gera um lote pequeno de casos de teste novos (nao reciclados do teste cego)
cobrindo as 10 classes de falta suportadas, roda o ATP de verdade para cada um,
e classifica com o pipeline congelado. Serve como validacao independente e
rapida, nao substitui o teste cego oficial documentado em RELATORIO_FINAL.md.
"""

from __future__ import annotations

import json
from pathlib import Path

import random

from fault_case_generator import FAULT_SWITCHES
from simulation_generator import SimulationParameters, generate_simulation
from infer_fault import infer_fault
from pilot_planner import (
    TRAIN_DISTANCES, TRAIN_RFAULTS, TRAIN_ANGLES, TRAIN_REMOTES,
    VALIDATION_DISTANCES, VALIDATION_RFAULTS, VALIDATION_ANGLES, VALIDATION_REMOTES,
    TEST_DISTANCES, TEST_RFAULTS, TEST_ANGLES, TEST_REMOTES,
)

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\quick_check")
CLASSIFIER = Path(__file__).resolve().parent.parent / "modelos" / "robust_classifier_v5_calibrated.joblib"
FREEZE = Path(__file__).resolve().parent.parent / "modelos" / "FINAL_PIPELINE_FREEZE_V8.json"

# Parametros novos, gerados por amostragem aleatoria continua (nao os valores
# discretos usados no treino/validacao/teste cego oficial), 15-450km (faixa
# comprovada), Rfault 0.01-100 ohm, angulo 0-360, remoto o suficiente para
# distancia+remoto >= 100km. 5 casos por classe = 50 no total.
_USED_DISTANCES = set(TRAIN_DISTANCES) | set(VALIDATION_DISTANCES) | set(TEST_DISTANCES)
_USED_RFAULTS = set(TRAIN_RFAULTS) | set(VALIDATION_RFAULTS) | set(TEST_RFAULTS)
_USED_ANGLES = set(TRAIN_ANGLES) | set(VALIDATION_ANGLES) | set(TEST_ANGLES)
CASES_PER_CLASS = 5
SEED = 20260807


def _build_cases() -> list[tuple[str, float, float, float, float]]:
    rng = random.Random(SEED)
    cases: list[tuple[str, float, float, float, float]] = []
    for fault_class in sorted(FAULT_SWITCHES):
        for _ in range(CASES_PER_CLASS):
            while True:
                distance_km = round(rng.uniform(15.0, 450.0), 2)
                if distance_km not in _USED_DISTANCES:
                    break
            remote_length_km = round(rng.uniform(max(1.0, 100.0 - distance_km), 500.0), 2)
            while True:
                rfault_ohm = round(rng.uniform(0.01, 100.0), 3)
                if rfault_ohm not in _USED_RFAULTS:
                    break
            while True:
                angle_deg = round(rng.uniform(0.0, 359.9), 1)
                if angle_deg not in _USED_ANGLES:
                    break
            cases.append((fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg))
    return cases


CASES = _build_cases()


def main() -> None:
    results = []
    for index, (fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg) in enumerate(CASES, start=1):
        run_id = f"run_9{index:05d}"
        params = SimulationParameters(
            run_id=run_id,
            fault_class=fault_class,
            distance_km=distance_km,
            remote_length_km=remote_length_km,
            rfault_ohm=rfault_ohm,
            incidence_angle_deg=angle_deg,
            split="test_unseen",
        )
        print(f"[{index}/{len(CASES)}] Gerando {run_id} ({fault_class}, {distance_km}km)...")
        sim_result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
        pl4_path = Path(sim_result["file_path"])

        prediction = infer_fault(pl4_path, CLASSIFIER, FREEZE)
        predicted_class = prediction["classification"]["fault_class"]
        vote_fraction = prediction["classification"]["tree_vote_fraction"]
        location = prediction["location"]

        hit = predicted_class == fault_class
        results.append({
            "run_id": run_id,
            "true_class": fault_class,
            "predicted_class": predicted_class,
            "hit": hit,
            "vote_fraction": vote_fraction,
            "true_distance_km": distance_km,
            "predicted_distance_km": location["distance_from_PDT_km"],
            "location_conclusive": location["conclusive"],
        })
        status = "OK" if hit else "ERRO"
        print(
            f"    -> previsto={predicted_class} ({vote_fraction:.1%}) "
            f"real={fault_class}  [{status}]  dist_real={distance_km}km "
            f"dist_prevista={location['distance_from_PDT_km']}"
        )

    accuracy = sum(r["hit"] for r in results) / len(results)
    print(f"\nAcuracia no lote novo (10 casos, 1 por classe): {accuracy:.1%}")

    report_path = OUTPUT_ROOT / "quick_check_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps({"accuracy": accuracy, "results": results}, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Relatorio salvo em: {report_path}")


if __name__ == "__main__":
    main()

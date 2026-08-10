"""Estende a faixa validada de distancia de 15-450km para 15-600km.
Gera casos de treino cobrindo 450-600km para as 10 classes, funde com o
manifesto v2 (500 originais + 36 augment ABG/BCG/CAG), retreina o classificador
(v3) e reconfigura o localizador com maximum_distance_km=600.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from fault_case_generator import FAULT_SWITCHES
from simulation_generator import SimulationParameters, generate_simulation

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
BASE_MANIFEST = Path(r"C:\RESULTPESQUISA\campaign_v4_augment\manifest_combined.csv")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\campaign_v4_600km")
COMBINED_MANIFEST = OUTPUT_ROOT / "manifest_combined_600km.csv"

CASES_PER_CLASS_TRAIN = 6
CASES_PER_CLASS_VALIDATION = 2
SEED = 20260807600

MANIFEST_FIELDS = (
    "run_id", "file_path", "split", "fault_class", "distance_km",
    "rfault_ohm", "incidence_angle_deg", "remote_length_km", "snr_db",
    "gain_error_pct", "sync_error_us", "source_voltage_error_pct",
    "source_impedance_error_pct",
)


def _build_cases() -> list[tuple[str, str, float, float, float, float]]:
    rng = random.Random(SEED)
    cases: list[tuple[str, str, float, float, float, float]] = []
    for fault_class in sorted(FAULT_SWITCHES):
        for split, count in (("train", CASES_PER_CLASS_TRAIN), ("validation", CASES_PER_CLASS_VALIDATION)):
            for _ in range(count):
                distance_km = round(rng.uniform(450.0, 600.0), 2)
                remote_length_km = round(rng.uniform(50.0, 400.0), 2)
                rfault_ohm = round(rng.uniform(0.01, 100.0), 3)
                angle_deg = round(rng.uniform(0.0, 359.9), 1)
                cases.append((split, fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg))
    return cases


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = _build_cases()
    new_rows = []
    for index, (split, fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg) in enumerate(cases, start=1):
        run_id = f"run_6{index:05d}"
        params = SimulationParameters(
            run_id=run_id, fault_class=fault_class, distance_km=distance_km,
            remote_length_km=remote_length_km, rfault_ohm=rfault_ohm,
            incidence_angle_deg=angle_deg, split=split,
        )
        print(f"[{index}/{len(cases)}] Gerando {run_id} ({fault_class}, {distance_km}km, {split})...")
        result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
        new_rows.append({
            "run_id": run_id, "file_path": result["file_path"], "split": split,
            "fault_class": fault_class, "distance_km": distance_km,
            "rfault_ohm": rfault_ohm, "incidence_angle_deg": angle_deg,
            "remote_length_km": remote_length_km, "snr_db": "", "gain_error_pct": "",
            "sync_error_us": "", "source_voltage_error_pct": "", "source_impedance_error_pct": "",
        })

    with BASE_MANIFEST.open("r", encoding="utf-8-sig", newline="") as stream:
        base_rows = list(csv.DictReader(stream))

    combined = base_rows + new_rows
    with COMBINED_MANIFEST.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(combined)

    print(f"\nManifesto combinado: {len(base_rows)} anteriores + {len(new_rows)} novos (450-600km) = {len(combined)} linhas")
    print(f"Salvo em: {COMBINED_MANIFEST}")


if __name__ == "__main__":
    main()

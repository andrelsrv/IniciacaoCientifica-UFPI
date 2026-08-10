"""Gera casos extras das 4 classes sem terra (AB/BC/CA/ABC), com enfase em
distancias curtas (15-80km), onde o erro residual do v4 se concentrou
(transiente de curto alcance imita assinatura de terra)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from simulation_generator import SimulationParameters, generate_simulation

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
BASE_MANIFEST = Path(r"C:\RESULTPESQUISA\campaign_v5_900\manifest_combined_v5.csv")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\campaign_v6_ab_focus")
COMBINED_MANIFEST = OUTPUT_ROOT / "manifest_combined_v6.csv"

TARGET_CLASSES = ("AB", "BC", "CA", "ABC")
CASES_PER_CLASS = 20
SEED = 20260808700

MANIFEST_FIELDS = (
    "run_id", "file_path", "split", "fault_class", "distance_km",
    "rfault_ohm", "incidence_angle_deg", "remote_length_km", "snr_db",
    "gain_error_pct", "sync_error_us", "source_voltage_error_pct",
    "source_impedance_error_pct",
)


def _build_cases() -> list[tuple[str, float, float, float, float]]:
    rng = random.Random(SEED)
    cases: list[tuple[str, float, float, float, float]] = []
    for fault_class in TARGET_CLASSES:
        for _ in range(CASES_PER_CLASS):
            distance_km = round(rng.uniform(15.0, 80.0), 2)
            remote_length_km = round(rng.uniform(max(1.0, 100.0 - distance_km), 500.0), 2)
            rfault_ohm = round(rng.uniform(0.01, 100.0), 3)
            angle_deg = round(rng.uniform(0.0, 359.9), 1)
            cases.append((fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg))
    return cases


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = _build_cases()
    new_rows = []
    for index, (fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg) in enumerate(cases, start=1):
        run_id = f"run_7{index:05d}"
        params = SimulationParameters(
            run_id=run_id, fault_class=fault_class, distance_km=distance_km,
            remote_length_km=remote_length_km, rfault_ohm=rfault_ohm,
            incidence_angle_deg=angle_deg, split="train",
        )
        print(f"[{index}/{len(cases)}] Gerando {run_id} ({fault_class}, {distance_km}km)...", flush=True)
        result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
        new_rows.append({
            "run_id": run_id, "file_path": result["file_path"], "split": "train",
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

    print(f"\nManifesto v6: {len(base_rows)} anteriores + {len(new_rows)} novos = {len(combined)} linhas")
    print(f"Salvo em: {COMBINED_MANIFEST}")


if __name__ == "__main__":
    main()

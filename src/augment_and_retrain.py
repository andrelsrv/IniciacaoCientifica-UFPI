"""Gera casos de treino extras para as classes bifasicas-terra (ABG/BCG/CAG)
em longa distancia (regiao onde o classificador v1 confundia com AG/BG/CG),
funde com o manifesto de treino/validacao congelado (campaign_v4) e retreina
uma nova versao (v2) do classificador. Nao sobrescreve o artefato congelado.
"""

from __future__ import annotations

import csv
import random
from pathlib import Path

from simulation_generator import SimulationParameters, generate_simulation

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
BASE_MANIFEST = Path(
    r"C:\Users\sams\Desktop\PEQUISA ACADEMICA\PesquisaAcademicaUFPI-20260801T132603Z-1-001"
    r"\pilot\campaign_v4\manifest.csv"
)
AUGMENT_ROOT = Path(r"C:\RESULTPESQUISA\campaign_v4_augment")
COMBINED_MANIFEST = AUGMENT_ROOT / "manifest_combined.csv"

TARGET_CLASSES = ("ABG", "BCG", "CAG")
CASES_PER_CLASS = 12
SEED = 20260807003

MANIFEST_FIELDS = (
    "run_id", "file_path", "split", "fault_class", "distance_km",
    "rfault_ohm", "incidence_angle_deg", "remote_length_km", "snr_db",
    "gain_error_pct", "sync_error_us", "source_voltage_error_pct",
    "source_impedance_error_pct",
)


def _build_extra_cases() -> list[tuple[str, float, float, float, float]]:
    rng = random.Random(SEED)
    cases: list[tuple[str, float, float, float, float]] = []
    for fault_class in TARGET_CLASSES:
        for _ in range(CASES_PER_CLASS):
            distance_km = round(rng.uniform(200.0, 450.0), 2)
            remote_length_km = round(rng.uniform(max(1.0, 100.0 - distance_km), 400.0), 2)
            rfault_ohm = round(rng.uniform(0.01, 100.0), 3)
            angle_deg = round(rng.uniform(0.0, 359.9), 1)
            cases.append((fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg))
    return cases


def main() -> None:
    AUGMENT_ROOT.mkdir(parents=True, exist_ok=True)
    cases = _build_extra_cases()
    new_rows = []
    for index, (fault_class, distance_km, remote_length_km, rfault_ohm, angle_deg) in enumerate(cases, start=1):
        run_id = f"run_8{index:05d}"
        params = SimulationParameters(
            run_id=run_id, fault_class=fault_class, distance_km=distance_km,
            remote_length_km=remote_length_km, rfault_ohm=rfault_ohm,
            incidence_angle_deg=angle_deg, split="train",
        )
        print(f"[{index}/{len(cases)}] Gerando {run_id} ({fault_class}, {distance_km}km)...")
        result = generate_simulation(params, TEMPLATE_ATP, AUGMENT_ROOT)
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

    print(f"\nManifesto combinado: {len(base_rows)} originais + {len(new_rows)} novos = {len(combined)} linhas")
    print(f"Salvo em: {COMBINED_MANIFEST}")


if __name__ == "__main__":
    main()

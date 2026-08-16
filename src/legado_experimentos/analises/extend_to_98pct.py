"""Gera ~450 casos de treino extras (10 classes), com enfase na regiao
historicamente dificil (200-600km, Rfault medio-alto), buscando dobrar o
volume de treino (446 -> ~900) para reduzir o erro residual observado na
curva de aprendizado (94,4% em 446 casos)."""

from __future__ import annotations

import csv
import random
from pathlib import Path

from fault_case_generator import FAULT_SWITCHES
from simulation_generator import SimulationParameters, generate_simulation

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
BASE_MANIFEST = Path(r"C:\RESULTPESQUISA\campaign_v4_600km\manifest_combined_600km.csv")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\campaign_v5_900")
COMBINED_MANIFEST = OUTPUT_ROOT / "manifest_combined_v5.csv"

CASES_PER_CLASS = 45
SEED = 20260808980

MANIFEST_FIELDS = (
    "run_id", "file_path", "split", "fault_class", "distance_km",
    "rfault_ohm", "incidence_angle_deg", "remote_length_km", "snr_db",
    "gain_error_pct", "sync_error_us", "source_voltage_error_pct",
    "source_impedance_error_pct",
)


def _build_cases() -> list[tuple[str, float, float, float, float]]:
    rng = random.Random(SEED)
    cases: list[tuple[str, float, float, float, float]] = []
    for fault_class in sorted(FAULT_SWITCHES):
        for i in range(CASES_PER_CLASS):
            # 60% enfase na regiao dificil (200-600km), 40% cobertura geral (15-600km)
            if i < int(CASES_PER_CLASS * 0.6):
                distance_km = round(rng.uniform(200.0, 600.0), 2)
            else:
                distance_km = round(rng.uniform(15.0, 600.0), 2)
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
        run_id = f"run_9{index:05d}"
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

    print(f"\nManifesto v5: {len(base_rows)} anteriores + {len(new_rows)} novos = {len(combined)} linhas")
    print(f"Salvo em: {COMBINED_MANIFEST}")


if __name__ == "__main__":
    main()

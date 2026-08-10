"""Gera e valida uma simulação ATP completa a partir de parâmetros físicos."""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path

from fault_case_generator import FAULT_SWITCHES, FaultParameters, configure_fault_deck
from jmarti_generator import JMartiConfig, generate as generate_jmarti
from pl4_reader import read_pl4
from rebuild_reference_case import rebuild


RUN_ID_RE = re.compile(r"^run_\d{6}$")


@dataclass(frozen=True)
class SimulationParameters:
    run_id: str
    fault_class: str
    distance_km: float
    remote_length_km: float
    rfault_ohm: float
    incidence_angle_deg: float
    split: str = "train"


def validate_parameters(params: SimulationParameters) -> None:
    if not RUN_ID_RE.fullmatch(params.run_id):
        raise ValueError("run_id deve seguir o formato opaco run_000001")
    if params.fault_class not in FAULT_SWITCHES:
        raise ValueError(f"Classe inválida: {params.fault_class}")
    if not 1 <= params.distance_km <= 600:
        raise ValueError("Distância deve estar entre 1 e 600 km")
    if not 0 < params.remote_length_km <= 600:
        raise ValueError("Trecho remoto deve estar entre 0 e 600 km")
    if params.distance_km + params.remote_length_km < 100:
        raise ValueError("O comprimento total da linha deve ser de pelo menos 100 km")
    if params.split not in {"train", "validation", "test_combination", "test_unseen"}:
        raise ValueError(f"Split inválido: {params.split}")
    # Reutiliza as validações congeladas de Rfault, ângulo e frequência.
    configure_fault_deck
    FaultParameters(params.fault_class, params.rfault_ohm, params.incidence_angle_deg)
    if not 0.01 <= params.rfault_ohm <= 3000:
        raise ValueError("Rfault deve estar entre 0,01 e 3000 ohms")
    if not 0 <= params.incidence_angle_deg < 360:
        raise ValueError("Ângulo deve estar no intervalo [0, 360)")


def generate_simulation(params: SimulationParameters, template_atp: Path,
                        output_root: Path) -> dict[str, object]:
    validate_parameters(params)
    run_dir = output_root / params.run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    pdt_name = f"{params.run_id}_pdt"
    bea_name = f"{params.run_id}_bea"
    generate_jmarti(
        JMartiConfig(
            params.distance_km,
            ("LOCA", "LOCB", "LOCC"),
            ("X0001A", "X0001B", "X0001C"),
        ), run_dir, pdt_name,
    )
    generate_jmarti(
        JMartiConfig(
            params.remote_length_km,
            ("X0001A", "X0001B", "X0001C"),
            ("X0004A", "X0004B", "X0004C"),
        ), run_dir, bea_name,
    )

    configured_path = run_dir / f"{params.run_id}_configured.atp"
    configured_path.write_text(
        configure_fault_deck(
            template_atp.read_text(encoding="latin-1"),
            FaultParameters(
                params.fault_class, params.rfault_ohm, params.incidence_angle_deg
            ),
        ), encoding="latin-1", newline="\r\n",
    )
    rebuild(
        configured_path,
        run_dir / f"{pdt_name}.pch",
        run_dir / f"{bea_name}.pch",
        run_dir,
        params.run_id,
    )
    pl4_path = run_dir / f"{params.run_id}.pl4"
    signals = read_pl4(pl4_path, expected_channels=12)
    result = {
        "parameters": asdict(params),
        "file_path": str(pl4_path.resolve()),
        "sample_count": signals.metadata.sample_count,
        "timestep_s": signals.metadata.timestep_s,
        "end_time_s": signals.metadata.end_time_s,
        "all_finite": True,
    }
    (run_dir / f"{params.run_id}.case.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8", newline="\n"
    )
    return result


def write_manifest_row(result: dict[str, object], output: Path) -> None:
    p = result["parameters"]
    row = {
        "run_id": p["run_id"], "file_path": result["file_path"],
        "split": p["split"], "fault_class": p["fault_class"],
        "distance_km": p["distance_km"], "rfault_ohm": p["rfault_ohm"],
        "incidence_angle_deg": p["incidence_angle_deg"],
        "remote_length_km": p["remote_length_km"],
        "snr_db": "", "gain_error_pct": "", "sync_error_us": "",
        "source_voltage_error_pct": "", "source_impedance_error_pct": "",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=row.keys())
        writer.writeheader()
        writer.writerow(row)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-atp", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--fault-class", choices=sorted(FAULT_SWITCHES), required=True)
    parser.add_argument("--distance-km", type=float, required=True)
    parser.add_argument("--remote-length-km", type=float, required=True)
    parser.add_argument("--rfault-ohm", type=float, required=True)
    parser.add_argument("--angle-deg", type=float, required=True)
    parser.add_argument("--split", default="train")
    parser.add_argument("--manifest-row", type=Path)
    args = parser.parse_args()
    params = SimulationParameters(
        args.run_id, args.fault_class, args.distance_km,
        args.remote_length_km, args.rfault_ohm, args.angle_deg, args.split,
    )
    result = generate_simulation(params, args.template_atp, args.output_root)
    if args.manifest_row:
        write_manifest_row(result, args.manifest_row)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

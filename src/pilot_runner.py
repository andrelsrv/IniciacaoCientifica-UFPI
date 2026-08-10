"""Executa o plano do piloto de forma incremental e retomável."""

from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

from pl4_reader import read_pl4
from simulation_generator import SimulationParameters, generate_simulation


MANIFEST_FIELDS = (
    "run_id", "file_path", "split", "fault_class", "distance_km",
    "rfault_ohm", "incidence_angle_deg", "remote_length_km", "snr_db",
    "gain_error_pct", "sync_error_us", "source_voltage_error_pct",
    "source_impedance_error_pct",
)


def load_plan(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or len({row["run_id"] for row in rows}) != len(rows):
        raise ValueError("Plano vazio ou com run_id duplicado")
    return rows


def valid_existing_pl4(path: Path) -> bool:
    try:
        data = read_pl4(path, expected_channels=12)
        return data.metadata.sample_count == 150001
    except (OSError, ValueError):
        return False


def manifest_row(plan: dict[str, str], pl4_path: Path) -> dict[str, object]:
    return {
        "run_id": plan["run_id"], "file_path": str(pl4_path.resolve()),
        "split": plan["split"], "fault_class": plan["fault_class"],
        "distance_km": plan["distance_km"], "rfault_ohm": plan["rfault_ohm"],
        "incidence_angle_deg": plan["incidence_angle_deg"],
        "remote_length_km": plan["remote_length_km"], "snr_db": "",
        "gain_error_pct": "", "sync_error_us": "",
        "source_voltage_error_pct": "", "source_impedance_error_pct": "",
    }


def write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=MANIFEST_FIELDS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def run(plan_path: Path, template_atp: Path, output_root: Path,
        limit: int | None = None) -> dict[str, object]:
    plan = load_plan(plan_path)
    if limit is not None:
        plan = plan[:limit]
    output_root.mkdir(parents=True, exist_ok=True)
    manifest_path = output_root / "manifest.csv"
    completed: list[dict[str, object]] = []
    generated = resumed = 0
    start = time.monotonic()

    for index, row in enumerate(plan, start=1):
        run_id = row["run_id"]
        pl4_path = output_root / run_id / f"{run_id}.pl4"
        if valid_existing_pl4(pl4_path):
            resumed += 1
            status = "validado existente"
        else:
            params = SimulationParameters(
                run_id, row["fault_class"], float(row["distance_km"]),
                float(row["remote_length_km"]), float(row["rfault_ohm"]),
                float(row["incidence_angle_deg"]), row["split"],
            )
            generate_simulation(params, template_atp, output_root)
            if not valid_existing_pl4(pl4_path):
                raise RuntimeError(f"PL4 inválido após geração: {run_id}")
            generated += 1
            status = "gerado"
        completed.append(manifest_row(row, pl4_path))
        write_manifest(manifest_path, completed)
        elapsed = time.monotonic() - start
        print(
            f"[{index}/{len(plan)}] {run_id} {row['fault_class']} {status} "
            f"({elapsed:.1f}s)", flush=True,
        )

    summary = {
        "planned": len(plan), "completed": len(completed),
        "generated": generated, "resumed": resumed,
        "elapsed_s": time.monotonic() - start,
        "manifest": str(manifest_path.resolve()),
    }
    (output_root / "pilot_status.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8", newline="\n"
    )
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--template-atp", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    print(json.dumps(run(args.plan, args.template_atp, args.output_root, args.limit), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

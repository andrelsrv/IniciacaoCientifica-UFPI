"""Valida o manifesto e inspeciona PL4 em lote sem expor rotulos ao modelo."""

from __future__ import annotations

import argparse
import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from manifest import ManifestRow, read_manifest, resolve_pl4_path
from signal_io import SignalData, read_canonical_pl4


@dataclass(frozen=True)
class InferenceRun:
    """Unica estrutura permitida depois da fronteira de ingestao."""

    run_id: str
    signals: SignalData


def load_inference_run(row: ManifestRow, manifest_path: str | Path) -> InferenceRun:
    # O caminho e o rotulo terminam nesta funcao. Nenhum deles e armazenado no
    # objeto entregue futuramente ao classificador/localizador.
    signals = read_canonical_pl4(resolve_pl4_path(row, manifest_path))
    return InferenceRun(run_id=row.run_id, signals=signals)


def inspect_manifest(manifest_path: str | Path) -> list[dict[str, object]]:
    rows = read_manifest(manifest_path)
    summaries: list[dict[str, object]] = []
    for row in rows:
        run = load_inference_run(row, manifest_path)
        timestep = float((run.signals.time_s[-1] - run.signals.time_s[0]) / (
            run.signals.time_s.size - 1
        ))
        summaries.append(
            {
                "run_id": run.run_id,
                "sample_count": int(run.signals.time_s.size),
                "timestep_s": timestep,
                "start_time_s": float(run.signals.time_s[0]),
                "end_time_s": float(run.signals.time_s[-1]),
                "all_finite": bool(np.all(np.isfinite(run.signals.values))),
            }
        )
    return summaries


def write_summaries(rows: list[dict[str, object]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.suffix.casefold() == ".json":
        output.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
        return
    if output.suffix.casefold() == ".csv":
        with output.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=tuple(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        return
    raise ValueError("A saida deve terminar em .json ou .csv.")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    validate = subparsers.add_parser("validate", help="Valida somente o manifesto.")
    validate.add_argument("manifest", type=Path)
    inspect = subparsers.add_parser("inspect", help="Valida e le todos os PL4.")
    inspect.add_argument("manifest", type=Path)
    inspect.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.command == "validate":
        rows = read_manifest(args.manifest)
        print(json.dumps({"valid": True, "runs": len(rows)}, indent=2))
        return 0

    summaries = inspect_manifest(args.manifest)
    write_summaries(summaries, args.output)
    print(json.dumps({"valid": True, "runs": len(summaries), "output": str(args.output)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

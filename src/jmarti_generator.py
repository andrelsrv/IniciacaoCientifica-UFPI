"""Geração reproduzível de trechos JMarti pelo solver ATP autorizado."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from dataclasses import asdict, dataclass
from pathlib import Path


CONDUCTOR_CARDS = (
    "  1.3156  .08998 4           2.514     0.0     24.     12.     40.   45.",
    "  2.3156  .08998 4           2.514     11.     24.     12.     40.   45.",
    "  3.3156  .08998 4           2.514     22.     24.     12.     40.   45.",
    "  0   .5   4.188 4          0.9144      5.     33.     26.     0.0   0.0",
    "  0   .5   4.188 4          0.9144     16.     33.     26.     0.0   0.0",
)


@dataclass(frozen=True)
class JMartiConfig:
    length_km: float
    from_nodes: tuple[str, str, str]
    to_nodes: tuple[str, str, str]
    soil_resistivity_ohm_m: float = 250.0
    matrix_frequency_hz: float = 450_000.0
    steady_state_frequency_hz: float = 60.0
    decades: int = 10
    points_per_decade: int = 20


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


_SHA256_CACHE: dict[tuple[str, int], str] = {}


def cached_sha256(path: Path) -> str:
    """Como sha256(), mas evita rehash de arquivos reutilizados entre casos
    (template ATP, solver, .pch de um pool de modelos de linha) -- o cache
    e' invalidado automaticamente se o arquivo mudar (chave inclui mtime)."""
    key = (str(path), path.stat().st_mtime_ns)
    cached = _SHA256_CACHE.get(key)
    if cached is not None:
        return cached
    digest = sha256(path)
    _SHA256_CACHE[key] = digest
    return digest


def _node_field(nodes: tuple[str, str, str]) -> str:
    if len(nodes) != 3 or any(not node or len(node) > 6 for node in nodes):
        raise ValueError("Cada terminal deve ter exatamente três nós ATP de 1 a 6 caracteres")
    return "".join(f"{node:<6}" for node in nodes)


def render_deck(config: JMartiConfig) -> str:
    if config.length_km <= 0:
        raise ValueError("O comprimento JMarti deve ser positivo")
    _node_field(config.from_nodes)
    _node_field(config.to_nodes)
    branch = "BRANCH  " + "".join(
        f"{left:<6}{right:<6}"
        for left, right in zip(config.from_nodes, config.to_nodes, strict=True)
    )
    rho = f"{config.soil_resistivity_ohm_m:g}"
    length = f"{config.length_km:g}"
    matrix = f"{config.matrix_frequency_hz:g}"
    steady = f"{config.steady_state_frequency_hz:g}"
    frequency_cards = (
        f"{rho:>8}{matrix:>10}{'':26}{length:>8}{'':17}1",
        f"{rho:>8}{steady:>10}{'':26}{length:>8}{'':17}1",
        f"{rho:>8}{steady:>10}{'':26}{length:>8}{config.decades:>10}{config.points_per_decade:>3}{'':4}1",
    )
    lines = [
        "BEGIN NEW DATA CASE", "JMARTI SETUP", "$ERASE", branch,
        "LINE CONSTANTS", "METRIC", *CONDUCTOR_CARDS, "",
        *frequency_cards, "", "", "DEFAULT", "$PUNCH", "",
        "BEGIN NEW DATA CASE", "BLANK", "",
    ]
    return "\n".join(lines)


def generate(config: JMartiConfig, output_dir: Path, name: str,
             solver: Path = Path(r"C:\ATP\atpmingw\tpbig.exe")) -> dict[str, object]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if not solver.is_file():
        raise FileNotFoundError(f"Solver ATP não encontrado: {solver}")
    runtime_dir = solver.parent
    work_dir = Path(tempfile.mkdtemp(prefix=f".{name}_", dir=output_dir))
    try:
        for asset in ("startup", "graphics", "graphics.aux"):
            source = runtime_dir / asset
            if not source.is_file():
                raise FileNotFoundError(f"Arquivo de runtime ausente: {source}")
            shutil.copy2(source, work_dir / asset)

        work_deck = work_dir / f"{name}.atp"
        work_deck.write_text(render_deck(config), encoding="ascii", newline="\r\n")
        result = subprocess.run(
            [str(solver), "disk", f".\\{work_deck.name}", "s", "-r"],
            cwd=work_dir, capture_output=True, text=True, timeout=120, check=False,
        )
        work_pch = work_dir / f"{name}.pch"
        if result.returncode != 0 or not work_pch.is_file() or work_pch.stat().st_size == 0:
            raise RuntimeError(
                f"Falha ao gerar JMarti ({result.returncode}):\n{result.stdout}\n{result.stderr}"
            )
        for suffix in (".atp", ".pch", ".lis", ".dbg"):
            source = work_dir / f"{name}{suffix}"
            if source.is_file():
                shutil.copy2(source, output_dir / source.name)
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    deck_path = output_dir / f"{name}.atp"
    pch_path = output_dir / f"{name}.pch"

    record = {
        "schema_version": 1,
        "generator": "ATP JMARTI SETUP",
        "config": asdict(config),
        "solver_path": str(solver.resolve()),
        "solver_sha256": sha256(solver),
        "deck_sha256": sha256(deck_path),
        "pch_sha256": sha256(pch_path),
    }
    (output_dir / f"{name}.provenance.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8", newline="\n"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--length-km", type=float, required=True)
    parser.add_argument("--from-nodes", nargs=3, required=True)
    parser.add_argument("--to-nodes", nargs=3, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", required=True)
    args = parser.parse_args()
    config = JMartiConfig(args.length_km, tuple(args.from_nodes), tuple(args.to_nodes))
    print(json.dumps(generate(config, args.output_dir, args.name), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

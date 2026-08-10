"""Reconstrói e executa o caso de referência usando dois PCH rastreáveis."""

from __future__ import annotations

import argparse
import json
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from jmarti_generator import sha256


INSERT_RE = re.compile(r"^\$INSERT,.*$", re.MULTILINE)


def replace_line_inserts(template: str, first_pch: Path, second_pch: Path) -> str:
    inserts = INSERT_RE.findall(template)
    if len(inserts) != 2:
        raise ValueError(f"Esperadas exatamente 2 linhas $INSERT; encontradas {len(inserts)}")
    replacements = iter((f"$INSERT, {first_pch}", f"$INSERT, {second_pch}"))
    return INSERT_RE.sub(lambda _: next(replacements), template)


def rebuild(template_atp: Path, first_pch: Path, second_pch: Path,
            output_dir: Path, name: str,
            solver: Path = Path(r"C:\ATP\atpmingw\tpbig.exe")) -> dict[str, object]:
    for path in (template_atp, first_pch, second_pch, solver):
        if not path.is_file():
            raise FileNotFoundError(path)
    output_dir.mkdir(parents=True, exist_ok=True)
    work_dir = Path(tempfile.mkdtemp(prefix=f".{name}_", dir=output_dir))
    try:
        for asset in ("startup", "graphics", "graphics.aux"):
            shutil.copy2(solver.parent / asset, work_dir / asset)
        shutil.copy2(first_pch, work_dir / "pdt.pch")
        shutil.copy2(second_pch, work_dir / "bea.pch")
        deck = replace_line_inserts(
            template_atp.read_text(encoding="latin-1"), Path("pdt.pch"), Path("bea.pch")
        )
        work_atp = work_dir / f"{name}.atp"
        work_atp.write_text(deck, encoding="latin-1", newline="\r\n")
        result = subprocess.run(
            [str(solver), "disk", f".\\{work_atp.name}", "s", "-r"],
            cwd=work_dir, capture_output=True, text=True, timeout=180, check=False,
        )
        work_pl4 = work_dir / f"{name}.pl4"
        for suffix in (".atp", ".pl4", ".lis", ".dbg"):
            source = work_dir / f"{name}{suffix}"
            if source.is_file():
                shutil.copy2(source, output_dir / source.name)
        if result.returncode != 0 or not work_pl4.is_file() or work_pl4.stat().st_size == 0:
            raise RuntimeError(f"ATP não gerou PL4:\n{result.stdout}\n{result.stderr}")
    finally:
        shutil.rmtree(work_dir, ignore_errors=True)

    record = {
        "schema_version": 1,
        "template_atp_sha256": sha256(template_atp),
        "first_pch_sha256": sha256(first_pch),
        "second_pch_sha256": sha256(second_pch),
        "solver_sha256": sha256(solver),
        "pl4_sha256": sha256(output_dir / f"{name}.pl4"),
    }
    (output_dir / f"{name}.provenance.json").write_text(
        json.dumps(record, indent=2), encoding="utf-8", newline="\n"
    )
    return record


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template-atp", type=Path, required=True)
    parser.add_argument("--first-pch", type=Path, required=True)
    parser.add_argument("--second-pch", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--name", default="reference_rebuilt")
    args = parser.parse_args()
    print(json.dumps(rebuild(**vars(args)), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

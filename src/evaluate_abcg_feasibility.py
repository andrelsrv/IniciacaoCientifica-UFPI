"""Teste de viabilidade: ABC-G e separavel de ABC no espaco de atributos?

Estrategia: gera um caso ABC valido (via simulation_generator, que ja produz
o .atp configurado corretamente para a distancia/rfault pedidos), depois
edita o deck configurado para TAMBEM fechar uma chave de terra (mesmo t_cl,
T-op=2), roda o ATP de novo, e compara os atributos de sequencia (que
capturam presenca de corrente de retorno pela terra) entre a versao ABC e a
versao ABC-G derivada, para varias distancias/rfault.
"""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from fault_case_generator import FaultParameters
from simulation_generator import SimulationParameters, generate_simulation
from signal_io import read_canonical_pl4
from feature_extraction import extract_features

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\abcg_feasibility")
SOLVER = Path(r"C:\ATP\atpmingw\tpbig.exe")

CASES = [
    (15.0, 200.0, 0.5, 0.0),
    (100.0, 200.0, 5.0, 45.0),
    (300.0, 200.0, 20.0, 90.0),
    (600.0, 100.0, 50.0, 180.0),
]


def _patch_to_abcg(configured_atp: Path, tclose: float) -> Path:
    data = configured_atp.read_bytes()
    lines = data.split(b"\r\n")
    out = []
    for line in lines:
        if line.startswith(b"  XX0006X0001A"):
            field = f"{tclose:.6f}".rstrip("0").rstrip(".")
            tclose_str = field.encode()
            new_line = line[:14] + tclose_str.rjust(10) + b"        2." + line[34:]
            out.append(new_line)
        else:
            out.append(line)
    patched_path = configured_atp.with_name(configured_atp.stem + "_abcg.atp")
    patched_path.write_bytes(b"\r\n".join(out))
    return patched_path


def _run_atp(atp_path: Path) -> Path:
    work_dir = atp_path.parent
    for asset in ("startup", "graphics", "graphics.aux"):
        source = SOLVER.parent / asset
        target = work_dir / asset
        if not target.is_file():
            shutil.copy2(source, target)
    subprocess.run(
        [str(SOLVER), "disk", atp_path.name, "s", "-r"],
        cwd=work_dir, capture_output=True, text=True, timeout=60, check=False,
    )
    pl4_path = atp_path.with_suffix(".pl4")
    if not pl4_path.is_file():
        raise RuntimeError(f"PL4 nao gerado para {atp_path}")
    return pl4_path


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    print(f"{'dist':>6s} {'rfault':>7s}  {'I0_PDT (ABC)':>13s} {'I0_PDT (ABCG)':>14s}  {'I0_BEA (ABC)':>13s} {'I0_BEA (ABCG)':>14s}")
    for index, (distance_km, remote_km, rfault_ohm, angle_deg) in enumerate(CASES, start=1):
        run_id = f"run_2{index:05d}"
        params = SimulationParameters(
            run_id=run_id, fault_class="ABC", distance_km=distance_km,
            remote_length_km=remote_km, rfault_ohm=rfault_ohm,
            incidence_angle_deg=angle_deg, split="train",
        )
        result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
        abc_pl4 = Path(result["file_path"])

        params_for_tclose = FaultParameters("ABC", rfault_ohm, angle_deg)
        tclose = params_for_tclose.tclose_s

        configured_atp = OUTPUT_ROOT / run_id / f"{run_id}_configured.atp"
        abcg_atp = _patch_to_abcg(configured_atp, tclose)
        abcg_pl4 = _run_atp(abcg_atp)

        signals_abc = read_canonical_pl4(abc_pl4)
        signals_abcg = read_canonical_pl4(abcg_pl4)
        f_abc = extract_features(signals_abc)
        f_abcg = extract_features(signals_abcg)

        i0_pdt_abc = f_abc.values[f_abc.names.index("sequence_ratio__PDT_I_zero")]
        i0_pdt_abcg = f_abcg.values[f_abcg.names.index("sequence_ratio__PDT_I_zero")]
        i0_bea_abc = f_abc.values[f_abc.names.index("sequence_ratio__BEA_I_zero")]
        i0_bea_abcg = f_abcg.values[f_abcg.names.index("sequence_ratio__BEA_I_zero")]

        print(f"{distance_km:6.0f} {rfault_ohm:7.2f}  {i0_pdt_abc:13.3f} {i0_pdt_abcg:14.3f}  {i0_bea_abc:13.3f} {i0_bea_abcg:14.3f}")


if __name__ == "__main__":
    main()

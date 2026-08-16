"""Teste de viabilidade completo: ABC vs ABC-G, usando todos os 61 atributos
e validacao cruzada, para estimar honestamente se essas duas classes sao
separaveis o suficiente para virar uma 11a classe do classificador."""

from __future__ import annotations

import random
import shutil
import subprocess
from pathlib import Path

import numpy as np
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import cross_val_score, StratifiedKFold

from fault_case_generator import FaultParameters
from simulation_generator import SimulationParameters, generate_simulation
from signal_io import read_canonical_pl4
from feature_extraction import extract_features

TEMPLATE_ATP = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")
OUTPUT_ROOT = Path(r"C:\RESULTPESQUISA\abcg_feasibility_full")
SOLVER = Path(r"C:\ATP\atpmingw\tpbig.exe")
N_PAIRS = 30
SEED = 20260810500


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
        target = work_dir / asset
        if not target.is_file():
            shutil.copy2(SOLVER.parent / asset, target)
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
    rng = random.Random(SEED)

    x_all, y_all = [], []
    for index in range(1, N_PAIRS + 1):
        distance_km = round(rng.uniform(15.0, 600.0), 2)
        remote_km = round(rng.uniform(max(1.0, 100.0 - distance_km), 400.0), 2)
        rfault_ohm = round(rng.uniform(0.01, 100.0), 3)
        angle_deg = round(rng.uniform(0.0, 359.9), 1)

        run_id = f"run_1{index:05d}"
        params = SimulationParameters(
            run_id=run_id, fault_class="ABC", distance_km=distance_km,
            remote_length_km=remote_km, rfault_ohm=rfault_ohm,
            incidence_angle_deg=angle_deg, split="train",
        )
        print(f"[{index}/{N_PAIRS}] dist={distance_km:.1f}km rfault={rfault_ohm:.2f} ang={angle_deg:.1f}", flush=True)
        result = generate_simulation(params, TEMPLATE_ATP, OUTPUT_ROOT)
        abc_pl4 = Path(result["file_path"])

        tclose = FaultParameters("ABC", rfault_ohm, angle_deg).tclose_s
        configured_atp = OUTPUT_ROOT / run_id / f"{run_id}_configured.atp"
        abcg_atp = _patch_to_abcg(configured_atp, tclose)
        abcg_pl4 = _run_atp(abcg_atp)

        f_abc = extract_features(read_canonical_pl4(abc_pl4))
        f_abcg = extract_features(read_canonical_pl4(abcg_pl4))

        x_all.append(f_abc.values)
        y_all.append("ABC")
        x_all.append(f_abcg.values)
        y_all.append("ABCG")

    x = np.vstack(x_all)
    y = np.asarray(y_all)

    model = ExtraTreesClassifier(
        n_estimators=400, min_samples_leaf=1, max_features="sqrt",
        class_weight="balanced", random_state=SEED, n_jobs=-1,
    )
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED)
    scores = cross_val_score(model, x, y, cv=cv, scoring="accuracy")

    print(f"\nAmostras: {len(y)} ({N_PAIRS} pares ABC/ABC-G)")
    print(f"Acuracia por fold (validacao cruzada 5x): {[f'{s:.1%}' for s in scores]}")
    print(f"Acuracia media: {scores.mean():.1%}  desvio: {scores.std():.1%}")

    # Importancia de atributos treinando no conjunto inteiro, so para diagnostico
    model.fit(x, y)
    names = extract_features(read_canonical_pl4(abc_pl4)).names
    importances = sorted(zip(names, model.feature_importances_), key=lambda t: -t[1])[:10]
    print("\nAtributos mais importantes para distinguir ABC de ABC-G:")
    for name, imp in importances:
        print(f"  {name:40s} {imp:.4f}")


if __name__ == "__main__":
    main()

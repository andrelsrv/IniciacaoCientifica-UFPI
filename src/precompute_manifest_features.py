"""Pre-calcula e cacheia as features do manifesto (todas as condicoes de
robustez) em um .npz, para nao precisar reextrair a cada retreino. A
extracao roda em paralelo (ProcessPoolExecutor) em vez de sequencial."""

from __future__ import annotations

import argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np


def _extract_one(args):
    manifest_path, run_id, file_path, fault_class, split, condition_name = args
    import sys
    sys.path.insert(0, r"C:\Users\sams\Desktop\PEQUISAACADEMICA\PesquisaAcademicaUFPIV2\PesquisaAcademicaUFPI\src")
    from feature_extraction import extract_features
    from robustness_evaluation import CONDITIONS, perturb
    from signal_io import read_canonical_pl4

    condition = next(c for c in CONDITIONS if c.name == condition_name)
    signals = read_canonical_pl4(Path(file_path))
    result = extract_features(perturb(signals, run_id, condition))
    return run_id, fault_class, split, condition_name, result.names, result.values


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("manifest", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=6)
    args = parser.parse_args()

    import sys
    sys.path.insert(0, r"C:\Users\sams\Desktop\PEQUISAACADEMICA\PesquisaAcademicaUFPIV2\PesquisaAcademicaUFPI\src")
    from manifest import read_manifest, resolve_pl4_path
    from robustness_evaluation import CONDITIONS

    rows = read_manifest(args.manifest)
    development = [row for row in rows if row.split in {"train", "validation"}]

    jobs = []
    for row in development:
        file_path = str(resolve_pl4_path(row, args.manifest))
        for condition in CONDITIONS:
            jobs.append((str(args.manifest), row.run_id, file_path, row.fault_class, row.split, condition.name))

    total = len(jobs)
    print(f"Total de extracoes: {total} ({len(development)} casos x {len(CONDITIONS)} condicoes)")

    xs, ys, splits, conditions, run_ids = [], [], [], [], []
    names_ref = None
    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(_extract_one, j) for j in jobs]
        for fut in as_completed(futures):
            run_id, fault_class, split, condition_name, names, values = fut.result()
            if names_ref is None:
                names_ref = names
            xs.append(values)
            ys.append(fault_class)
            splits.append(split)
            conditions.append(condition_name)
            run_ids.append(run_id)
            done += 1
            if done % 200 == 0 or done == total:
                print(f"{done}/{total} ({100*done/total:.0f}%)", flush=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        X=np.vstack(xs),
        y=np.array(ys),
        split=np.array(splits),
        condition=np.array(conditions),
        run_id=np.array(run_ids),
        names=np.array(names_ref),
    )
    print(f"Cache salvo em: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Treina um regressor de distancia (a partir dos mesmos 61 atributos do
classificador) para servir como checagem de sanidade independente do
localizador por ondas viajantes.

Reaproveita todos os casos ja gerados em C:\\RESULTPESQUISA (identificados
por *.case.json), sem rodar ATP de novo.
"""
import glob
import json
import sys
import time
from pathlib import Path

ROOT = Path(r"C:\Users\sams\Desktop\PEQUISAACADEMICA\PesquisaAcademicaUFPIV2\PesquisaAcademicaUFPI")
sys.path.insert(0, str(ROOT / "src"))

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score

from feature_extraction import extract_features
from signal_io import read_canonical_pl4

SEARCH_ROOT = r"C:\RESULTPESQUISA"
OUTPUT_PATH = ROOT / "modelos" / "distance_sanity_regressor.joblib"

case_files = glob.glob(SEARCH_ROOT + "/**/*.case.json", recursive=True)
print(f"Encontrados {len(case_files)} arquivos case.json")

sources: list[tuple[str, float]] = []
for cf in case_files:
    try:
        meta = json.loads(Path(cf).read_text(encoding="utf-8"))
        sources.append((meta["file_path"], meta["parameters"]["distance_km"]))
    except Exception:
        pass

# Complementa com o lote t_cl=125-475ms (Tmax=0.5s), que nao usa case.json
# (gerado via helper de baixo nivel, fora do pipeline oficial congelado).
wide_tcl_manifest = Path(SEARCH_ROOT) / "wide_tcl_batch" / "manifest.json"
if wide_tcl_manifest.exists():
    wide_rows = json.loads(wide_tcl_manifest.read_text(encoding="utf-8"))
    for pl4_path, fault_class, distance_km in wide_rows:
        sources.append((pl4_path, distance_km))
    print(f"+ {len(wide_rows)} casos do lote t_cl 125-475ms")

print(f"Total de fontes: {len(sources)}")

rows = []
skipped = 0
start = time.time()
for i, (file_path, distance_km) in enumerate(sources):
    try:
        pl4_path = Path(file_path)
        if not pl4_path.exists():
            skipped += 1
            continue
        signals = read_canonical_pl4(pl4_path)
        features = extract_features(signals)
        rows.append((features.values, distance_km, str(pl4_path)))
    except Exception:
        skipped += 1
    if (i + 1) % 200 == 0:
        print(f"{i + 1}/{len(sources)} processados, {len(rows)} validos, {skipped} pulados, {time.time()-start:.0f}s")

print(f"\nTotal utilizavel: {len(rows)} casos ({skipped} pulados)")

X = np.asarray([r[0] for r in rows])
y = np.asarray([r[1] for r in rows])
paths = [r[2] for r in rows]

X_train, X_test, y_train, y_test, paths_train, paths_test = train_test_split(
    X, y, paths, test_size=0.2, random_state=20260810
)

model = RandomForestRegressor(
    n_estimators=400, max_depth=None, min_samples_leaf=2,
    n_jobs=-1, random_state=20260810,
)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
mae = mean_absolute_error(y_test, pred_test)
r2 = r2_score(y_test, pred_test)
print(f"\n=== Avaliacao (holdout, {len(X_test)} casos) ===")
print(f"MAE={mae:.2f}km  R2={r2:.4f}  median_abs_err={np.median(np.abs(pred_test - y_test)):.2f}km  "
      f"p90={np.percentile(np.abs(pred_test - y_test), 90):.2f}km  max={np.max(np.abs(pred_test - y_test)):.2f}km")

# feature_names precisam ser as mesmas do classificador para reuso do vetor
sample_signals = read_canonical_pl4(Path(rows[0][2]))
feature_names = extract_features(sample_signals).names
artifact = {
    "model": model,
    "feature_names": list(feature_names),
    "trained_on_cases": len(rows),
    "holdout_mae_km": mae,
    "holdout_r2": r2,
}
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
joblib.dump(artifact, OUTPUT_PATH)
print(f"\nModelo salvo em {OUTPUT_PATH}")

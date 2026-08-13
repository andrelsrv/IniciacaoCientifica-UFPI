"""Calibra a confianca (fracao de votos das arvores) do classificador v13
usando o conjunto de validacao (nunca usado no treino), com CalibratedClassifierCV
(sigmoid/Platt). Compara confianca crua vs calibrada nos 300 casos independentes,
usando um diagrama de confiabilidade (bins de confianca x acuracia real)."""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.calibration import CalibratedClassifierCV

from feature_extraction import extract_features
from manifest import read_manifest, resolve_pl4_path
from signal_io import read_canonical_pl4

MANIFEST = Path(r"C:\RESULTPESQUISA\campaign_v9_boundary\manifest_combined_v9.csv")
CLASSIFIER_PATH = Path(r"C:\RESULTPESQUISA\campaign_v6_ab_focus\classifier_robust_v5\robust_classifier_500100.joblib")
OUTPUT_DIR = Path(r"C:\RESULTPESQUISA\retrain_v18_500_100\classifier_calibrated")

TEST_BATCHES = [
    (Path(r"C:\RESULTPESQUISA\quick_check"), "quick_check_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v2"), "quick_check_v2_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v3_600km"), "quick_check_v3_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v4_final"), "quick_check_v4_report.json"),
    (Path(r"C:\RESULTPESQUISA\quick_check_v5_final"), "quick_check_v5_report.json"),
]


def load_independent_test_set() -> tuple[np.ndarray, np.ndarray]:
    xs, ys = [], []
    for root, fname in TEST_BATCHES:
        report = json.loads((root / fname).read_text(encoding="utf-8"))
        for r in report["results"]:
            run_id = r["run_id"]
            pl4_path = root / run_id / f"{run_id}.pl4"
            if not pl4_path.is_file():
                continue
            signals = read_canonical_pl4(pl4_path)
            f = extract_features(signals)
            xs.append(f.values)
            ys.append(r["true_class"])
    return np.vstack(xs), np.asarray(ys)


def reliability_table(y_true: np.ndarray, confidences: np.ndarray, hits: np.ndarray, label: str) -> None:
    bins = [(0.0, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
    print(f"\n{label}")
    print(f"{'faixa de confianca':20s} {'n':>5s} {'acuracia real':>15s}")
    for lo, hi in bins:
        mask = (confidences >= lo) & (confidences < hi)
        n = int(mask.sum())
        acc = float(hits[mask].mean()) if n else float("nan")
        print(f"{lo:.1f}-{hi:.1f}{'':13s} {n:5d} {acc:14.1%}" if n else f"{lo:.1f}-{hi:.1f}{'':13s} {n:5d} {'--':>15s}")


def main() -> None:
    artifact = joblib.load(CLASSIFIER_PATH)
    base_model = artifact["classifier"]

    rows = read_manifest(MANIFEST)
    val_rows = [r for r in rows if r.split == "validation"]
    print("Extraindo atributos de", len(val_rows), "casos de validacao (calibracao)...")
    x_val, y_val = [], []
    for row in val_rows:
        signals = read_canonical_pl4(resolve_pl4_path(row, MANIFEST))
        f = extract_features(signals)
        x_val.append(f.values)
        y_val.append(row.fault_class)
    x_val = np.vstack(x_val)
    y_val = np.asarray(y_val)

    calibrated = CalibratedClassifierCV(base_model, method="sigmoid", cv="prefit")
    calibrated.fit(x_val, y_val)

    print("Carregando conjunto de teste independente (300 casos)...")
    x_test, y_test = load_independent_test_set()

    raw_probs = base_model.predict_proba(x_test)
    raw_pred = base_model.classes_[np.argmax(raw_probs, axis=1)]
    raw_conf = np.max(raw_probs, axis=1)
    raw_hits = (raw_pred == y_test)

    cal_probs = calibrated.predict_proba(x_test)
    cal_pred = calibrated.classes_[np.argmax(cal_probs, axis=1)]
    cal_conf = np.max(cal_probs, axis=1)
    cal_hits = (cal_pred == y_test)

    print(f"\nAcuracia (previsao) crua: {raw_hits.mean():.1%}   calibrada: {cal_hits.mean():.1%}")
    reliability_table(y_test, raw_conf, raw_hits, "Confianca CRUA (fracao de votos das arvores)")
    reliability_table(y_test, cal_conf, cal_hits, "Confianca CALIBRADA (Platt/sigmoid)")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    artifact_calibrated = dict(artifact)
    artifact_calibrated["classifier"] = calibrated
    artifact_calibrated["calibration_method"] = "sigmoid (Platt), fit on validation split (v13 dense high-Z)"
    joblib.dump(artifact_calibrated, OUTPUT_DIR / "calibrated_classifier.joblib")
    print("\nSalvo em:", OUTPUT_DIR / "calibrated_classifier.joblib")


if __name__ == "__main__":
    main()

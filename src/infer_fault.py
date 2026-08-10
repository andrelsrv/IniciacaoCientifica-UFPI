"""Inferência final de classe e distância a partir de um PL4 novo."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import joblib
import numpy as np

from adaptive_localizer import estimate_prefault_snr_db, locate_adaptive
from feature_extraction import extract_features
from signal_io import read_canonical_pl4
from traveling_wave_localizer import TravelingWaveConfig

# O classificador agora aceita o evento de falta em qualquer ponto de
# ~0.025-0.125s (ver feature_extraction.py). O localizador, porem, ainda usa
# uma janela de referencia fixa (0.03-0.075s) internamente e nao foi
# revalidado fora do t_cl classico (0.0833-0.100s) — testes confirmaram que,
# fora dessa faixa, ele pode devolver uma distancia errada marcada como
# "conclusiva". Por seguranca, a localizacao fica bloqueada fora dela ate
# que o localizador seja revisado e revalidado separadamente.
LOCATION_SAFE_EVENT_WINDOW_S = (0.080, 0.105)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def infer_fault(pl4: Path, classifier_path: Path, freeze_path: Path) -> dict[str, object]:
    """Executa a inferência congelada e retorna um objeto serializável."""
    freeze = json.loads(freeze_path.read_text(encoding="utf-8"))
    actual_hash = _sha256(classifier_path)
    expected_hash = freeze["classifier"]["sha256"]
    if actual_hash != expected_hash:
        raise ValueError("O classificador não corresponde ao artefato congelado.")
    artifact = joblib.load(classifier_path)
    signals = read_canonical_pl4(pl4)
    features = extract_features(signals)
    if tuple(artifact["feature_names"]) != features.names:
        raise ValueError("A ordem de atributos difere do modelo congelado.")
    probabilities = artifact["classifier"].predict_proba(features.values.reshape(1, -1))[0]
    best = int(np.argmax(probabilities))
    predicted_class = str(artifact["classifier"].classes_[best])
    vote_fraction = float(probabilities[best])
    snr_db = estimate_prefault_snr_db(signals)

    event_lo, event_hi = LOCATION_SAFE_EVENT_WINDOW_S
    event_in_safe_window = event_lo <= features.event_time_s <= event_hi

    # Guarda operacional acrescentada depois da avaliação cega: abaixo de
    # 50 dB o teste final revelou falsas reflexões conclusivas. A classificação
    # continua disponível, mas nenhuma distância é divulgada nesse domínio.
    if not event_in_safe_window:
        location = {
            "conclusive": False,
            "distance_from_PDT_km": None,
            "reason": (
                f"Evento detectado em {features.event_time_s * 1000:.1f}ms, fora da janela "
                f"{event_lo * 1000:.0f}-{event_hi * 1000:.0f}ms em que o localizador foi validado. "
                "A classificação permanece confiável; a localização não."
            ),
            "method": None,
        }
    elif snr_db < 50.0:
        location = {
            "conclusive": False,
            "distance_from_PDT_km": None,
            "reason": "SNR abaixo de 50 dB; localização fora do domínio seguro validado",
            "method": None,
        }
    else:
        result = locate_adaptive(
            signals,
            TravelingWaveConfig(
                velocity_km_per_us=float(freeze["traveling_wave_velocity_km_per_us"])
            ),
        )
        location = {
            "conclusive": result.conclusive,
            "distance_from_PDT_km": result.distance_km,
            "reason": result.inconclusive_reason,
            "method": result.method,
        }
    output = {
        "input": str(pl4),
        "classification": {
            "fault_class": predicted_class,
            "tree_vote_fraction": vote_fraction,
            "conclusive": vote_fraction >= 0.60,
            "note": "Fração de votos; não é probabilidade calibrada.",
        },
        "location": location,
        "location_domain_warning": (
            "A avaliação cega cobriu 15 a 450 km. Faltas muito próximas do "
            "terminal (por exemplo, 1 km) não são resolvidas com segurança "
            "pela janela atual de 40 us."
        ),
        "estimated_prefault_snr_db": snr_db,
        "classifier_sha256": actual_hash,
        "freeze_sha256": _sha256(freeze_path),
    }
    return output


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("pl4", type=Path)
    parser.add_argument("--classifier", type=Path, required=True)
    parser.add_argument("--freeze", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = infer_fault(args.pl4, args.classifier, args.freeze)
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

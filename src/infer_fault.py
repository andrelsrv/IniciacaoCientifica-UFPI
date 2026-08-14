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

# A janela de baseline do localizador (traveling_wave_localizer.py e
# adaptive_localizer.py) era fixa (0.03-0.075s) e por isso dava distancias
# erradas com falsa confianca fora do t_cl classico. Corrigido para usar a
# mesma janela de regime permanente relativa ao inicio da simulacao que
# feature_extraction.py usa (ver BASELINE_START_S/END_S) — validado com
# casos reais em t_cl=33ms e t_cl=117ms (erro de 0.31km em ambos, antes o
# erro chegava a 79-243km). A janela segura acompanha a mesma faixa de
# busca do classificador (feature_extraction.SEARCH_START_S/END_S), que
# por sua vez depende do Tmax do template ATP (0.7s a partir da v12 —
# testado ate 1.0s, mas o solver ATP corrompe resultados acima de ~800 mil
# passos de tempo; 0.7s fica com margem segura abaixo desse limite).
LOCATION_SAFE_EVENT_WINDOW_S = (0.025, 0.675)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest().upper()


def infer_fault(
    pl4: Path, classifier_path: Path, freeze_path: Path, regressor_path: Path | None = None
) -> dict[str, object]:
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

    regressor_info = freeze.get("distance_sanity_check")
    sanity_artifact = None
    if regressor_info is not None:
        candidate_path = regressor_path or (freeze_path.parent / regressor_info["path"])
        if _sha256(candidate_path) != regressor_info["sha256"]:
            raise ValueError("O regressor de sanidade não corresponde ao artefato congelado.")
        sanity_artifact = joblib.load(candidate_path)
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
        # Checagem de sanidade independente: um regressor de ML (treinado a
        # partir da atenuacao do sinal, nao da correlacao de ondas viajantes)
        # estima a distancia por um caminho fisico diferente. Validado em
        # 280 casos: reduz o risco de reflexao falsa aceita como conclusiva
        # de ~1.5% para ~0.4% dos casos conclusivos, ao custo de rejeitar
        # raros casos corretos com atenuacao atipica. Ver freeze v10.
        if location["conclusive"] and sanity_artifact is not None:
            # O regressor de sanidade nao foi retreinado desde o v12 (61
            # atributos, sem phase_asymmetry) -- excluir as 2 colunas novas
            # antes de prever, senao o vetor fica com o numero errado de
            # atributos e ordem incompativel com o que o regressor espera.
            sanity_mask = [not name.startswith("phase_asymmetry__") for name in features.names]
            sanity_values = features.values[sanity_mask]
            regressor_estimate = float(
                sanity_artifact["model"].predict(sanity_values.reshape(1, -1))[0]
            )
            disagreement_km = abs(location["distance_from_PDT_km"] - regressor_estimate)
            max_disagreement = float(regressor_info["max_disagreement_km"])
            if disagreement_km > max_disagreement:
                location = {
                    "conclusive": False,
                    "distance_from_PDT_km": None,
                    "reason": (
                        f"Reflexao rejeitada pela checagem de sanidade: onda viajante estimou "
                        f"{location['distance_from_PDT_km']:.1f}km, mas o regressor de atenuacao "
                        f"estimou {regressor_estimate:.1f}km (discordancia de {disagreement_km:.1f}km, "
                        f"acima do limite de {max_disagreement:.0f}km)."
                    ),
                    "method": None,
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
            "A avaliação cega oficial cobriu 15 a 450 km; 15-600 km foi "
            "revalidado informalmente após a correção da janela de "
            "normalização (freeze v9). Faltas muito próximas do terminal "
            "(por exemplo, 1 km) não são resolvidas com segurança pela "
            "janela atual de 40 us."
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
    parser.add_argument("--regressor", type=Path, default=None)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = infer_fault(args.pl4, args.classifier, args.freeze, args.regressor)
    rendered = json.dumps(output, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

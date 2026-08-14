"""Extração determinística de atributos físicos dos 12 sinais ATP."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from signal_io import CANONICAL_CHANNELS, SignalData


FEATURE_VERSION = "pilot_v6_phase_asymmetry"
# Janela de busca ampliada: cobre praticamente toda a simulacao, em vez de
# uma fatia fixa de 80-110ms. A linha de base (regime permanente) usa um
# trecho bem no inicio, pois a fonte ja parte em regime (solucao fasorial
# como condicao inicial no ATP) — nao ha transitorio de partida a esperar.
# Tmax do template ATP e 0.7s (700 mil passos de tempo com dt=1us). O
# solver ATP (tpbig.exe) tem um limite numerico proprio: um teste dedicado
# mostrou precisao correta ate 800 mil passos e corrompida a partir de 900
# mil (mesmo caso, so mudando Tmax, erro de localizacao pulou de <1km para
# 351km). 0.7s fica com boa margem abaixo desse limite. NAO AUMENTAR o
# Tmax sem revalidar esse limite do solver. A janela de busca acompanha o
# Tmax (folga de ~25ms antes do fim). Se o Tmax do seu .atp for diferente,
# ajuste SEARCH_END_S = Tmax - 0.025.
SEARCH_START_S = 0.025
SEARCH_END_S = 0.675
BASELINE_START_S = 0.005
BASELINE_END_S = 0.020


@dataclass(frozen=True)
class FeatureResult:
    values: np.ndarray
    names: tuple[str, ...]
    event_time_s: float
    pdt_arrival_s: float
    bea_arrival_s: float


def _rms(values: np.ndarray, axis: int = 0) -> np.ndarray:
    return np.sqrt(np.mean(np.square(values), axis=axis))


def _index(time_s: np.ndarray, value: float) -> int:
    return int(np.searchsorted(time_s, value, side="left"))


def _arrival_index(
    normalized: np.ndarray,
    start: int,
    end: int,
    cycle_samples: int,
    columns: tuple[int, ...],
) -> int:
    residual = normalized[cycle_samples:] - normalized[:-cycle_samples]
    energy = _rms(residual[:, columns], axis=1)
    baseline_start = max(0, _index_from_dt(BASELINE_START_S, normalized.shape[0], cycle_samples) - cycle_samples)
    baseline_end = max(baseline_start + 1, start - cycle_samples - 1000)
    base = energy[baseline_start:baseline_end]
    median = float(np.median(base))
    mad = float(np.median(np.abs(base - median)))
    threshold = max(0.01, median + 12.0 * max(mad, 1e-8))
    lo = max(0, start - cycle_samples)
    hi = min(energy.size, end - cycle_samples)
    # Uma média de 25 amostras rejeita impulsos numéricos isolados sem apagar
    # a frente de onda (25 us com o passo congelado em 1 us).
    window = 25
    smooth = np.convolve(energy[lo:hi], np.ones(window) / window, mode="same")
    candidates = np.flatnonzero(smooth > threshold)
    if candidates.size:
        return lo + int(candidates[0]) + cycle_samples
    return lo + int(np.argmax(smooth)) + cycle_samples


def _index_from_dt(seconds: float, sample_count: int, cycle_samples: int) -> int:
    # cycle_samples corresponde a 1/60 s e evita propagar pequenas diferenças
    # float32 do vetor de tempo para a linha de base.
    return min(sample_count - 1, max(0, round(seconds * 60.0 * cycle_samples)))


def _sequence_rms(window: np.ndarray, offset: int) -> np.ndarray:
    a = np.exp(2j * np.pi / 3)
    transform = np.array(
        [[1, 1, 1], [1, a, a**2], [1, a**2, a]], dtype=np.complex128
    ) / 3.0
    sequences = window[:, offset : offset + 3] @ transform.T
    return _rms(np.abs(sequences), axis=0)


def extract_features(signals: SignalData) -> FeatureResult:
    if signals.channel_names != CANONICAL_CHANNELS:
        raise ValueError("A extração exige a ordem canônica dos 12 canais.")
    time_s = signals.time_s
    values = signals.values
    dt = float(np.median(np.diff(time_s)))
    if not 0.5e-6 <= dt <= 1.5e-6:
        raise ValueError(f"Passo temporal fora do perfil do piloto: {dt} s")
    cycle_samples = int(round(1.0 / 60.0 / dt))
    base_lo, base_hi = _index(time_s, BASELINE_START_S), _index(time_s, BASELINE_END_S)
    scale = np.maximum(_rms(values[base_lo:base_hi]), 1e-9)
    normalized = values / scale
    search_lo, search_hi = _index(time_s, SEARCH_START_S), _index(time_s, SEARCH_END_S)

    pdt = _arrival_index(normalized, search_lo, search_hi, cycle_samples, (0, 1, 2, 6, 7, 8))
    bea = _arrival_index(normalized, search_lo, search_hi, cycle_samples, (3, 4, 5, 9, 10, 11))
    event = min(pdt, bea)

    pre_lo = max(0, event - cycle_samples)
    pre_hi = max(pre_lo + 10, event - int(round(0.0005 / dt)))
    post_lo = min(values.shape[0] - 10, event + int(round(0.005 / dt)))
    post_hi = min(values.shape[0], post_lo + cycle_samples)
    transient_hi = min(values.shape[0], event + int(round(0.005 / dt)))
    if post_hi - post_lo < cycle_samples // 2:
        raise ValueError("Sinal curto demais para a janela pós-falta.")

    pre = normalized[pre_lo:pre_hi]
    post = normalized[post_lo:post_hi]
    transient = normalized[event:transient_hi]
    pre_rms = np.maximum(_rms(pre), 1e-9)
    post_rms = _rms(post)
    rms_ratio = post_rms / pre_rms
    peak_ratio = np.max(np.abs(post), axis=0) / np.maximum(np.max(np.abs(pre), axis=0), 1e-9)
    delta_rms = _rms(post - np.mean(post, axis=0)) / pre_rms
    transient_step = np.max(np.abs(np.diff(transient, axis=0)), axis=0)

    feature_values: list[float] = []
    feature_names: list[str] = []
    for prefix, vector in (
        ("rms_ratio", rms_ratio),
        ("peak_ratio", peak_ratio),
        ("post_ac_rms", delta_rms),
        ("transient_max_step", transient_step),
    ):
        feature_values.extend(float(x) for x in vector)
        feature_names.extend(f"{prefix}__{name}" for name in CANONICAL_CHANNELS)

    for terminal, voltage_offset, current_offset in (("PDT", 0, 6), ("BEA", 3, 9)):
        for quantity, offset in (("V", voltage_offset), ("I", current_offset)):
            pre_seq = np.maximum(_sequence_rms(pre, offset), 1e-9)
            post_seq = _sequence_rms(post, offset)
            ratio = post_seq / pre_seq
            feature_values.extend(float(x) for x in ratio)
            feature_names.extend(
                f"sequence_ratio__{terminal}_{quantity}_{component}"
                for component in ("zero", "positive", "negative")
            )

    # Assimetria entre fases sas (max-min do rms_ratio das 3 fases de
    # corrente de um terminal). Faltas fase-fase-terra (ex. ABG) drenam
    # corrente extra pelo caminho de terra, fazendo as fases afetadas
    # SUBIREM acima do valor pre-falta (ratio > 1); faltas fase-fase puras
    # (ex. AB) redistribuem corrente sem caminho extra, tipicamente CAINDO
    # abaixo do pre-falta (ratio < 1). Essa diferenca de sinal amplia a
    # assimetria entre as fases faltosas exatamente onde a razao de
    # sequencia-zero perde forca (Rfault alto, corrente de terra pequena) --
    # validado empiricamente (AUC=1.0 em lote de diagnostico, Rfault
    # 900-3000 ohm, ver diag_featuretest em RESULTPESQUISA).
    for terminal, current_offset in (("PDT", 6), ("BEA", 9)):
        phase_ratios = rms_ratio[current_offset : current_offset + 3]
        feature_values.append(float(np.max(phase_ratios) - np.min(phase_ratios)))
        feature_names.append(f"phase_asymmetry__{terminal}_I")

    feature_values.append((float(time_s[bea]) - float(time_s[pdt])) * 1e6)
    feature_names.append("arrival_delay_BEA_minus_PDT_us")
    return FeatureResult(
        values=np.asarray(feature_values, dtype=np.float64),
        names=tuple(feature_names),
        event_time_s=float(time_s[event]),
        pdt_arrival_s=float(time_s[pdt]),
        bea_arrival_s=float(time_s[bea]),
    )


def add_phase_asymmetry(values: np.ndarray, names: tuple[str, ...]) -> tuple[np.ndarray, tuple[str, ...]]:
    """Deriva phase_asymmetry__{PDT,BEA}_I a partir de rms_ratio ja
    presentes, sem precisar reprocessar o sinal bruto. Usada para trazer
    datasets extraidos com uma FEATURE_VERSION anterior (sem essas 2
    colunas) para o mesmo espaco de atributos da extracao atual, sem
    resimular no ATP. Idempotente: se as colunas ja existirem, retorna os
    dados inalterados. `values` pode ser 1D (um caso) ou 2D (N casos x
    atributos), alinhado com `names`."""
    if "phase_asymmetry__PDT_I" in names:
        return values, names
    values_2d = values if values.ndim == 2 else values[np.newaxis, :]
    extra_cols = []
    extra_names = []
    for terminal in ("PDT", "BEA"):
        phase_idx = [names.index(f"rms_ratio__{terminal}_I{p}_A") for p in ("A", "B", "C")]
        phase_ratios = values_2d[:, phase_idx]
        extra_cols.append(np.max(phase_ratios, axis=1) - np.min(phase_ratios, axis=1))
        extra_names.append(f"phase_asymmetry__{terminal}_I")
    extra_block = np.column_stack(extra_cols)
    # Inserido na MESMA posicao que extract_features usa (antes de
    # arrival_delay_BEA_minus_PDT_us), para que dados aumentados aqui e
    # dados extraidos ao vivo fiquem com ordem de colunas identica --
    # caso contrario um concat silenciosamente desalinha os atributos.
    if "arrival_delay_BEA_minus_PDT_us" in names:
        insert_at = names.index("arrival_delay_BEA_minus_PDT_us")
    else:
        insert_at = len(names)
    augmented = np.hstack([values_2d[:, :insert_at], extra_block, values_2d[:, insert_at:]])
    new_names = tuple(names[:insert_at]) + tuple(extra_names) + tuple(names[insert_at:])
    if values.ndim == 1:
        augmented = augmented[0]
    return augmented, new_names

"""Localização híbrida por primeiras frentes e reflexões direcionais."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import find_peaks

from feature_extraction import FeatureResult, extract_features
from signal_io import SignalData


CLARKE_AERIAL = np.asarray(
    [[2 / 3, -1 / 3, -1 / 3], [0, 1 / np.sqrt(3), -1 / np.sqrt(3)]]
)


@dataclass(frozen=True)
class TravelingWaveConfig:
    velocity_km_per_us: float
    signature_samples: int = 40
    current_weight: float = 0.25
    minimum_correlation: float = 0.20
    minimum_terminal_correlation_for_conclusive: float = 0.30
    maximum_consistency_us: float = 15.0
    maximum_distance_km: float = 500.0


@dataclass(frozen=True)
class LocationResult:
    distance_km: float
    remote_distance_km: float
    pdt_reflection_delay_us: float
    bea_reflection_delay_us: float
    consistency_error_us: float
    correlation_pdt: float
    correlation_bea: float
    candidate_margin: float
    conclusive: bool
    inconclusive_reason: str | None
    arrivals: FeatureResult


def _directional_curve(
    signals: SignalData,
    arrival_s: float,
    voltage_offset: int,
    current_offset: int,
    config: TravelingWaveConfig,
) -> np.ndarray:
    time_s, values = signals.time_s, signals.values
    dt = float(np.median(np.diff(time_s)))
    cycle = int(round(1 / 60 / dt))
    baseline = (time_s >= 0.03) & (time_s < 0.075)
    voltage = values[:, voltage_offset : voltage_offset + 3] @ CLARKE_AERIAL.T
    current = values[:, current_offset : current_offset + 3] @ CLARKE_AERIAL.T
    voltage /= max(float(np.sqrt(np.mean(voltage[baseline] ** 2))), 1e-12)
    current /= max(float(np.sqrt(np.mean(current[baseline] ** 2))), 1e-12)
    voltage = np.diff(voltage[cycle:] - voltage[:-cycle], axis=0)
    current = np.diff(current[cycle:] - current[:-cycle], axis=0)
    start = int(np.searchsorted(time_s, arrival_s)) - cycle - 1
    positive = voltage + config.current_weight * current
    negative = voltage - config.current_weight * current
    width = config.signature_samples
    if np.linalg.norm(positive[start : start + width]) >= np.linalg.norm(
        negative[start : start + width]
    ):
        incident, reflected = positive, negative
    else:
        incident, reflected = negative, positive
    signature = incident[start : start + width].ravel()
    signature -= np.mean(signature)
    maximum_lag = int(np.ceil(
        2 * config.maximum_distance_km / config.velocity_km_per_us / (dt * 1e6)
    )) + width + 20
    search = reflected[start : start + maximum_lag]
    curve = np.empty(maximum_lag - width, dtype=np.float64)
    signature_norm = np.linalg.norm(signature)
    for lag in range(curve.size):
        candidate = search[lag : lag + width].ravel()
        candidate = candidate - np.mean(candidate)
        curve[lag] = abs(float(np.dot(signature, candidate))) / (
            signature_norm * np.linalg.norm(candidate) + 1e-20
        )
    return curve


def locate(signals: SignalData, config: TravelingWaveConfig) -> LocationResult:
    arrivals = extract_features(signals)
    pdt_curve = _directional_curve(signals, arrivals.pdt_arrival_s, 0, 6, config)
    bea_curve = _directional_curve(signals, arrivals.bea_arrival_s, 3, 9, config)
    pdt_peaks, _ = find_peaks(
        pdt_curve, height=config.minimum_correlation, distance=4
    )
    bea_peaks, _ = find_peaks(
        bea_curve, height=config.minimum_correlation, distance=4
    )
    dt_us = float(np.median(np.diff(signals.time_s))) * 1e6
    minimum_lag = max(2, int(np.floor(2 / config.velocity_km_per_us / dt_us)))
    maximum_lag = int(np.ceil(
        2 * config.maximum_distance_km / config.velocity_km_per_us / dt_us
    ))
    pdt_peaks = pdt_peaks[(pdt_peaks >= minimum_lag) & (pdt_peaks <= maximum_lag)]
    bea_peaks = bea_peaks[(bea_peaks >= minimum_lag) & (bea_peaks <= maximum_lag)]
    if not pdt_peaks.size or not bea_peaks.size:
        raise ValueError("Não foram encontradas reflexões candidatas nos dois terminais.")

    arrival_delta_samples = (
        arrivals.bea_arrival_s - arrivals.pdt_arrival_s
    ) / (dt_us * 1e-6)
    candidates: list[tuple[float, int, int, float]] = []
    for pdt_lag in pdt_peaks:
        target_bea_lag = pdt_lag + 2 * arrival_delta_samples
        position = int(np.argmin(np.abs(bea_peaks - target_bea_lag)))
        bea_lag = int(bea_peaks[position])
        consistency_samples = abs(
            (bea_lag - pdt_lag) - 2 * arrival_delta_samples
        )
        consistency_us = consistency_samples * dt_us
        if consistency_us > config.maximum_consistency_us:
            continue
        # Favorece reflexões fortes e fisicamente coerentes; a penalização
        # temporal evita escolher múltiplas voltas da mesma onda.
        score = (
            float(pdt_curve[pdt_lag] + bea_curve[bea_lag])
            - 0.0003 * (pdt_lag + bea_lag) * dt_us
            - 0.01 * consistency_us
        )
        candidates.append((score, int(pdt_lag), bea_lag, consistency_us))
    if not candidates:
        raise ValueError("Nenhum par de reflexões satisfez a coerência entre terminais.")
    candidates.sort(reverse=True)
    score, pdt_lag, bea_lag, consistency_us = candidates[0]
    margin = score - candidates[1][0] if len(candidates) > 1 else float("inf")
    distance = config.velocity_km_per_us * pdt_lag * dt_us / 2
    remote = config.velocity_km_per_us * bea_lag * dt_us / 2
    minimum_terminal_correlation = min(
        float(pdt_curve[pdt_lag]), float(bea_curve[bea_lag])
    )
    conclusive = (
        minimum_terminal_correlation
        >= config.minimum_terminal_correlation_for_conclusive
    )
    return LocationResult(
        distance_km=float(distance),
        remote_distance_km=float(remote),
        pdt_reflection_delay_us=float(pdt_lag * dt_us),
        bea_reflection_delay_us=float(bea_lag * dt_us),
        consistency_error_us=float(consistency_us),
        correlation_pdt=float(pdt_curve[pdt_lag]),
        correlation_bea=float(bea_curve[bea_lag]),
        candidate_margin=float(margin),
        conclusive=conclusive,
        inconclusive_reason=(
            None if conclusive else "reflexão fraca em pelo menos um terminal"
        ),
        arrivals=arrivals,
    )

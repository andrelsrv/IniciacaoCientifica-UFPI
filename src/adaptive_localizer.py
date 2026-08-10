"""Seleciona automaticamente localização simples ou multiescala pelo SNR."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from multiscale_localizer import MultiscaleConfig, MultiscaleLocationResult, locate_multiscale
from signal_io import SignalData
from traveling_wave_localizer import LocationResult, TravelingWaveConfig, locate


@dataclass(frozen=True)
class AdaptiveLocationResult:
    distance_km: float | None
    conclusive: bool
    inconclusive_reason: str | None
    estimated_snr_db: float
    method: str
    detail: LocationResult | MultiscaleLocationResult


def estimate_prefault_snr_db(signals: SignalData) -> float:
    mask = (signals.time_s >= 0.02) & (signals.time_s < 0.075)
    time = signals.time_s[mask]
    values = signals.values[mask]
    design = np.column_stack((
        np.sin(2 * np.pi * 60 * time),
        np.cos(2 * np.pi * 60 * time),
        np.ones(time.size),
    ))
    coefficients = np.linalg.lstsq(design, values, rcond=None)[0]
    fitted = design @ coefficients
    signal_rms = np.sqrt(np.mean(fitted**2, axis=0))
    residual_rms = np.sqrt(np.mean((values - fitted) ** 2, axis=0))
    channel_snr = 20 * np.log10(
        np.maximum(signal_rms, 1e-12) / np.maximum(residual_rms, 1e-12)
    )
    return float(np.median(channel_snr))


def locate_adaptive(
    signals: SignalData,
    traveling_wave_config: TravelingWaveConfig,
    *,
    multiscale_config: MultiscaleConfig = MultiscaleConfig(),
    multiscale_below_snr_db: float = 50.0,
) -> AdaptiveLocationResult:
    snr = estimate_prefault_snr_db(signals)
    if snr < multiscale_below_snr_db:
        detail = locate_multiscale(signals, traveling_wave_config, multiscale_config)
        return AdaptiveLocationResult(
            detail.distance_km, detail.conclusive, detail.inconclusive_reason,
            snr, "multiscale_consensus", detail,
        )
    try:
        detail = locate(signals, traveling_wave_config)
    except ValueError as error:
        empty = MultiscaleLocationResult(None, False, str(error), None, tuple())
        return AdaptiveLocationResult(None, False, str(error), snr, "single_scale", empty)
    return AdaptiveLocationResult(
        detail.distance_km if detail.conclusive else None,
        detail.conclusive,
        detail.inconclusive_reason,
        snr,
        "single_scale",
        detail,
    )

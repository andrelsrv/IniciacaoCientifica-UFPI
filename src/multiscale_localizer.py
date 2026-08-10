"""Consenso de localizadores direcionais em múltiplas bandas."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.signal import butter, sosfiltfilt

from signal_io import SignalData
from traveling_wave_localizer import LocationResult, TravelingWaveConfig, locate


@dataclass(frozen=True)
class MultiscaleConfig:
    cutoff_hz: tuple[float, ...] = (60_000, 80_000, 100_000, 150_000)
    maximum_spread_km: float = 3.0


@dataclass(frozen=True)
class MultiscaleLocationResult:
    distance_km: float | None
    conclusive: bool
    inconclusive_reason: str | None
    spread_km: float | None
    scale_results: tuple[LocationResult | None, ...]


def locate_multiscale(
    signals: SignalData,
    traveling_wave_config: TravelingWaveConfig,
    multiscale_config: MultiscaleConfig = MultiscaleConfig(),
) -> MultiscaleLocationResult:
    dt = float(np.median(np.diff(signals.time_s)))
    sample_rate = 1.0 / dt
    results: list[LocationResult | None] = []
    for cutoff in multiscale_config.cutoff_hz:
        sos = butter(4, cutoff, fs=sample_rate, output="sos")
        filtered = SignalData(
            signals.time_s,
            sosfiltfilt(sos, signals.values, axis=0),
            signals.channel_names,
        )
        try:
            result = locate(filtered, traveling_wave_config)
        except ValueError:
            result = None
        results.append(result)
    valid = [result for result in results if result is not None and result.conclusive]
    if len(valid) != len(multiscale_config.cutoff_hz):
        return MultiscaleLocationResult(
            None, False, "nem todas as bandas produziram reflexão confiável",
            None, tuple(results),
        )
    distances = np.asarray([result.distance_km for result in valid])
    spread = float(np.ptp(distances))
    if spread > multiscale_config.maximum_spread_km:
        return MultiscaleLocationResult(
            None, False, "estimativas multiescala discordantes",
            spread, tuple(results),
        )
    return MultiscaleLocationResult(
        float(np.median(distances)), True, None, spread, tuple(results)
    )

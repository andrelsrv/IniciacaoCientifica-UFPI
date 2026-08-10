"""Mapeamento validado dos sinais eletricos do circuito ATPDraw."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from pl4_reader import Pl4Data, Pl4FormatError, read_pl4


CANONICAL_CHANNELS = (
    "PDT_VA_V",
    "PDT_VB_V",
    "PDT_VC_V",
    "BEA_VA_V",
    "BEA_VB_V",
    "BEA_VC_V",
    "PDT_IA_A",
    "PDT_IB_A",
    "PDT_IC_A",
    "BEA_IA_A",
    "BEA_IB_A",
    "BEA_IC_A",
)

# Perfil congelado a partir de SIMULACAOUSADA.pl4 e validado contra
# ARCHIVEADFFALTAAG.adf em 2026-08-01. A ordem e intencionalmente explicita.
REFERENCE_PL4_PROFILE = (
    ("4", "LOCA"),
    ("4", "LOCB"),
    ("4", "LOCC"),
    ("4", "X0007A"),
    ("4", "X0007B"),
    ("4", "X0007C"),
    ("9", "X0003ALOCA"),
    ("9", "X0003BLOCB"),
    ("9", "X0003CLOCC"),
    ("9", "X0007AX0004A"),
    ("9", "X0007BX0004B"),
    ("9", "X0007CX0004C"),
)

REFERENCE_ADF_COLUMNS = (
    "t",
    "vLoca",
    "vLocb",
    "vLocc",
    "vX0007a",
    "vX0007b",
    "vX0007c",
    "iX0003aLoca",
    "iX0003bLocb",
    "iX0003cLocc",
    "iX0007aX0004a",
    "iX0007bX0004b",
    "iX0007cX0004c",
)


@dataclass(frozen=True)
class SignalData:
    time_s: np.ndarray
    values: np.ndarray
    channel_names: tuple[str, ...] = CANONICAL_CHANNELS

    def __post_init__(self) -> None:
        if self.values.ndim != 2 or self.values.shape[1] != len(self.channel_names):
            raise ValueError("Matriz de sinais nao corresponde aos canais canonicos.")
        if self.values.shape[0] != self.time_s.shape[0]:
            raise ValueError("Tempo e sinais possuem quantidades de amostras diferentes.")


def _validate_pl4_profile(pl4: Pl4Data) -> None:
    actual = tuple((channel.type_code, channel.label) for channel in pl4.channels)
    if actual != REFERENCE_PL4_PROFILE:
        lines = ["Perfil de canais PL4 diferente do perfil validado."]
        for index, (expected, found) in enumerate(
            zip(REFERENCE_PL4_PROFILE, actual, strict=False)
        ):
            if expected != found:
                lines.append(f"canal {index}: esperado={expected!r}, encontrado={found!r}")
        if len(actual) != len(REFERENCE_PL4_PROFILE):
            lines.append(
                f"quantidade: esperada={len(REFERENCE_PL4_PROFILE)}, encontrada={len(actual)}"
            )
        raise Pl4FormatError("\n".join(lines))


def read_canonical_pl4(path: str | Path) -> SignalData:
    pl4 = read_pl4(path, expected_channels=len(CANONICAL_CHANNELS))
    _validate_pl4_profile(pl4)
    return SignalData(time_s=pl4.time_s, values=pl4.values)


def read_reference_adf(path: str | Path) -> SignalData:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig") as stream:
        first_line = stream.readline()
        header_line = stream.readline()
    if "ADF file" not in first_line:
        raise ValueError("Arquivo nao possui o cabecalho esperado do PlotXY ADF.")
    columns = tuple(header_line.strip().split())
    if columns != REFERENCE_ADF_COLUMNS:
        raise ValueError(
            "Colunas ADF diferentes do perfil validado:\n"
            f"esperadas={REFERENCE_ADF_COLUMNS!r}\nrecebidas={columns!r}"
        )
    matrix = np.atleast_2d(np.loadtxt(source, skiprows=2, dtype=np.float64))
    if matrix.shape[1] != len(REFERENCE_ADF_COLUMNS):
        raise ValueError("Quantidade de colunas numericas inesperada no ADF.")
    if not np.all(np.isfinite(matrix)):
        raise ValueError("ADF contem NaN ou infinito.")
    return SignalData(time_s=matrix[:, 0], values=matrix[:, 1:])


def compare_signals(reference: SignalData, candidate: SignalData) -> dict[str, object]:
    if reference.channel_names != candidate.channel_names:
        raise ValueError("Os conjuntos usam nomes ou ordens de canais diferentes.")
    if reference.values.shape != candidate.values.shape:
        raise ValueError(
            f"Dimensoes diferentes: {reference.values.shape} != {candidate.values.shape}."
        )
    time_error = np.abs(reference.time_s - candidate.time_s)
    value_error = np.abs(reference.values - candidate.values)
    peak = np.maximum(1.0, np.max(np.abs(reference.values), axis=0))
    return {
        "sample_count": int(reference.time_s.size),
        "time_max_abs_s": float(np.max(time_error)),
        "channels": {
            name: {
                "max_abs": float(np.max(value_error[:, index])),
                "max_relative_to_reference_peak": float(
                    np.max(value_error[:, index]) / peak[index]
                ),
            }
            for index, name in enumerate(reference.channel_names)
        },
    }

"""Leitor para arquivos binarios ATP PL4 no formato NEWPL4=2.

O leitor preserva os descritores de canal gravados pelo ATP. A associacao entre
esses descritores e nomes de dominio (PDT_VA_V, BEA_IA_A etc.) deve ser feita
explicitamente depois, para impedir trocas silenciosas de fase ou terminal.
"""

from __future__ import annotations

import argparse
import json
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


FIXED_HEADER_SIZE = 80
CHANNEL_DESCRIPTOR_SIZE = 16
FLOAT_SIZE = 4


class Pl4FormatError(ValueError):
    """Indica que um PL4 nao corresponde ao layout suportado ou esta corrompido."""


@dataclass(frozen=True)
class Pl4Channel:
    index: int
    type_code: str
    label: str
    descriptor: str


@dataclass(frozen=True)
class Pl4Metadata:
    producer_stamp: str
    channel_count: int
    sample_count: int
    header_size: int
    timestep_s: float
    start_time_s: float
    end_time_s: float


@dataclass(frozen=True)
class Pl4Data:
    metadata: Pl4Metadata
    channels: tuple[Pl4Channel, ...]
    time_s: np.ndarray
    values: np.ndarray

    def channel(self, index: int) -> np.ndarray:
        """Retorna um canal pelo indice zero-based preservado no cabecalho."""
        return self.values[:, index]

    def summary(self) -> dict[str, object]:
        return {
            "producer_stamp": self.metadata.producer_stamp,
            "channel_count": self.metadata.channel_count,
            "sample_count": self.metadata.sample_count,
            "header_size": self.metadata.header_size,
            "timestep_s": self.metadata.timestep_s,
            "start_time_s": self.metadata.start_time_s,
            "end_time_s": self.metadata.end_time_s,
            "channels": [
                {
                    "index": channel.index,
                    "type_code": channel.type_code,
                    "label": channel.label,
                    "descriptor": channel.descriptor,
                }
                for channel in self.channels
            ],
        }


def _decode_channel(index: int, raw: bytes) -> Pl4Channel:
    descriptor = raw.decode("ascii", errors="replace")
    compact = descriptor.strip()
    if len(compact) < 2:
        raise Pl4FormatError(f"Descritor vazio ou invalido no canal {index}: {raw!r}")
    return Pl4Channel(
        index=index,
        type_code=compact[0],
        label=compact[1:].strip(),
        descriptor=descriptor,
    )


def read_pl4(
    path: str | Path,
    *,
    expected_channels: int | None = None,
    require_uniform_time: bool = True,
) -> Pl4Data:
    """Le um PL4 NEWPL4=2 e executa validacoes estruturais e temporais.

    O layout suportado foi confirmado no arquivo produzido pelo ATP usado no
    projeto: 80 bytes fixos, N descritores ASCII de 16 bytes e registros
    little-endian float32 contendo tempo seguido por N canais.
    """
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(source)

    raw = source.read_bytes()
    if len(raw) < FIXED_HEADER_SIZE:
        raise Pl4FormatError(
            f"Arquivo vazio ou curto demais: {len(raw)} bytes; minimo {FIXED_HEADER_SIZE}."
        )

    encoded_header_size = struct.unpack_from("<I", raw, 52)[0]
    header_size = encoded_header_size - 1
    if (
        header_size < FIXED_HEADER_SIZE
        or (header_size - FIXED_HEADER_SIZE) % CHANNEL_DESCRIPTOR_SIZE != 0
    ):
        raise Pl4FormatError(
            f"Tamanho de cabecalho NEWPL4=2 invalido: {encoded_header_size}."
        )
    channel_count = (header_size - FIXED_HEADER_SIZE) // CHANNEL_DESCRIPTOR_SIZE
    if channel_count <= 0 or channel_count > 10_000:
        raise Pl4FormatError(f"Quantidade de canais implausivel: {channel_count}.")
    if expected_channels is not None and channel_count != expected_channels:
        raise Pl4FormatError(
            f"Esperados {expected_channels} canais, mas o PL4 declara {channel_count}."
        )

    if len(raw) <= header_size:
        raise Pl4FormatError("PL4 nao contem registros de amostras depois do cabecalho.")

    encoded_file_size = struct.unpack_from("<I", raw, 56)[0]
    if encoded_file_size not in (0, len(raw) + 1):
        raise Pl4FormatError(
            "PL4 truncado ou alterado: tamanho declarado difere do arquivo: "
            f"declarado={encoded_file_size - 1}, real={len(raw)}."
        )

    record_size = (channel_count + 1) * FLOAT_SIZE
    payload_size = len(raw) - header_size
    if payload_size % record_size != 0:
        raise Pl4FormatError(
            "PL4 truncado ou layout incompativel: "
            f"{payload_size} bytes de dados nao sao multiplos de {record_size}."
        )

    sample_count = payload_size // record_size
    descriptors = raw[FIXED_HEADER_SIZE:header_size]
    channels = tuple(
        _decode_channel(
            index,
            descriptors[
                index * CHANNEL_DESCRIPTOR_SIZE : (index + 1) * CHANNEL_DESCRIPTOR_SIZE
            ],
        )
        for index in range(channel_count)
    )

    matrix = np.frombuffer(raw, dtype="<f4", offset=header_size).reshape(
        sample_count, channel_count + 1
    )
    time_s = matrix[:, 0].astype(np.float64, copy=True)
    values = matrix[:, 1:].astype(np.float64, copy=True)

    if not np.all(np.isfinite(time_s)):
        raise Pl4FormatError("Vetor de tempo contem NaN ou infinito.")
    if not np.all(np.isfinite(values)):
        raise Pl4FormatError("Um ou mais canais contem NaN ou infinito.")
    if sample_count < 2:
        raise Pl4FormatError("Sao necessarias pelo menos duas amostras.")

    time_steps = np.diff(time_s)
    if np.any(time_steps <= 0):
        raise Pl4FormatError("O vetor de tempo nao e estritamente crescente.")
    # Diferencas consecutivas de tempos float32 oscilam conforme o expoente do
    # valor absoluto cresce. O passo pelo intervalo total e mais estavel; a
    # uniformidade e testada contra a grade ideal com tolerancia de dois ULPs.
    timestep_s = float((time_s[-1] - time_s[0]) / (sample_count - 1))
    if require_uniform_time:
        expected_time = time_s[0] + np.arange(sample_count, dtype=np.float64) * timestep_s
        max_abs_time = np.float32(np.max(np.abs(time_s)))
        float32_ulp = float(np.spacing(max_abs_time)) if max_abs_time != 0 else 0.0
        time_tolerance = max(1e-12, 2.0 * abs(float32_ulp))
        if not np.allclose(time_s, expected_time, rtol=0.0, atol=time_tolerance):
            raise Pl4FormatError("O passo temporal do PL4 nao e uniforme.")

    stamp_bytes = raw[1:20] if raw[0] >= 0x80 else raw[:20]
    producer_stamp = stamp_bytes.decode("ascii", errors="replace").strip()
    metadata = Pl4Metadata(
        producer_stamp=producer_stamp,
        channel_count=channel_count,
        sample_count=sample_count,
        header_size=header_size,
        timestep_s=timestep_s,
        start_time_s=float(time_s[0]),
        end_time_s=float(time_s[-1]),
    )
    return Pl4Data(
        metadata=metadata,
        channels=channels,
        time_s=time_s,
        values=values,
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspeciona um ATP PL4 NEWPL4=2.")
    parser.add_argument("pl4", type=Path, help="Caminho do arquivo .pl4")
    parser.add_argument(
        "--expected-channels",
        type=int,
        default=None,
        help="Falha se a quantidade de canais for diferente.",
    )
    parser.add_argument("--json", action="store_true", help="Imprime resumo em JSON.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    data = read_pl4(args.pl4, expected_channels=args.expected_channels)
    summary = data.summary()
    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"Arquivo: {args.pl4}")
        print(
            f"Amostras: {data.metadata.sample_count} | "
            f"Canais: {data.metadata.channel_count} | "
            f"dt: {data.metadata.timestep_s:.12g} s"
        )
        print(
            f"Tempo: {data.metadata.start_time_s:.12g} a "
            f"{data.metadata.end_time_s:.12g} s"
        )
        for channel in data.channels:
            print(
                f"[{channel.index:02d}] tipo={channel.type_code!r} "
                f"rotulo={channel.label!r} descritor={channel.descriptor!r}"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

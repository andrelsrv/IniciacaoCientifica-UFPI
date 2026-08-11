"""Parametrização segura de faltas no circuito ATP de referência."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path


# O nome foi preservado porque os demais módulos usam suas chaves como a
# taxonomia oficial. Os valores representam as fases envolvidas e não as
# antigas chaves gráficas do ACP.
FAULT_SWITCHES = {
    "AG": ("A",), "BG": ("B",), "CG": ("C",),
    "AB": ("A", "B"), "BC": ("B", "C"), "CA": ("C", "A"),
    "ABG": ("A", "B"), "BCG": ("B", "C"), "CAG": ("C", "A"),
    "ABC": ("A", "B", "C"),
}

GROUNDED_FAULTS = frozenset({"AG", "BG", "CG", "ABG", "BCG", "CAG"})

SWITCH_NODE_PAIRS = {
    "AB": "X0001BX0001A",
    "BC": "X0001CX0001B",
    "BG": "XX0006X0001B",
    "AG": "XX0006X0001A",
    "CG": "XX0006X0001C",
    "CA": "X0001CX0001A",
}

LEGACY_SWITCH_PAIRS = frozenset(SWITCH_NODE_PAIRS.values())
PHASE_NODES = {phase: f"X0001{phase}" for phase in "ABC"}
FAULT_NODES = {phase: f"XF000{phase}" for phase in "ABC"}
STAR_NODE = "XFSTAR"


@dataclass(frozen=True)
class FaultParameters:
    fault_class: str
    rfault_ohm: float
    incidence_angle_deg: float
    frequency_hz: float = 60.0
    prefault_cycles: int = 5

    @property
    def tclose_s(self) -> float:
        return self.prefault_cycles / self.frequency_hz + (
            self.incidence_angle_deg / 360.0 / self.frequency_hz
        )


def _format_field(value: float, width: int, decimals: int) -> str:
    text = f"{value:.{decimals}f}".rstrip("0").rstrip(".")
    if text.startswith("0."):
        text = text[1:]
    if len(text) > width:
        text = f"{value:.{max(1, width - 5)}g}"
    if len(text) > width:
        raise ValueError(f"Valor {value} não cabe no campo ATP de {width} colunas")
    return f"{text:>{width}}"


def _fault_network(fault_class: str) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    """Retorna ramos R e chaves da rede canônica de falta.

    Cada ramo da primeira lista recebe exatamente Rfault. A segunda lista liga
    a fase ao lado correspondente do resistor somente em tclose.
    """
    phases = FAULT_SWITCHES[fault_class]
    if fault_class in GROUNDED_FAULTS:
        if len(phases) == 1:
            phase = phases[0]
            branches = [(FAULT_NODES[phase], "")]
            switches = [(PHASE_NODES[phase], FAULT_NODES[phase])]
        else:
            # Falta dupla fase-terra (ABG/BCG/CAG): no modelo classico de
            # componentes simetricas (fault LLG) as fases envolvidas ficam
            # curto-circuitadas entre si no ponto da falta, e esse ponto
            # comum tem uma unica impedancia ate a terra -- mesma topologia
            # usada em desenhos manuais no ATPDraw. A versao anterior dava a
            # cada fase seu proprio resistor independente ate a terra, sem
            # ligar as fases entre si, o que diverge do modelo padrao.
            shared_node = FAULT_NODES[phases[0]]
            branches = [(shared_node, "")]
            switches = [(PHASE_NODES[phase], shared_node) for phase in phases]
    elif fault_class == "ABC":
        branches = [(FAULT_NODES[phase], STAR_NODE) for phase in phases]
        switches = [(PHASE_NODES[phase], FAULT_NODES[phase]) for phase in phases]
    else:
        first, second = phases
        branches = [(FAULT_NODES[first], PHASE_NODES[second])]
        switches = [(PHASE_NODES[first], FAULT_NODES[first])]
    return branches, switches


def _branch_card(skeleton: str, node1: str, node2: str, resistance: float) -> str:
    return (
        skeleton[:2] + f"{node1:<6}{node2:<6}" + skeleton[14:26]
        + _format_field(resistance, 6, 4) + skeleton[32:]
    )


def _switch_card(skeleton: str, node1: str, node2: str, tclose: float) -> str:
    return (
        skeleton[:2] + f"{node1:<6}{node2:<6}"
        + _format_field(tclose, 10, 7)
        + _format_field(2.0, 10, 7)
        + skeleton[34:]
    )


def configure_fault_deck(template: str, params: FaultParameters) -> str:
    fault_class = params.fault_class.upper()
    if fault_class not in FAULT_SWITCHES:
        raise ValueError(f"Classe de falta inválida: {params.fault_class}")
    if not 0.01 <= params.rfault_ohm <= 3000.0:
        raise ValueError("Rfault deve estar entre 0,01 e 3000 ohms")
    if not 0.0 <= params.incidence_angle_deg < 360.0:
        raise ValueError("O ângulo deve estar no intervalo [0, 360) graus")
    if params.frequency_hz != 60.0 or params.prefault_cycles != 5:
        raise ValueError("Esta campanha foi congelada em 60 Hz e cinco ciclos pré-falta")

    branches, switches = _fault_network(fault_class)
    legacy_switches_seen = 0
    network_switches_written = False
    rfault_seen = 0
    output: list[str] = []
    for line in template.splitlines():
        if len(line) == 80 and line[2:14] in LEGACY_SWITCH_PAIRS:
            legacy_switches_seen += 1
            if not network_switches_written:
                output.extend(
                    _switch_card(line, node1, node2, params.tclose_s)
                    for node1, node2 in switches
                )
                network_switches_written = True
            continue
        elif len(line) == 80 and line[2:8] == "XX0006" and not line[8:14].strip():
            rfault_seen += 1
            output.extend(
                _branch_card(line, node1, node2, params.rfault_ohm)
                for node1, node2 in branches
            )
            continue
        output.append(line)

    if legacy_switches_seen != len(LEGACY_SWITCH_PAIRS) or rfault_seen != 1:
        raise ValueError(
            "Template incompatível: "
            f"chaves de falta={legacy_switches_seen}/{len(LEGACY_SWITCH_PAIRS)}, "
            f"ramos Rfault={rfault_seen}"
        )
    return "\n".join(output) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--template", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fault-class", choices=sorted(FAULT_SWITCHES), required=True)
    parser.add_argument("--rfault-ohm", type=float, required=True)
    parser.add_argument("--angle-deg", type=float, required=True)
    args = parser.parse_args()
    configured = configure_fault_deck(
        args.template.read_text(encoding="latin-1"),
        FaultParameters(args.fault_class, args.rfault_ohm, args.angle_deg),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(configured, encoding="latin-1", newline="\r\n")
    print(f"ATP configurado: {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

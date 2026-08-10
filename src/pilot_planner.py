"""Planeja o piloto pareado e bloqueia treino, validação e teste antecipadamente."""

from __future__ import annotations

import argparse
import csv
import itertools
import random
from dataclasses import asdict, dataclass
from pathlib import Path

from fault_case_generator import FAULT_SWITCHES


SEED = 20260802
TRAIN_DISTANCES = (1, 5, 10, 30, 50, 80, 120, 160, 200, 230, 300, 400, 500)
TRAIN_RFAULTS = (0.01, 0.1, 0.5, 1, 5, 10, 25, 50, 75, 100)
TRAIN_ANGLES = (0, 30, 60, 90, 180, 210, 270, 330)
TRAIN_REMOTES = (50, 150, 300, 500)

VALIDATION_DISTANCES = (20, 40, 60, 100, 140, 220, 320, 450)
VALIDATION_RFAULTS = (0.02, 0.2, 2, 20)
VALIDATION_ANGLES = (120, 240)
VALIDATION_REMOTES = (200, 75)

TEST_DISTANCES = (15, 35, 75, 125, 175, 275, 375)
TEST_RFAULTS = (0.05, 0.25, 0.75, 2.5, 7.5, 17.5, 37.5)
TEST_ANGLES = (150, 300)
TEST_REMOTES = (100, 225, 400)


@dataclass(frozen=True)
class Scenario:
    scenario_id: str
    split: str
    distance_km: float
    remote_length_km: float
    rfault_ohm: float
    incidence_angle_deg: float


@dataclass(frozen=True)
class PlannedRun:
    run_id: str
    scenario_id: str
    split: str
    fault_class: str
    distance_km: float
    remote_length_km: float
    rfault_ohm: float
    incidence_angle_deg: float


def _sample_training() -> list[tuple[float, float, float, float]]:
    candidates = [
        item for item in itertools.product(
            TRAIN_DISTANCES, TRAIN_REMOTES, TRAIN_RFAULTS, TRAIN_ANGLES
        ) if item[0] + item[1] >= 100
    ]
    rng = random.Random(SEED)
    rng.shuffle(candidates)
    # Trocas determinísticas garantem a presença das distâncias curtas críticas.
    mandatory = [
        (1, 150, 0.01, 0),
        (5, 150, 100, 330),
        (500, 50, 10, 90),
    ]
    selected = [item for item in candidates if item not in mandatory][:32]
    return mandatory + selected


def build_scenarios() -> list[Scenario]:
    raw: list[tuple[str, float, float, float, float]] = []
    raw.extend(("train", *item) for item in _sample_training())
    for i in range(8):
        raw.append((
            "validation",
            VALIDATION_DISTANCES[i],
            VALIDATION_REMOTES[(i * 3) % len(VALIDATION_REMOTES)],
            VALIDATION_RFAULTS[(i * 3) % len(VALIDATION_RFAULTS)],
            VALIDATION_ANGLES[i % len(VALIDATION_ANGLES)],
        ))
    for i in range(7):
        raw.append((
            "test_unseen", TEST_DISTANCES[i],
            TEST_REMOTES[(i * 2) % len(TEST_REMOTES)], TEST_RFAULTS[i],
            TEST_ANGLES[i % len(TEST_ANGLES)],
        ))
    return [
        Scenario(f"scenario_{index:03d}", *values)
        for index, values in enumerate(raw, start=1)
    ]


def build_runs() -> list[PlannedRun]:
    runs: list[PlannedRun] = []
    index = 1
    for scenario in build_scenarios():
        for fault_class in FAULT_SWITCHES:
            runs.append(PlannedRun(
                f"run_{index:06d}", scenario.scenario_id, scenario.split,
                fault_class, scenario.distance_km, scenario.remote_length_km,
                scenario.rfault_ohm, scenario.incidence_angle_deg,
            ))
            index += 1
    return runs


def write_plan(path: Path) -> None:
    rows = [asdict(run) for run in build_runs()]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    write_plan(args.output)
    print(f"Plano bloqueado: {args.output} (50 cenários, 500 execuções)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

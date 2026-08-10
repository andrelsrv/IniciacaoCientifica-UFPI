"""Esquema e auditoria do manifesto da campanha ATPDraw."""

from __future__ import annotations

import csv
import math
import re
from dataclasses import dataclass, fields
from pathlib import Path


FAULT_CLASSES = frozenset(
    {"AG", "BG", "CG", "AB", "BC", "CA", "ABG", "BCG", "CAG", "ABC"}
)
SPLITS = frozenset({"train", "validation", "test_combination", "test_unseen"})
RUN_ID_PATTERN = re.compile(r"^run_[0-9]{6}$")


class ManifestError(ValueError):
    """Indica manifesto invalido ou vazamento entre divisoes."""


def _optional_float(value: str) -> float | None:
    value = value.strip()
    return None if value == "" else float(value)


@dataclass(frozen=True)
class ManifestRow:
    run_id: str
    file_path: str
    split: str
    fault_class: str
    distance_km: float
    rfault_ohm: float
    incidence_angle_deg: float
    remote_length_km: float
    snr_db: float | None = None
    gain_error_pct: float | None = None
    sync_error_us: float | None = None
    source_voltage_error_pct: float | None = None
    source_impedance_error_pct: float | None = None

    @property
    def physical_key(self) -> tuple[object, ...]:
        return (
            self.fault_class,
            self.distance_km,
            self.rfault_ohm,
            self.incidence_angle_deg,
            self.remote_length_km,
            self.snr_db,
            self.gain_error_pct,
            self.sync_error_us,
            self.source_voltage_error_pct,
            self.source_impedance_error_pct,
        )

    def validate(self) -> None:
        if not RUN_ID_PATTERN.fullmatch(self.run_id):
            raise ManifestError(f"run_id invalido: {self.run_id!r}.")
        if not self.file_path.strip():
            raise ManifestError(f"{self.run_id}: file_path vazio.")
        if self.split not in SPLITS:
            raise ManifestError(f"{self.run_id}: split invalido: {self.split!r}.")
        if self.fault_class not in FAULT_CLASSES:
            raise ManifestError(f"{self.run_id}: classe invalida: {self.fault_class!r}.")
        if not 1.0 <= self.distance_km <= 600.0:
            raise ManifestError(f"{self.run_id}: distance_km fora de 1..600.")
        if not 0.01 <= self.rfault_ohm <= 3000.0:
            raise ManifestError(f"{self.run_id}: rfault_ohm fora de 0,01..3000.")
        if not 0.0 <= self.incidence_angle_deg < 360.0:
            raise ManifestError(f"{self.run_id}: incidence_angle_deg fora de 0..<360.")
        if not 0.0 < self.remote_length_km <= 600.0:
            raise ManifestError(f"{self.run_id}: remote_length_km fora de 0..600.")
        if self.distance_km + self.remote_length_km < 100.0:
            raise ManifestError(f"{self.run_id}: comprimento total inferior a 100 km.")
        if self.snr_db is not None and self.snr_db not in {30.0, 40.0, 60.0}:
            raise ManifestError(f"{self.run_id}: snr_db deve ser 30, 40, 60 ou vazio.")
        bounded = (
            ("gain_error_pct", self.gain_error_pct, 1.0),
            ("sync_error_us", self.sync_error_us, 1.0),
            ("source_voltage_error_pct", self.source_voltage_error_pct, 5.0),
            ("source_impedance_error_pct", self.source_impedance_error_pct, 10.0),
        )
        for name, value, limit in bounded:
            if value is not None and (not math.isfinite(value) or abs(value) > limit):
                raise ManifestError(f"{self.run_id}: {name} fora de +/-{limit}.")


MANIFEST_COLUMNS = tuple(field.name for field in fields(ManifestRow))


def read_manifest(path: str | Path) -> list[ManifestRow]:
    source = Path(path)
    with source.open("r", encoding="utf-8-sig", newline="") as stream:
        reader = csv.DictReader(stream)
        if tuple(reader.fieldnames or ()) != MANIFEST_COLUMNS:
            raise ManifestError(
                f"Cabecalho invalido. Esperado={MANIFEST_COLUMNS!r}; "
                f"recebido={tuple(reader.fieldnames or ())!r}."
            )
        rows = [
            ManifestRow(
                run_id=item["run_id"].strip(),
                file_path=item["file_path"].strip(),
                split=item["split"].strip(),
                fault_class=item["fault_class"].strip().upper(),
                distance_km=float(item["distance_km"]),
                rfault_ohm=float(item["rfault_ohm"]),
                incidence_angle_deg=float(item["incidence_angle_deg"]),
                remote_length_km=float(item["remote_length_km"]),
                snr_db=_optional_float(item["snr_db"]),
                gain_error_pct=_optional_float(item["gain_error_pct"]),
                sync_error_us=_optional_float(item["sync_error_us"]),
                source_voltage_error_pct=_optional_float(item["source_voltage_error_pct"]),
                source_impedance_error_pct=_optional_float(
                    item["source_impedance_error_pct"]
                ),
            )
            for item in reader
        ]
    if not rows:
        raise ManifestError("Manifesto vazio.")
    validate_manifest(rows)
    return rows


def validate_manifest(rows: list[ManifestRow]) -> None:
    run_ids: set[str] = set()
    paths: set[str] = set()
    keys: dict[tuple[object, ...], str] = {}
    for row in rows:
        row.validate()
        normalized_path = str(Path(row.file_path)).casefold()
        if row.run_id in run_ids:
            raise ManifestError(f"run_id duplicado: {row.run_id}.")
        if normalized_path in paths:
            raise ManifestError(f"file_path duplicado: {row.file_path}.")
        if row.physical_key in keys:
            raise ManifestError(
                f"Cenario fisico duplicado: {row.run_id} e {keys[row.physical_key]}."
            )
        run_ids.add(row.run_id)
        paths.add(normalized_path)
        keys[row.physical_key] = row.run_id
    audit_split_leakage(rows)


def audit_split_leakage(rows: list[ManifestRow]) -> None:
    train = [row for row in rows if row.split == "train"]
    train_keys = {row.physical_key for row in train}
    for row in rows:
        if row.split != "train" and row.physical_key in train_keys:
            raise ManifestError(f"{row.run_id}: combinacao fisica tambem existe no treino.")

    unseen = [row for row in rows if row.split == "test_unseen"]
    if not train or not unseen:
        return
    dimensions = (
        "distance_km",
        "rfault_ohm",
        "incidence_angle_deg",
        "remote_length_km",
    )
    train_values = {name: {getattr(row, name) for row in train} for name in dimensions}
    for row in unseen:
        repeated = [name for name in dimensions if getattr(row, name) in train_values[name]]
        if repeated:
            raise ManifestError(
                f"{row.run_id}: test_unseen reutiliza valores de treino em {repeated}."
            )


def resolve_pl4_path(row: ManifestRow, manifest_path: str | Path) -> Path:
    candidate = Path(row.file_path)
    if not candidate.is_absolute():
        candidate = Path(manifest_path).resolve().parent / candidate
    return candidate.resolve()

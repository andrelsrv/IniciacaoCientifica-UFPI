import unittest

from batch_pipeline import InferenceRun
from manifest import ManifestError, ManifestRow, validate_manifest


def row(**changes) -> ManifestRow:
    values = dict(
        run_id="run_000001",
        file_path="one.pl4",
        split="train",
        fault_class="AG",
        distance_km=230.0,
        rfault_ohm=0.01,
        incidence_angle_deg=0.0,
        remote_length_km=70.0,
    )
    values.update(changes)
    return ManifestRow(**values)


class ManifestTests(unittest.TestCase):
    def test_accepts_valid_row(self) -> None:
        validate_manifest([row()])

    def test_rejects_short_total_line(self) -> None:
        with self.assertRaisesRegex(ManifestError, "inferior a 100 km"):
            validate_manifest([row(distance_km=1.0, remote_length_km=50.0)])

    def test_rejects_unseen_value_reused_from_train(self) -> None:
        rows = [
            row(),
            row(
                run_id="run_000002",
                file_path="two.pl4",
                split="test_unseen",
                fault_class="BG",
                distance_km=230.0,
                rfault_ohm=0.5,
                incidence_angle_deg=30.0,
                remote_length_km=100.0,
            ),
        ]
        with self.assertRaisesRegex(ManifestError, "distance_km"):
            validate_manifest(rows)

    def test_inference_boundary_has_no_label_or_path(self) -> None:
        self.assertEqual(set(InferenceRun.__dataclass_fields__), {"run_id", "signals"})


if __name__ == "__main__":
    unittest.main()

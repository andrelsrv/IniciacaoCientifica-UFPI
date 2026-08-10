import tempfile
import unittest
from pathlib import Path

from pilot_runner import load_plan, valid_existing_pl4


class PilotRunnerTests(unittest.TestCase):
    def test_rejects_duplicate_ids(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "plan.csv"
            path.write_text(
                "run_id,fault_class\nrun_000001,AG\nrun_000001,BG\n",
                encoding="utf-8",
            )
            with self.assertRaises(ValueError):
                load_plan(path)

    def test_missing_pl4_is_not_valid(self):
        self.assertFalse(valid_existing_pl4(Path("does_not_exist.pl4")))


if __name__ == "__main__":
    unittest.main()

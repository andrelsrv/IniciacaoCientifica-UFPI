import unittest

from simulation_generator import SimulationParameters, validate_parameters


class SimulationGeneratorTests(unittest.TestCase):
    def test_accepts_valid_case(self):
        validate_parameters(SimulationParameters("run_000002", "ABG", 120, 80, 10, 60))

    def test_rejects_nonopaque_run_id(self):
        with self.assertRaises(ValueError):
            validate_parameters(SimulationParameters("AG_120km", "AG", 120, 80, 1, 0))

    def test_rejects_short_total_line(self):
        with self.assertRaises(ValueError):
            validate_parameters(SimulationParameters("run_000003", "AG", 5, 50, 1, 0))


if __name__ == "__main__":
    unittest.main()

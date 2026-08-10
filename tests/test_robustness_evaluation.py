import unittest

import numpy as np

from robustness_evaluation import Perturbation, perturb
from signal_io import SignalData


class RobustnessEvaluationTests(unittest.TestCase):
    def test_perturbation_is_deterministic_and_preserves_shape(self):
        time = np.arange(100, dtype=float) * 1e-6
        signals = SignalData(time, np.ones((100, 12)))
        condition = Perturbation("combined", 30, 1, 1)
        first = perturb(signals, "run_000001", condition)
        second = perturb(signals, "run_000001", condition)
        self.assertEqual(first.values.shape, signals.values.shape)
        np.testing.assert_array_equal(first.values, second.values)
        self.assertFalse(np.array_equal(first.values, signals.values))


if __name__ == "__main__":
    unittest.main()

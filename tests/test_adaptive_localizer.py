import unittest

import numpy as np

from adaptive_localizer import estimate_prefault_snr_db
from signal_io import SignalData


class AdaptiveLocalizerTests(unittest.TestCase):
    def test_estimates_synthetic_snr(self):
        rng = np.random.default_rng(42)
        time = np.arange(150001) * 1e-6
        base = np.sin(2 * np.pi * 60 * time)[:, None] * np.ones((1, 12))
        noise_rms = np.sqrt(np.mean(base**2)) / (10 ** (40 / 20))
        signals = SignalData(time, base + rng.normal(size=base.shape) * noise_rms)
        self.assertAlmostEqual(estimate_prefault_snr_db(signals), 40, delta=0.5)


if __name__ == "__main__":
    unittest.main()

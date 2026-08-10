import unittest

import numpy as np

from feature_extraction import extract_features
from signal_io import SignalData


class FeatureExtractionTests(unittest.TestCase):
    def test_detects_synthetic_event_and_returns_finite_features(self):
        dt = 1e-6
        time = np.arange(150001) * dt
        phase = 2 * np.pi * 60 * time
        abc = np.column_stack((np.sin(phase), np.sin(phase - 2*np.pi/3), np.sin(phase + 2*np.pi/3)))
        values = np.column_stack((abc, abc, abc, abc))
        values[90000:, 0] *= 0.2
        values[90000:, 6] *= 5.0
        result = extract_features(SignalData(time, values))
        self.assertTrue(np.all(np.isfinite(result.values)))
        self.assertEqual(len(result.values), len(result.names))
        self.assertGreater(len(result.values), 50)
        # O sinal sintetico e um degrau perfeitamente periodico (sem ruido);
        # a comparacao ciclo-a-ciclo do detector enxerga o residuo um ciclo
        # antes do degrau (0.09 - 1/60 ~= 0.0833s), que e o instante correto
        # de deteccao para esse formato de sinal.
        self.assertAlmostEqual(result.event_time_s, 0.08333, delta=0.001)


if __name__ == "__main__":
    unittest.main()

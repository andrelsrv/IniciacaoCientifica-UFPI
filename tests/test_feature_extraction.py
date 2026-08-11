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
        # O sinal sintetico tem so 150001 amostras (0.15s), mais curto que a
        # SEARCH_END_S atual (0.475s, calibrada para o Tmax real de 0.5s do
        # template ATP) — a busca fica limitada ao fim do sinal disponivel,
        # o que agora captura o degrau real em ~0.09s (em vez do residuo um
        # ciclo antes, que aparecia quando a janela de busca era mais curta).
        self.assertAlmostEqual(result.event_time_s, 0.09, delta=0.001)


if __name__ == "__main__":
    unittest.main()

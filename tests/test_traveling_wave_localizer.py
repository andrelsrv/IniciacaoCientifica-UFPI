import unittest

from traveling_wave_localizer import TravelingWaveConfig


class TravelingWaveLocalizerTests(unittest.TestCase):
    def test_config_keeps_physical_velocity_range(self):
        config = TravelingWaveConfig(velocity_km_per_us=0.299)
        self.assertAlmostEqual(config.velocity_km_per_us * 1000, 299.0)
        self.assertEqual(config.maximum_distance_km, 500.0)
        self.assertGreater(
            config.minimum_terminal_correlation_for_conclusive,
            config.minimum_correlation,
        )


if __name__ == "__main__":
    unittest.main()

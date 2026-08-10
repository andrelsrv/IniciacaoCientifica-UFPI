import unittest

from multiscale_localizer import MultiscaleConfig


class MultiscaleLocalizerTests(unittest.TestCase):
    def test_requires_four_ordered_bands(self):
        config = MultiscaleConfig()
        self.assertEqual(len(config.cutoff_hz), 4)
        self.assertEqual(tuple(sorted(config.cutoff_hz)), config.cutoff_hz)
        self.assertEqual(config.maximum_spread_km, 3.0)


if __name__ == "__main__":
    unittest.main()

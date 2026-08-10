import tempfile
import unittest
from pathlib import Path

import numpy as np

from signal_io import CANONICAL_CHANNELS, SignalData, compare_signals, read_reference_adf


class SignalIoTests(unittest.TestCase):
    def test_reads_reference_adf(self) -> None:
        header = (
            "//ADF file created by \"MC's PlotXY\" program\n"
            "t\tvLoca\tvLocb\tvLocc\tvX0007a\tvX0007b\tvX0007c\t"
            "iX0003aLoca\tiX0003bLocb\tiX0003cLocc\t"
            "iX0007aX0004a\tiX0007bX0004b\tiX0007cX0004c\t\n"
        )
        row = "0\t" + "\t".join(str(value) for value in range(1, 13)) + "\t\n"
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.adf"
            path.write_text(header + row, encoding="utf-8")
            data = read_reference_adf(path)

        self.assertEqual(data.channel_names, CANONICAL_CHANNELS)
        np.testing.assert_allclose(data.values[0], np.arange(1, 13))

    def test_compares_equal_shapes(self) -> None:
        first = SignalData(np.array([0.0, 1.0]), np.zeros((2, 12)))
        second = SignalData(np.array([0.0, 1.1]), np.ones((2, 12)))
        result = compare_signals(first, second)
        self.assertAlmostEqual(result["time_max_abs_s"], 0.1)
        self.assertEqual(result["channels"]["PDT_VA_V"]["max_abs"], 1.0)


if __name__ == "__main__":
    unittest.main()

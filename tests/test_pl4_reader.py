import struct
import tempfile
import unittest
from pathlib import Path

import numpy as np

from pl4_reader import Pl4FormatError, read_pl4


def write_synthetic_pl4(path: Path, *, truncate: bool = False) -> None:
    channels = (b"   4LOCA        ", b"   9LEFT  RIGHT ")
    fixed = bytearray(80)
    stamp = b"\x8411-Nov-18  11.00.00"
    fixed[: len(stamp)] = stamp
    struct.pack_into("<I", fixed, 44, len(channels))
    struct.pack_into("<I", fixed, 48, len(channels) * 2)
    struct.pack_into("<I", fixed, 52, 80 + len(channels) * 16 + 1)
    matrix = np.array(
        [[0.0, 1.0, -1.0], [1e-6, 2.0, -2.0], [2e-6, 3.0, -3.0]],
        dtype="<f4",
    )
    payload = bytes(fixed) + b"".join(channels) + matrix.tobytes()
    payload = bytearray(payload)
    struct.pack_into("<I", payload, 56, len(payload) + 1)
    payload = bytes(payload)
    path.write_bytes(payload[:-1] if truncate else payload)


class Pl4ReaderTests(unittest.TestCase):
    def test_reads_channels_and_samples(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pl4"
            write_synthetic_pl4(path)
            data = read_pl4(path, expected_channels=2)

        self.assertEqual(data.metadata.sample_count, 3)
        self.assertAlmostEqual(data.metadata.timestep_s, 1e-6, places=12)
        self.assertEqual([channel.type_code for channel in data.channels], ["4", "9"])
        self.assertEqual([channel.label for channel in data.channels], ["LOCA", "LEFT  RIGHT"])
        np.testing.assert_allclose(data.values[:, 0], [1.0, 2.0, 3.0])

    def test_rejects_wrong_channel_count(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.pl4"
            write_synthetic_pl4(path)
            with self.assertRaisesRegex(Pl4FormatError, "Esperados 12 canais"):
                read_pl4(path, expected_channels=12)

    def test_rejects_truncated_payload(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "truncated.pl4"
            write_synthetic_pl4(path, truncate=True)
            with self.assertRaisesRegex(Pl4FormatError, "truncado"):
                read_pl4(path)


if __name__ == "__main__":
    unittest.main()

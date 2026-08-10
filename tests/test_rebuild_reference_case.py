import unittest
from pathlib import Path

from rebuild_reference_case import replace_line_inserts


class RebuildReferenceCaseTests(unittest.TestCase):
    def test_replaces_exactly_two_inserts_in_order(self):
        source = "$INSERT, old1.pch\n$INSERT, old2.pch\n/OUTPUT\n"
        result = replace_line_inserts(source, Path("first.pch"), Path("second.pch"))
        lines = result.splitlines()
        self.assertTrue(lines[0].endswith("first.pch"))
        self.assertTrue(lines[1].endswith("second.pch"))

    def test_rejects_unexpected_insert_count(self):
        with self.assertRaises(ValueError):
            replace_line_inserts("$INSERT, only.pch\n", Path("a"), Path("b"))


if __name__ == "__main__":
    unittest.main()

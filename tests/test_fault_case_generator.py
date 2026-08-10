import unittest
from pathlib import Path

from fault_case_generator import FaultParameters, configure_fault_deck


TEMPLATE = Path(r"C:\RESULTPESQUISA\SIMULACAOUSADA.atp")


def cards(deck, section):
    lines = deck.splitlines()
    start = lines.index(section) + 2
    end = next(i for i in range(start, len(lines)) if lines[i].startswith("/"))
    return [line for line in lines[start:end] if len(line) == 80]


class FaultCaseGeneratorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.template = TEMPLATE.read_text(encoding="latin-1")

    def test_ag_has_one_switched_resistor_to_ground(self):
        deck = configure_fault_deck(self.template, FaultParameters("AG", 0.01, 0))
        branches = cards(deck, "/BRANCH")
        switches = cards(deck, "/SWITCH")
        resistor = next(x for x in branches if x[2:8] == "XF000A")
        switch = next(x for x in switches if x[2:14] == "X0001AXF000A")
        self.assertFalse(resistor[8:14].strip())
        self.assertEqual(float(resistor[26:32]), 0.01)
        self.assertAlmostEqual(float(switch[14:24]), 5 / 60, places=7)
        self.assertEqual(float(switch[24:34]), 2.0)

    def test_ab_places_rfault_between_a_and_b(self):
        deck = configure_fault_deck(self.template, FaultParameters("AB", 25, 30))
        branches = cards(deck, "/BRANCH")
        switches = cards(deck, "/SWITCH")
        resistor = next(x for x in branches if x[2:14] == "XF000AX0001B")
        self.assertEqual(float(resistor[26:32]), 25.0)
        self.assertTrue(any(x[2:14] == "X0001AXF000A" for x in switches))

    def test_abg_uses_two_independent_resistors_to_ground(self):
        deck = configure_fault_deck(self.template, FaultParameters("ABG", 1, 30))
        branches = cards(deck, "/BRANCH")
        fault_resistors = [x for x in branches if x[2:8] in {"XF000A", "XF000B"}]
        self.assertEqual(len(fault_resistors), 2)
        self.assertTrue(all(not x[8:14].strip() for x in fault_resistors))
        self.assertTrue(all(float(x[26:32]) == 1.0 for x in fault_resistors))

    def test_abc_uses_three_resistors_in_floating_star(self):
        deck = configure_fault_deck(self.template, FaultParameters("ABC", 10, 330))
        branches = cards(deck, "/BRANCH")
        resistors = [x for x in branches if x[8:14] == "XFSTAR"]
        self.assertEqual({x[2:8] for x in resistors}, {"XF000A", "XF000B", "XF000C"})
        self.assertTrue(all(float(x[26:32]) == 10.0 for x in resistors))

    def test_removes_all_legacy_fault_switches_and_branch(self):
        deck = configure_fault_deck(self.template, FaultParameters("BG", 1, 0))
        self.assertNotIn("XX0006X0001", deck)
        self.assertNotIn("X0001BX0001A", deck)
        self.assertNotIn("  XX0006                    ", deck)

    def test_rejects_invalid_class(self):
        with self.assertRaises(ValueError):
            configure_fault_deck(self.template, FaultParameters("NO_FAULT", 1, 0))


if __name__ == "__main__":
    unittest.main()

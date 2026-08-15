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

    def test_ab_is_bolted_direct_switch_no_resistor(self):
        # Falta fase-fase (sem terra) nao tem caminho ate o Rfault nesta
        # topologia (confirmado contra o template de referencia do
        # ATPDraw) -- e' sempre uma ligacao direta A-B, sem resistor.
        deck = configure_fault_deck(self.template, FaultParameters("AB", 0.0, 30))
        branches = cards(deck, "/BRANCH")
        switches = cards(deck, "/SWITCH")
        self.assertFalse(any(x[2:8] in {"XF000A", "XF000B"} for x in branches))
        self.assertTrue(any(x[2:14] == "X0001AX0001B" for x in switches))

    def test_ab_rejects_nonzero_rfault(self):
        with self.assertRaises(ValueError):
            configure_fault_deck(self.template, FaultParameters("AB", 25, 30))

    def test_abg_uses_two_independent_resistors_to_ground(self):
        deck = configure_fault_deck(self.template, FaultParameters("ABG", 1, 30))
        branches = cards(deck, "/BRANCH")
        fault_resistors = [x for x in branches if x[2:8] in {"XF000A", "XF000B"}]
        self.assertEqual(len(fault_resistors), 2)
        self.assertTrue(all(not x[8:14].strip() for x in fault_resistors))
        self.assertTrue(all(float(x[26:32]) == 1.0 for x in fault_resistors))

    def test_abc_is_bolted_star_no_resistor(self):
        # Estrela (3 chaves ate' um no comum), nao triangulo -- fechar as
        # 3 chaves fase-fase entre si simultaneamente cria um loop de
        # chaves ideais que o ATP rejeita (KILL 363).
        deck = configure_fault_deck(self.template, FaultParameters("ABC", 0.0, 330))
        branches = cards(deck, "/BRANCH")
        switches = cards(deck, "/SWITCH")
        self.assertFalse(any(x[2:8] in {"XF000A", "XF000B", "XF000C"} for x in branches))
        star_switches = [x for x in switches if x[8:14] == "XFSTAR"]
        self.assertEqual({x[2:8] for x in star_switches}, {"X0001A", "X0001B", "X0001C"})

    def test_abc_rejects_nonzero_rfault(self):
        with self.assertRaises(ValueError):
            configure_fault_deck(self.template, FaultParameters("ABC", 10, 330))

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

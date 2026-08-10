import unittest

from jmarti_generator import JMartiConfig, render_deck


class JMartiGeneratorTests(unittest.TestCase):
    def test_renders_reproducible_parameters_and_nodes(self):
        deck = render_deck(JMartiConfig(
            230.0, ("LOCA", "LOCB", "LOCC"),
            ("X0001A", "X0001B", "X0001C"),
        ))
        self.assertIn("BRANCH  LOCA  X0001ALOCB  X0001BLOCC  X0001C", deck)
        self.assertIn("450000", deck)
        self.assertIn("      230", deck)
        self.assertIn("$PUNCH\n\nBEGIN NEW DATA CASE", deck)
        frequency_cards = deck.splitlines()[12:15]
        self.assertTrue(all(len(card) == 70 for card in frequency_cards))
        self.assertTrue(all(card[44:52].strip() == "230" for card in frequency_cards))

    def test_rejects_invalid_length(self):
        with self.assertRaises(ValueError):
            render_deck(JMartiConfig(0, ("A", "B", "C"), ("D", "E", "F")))

    def test_rejects_node_longer_than_six_characters(self):
        with self.assertRaises(ValueError):
            render_deck(JMartiConfig(70, ("TOO_LONG", "B", "C"), ("D", "E", "F")))


if __name__ == "__main__":
    unittest.main()

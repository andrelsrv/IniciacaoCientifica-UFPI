import unittest
from collections import Counter, defaultdict

from pilot_planner import build_runs, build_scenarios


class PilotPlannerTests(unittest.TestCase):
    def test_counts_and_pairing(self):
        scenarios = build_scenarios()
        runs = build_runs()
        self.assertEqual(len(scenarios), 50)
        self.assertEqual(len(runs), 500)
        self.assertEqual(Counter(x.split for x in scenarios), {
            "train": 35, "validation": 8, "test_unseen": 7,
        })
        per_scenario = Counter(x.scenario_id for x in runs)
        self.assertTrue(all(count == 10 for count in per_scenario.values()))

    def test_unseen_scalars_do_not_appear_in_training(self):
        grouped = defaultdict(list)
        for scenario in build_scenarios():
            grouped[scenario.split].append(scenario)
        train = grouped["train"]
        test = grouped["test_unseen"]
        for field in (
            "distance_km", "remote_length_km", "rfault_ohm", "incidence_angle_deg"
        ):
            self.assertTrue(
                {getattr(x, field) for x in train}.isdisjoint(
                    {getattr(x, field) for x in test}
                )
            )

    def test_all_lines_have_realistic_minimum_length(self):
        self.assertTrue(all(
            x.distance_km + x.remote_length_km >= 100 for x in build_scenarios()
        ))

    def test_no_duplicate_physical_scenarios_within_split(self):
        keys = [(
            x.split, x.distance_km, x.remote_length_km,
            x.rfault_ohm, x.incidence_angle_deg,
        ) for x in build_scenarios()]
        self.assertEqual(len(keys), len(set(keys)))

    def test_run_ids_are_opaque_and_unique(self):
        ids = [x.run_id for x in build_runs()]
        self.assertEqual(len(ids), len(set(ids)))
        self.assertEqual(ids[0], "run_000001")
        self.assertEqual(ids[-1], "run_000500")


if __name__ == "__main__":
    unittest.main()

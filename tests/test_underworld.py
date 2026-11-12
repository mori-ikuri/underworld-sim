"""Tests for the Underworld simulation.

Run with the standard library only:
    python -m unittest discover -s tests -v
(also compatible with pytest if available)
"""

from __future__ import annotations

import json
import sys
import unittest
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from underworld.config import load_config
from underworld.emotions import EmotionState
from underworld.episodes import detect_episodes
from underworld.simulation import Simulation


def _serialized(events: list[dict]) -> str:
    return "\n".join(json.dumps(e, ensure_ascii=False, sort_keys=True) for e in events)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_log(self):
        config = load_config()
        sim1 = Simulation(config, seed=42)
        sim1.run_days(5)
        sim2 = Simulation(config, seed=42)
        sim2.run_days(5)
        self.assertEqual(_serialized(sim1.events), _serialized(sim2.events))
        self.assertEqual(sim1.state_dict()["individuals"],
                         sim2.state_dict()["individuals"])

    def test_different_seed_diverges(self):
        config = load_config()
        sim1 = Simulation(config, seed=1)
        sim1.run_days(5)
        sim2 = Simulation(config, seed=2)
        sim2.run_days(5)
        self.assertNotEqual(_serialized(sim1.events), _serialized(sim2.events))

    def test_resume_matches_continuous_run(self):
        config = load_config()
        whole = Simulation(config, seed=7)
        whole.run_days(4)
        part = Simulation(config, seed=7)
        part.run_days(2)
        resumed = Simulation.from_state(config, part.state_dict())
        resumed.run_days(2)
        self.assertEqual(part.events + resumed.events, whole.events)


class TestEmotionEngine(unittest.TestCase):
    def setUp(self):
        self.config = load_config()["emotions"]

    def test_decay_half_life(self):
        state = EmotionState(self.config)
        state.set("anger", 0.83)  # excess 0.80 over baseline 0.03
        state.decay(1.5)  # one half-life of anger
        self.assertAlmostEqual(state.get("anger"), 0.03 + 0.40, places=6)

    def test_decay_regresses_to_baseline_only(self):
        state = EmotionState(self.config)
        state.set("affection", 0.0)  # below baseline: must rise toward it
        state.decay(24)
        self.assertAlmostEqual(state.get("affection"), 0.15, places=6)
        state.set("trust", 1.0)
        state.decay(720)  # long enough to (nearly) reach baseline, never below
        self.assertGreaterEqual(state.get("trust"), state.baseline("trust"))

    def test_dominant_and_calm(self):
        state = EmotionState(self.config)
        self.assertEqual(state.dominant()[0], None)  # everything at baseline
        state.apply("anger", 5)  # +0.15
        dim, excess = state.dominant()
        self.assertEqual(dim, "anger")
        self.assertAlmostEqual(excess, 0.15, places=6)


class TestDecisions(unittest.TestCase):
    def setUp(self):
        self.config = load_config()

    def test_survival_overrides_emotional_interrupt(self):
        sim = Simulation(self.config, seed=3)
        a = sim.individuals["A"]
        a.emotions.set("affection", 1.0)  # way over A's interrupt threshold
        a.hunger = 0.95  # ... but starving: survival must win
        dest, reason, _ = sim.decide_destination("A")
        self.assertEqual(reason, "survival_food")
        self.assertIn(dest, [list(s) for s in self.config["world"]["food_spots"]])

    def test_emotional_interrupt_when_not_starving(self):
        sim = Simulation(self.config, seed=3)
        a = sim.individuals["A"]
        b = sim.individuals["B"]
        a.emotions.set("affection", 0.9)
        a.hunger = 0.2
        dest, reason, meta = sim.decide_destination("A")
        self.assertEqual(reason, "interrupt_approach")
        self.assertEqual(dest, b.pos)
        self.assertEqual(meta["emotion"], "affection")

    def test_avoidance_suppresses_interrupt(self):
        sim = Simulation(self.config, seed=3)
        a = sim.individuals["A"]
        a.emotions.set("affection", 0.9)
        a.avoid = True
        _dest, reason, _ = sim.decide_destination("A")
        self.assertEqual(reason, "avoid")


class TestEpisodeDetection(unittest.TestCase):
    def test_reconcile_chain(self):
        events = [
            {"step": 0, "day": 1, "clock": 7, "type": "FIGHT",
             "actor": "A", "target": "B"},
            {"step": 1, "day": 1, "clock": 8, "type": "AVOID",
             "actor": "A", "target": "B", "kind": "start"},
            {"step": 20, "day": 1, "clock": 3, "type": "APOLOGY",
             "actor": "B", "target": "A", "detail": {"result": "accepted"}},
        ]
        episodes = detect_episodes(events)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["pattern"], "reconcile")

    def test_pattern_a_chain(self):
        events = [
            {"step": 0, "day": 1, "clock": 7, "type": "INTERRUPT",
             "actor": "A", "target": "B", "kind": "approach",
             "detail": {"emotion": "anger"}},
            {"step": 2, "day": 1, "clock": 9, "type": "REACT",
             "actor": "A", "target": "B", "kind": "avoided",
             "detail": {"approach_emotion": "anger"}},
        ]
        episodes = detect_episodes(events)
        self.assertEqual(len(episodes), 1)
        self.assertEqual(episodes[0]["pattern"], "A")


class TestLongRun(unittest.TestCase):
    def test_pattern_a_occurs_on_seed_42(self):
        # Regression test for the anger tuning (2026-11-12): with the tuned
        # parameters, individual A -- the patient one, threshold 0.55 --
        # crosses the limit, approaches in anger, gets fled from, and the
        # detector cuts it out as a pattern A episode (day 25, step ~584).
        config = load_config()
        sim = Simulation(config, seed=42)
        sim.run_days(30)
        episodes = detect_episodes(sim.events)
        pattern_a = [e for e in episodes if e["pattern"] == "A"]
        self.assertGreaterEqual(
            len(pattern_a), 1,
            f"expected at least one pattern A episode in 30 days on seed 42, "
            f"got episodes: {episodes}")
        self.assertEqual(pattern_a[0]["actor"], "A")

    def test_thirty_days_produce_key_events(self):
        config = load_config()
        sim = Simulation(config, seed=42)
        sim.run_days(30)
        counts = Counter(e["type"] for e in sim.events)
        for expected in ("FRICTION", "APOLOGY", "FIGHT", "SURVIVAL"):
            self.assertGreaterEqual(
                counts[expected], 1,
                f"expected at least one {expected} in 30 days, got {dict(counts)}")
        # diaries are written daily per individual (template provider)
        self.assertGreaterEqual(counts["DIARY"], 50)


if __name__ == "__main__":
    unittest.main()

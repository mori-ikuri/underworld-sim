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


class TestLogV2(unittest.TestCase):
    """Log v2 (appraisal sample spec v0.7, sections 9.1-9.3, 11.1-11.2)."""

    def test_seed42_regression_without_appraisal_key(self):
        # Section 9.2 regression: default config (all coefficients 1.0,
        # per-person friction), seed 42, 100 days; stripping the added
        # `appraisal` key must reproduce examples/ byte for byte.
        config = load_config()
        sim = Simulation(config, seed=42)
        sim.run_days(100)
        example_path = (Path(__file__).resolve().parent.parent
                        / "examples" / "run-seed42-100d" / "log.jsonl")
        example_lines = example_path.read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(example_lines), len(sim.events))
        for event, line in zip(sim.events, example_lines):
            stripped = {k: v for k, v in event.items() if k != "appraisal"}
            self.assertEqual(json.dumps(stripped, ensure_ascii=False), line)

    def test_appraisal_involved_individuals(self):
        config = load_config()
        sim = Simulation(config, seed=42)
        sim.run_days(30)
        seen = Counter()
        for event in sim.events:
            if "appraisal" not in event:
                continue
            record = event["appraisal"]
            if event["type"] in ("CONTACT", "FRICTION", "FIGHT", "APOLOGY"):
                self.assertEqual(set(record), {"A", "B"}, event)
            if event["type"] in ("NOTICE", "LOST_SIGHT", "REACT"):
                self.assertEqual(set(record), {event["actor"]}, event)
            for rec in record.values():
                self.assertEqual(set(rec["state_before"]),
                                 {"joy", "sadness", "anger", "fear", "surprise",
                                  "affection", "trust", "fatigue"})
                self.assertIn("points_pre", rec)
                self.assertIn("points_post", rec)
                self.assertIn("grudge", rec)
            seen[event["type"]] += 1
        for expected in ("CONTACT", "FRICTION", "FIGHT", "NOTICE"):
            self.assertGreaterEqual(seen[expected], 1)

    def test_tables_recorded_in_appraisal(self):
        # Fix 2: each appraisal record names the delta tables applied
        # to that individual (first-application order, no duplicates);
        # direct point additions (NOTICE) leave the list empty.
        config = load_config()
        sim = Simulation(config, seed=42)
        sim.run_days(30)
        greet_tables = set()
        for event in sim.events:
            if "appraisal" not in event:
                continue
            for rec in event["appraisal"].values():
                self.assertIn("tables", rec)
                self.assertIsInstance(rec["tables"], list)
                self.assertEqual(len(rec["tables"]),
                                 len(set(rec["tables"])))
            if event["type"] == "CONTACT" and event.get("kind") == "greet":
                for rec in event["appraisal"].values():
                    greet_tables.update(rec["tables"])
            if event["type"] == "CONTACT" \
                    and event.get("kind") == "keep_distance":
                all_tables = [t for rec in event["appraisal"].values()
                              for t in rec["tables"]]
                self.assertIn("keep_distance_self", all_tables)
        self.assertTrue(greet_tables & {"greet_warm", "greet_met",
                                        "greet_calm"}, greet_tables)
        self.assertTrue(greet_tables <= {"greet_warm", "greet_met",
                                         "greet_calm"}, greet_tables)

    def test_coefficient_scales_points_only(self):
        config = load_config()
        config["individuals"]["A"]["coefficients"]["anger"] = 1.5
        sim = Simulation(config, seed=3)
        a = sim.individuals["A"]
        sim._begin_capture(["A"])
        delta = sim.apply_points(a, "anger", 4.0)
        record = sim._take_capture()["A"]
        self.assertEqual(record["points_pre"]["anger"], 4.0)
        self.assertEqual(record["points_post"]["anger"], 6.0)
        self.assertAlmostEqual(delta, 6.0 * a.emotions.point_scale, places=9)

    def test_grudge_flag_and_discount_recorded(self):
        config = load_config()
        sim = Simulation(config, seed=3)
        a = sim.individuals["A"]
        a.grudge_until = sim.step_count + 10
        sim._begin_capture(["A"])
        sim.apply_points(a, "affection", 5.0)
        record = sim._take_capture()["A"]
        self.assertTrue(record["grudge"])
        self.assertAlmostEqual(record["points_pre"]["affection"], 2.0)

    def test_friction_strength_mode(self):
        # Spec 9.3-1 / 11.1 (method B; updated from the v0.6 form of
        # this test, where the same anger landed on both sides): the
        # bumped side takes anger (stage points 2/5/8) and trust -2;
        # the side that bumped takes sadness = stage only -- no anger,
        # no trust change. The bumper is recorded in the detail.
        config = load_config()
        config["contact"]["friction_mode"] = "strength"
        sim = Simulation(config, seed=42)
        sim.run_days(30)
        frictions = [e for e in sim.events if e["type"] == "FRICTION"]
        self.assertGreaterEqual(len(frictions), 1)
        for event in frictions:
            stage = event["detail"]["strength_stage"]
            self.assertIn(stage, (1, 2, 3))
            points = event["detail"]["strength_anger_points"]
            self.assertEqual(points, [2, 5, 8][stage - 1])
            self.assertEqual(event["detail"]["strength_sadness_points"],
                             stage)
            bumper = event["detail"]["bumper"]
            self.assertIn(bumper, ("A", "B"))
            bumped = "B" if bumper == "A" else "A"
            rec_bumped = event["appraisal"][bumped]
            self.assertEqual(rec_bumped["points_pre"]["anger"],
                             float(points))
            self.assertEqual(rec_bumped["points_pre"]["trust"], -2.0)
            self.assertNotIn("sadness", rec_bumped["points_pre"])
            rec_bumper = event["appraisal"][bumper]
            self.assertEqual(rec_bumper["points_pre"],
                             {"sadness": float(stage)})
            self.assertIn("friction_bumper", rec_bumper["tables"])
            self.assertIn("friction", rec_bumped["tables"])

    def test_friction_bumper_coin_deterministic(self):
        # Spec 11.2: the bumper coin is drawn from the seeded rng --
        # the same seed reproduces the same (stage, bumper) sequence.
        def run():
            config = load_config()
            config["contact"]["friction_mode"] = "strength"
            sim = Simulation(config, seed=7)
            sim.run_days(30)
            return [(e["detail"]["strength_stage"],
                     e["detail"]["bumper"])
                    for e in sim.events if e["type"] == "FRICTION"]
        first = run()
        self.assertGreaterEqual(len(first), 1)
        self.assertEqual(first, run())

    def test_friction_bumper_roughly_half(self):
        # Spec 11.2: over 100 days in strength mode, the coin lands
        # on each side roughly half of the time.
        config = load_config()
        config["contact"]["friction_mode"] = "strength"
        sim = Simulation(config, seed=42)
        sim.run_days(100)
        bumpers = [e["detail"]["bumper"] for e in sim.events
                   if e["type"] == "FRICTION"]
        self.assertGreaterEqual(len(bumpers), 20)
        share_a = bumpers.count("A") / len(bumpers)
        self.assertGreater(share_a, 0.3)
        self.assertLess(share_a, 0.7)


class TestDatagenConfig(unittest.TestCase):
    """Data-generation config (spec 11.3)."""

    def test_datagen_config_values(self):
        root = Path(__file__).resolve().parent.parent
        config = load_config(root / "config" / "datagen.json")
        self.assertEqual(config["contact"]["friction_mode"], "strength")
        self.assertEqual(config["apology"]["fail_anger"], 0.25)

    def test_default_config_unchanged(self):
        config = load_config()
        self.assertEqual(config["contact"]["friction_mode"], "per_person")
        self.assertEqual(config["apology"]["fail_anger"], 0.5)


if __name__ == "__main__":
    unittest.main()

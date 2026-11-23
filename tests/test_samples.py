"""Tests for the appraisal sample builder (spec v0.7 sections 1-6, 9.4-9.5, 11).

Run with:  python -m unittest discover -s tests -v
The schema test additionally needs the dev-only dependency:
    pip install -r requirements-dev.txt
"""

from __future__ import annotations

import json
import re
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from underworld.decoder import decode, decode_context
from underworld.labels import make_label, round_half_away_from_zero
from underworld.samples import assign_splits, build_samples, combo_signature
from underworld.templates import FRICTION_STRENGTH_ADVERBS, TEMPLATES

COEF_FLAT = {"joy": 1.0, "sadness": 1.0, "anger": 1.0, "fear": 1.0,
             "surprise": 1.0, "affection": 1.0, "trust": 1.0, "fatigue": 1.0}
STATE = {"joy": 0.25, "sadness": 0.05, "anger": 0.03, "fear": 0.05,
         "surprise": 0.0, "affection": 0.3, "trust": 0.4, "fatigue": 0.1}

BANNED_IN_EVENT_TEXT = [
    "喜び", "悲しみ", "怒り", "不安", "好意", "信頼", "疲労", "驚き",
    "怒っ", "喜ん", "悲し", "怖", "恐れ", "イライラ", "好き", "嫌い", "安心",
]


def _record(points=None, grudge=False, state=None, tables=None):
    return {"state_before": dict(state or STATE),
            "points_pre": dict(points or {}),
            "points_post": dict(points or {}),
            "grudge": grudge,
            "tables": list(tables or [])}


def _event(type_, kind=None, actor="A", target="B", appraisal=None,
           detail=None, step=10, day=1):
    event = {"step": step, "day": day, "clock": 7, "type": type_,
             "actor": actor, "target": target}
    if kind is not None:
        event["kind"] = kind
    if detail:
        event["detail"] = detail
    if appraisal is not None:
        event["appraisal"] = appraisal
    return event


def _meta(run_id="r1", coefficients=None, seed=1):
    coefficients = coefficients or {"A": dict(COEF_FLAT), "B": dict(COEF_FLAT)}
    return {"run_id": run_id, "seed": seed, "coefficients": coefficients,
            "thresholds": {"A": 0.55, "B": 0.35},
            "config_hash": "testhash", "generator_commit": "test"}


def _all_combo_events():
    """One synthetic event per type x kind combo, with slot variants."""
    pair = {"A": _record({"affection": 1.0}), "B": _record({"affection": 1.0})}
    events = [
        _event("NOTICE", "spot", appraisal={"A": _record()}),
        _event("NOTICE", "eye_contact", appraisal={"B": _record()}),
        _event("LOST_SIGHT", appraisal={"A": _record()}),
        _event("REACT", "avoided", appraisal={"A": _record({"anger": 5.0})}),
        _event("REACT", "mutual_approach", appraisal={"B": _record()}),
        _event("CONTACT", "greet", appraisal={
            "A": _record({"affection": 2.0, "joy": 1.0},
                         tables=["greet_met"]),
            "B": _record({"affection": 2.0, "joy": 2.0, "trust": 1.0},
                         tables=["greet_warm"])}),
        _event("CONTACT", "greet", appraisal={
            "A": _record({"affection": 1.0, "joy": 1.0, "trust": 1.0},
                         tables=["greet_calm"]),
            "B": _record({"affection": 1.0, "joy": 1.0, "trust": 1.0},
                         tables=["greet_calm"])},
            step=11),
        _event("CONTACT", "together", appraisal=dict(pair),
               detail={"consecutive_contacts": 9,
                       "habituation_factor": 0.25}),
        _event("CONTACT", "keep_distance", appraisal={
            "A": _record({"fear": 1.0, "sadness": 1.0},
                         tables=["keep_distance_self"]),
            "B": _record({"fear": 1.0},
                         tables=["keep_distance_other"])}),
        _event("CONTACT", "keep_distance", appraisal={
            "A": _record({"fear": 1.0},
                         tables=["keep_distance_other"]),
            "B": _record({"fear": 1.0, "sadness": 1.0},
                         tables=["keep_distance_self"])}, step=12),
        _event("FRICTION", appraisal={
            "A": _record({"sadness": 1.0}, tables=["friction_bumper"]),
            "B": _record({"anger": 2.0, "trust": -2.0},
                         tables=["friction"])},
            detail={"strength_stage": 1, "strength_anger_points": 2,
                    "strength_sadness_points": 1, "bumper": "A",
                    "consecutive_contacts": 1}),
        _event("FRICTION", appraisal={
            "A": _record({"anger": 8.0, "trust": -2.0},
                         tables=["friction"]),
            "B": _record({"sadness": 3.0}, tables=["friction_bumper"])},
            detail={"strength_stage": 3, "strength_anger_points": 8,
                    "strength_sadness_points": 3, "bumper": "B",
                    "consecutive_contacts": 1}, step=13),
        _event("FIGHT", appraisal=dict(pair)),
        _event("APOLOGY", actor="B", target="A", appraisal=dict(pair),
               detail={"result": "accepted"}),
        _event("APOLOGY", actor="A", target="B", appraisal=dict(pair),
               detail={"result": "rejected"}, step=14),
    ]
    return events


class TestRounding(unittest.TestCase):
    def test_half_away_from_zero(self):
        cases = [(0.5, 1), (-0.5, -1), (1.5, 2), (-1.5, -2), (2.5, 3),
                 (-2.5, -3), (0.49, 0), (-0.49, 0), (0.0, 0), (3.0, 3),
                 (4.5, 5), (-4.5, -5)]
        for value, expected in cases:
            self.assertEqual(round_half_away_from_zero(value), expected,
                             f"round({value})")


class TestLabelProcedure(unittest.TestCase):
    def test_friction_stage_examples(self):
        # Spec 9.4 / 10.1: stage as pre-coefficient anger points.
        # stage: {coef: expected anger delta}
        expected = {1: {0.5: 1, 1.0: 1, 1.5: 2},
                    2: {0.5: 1, 1.0: 2, 1.5: 3},
                    3: {0.5: 2, 1.0: 3, 1.5: 4}}
        for stage, by_coef in expected.items():
            for coef, anger_expected in by_coef.items():
                coefs = dict(COEF_FLAT, anger=coef, trust=1.0)
                verdict, deltas, _ = make_label(
                    {"anger": 8.0, "trust": -2.0}, coefs,
                    friction_stage=stage)
                self.assertEqual(verdict, "change")
                self.assertEqual(deltas["anger"], anger_expected,
                                 f"stage={stage} coef={coef}")
                self.assertEqual(deltas["trust"], -2)

    def test_bumper_sadness_examples(self):
        # Spec 9.4 / 11.1: the bumper's sadness is the stage itself as
        # pre-coefficient points, so it follows the same worked
        # examples as the bumped side's anger (coef 0.5 / 1.0 / 1.5).
        # No anger or trust dimension is created for the bumper.
        expected = {1: {0.5: 1, 1.0: 1, 1.5: 2},
                    2: {0.5: 1, 1.0: 2, 1.5: 3},
                    3: {0.5: 2, 1.0: 3, 1.5: 4}}
        for stage, by_coef in expected.items():
            for coef, sadness_expected in by_coef.items():
                coefs = dict(COEF_FLAT, sadness=coef)
                verdict, deltas, _ = make_label(
                    {"sadness": float(stage)}, coefs)
                self.assertEqual(verdict, "change")
                self.assertEqual(deltas, {"sadness": sadness_expected},
                                 f"stage={stage} coef={coef}")

    def test_clamps_and_zero_drop(self):
        verdict, deltas, info = make_label({"anger": 9.0}, dict(COEF_FLAT))
        self.assertEqual(deltas, {"anger": 3})  # pre-clamped to 3
        self.assertEqual(info["pre_clamp_hits"], 1)
        verdict, deltas, info = make_label(
            {"anger": 9.0}, dict(COEF_FLAT, anger=1.5))
        self.assertEqual(deltas, {"anger": 4})  # 3*1.5=4.5 -> 5 -> clamp 4
        self.assertEqual(info["post_clamp_hits"], 1)
        verdict, deltas, _ = make_label(
            {"joy": 0.4}, dict(COEF_FLAT, joy=0.5))  # 0.2 -> rounds to 0
        self.assertEqual(verdict, "no_change")
        self.assertEqual(deltas, {})


class TestDecoder(unittest.TestCase):
    def test_no_emotion_words_or_numbers_in_event_text(self):
        events = _all_combo_events()
        for side in ("train", "val", "test"):
            for event in events:
                for perspective in (event.get("appraisal") or {}):
                    text, _tid, _axis = decode(event, perspective, side, "r1")
                    for banned in BANNED_IN_EVENT_TEXT:
                        self.assertNotIn(banned, text,
                                         f"{banned} in {text!r}")
                    self.assertIsNone(re.search(r"[0-9]", text), text)

    def test_grudged_partner_greet_warm_is_smiling(self):
        # Sensei fix 2: B applied greet_warm but holds a grudge, so the
        # warm points are discounted x0.4 and no longer match the
        # undiscounted greet_warm shape. The demeanour must come from
        # the recorded table name, so A still sees B as smiling.
        event = _event("CONTACT", "greet", appraisal={
            "A": _record({"affection": 1.0, "joy": 1.0, "trust": 1.0},
                         tables=["greet_calm"]),
            "B": _record({"affection": 0.8, "joy": 0.8, "trust": 1.0},
                         tables=["greet_warm"])})
        text, _tid, _axis = decode(event, "A", "train", "r1")
        self.assertIn("にこやか", text)
        # And from B's side, A (greet_calm) is plain.
        text_b, _tid, _axis = decode(event, "B", "train", "r1")
        self.assertIn("ふつうの様子", text_b)

    def test_friction_text_shared_by_both_perspectives(self):
        # Spec 11.6: friction is written from an observer's viewpoint
        # with both names and who bumped into whom; one event gets one
        # sentence, whichever perspective the sample answers for --
        # even on sides that carry two or more friction templates.
        event = _event("FRICTION", appraisal={
            "A": _record({"sadness": 2.0}, tables=["friction_bumper"]),
            "B": _record({"anger": 5.0, "trust": -2.0},
                         tables=["friction"])},
            detail={"strength_stage": 2, "strength_anger_points": 5,
                    "strength_sadness_points": 2, "bumper": "A",
                    "consecutive_contacts": 1})
        for side in ("train", "val", "test"):
            side_templates = [t for t in TEMPLATES["friction"]
                              if t["side"] == side]
            self.assertGreaterEqual(len(side_templates), 2, side)
            text_a, tid_a, axis_a = decode(event, "A", side, "r1")
            text_b, tid_b, axis_b = decode(event, "B", side, "r1")
            self.assertEqual(text_a, text_b)
            self.assertEqual(tid_a, tid_b)
            self.assertIn("A", text_a)
            self.assertIn("B", text_a)
            self.assertFalse(axis_a["subject_drop"])
            for banned in ("しまう", "しまっ", "られ", "ふつうに"):
                self.assertNotIn(banned, text_a)

    def _friction_event(self, stage):
        points = {1: 2.0, 2: 5.0, 3: 8.0}[stage]
        return _event("FRICTION", appraisal={
            "A": _record({"sadness": float(stage)},
                         tables=["friction_bumper"]),
            "B": _record({"anger": points, "trust": -2.0},
                         tables=["friction"])},
            detail={"strength_stage": stage,
                    "strength_anger_points": int(points),
                    "strength_sadness_points": stage, "bumper": "A",
                    "consecutive_contacts": 1})

    def test_friction_middle_stage_has_no_strength_word(self):
        # Decision (patch 4): the middle stage carries no strength
        # word at all -- the sentence is just "...ぶつかった。".
        # The light / strong stages keep their words.
        event = self._friction_event(2)
        for side in ("train", "val", "test"):
            text, _tid, _axis = decode(event, "A", side, "r1")
            for word in ("軽く", "強く", "ふつうに"):
                self.assertNotIn(word, text, (side, text))
            self.assertTrue(text.endswith("ぶつかった。"), text)
        for stage, word in ((1, "軽く"), (3, "強く")):
            text, _tid, _axis = decode(
                self._friction_event(stage), "A", "train", "r1")
            self.assertIn(word, text)

    def test_friction_context_has_no_banned_or_strength_words(self):
        # recent_context friction uses the same shape: no middle
        # adverb anywhere, and no strength word at the middle stage.
        for stage in (1, 2, 3):
            text = decode_context(self._friction_event(stage))
            self.assertIsNotNone(text)
            for banned in ("しまう", "しまっ", "られ", "ふつうに"):
                self.assertNotIn(banned, text, (stage, text))
            if stage == 2:
                for word in ("軽く", "強く"):
                    self.assertNotIn(word, text)

    def test_friction_texts_have_no_chikazuku(self):
        # Audit fix (patch 4): friction sentences must not use the
        # word "近づ" -- most frictions happen while the two are
        # already next to each other with contact continuing, and
        # "近づく" is REACT's word, so it would read as a different
        # event mixed in. Checked on both the event text (every
        # friction template, every stage, plus the decoded text of
        # each side) and the recent_context friction form.
        for stage in (1, 2, 3):
            strength = FRICTION_STRENGTH_ADVERBS[stage]
            for t in TEMPLATES["friction"]:
                text = t["pattern"].format(
                    bumper="A", bumped="B", strength=strength)
                self.assertNotIn("近づ", text, (stage, t["id"], text))
            event = self._friction_event(stage)
            for side in ("train", "val", "test"):
                text, _tid, _axis = decode(event, "A", side, "r1")
                self.assertNotIn("近づ", text, (stage, side, text))
            context = decode_context(event)
            self.assertIsNotNone(context)
            self.assertNotIn("近づ", context, (stage, context))

    def test_friction_texts_disjoint_across_sides_at_every_stage(self):
        # At every stage, no friction sentence of one split side may
        # equal a sentence of another side (the middle stage has no
        # strength word, so the per-side wording itself must differ).
        for stage in (1, 2, 3):
            strength = FRICTION_STRENGTH_ADVERBS[stage]
            texts_by_side = {}
            for side in ("train", "val", "test"):
                texts = {
                    t["pattern"].format(bumper="A", bumped="B",
                                        strength=strength)
                    for t in TEMPLATES["friction"] if t["side"] == side}
                texts_by_side[side] = texts
            sides = list(texts_by_side)
            for i, side in enumerate(sides):
                for other in sides[i + 1:]:
                    self.assertFalse(
                        texts_by_side[side] & texts_by_side[other],
                        (stage, side, other))

    def test_template_choice_ignores_perspective_state(self):
        event = _event("NOTICE", "spot", appraisal={"A": _record()})
        text1, tid1, _ = decode(event, "A", "train", "r1")
        altered = _event("NOTICE", "spot", appraisal={
            "A": _record(state=dict(STATE, affection=0.99, anger=0.9))})
        text2, tid2, _ = decode(altered, "A", "train", "r1")
        self.assertEqual(tid1, tid2)
        self.assertEqual(text1, text2)

    def test_templates_cover_each_side(self):
        for combo, templates in TEMPLATES.items():
            self.assertGreaterEqual(len(templates), 3, combo)
            sides = {t["side"] for t in templates}
            self.assertEqual(sides, {"train", "val", "test"}, combo)
            for template in templates:
                self.assertNotIn(template["id"].split("/")[0],
                                 ("react_avoided_affection",))


class TestBuildSamples(unittest.TestCase):
    def test_state_before_and_streak_rules(self):
        events = _all_combo_events()
        samples, stats, _ = build_samples(events, _meta(), "train")
        by_type = {}
        for sample in samples:
            src = sample["provenance"]["source_event"]
            by_type.setdefault((src["type"], src["kind"]), []).append(sample)
        # NOTICE always carries state_before (spec section 3 list)
        for sample in by_type[("NOTICE", "spot")]:
            self.assertIn("state_before", sample["input"])
        # FIGHT carries it only under the 20% context treatment
        for sample in by_type[("FIGHT", None)]:
            if sample["provenance"].get("with_context_fields"):
                self.assertIn("state_before", sample["input"])
                self.assertIn("recent_context", sample["input"])
            else:
                self.assertNotIn("state_before", sample["input"])
        # contact_streak is capped at 6 ("6 or more")
        for sample in by_type[("CONTACT", "together")]:
            self.assertEqual(sample["input"]["contact_streak"], 6)

    def test_grudge_samples_excluded(self):
        events = [_event("FIGHT", appraisal={
            "A": _record({"anger": 9.0}, grudge=True),
            "B": _record({"anger": 9.0})})]
        samples, stats, _ = build_samples(events, _meta(), "train")
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0]["input"]["perspective"], "B")
        self.assertEqual(stats["grudge_excluded"], 1)

    def test_friction_bumper_and_bumped_labels(self):
        # End to end (spec 11.1 / 9.4): A bumped into B at stage 2.
        # A's label is sadness only (the recorded stage points, no
        # anger, no trust); B's label is the stage as anger plus
        # trust -2. Both samples share one event text (spec 11.6).
        events = [_event("FRICTION", appraisal={
            "A": _record({"sadness": 2.0}, tables=["friction_bumper"]),
            "B": _record({"anger": 5.0, "trust": -2.0},
                         tables=["friction"])},
            detail={"strength_stage": 2, "strength_anger_points": 5,
                    "strength_sadness_points": 2, "bumper": "A",
                    "consecutive_contacts": 1})]
        samples, _, _ = build_samples(events, _meta(), "train")
        self.assertEqual(len(samples), 2)
        by_perspective = {s["input"]["perspective"]: s for s in samples}
        self.assertEqual(by_perspective["A"]["output"]["deltas"],
                         {"sadness": 2})
        self.assertEqual(by_perspective["B"]["output"]["deltas"],
                         {"anger": 2, "trust": -2})
        self.assertEqual(by_perspective["A"]["input"]["event_text"],
                         by_perspective["B"]["input"]["event_text"])
        for sample in samples:
            self.assertEqual(sample["spec_version"], "0.7")
            self.assertEqual(sample["label_version"], "rule-2")
            self.assertEqual(
                sample["provenance"]["decoder_version"], "0.3.0")

    def test_orientation(self):
        events = [_event("REACT", "avoided",
                         appraisal={"A": _record({"anger": 5.0})})]
        samples, _, _ = build_samples(events, _meta(), "train")
        sample = samples[0]
        self.assertEqual(sample["input"]["perspective"], "A")
        self.assertEqual(sample["input"]["actor"], "B")
        self.assertEqual(sample["input"]["target"], "A")
        self.assertEqual(sample["input"]["target_category"], "相棒へ")
        self.assertEqual(sample["output"]["deltas"], {"anger": 3})


class TestSplit(unittest.TestCase):
    def test_ratios_and_combo_holdout(self):
        metas = []
        for i in range(10):
            coefs = {"A": dict(COEF_FLAT, anger=0.5 + 0.5 * (i % 3)),
                     "B": dict(COEF_FLAT)}
            metas.append(_meta(run_id=f"r{i}", coefficients=coefs))
        # r0 and r3 share the exact same combination
        metas[3]["coefficients"] = metas[0]["coefficients"]
        assignment = assign_splits(metas)
        # Combinations are assigned whole (groups of 4/3/3 runs), so
        # the side sizes deviate from the 8/1/1 ratio by group size --
        # that deviation is accepted (sensei fix 1); what must hold is
        # that no combination straddles sides and no side is empty.
        sides = list(assignment.values())
        self.assertEqual(sorted([sides.count("train"), sides.count("val"),
                                 sides.count("test")]), [3, 3, 4])
        self.assertEqual(assignment["r0"], assignment["r3"])

    def _metas_for_group_sizes(self, sizes):
        metas = []
        for gi, size in enumerate(sizes):
            coefs = {"A": dict(COEF_FLAT, anger=0.5 * (gi + 1)),
                     "B": dict(COEF_FLAT)}
            for k in range(size):
                metas.append(_meta(run_id=f"g{gi}r{k}",
                                   coefficients=coefs))
        return metas

    def test_combo_groups_never_split_3_3_4(self):
        metas = self._metas_for_group_sizes([3, 3, 4])
        assignment = assign_splits(metas)
        by_combo = {}
        for meta in metas:
            by_combo.setdefault(
                combo_signature(meta["coefficients"]), set()).add(
                    assignment[meta["run_id"]])
        for sig, sides in by_combo.items():
            self.assertEqual(len(sides), 1, sig)
        # With 3 combinations every side gets exactly one group.
        self.assertEqual(set(assignment.values()),
                         {"train", "val", "test"})

    def test_combo_groups_never_split_5_5(self):
        metas = self._metas_for_group_sizes([5, 5])
        with self.assertWarns(UserWarning):
            assignment = assign_splits(metas)
        by_combo = {}
        for meta in metas:
            by_combo.setdefault(
                combo_signature(meta["coefficients"]), set()).add(
                    assignment[meta["run_id"]])
        for sig, sides in by_combo.items():
            self.assertEqual(len(sides), 1, sig)

    def test_largest_group_assigned_first(self):
        # Patch 3: groups are placed largest first, so the size-4
        # group lands on the side with the largest deficit (train,
        # target 4 after the minimum-side adjustment) instead of
        # being pushed into a small side after the small groups.
        metas = self._metas_for_group_sizes([4, 1, 1])
        assignment = assign_splits(metas)
        for k in range(4):
            self.assertEqual(assignment[f"g0r{k}"], "train")
        self.assertEqual({assignment["g1r0"], assignment["g2r0"]},
                         {"val", "test"})

    def test_warning_when_two_or_fewer_combos(self):
        metas = self._metas_for_group_sizes([5, 5])
        with self.assertWarns(UserWarning):
            assign_splits(metas)

    def test_same_event_perspectives_share_side(self):
        # Sensei fix 3: the A-view and B-view samples built from the
        # same source event belong to the same split side.
        metas = [_meta(run_id="r1"),
                 _meta(run_id="r2", coefficients={
                     "A": dict(COEF_FLAT, anger=0.5), "B": dict(COEF_FLAT)}),
                 _meta(run_id="r3", coefficients={
                     "A": dict(COEF_FLAT, anger=1.5), "B": dict(COEF_FLAT)})]
        assignment = assign_splits(metas)
        side = assignment["r1"]
        events = [_event("FIGHT", appraisal={
            "A": _record({"anger": 5.0}), "B": _record({"anger": 5.0})})]
        samples, _, _ = build_samples(events, metas[0], side)
        self.assertEqual(len(samples), 2)
        self.assertEqual({s["input"]["perspective"] for s in samples},
                         {"A", "B"})
        self.assertEqual(samples[0]["provenance"]["source_event"],
                         samples[1]["provenance"]["source_event"])
        side_by_template = {t["id"]: t["side"]
                            for templates in TEMPLATES.values()
                            for t in templates}
        for sample in samples:
            self.assertEqual(
                side_by_template[sample["provenance"]["template_id"]],
                side)

    def test_side_templates_never_cross(self):
        events = _all_combo_events()
        ids_by_side = {}
        for side in ("train", "val", "test"):
            samples, _, _ = build_samples(events, _meta(), side)
            ids_by_side[side] = {s["provenance"]["template_id"]
                                 for s in samples}
        self.assertFalse(ids_by_side["train"] & ids_by_side["test"])
        self.assertFalse(ids_by_side["train"] & ids_by_side["val"])
        self.assertFalse(ids_by_side["val"] & ids_by_side["test"])


class TestIncompatibleLogs(unittest.TestCase):
    """Sensei fix 4: incompatible logs stop with a reason, not silence."""

    def test_sample_event_without_appraisal_raises(self):
        events = [_event("FIGHT")]  # v1-style: no appraisal key
        with self.assertRaises(ValueError) as ctx:
            build_samples(events, _meta(), "train")
        self.assertIn("appraisal", str(ctx.exception))
        self.assertIn("FIGHT", str(ctx.exception))

    def test_friction_without_strength_stage_raises(self):
        events = [_event("FRICTION", appraisal={
            "A": _record({"anger": 5.0}), "B": _record({"anger": 5.0})},
            detail={"consecutive_contacts": 1})]  # per-person friction
        with self.assertRaises(ValueError) as ctx:
            build_samples(events, _meta(), "train")
        self.assertIn("strength_stage", str(ctx.exception))

    def test_decoder_friction_without_strength_stage_raises(self):
        event = _event("FRICTION", appraisal={
            "A": _record({"anger": 5.0}), "B": _record({"anger": 5.0})},
            detail={"consecutive_contacts": 1})
        with self.assertRaises(ValueError):
            decode(event, "A", "train", "r1")

    def test_friction_without_bumper_raises(self):
        # Spec 11.4: a strength-mode FRICTION without detail.bumper
        # (a log written before method B) stops with a reason, both
        # in the sample builder and in the decoder.
        events = [_event("FRICTION", appraisal={
            "A": _record({"anger": 5.0, "trust": -2.0}),
            "B": _record({"anger": 5.0, "trust": -2.0})},
            detail={"strength_stage": 2, "strength_anger_points": 5,
                    "consecutive_contacts": 1})]
        with self.assertRaises(ValueError) as ctx:
            build_samples(events, _meta(), "train")
        self.assertIn("bumper", str(ctx.exception))
        with self.assertRaises(ValueError) as ctx:
            decode(events[0], "A", "train", "r1")
        self.assertIn("bumper", str(ctx.exception))

    def test_decoder_keep_distance_unknown_who_raises(self):
        event = _event("CONTACT", "keep_distance", appraisal={
            "A": _record({"fear": 1.0}, tables=["keep_distance_other"]),
            "B": _record({"fear": 1.0}, tables=["keep_distance_other"])})
        with self.assertRaises(ValueError):
            decode(event, "A", "train", "r1")


class TestSchema(unittest.TestCase):
    def test_all_built_samples_validate(self):
        try:
            import jsonschema
        except ImportError:
            self.fail("jsonschema is not installed; install the dev "
                      "dependencies first: pip install -r "
                      "requirements-dev.txt")
        schema_path = (Path(__file__).resolve().parent.parent
                       / "docs" / "sample-schema.json")
        schema = json.loads(schema_path.read_text(encoding="utf-8"))
        validator = jsonschema.Draft202012Validator(schema)
        events = _all_combo_events()
        checked = 0
        for side in ("train", "val", "test"):
            samples, _, _ = build_samples(events, _meta(), side)
            for sample in samples:
                errors = list(validator.iter_errors(sample))
                self.assertEqual(errors, [], f"{sample['sample_id']}: "
                                             f"{errors[:1]}")
                checked += 1
        self.assertGreater(checked, 0)


if __name__ == "__main__":
    unittest.main()

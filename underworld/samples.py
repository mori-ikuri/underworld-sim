"""Building appraisal samples from log v2 runs (spec v0.7, sections 1-6, 9.4-9.5, 11).

A "run" is a directory (or in-memory bundle) holding a log v2 `log.jsonl`
plus a `run.json` metadata file:

    {"run_id": "seed42-100d", "seed": 42,
     "coefficients": {"A": {dim: coef, ...}, "B": {...}},
     "thresholds": {"A": 0.55, "B": 0.35},
     "config_hash": "<sha256 of the effective config>",
     "generator_commit": "<git commit of the generator>"}

Split assignment happens on runs first (spec section 4): runs are grouped
by their coefficient combination and each combination is assigned whole
to a single side, so a combination reserved for the test side never
appears in training/validation runs. Side sizes therefore deviate from
the requested ratios by up to a group's size.
"""

from __future__ import annotations

import hashlib
import json
import warnings

from .decoder import DECODER_VERSION, decode, decode_context, is_sample_event
from .emotions import DIMENSIONS
from .labels import make_label

SPEC_VERSION = "0.7"
LABEL_VERSION = "rule-2"
RELATIONSHIP_SIM = "同じ箱庭で暮らす相手。"
TARGET_CATEGORY_SIM = "相棒へ"

# Event types whose labels depend on state_before (spec section 3, frozen
# list in 9.5-3). REACT counts only for kind=avoided; CONTACT only for
# greet / keep_distance.
_STATE_BEFORE_TYPES = {"NOTICE", "LOST_SIGHT"}
_STATE_BEFORE_KINDS = {("REACT", "avoided"),
                       ("CONTACT", "greet"),
                       ("CONTACT", "keep_distance")}

CONTEXT_FRACTION_MODULO = 5  # 1 in 5 samples = the 20% of spec section 6
RECENT_CONTEXT_MAX = 5


def state_before_included(event: dict) -> bool:
    if event["type"] in _STATE_BEFORE_TYPES:
        return True
    return (event["type"], event.get("kind")) in _STATE_BEFORE_KINDS


def _context_bucket(run_id: str, step: int, perspective: str) -> bool:
    digest = hashlib.sha256(
        f"{run_id}:{step}:{perspective}:ctx".encode("utf-8")).hexdigest()
    return int(digest, 16) % CONTEXT_FRACTION_MODULO == 0


def _round_state(state: dict) -> dict:
    return {dim: round(float(state[dim]), 2) for dim in DIMENSIONS}


def combo_signature(coefficients: dict) -> str:
    return json.dumps(coefficients, sort_keys=True, ensure_ascii=False)


def assign_splits(run_metas: list[dict], *, train_ratio: float = 0.8,
                  val_ratio: float = 0.1,
                  split_seed: int = 0) -> dict[str, str]:
    """Assign each run to train / val / test (spec section 4).

    Assignment is by coefficient *combination*: every run sharing a
    combination lands on the same side, whole. Groups are taken
    largest first (ties broken by a deterministic (split_seed) hash
    order, so a large group is placed while every side is still open
    and cannot be pushed into a small side late, skewing the ratios)
    and each is given to the side with the largest remaining run
    deficit against the requested ratios, except that a group is never
    spent on an already-filled side when the groups left are exactly
    enough to give every still empty side one group -- so with at
    least 3 combinations, validation and test are never left empty.
    With 2 or fewer combinations some side unavoidably stays empty;
    that case is reported with a warning. Deterministic in split_seed.
    Returns {run_id: side}.
    """
    groups: dict[str, list[str]] = {}
    for meta in run_metas:
        groups.setdefault(combo_signature(meta["coefficients"]), []).append(
            meta["run_id"])
    if len(groups) <= 2:
        warnings.warn(
            f"assign_splits: only {len(groups)} coefficient "
            f"combination(s) for {len(run_metas)} run(s); one of the "
            f"validation / test sides will be empty",
            UserWarning, stacklevel=2)
    ordered_sigs = sorted(
        groups,
        key=lambda sig: (-len(groups[sig]),
                         hashlib.sha256(
                             f"{split_seed}:{sig}".encode(
                                 "utf-8")).hexdigest()))
    ordered_groups = [sorted(groups[sig]) for sig in ordered_sigs]
    n = sum(len(g) for g in ordered_groups)
    targets = {"train": int(round(n * train_ratio)),
               "val": int(round(n * val_ratio))}
    targets["test"] = n - targets["train"] - targets["val"]
    if len(ordered_groups) >= 3:
        # Keep every side reachable at the run-count level too.
        for side in ("train", "val", "test"):
            if targets[side] < 1:
                donor = max(targets, key=lambda k: targets[k])
                targets[donor] -= 1
                targets[side] = 1
    assigned_counts = {"train": 0, "val": 0, "test": 0}
    assigned_groups = {"train": 0, "val": 0, "test": 0}
    assignment: dict[str, str] = {}
    sides = ("train", "val", "test")
    for idx, group in enumerate(ordered_groups):
        remaining_after = len(ordered_groups) - idx - 1
        candidates = list(sides)
        if len(ordered_groups) >= 3:
            allowed = []
            for side in sides:
                empty_after = sum(
                    1 for k in sides
                    if assigned_groups[k] == 0 and k != side)
                if remaining_after >= empty_after:
                    allowed.append(side)
            if allowed:
                candidates = allowed
        side = max(candidates,
                   key=lambda k: (targets[k] - assigned_counts[k],
                                  -sides.index(k)))
        for run_id in group:
            assignment[run_id] = side
        assigned_counts[side] += len(group)
        assigned_groups[side] += 1
    return assignment


def build_samples(events: list[dict], run_meta: dict, side: str,
                  *, start_index: int = 1,
                  collect_details: bool = False) -> tuple[list[dict], dict, int]:
    """Build samples for one run. Returns (samples, stats, next_index)."""
    run_id = run_meta["run_id"]
    coefficients = run_meta["coefficients"]
    samples: list[dict] = []
    stats: dict = {
        "per_type": {},
        "grudge_excluded": 0,
        "no_appraisal": 0,
        "pre_clamp_hits": 0,
        "post_clamp_hits": 0,
        "label_details": [],
    }
    index = start_index
    day_contexts: dict[int, list[str]] = {}
    for event in events:
        prior_context = day_contexts.setdefault(event["day"], [])
        if is_sample_event(event):
            appraisal = event.get("appraisal")
            if not appraisal:
                # Incompatible log (e.g. v1, without appraisal): refuse
                # loudly instead of silently skipping sample events.
                raise ValueError(
                    f"sample event has no appraisal record: "
                    f"run_id={run_id} type={event['type']} "
                    f"kind={event.get('kind')} step={event['step']} "
                    f"day={event['day']}; samples require log v2")
            if event["type"] == "FRICTION" and (
                    (event.get("detail") or {}).get("strength_stage")
                    not in (1, 2, 3)):
                raise ValueError(
                    f"FRICTION event has no valid "
                    f"detail.strength_stage: run_id={run_id} "
                    f"step={event['step']} day={event['day']} "
                    f"detail={event.get('detail')}; per-person "
                    f"friction logs cannot be built into samples")
            if event["type"] == "FRICTION" and (
                    (event.get("detail") or {}).get("bumper")
                    not in ("A", "B")):
                # Spec 11.4: a strength-mode FRICTION without the
                # bumper field (a log written before method B) stops
                # with a reason, like one without a strength stage.
                raise ValueError(
                    f"FRICTION event has no valid detail.bumper: "
                    f"run_id={run_id} step={event['step']} "
                    f"day={event['day']} detail={event.get('detail')}; "
                    f"strength-mode friction logs without a bumper "
                    f"cannot be built into samples")
            for perspective in sorted(appraisal):
                record = appraisal[perspective]
                if record.get("grudge"):
                    stats["grudge_excluded"] += 1
                    continue
                other = "B" if perspective == "A" else "A"
                stage = None
                if event["type"] == "FRICTION" and perspective != (
                        event.get("detail") or {}).get("bumper"):
                    # Spec 11.1 / 9.4: the stage replaces the recorded
                    # anger points only for the side that was bumped
                    # into. The bumper's side makes no anger points;
                    # its sadness is already the stage value in
                    # points_pre and goes through as recorded.
                    stage = (event.get("detail") or {}).get(
                        "strength_stage")
                verdict, deltas, info = make_label(
                    record.get("points_pre") or {},
                    coefficients[perspective],
                    friction_stage=stage)
                stats["pre_clamp_hits"] += info["pre_clamp_hits"]
                stats["post_clamp_hits"] += info["post_clamp_hits"]
                text, template_id, axis_values = decode(
                    event, perspective, side, run_id)
                inp: dict = {
                    "event_text": text,
                    "perspective": perspective,
                    "actor": other,
                    "target": perspective,
                    "target_category": TARGET_CATEGORY_SIM,
                    "relationship": RELATIONSHIP_SIM,
                }
                attach_context = _context_bucket(
                    run_id, event["step"], perspective)
                if state_before_included(event):
                    inp["state_before"] = _round_state(
                        record["state_before"])
                elif attach_context:
                    inp["state_before"] = _round_state(
                        record["state_before"])
                if event["type"] == "CONTACT" \
                        and event.get("kind") == "together":
                    streak = int((event.get("detail") or {}).get(
                        "consecutive_contacts", 1))
                    inp["contact_streak"] = min(streak, 6)
                if attach_context:
                    inp["recent_context"] = list(
                        prior_context[-RECENT_CONTEXT_MAX:])
                inp["char_profile"] = {
                    dim: float(coefficients[perspective][dim])
                    for dim in DIMENSIONS}
                output: dict = {"verdict": verdict}
                if verdict != "abstain":
                    output["deltas"] = deltas
                provenance: dict = {
                    "generator": "underworld-sim",
                    "generator_commit": run_meta.get(
                        "generator_commit", "…"),
                    "decoder_version": DECODER_VERSION,
                    "seed": run_meta["seed"],
                    "run_id": run_id,
                    "config_hash": run_meta["config_hash"],
                    "template_id": template_id,
                    "axis_values": axis_values,
                    "char_thresholds": {
                        "interrupt": run_meta["thresholds"][perspective]},
                    "source_event": {
                        "type": event["type"],
                        "kind": event.get("kind"),
                        "step": event["step"]},
                }
                if attach_context:
                    provenance["with_context_fields"] = True
                samples.append({
                    "sample_id": f"uw-{index:06d}",
                    "spec_version": SPEC_VERSION,
                    "input": inp,
                    "output": output,
                    "label_source": "table",
                    "label_version": LABEL_VERSION,
                    "source": "sim",
                    "provenance": provenance,
                })
                index += 1
                key = event["type"] + (
                    ":" + event["kind"] if event.get("kind") else "")
                stats["per_type"][key] = stats["per_type"].get(key, 0) + 1
                if collect_details:
                    for dim in DIMENSIONS:
                        stats["label_details"].append({
                            "dim": dim,
                            "coef": float(coefficients[perspective][dim]),
                            "delta": deltas.get(dim, 0),
                            "type": key,
                            "run_id": run_id,
                            "perspective": perspective,
                        })
        rendered = decode_context(event)
        if rendered:
            prior_context.append(rendered)
    return samples, stats, index

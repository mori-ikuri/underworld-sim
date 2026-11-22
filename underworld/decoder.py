"""Decoding symbolic log events into sample text (spec sections 2, 9.5).

Template selection uses only the event type x kind, the split side of the
run, and a stable hash of the event's identity -- never the perspective
individual's state or points, so phrasing cannot leak the label. Visible
facts (greeting demeanour, apology result, friction strength, who took
distance, eye contact vs. spotting) are filled into slots. The greeting
demeanour and the distance-taker are read from the *names* of the delta
tables the simulation recorded in `appraisal.<id>.tables` -- never from
the shape of the points, which the grudge discount can distort.
Incompatible logs fail loudly instead of being silently decoded: a
FRICTION without a strength stage, a greet / keep_distance record
without `tables`, or a keep_distance where neither side's tables name
keep_distance_self all raise ValueError.
"""

from __future__ import annotations

import hashlib

from .templates import (
    CONTEXT_APOLOGY_RESULT,
    CONTEXT_TEMPLATES,
    DEMEANOUR_PLAIN,
    DEMEANOUR_SMILING,
    STRENGTH_PHRASES,
    TEMPLATES,
)

DECODER_VERSION = "0.1.0"

# Event types that become samples (spec 9.5-1).
SAMPLE_TYPES = ("NOTICE", "LOST_SIGHT", "REACT", "CONTACT",
                "FRICTION", "FIGHT", "APOLOGY")


def combo_key(event: dict) -> str:
    key = event["type"].lower()
    if event.get("kind"):
        key += "_" + event["kind"]
    return key


def is_sample_event(event: dict) -> bool:
    if event["type"] not in SAMPLE_TYPES:
        return False
    if event["type"] == "REACT" and event.get("kind") not in (
            "avoided", "mutual_approach"):
        return False
    return True


def _event_ref(event: dict) -> str:
    return (f"type={event.get('type')} kind={event.get('kind')} "
            f"step={event.get('step')}")


def _tables(event: dict, ind_id: str) -> list:
    """Delta-table names recorded for one individual in this event.

    Raises ValueError when the record carries no `tables` list (a log
    written before the field existed): guessing the visible behaviour
    from the points would silently misstate it.
    """
    record = (event.get("appraisal") or {}).get(ind_id) or {}
    tables = record.get("tables")
    if not isinstance(tables, list):
        raise ValueError(
            f"appraisal record for {ind_id} has no 'tables' list "
            f"({_event_ref(event)}); cannot tell which delta tables "
            f"were applied")
    return tables


def _friction_stage(event: dict) -> int:
    stage = (event.get("detail") or {}).get("strength_stage")
    if stage not in STRENGTH_PHRASES:
        raise ValueError(
            f"FRICTION event has no valid detail.strength_stage "
            f"({_event_ref(event)}); per-person friction logs cannot "
            f"be decoded into samples")
    return stage


def _other(ind_id: str) -> str:
    return "B" if ind_id == "A" else "A"


def visible_slots(event: dict, perspective: str) -> dict:
    """Slot values derived from visible facts of the event only."""
    other = _other(perspective)
    slots: dict[str, str] = {"p": perspective, "o": other}
    if event["type"] == "CONTACT" and event.get("kind") == "greet":
        partner_tables = _tables(event, other)
        slots["demeanour"] = (
            DEMEANOUR_SMILING if "greet_warm" in partner_tables
            else DEMEANOUR_PLAIN)
    if event["type"] == "CONTACT" and event.get("kind") == "keep_distance":
        p_self = "keep_distance_self" in _tables(event, perspective)
        o_self = "keep_distance_self" in _tables(event, other)
        if p_self and o_self:
            slots["who"] = f"{perspective}と{other}は互いに"
        elif o_self:
            slots["who"] = f"{other}は{perspective}から"
        elif p_self:
            slots["who"] = f"{perspective}は{other}から"
        else:
            raise ValueError(
                f"keep_distance event names no keep_distance_self "
                f"table for either individual ({_event_ref(event)}); "
                f"cannot tell who took distance")
    if event["type"] == "FRICTION":
        slots["strength"] = STRENGTH_PHRASES[_friction_stage(event)]
    if event["type"] == "APOLOGY":
        slots["ap"] = event["actor"]
        slots["at"] = event["target"]
    return slots


def _pick(candidates: list[dict], run_key: str, event: dict,
          perspective: str, combo: str) -> dict:
    seed_text = f"{run_key}:{event['step']}:{combo}:{perspective}"
    digest = hashlib.sha256(seed_text.encode("utf-8")).hexdigest()
    return candidates[int(digest, 16) % len(candidates)]


def decode(event: dict, perspective: str, side: str,
           run_key: str) -> tuple[str, str, dict]:
    """Decode one event for one perspective individual.

    Returns (event_text, template_id, axis_values). `side` is the split
    side of the run the event belongs to; only that side's templates are
    used (spec section 4).
    """
    combo = combo_key(event)
    candidates = [t for t in TEMPLATES[combo] if t["side"] == side]
    if event["type"] == "APOLOGY":
        result = (event.get("detail") or {}).get("result")
        candidates = [t for t in candidates if t.get("result") == result]
    if not candidates:
        raise ValueError(f"no template for combo={combo} side={side}")
    template = _pick(candidates, run_key, event, perspective, combo)
    text = template["pattern"].format(**visible_slots(event, perspective))
    return text, template["id"], dict(template["axis"])


def decode_context(event: dict) -> str | None:
    """Compact factual rendering of an event for recent_context lists."""
    if not is_sample_event(event):
        return None
    combo = combo_key(event)
    pattern = CONTEXT_TEMPLATES.get(combo)
    if pattern is None:
        return None
    slots = {"a": event["actor"], "t": event["target"]}
    if event["type"] == "FRICTION":
        slots["strength"] = STRENGTH_PHRASES[_friction_stage(event)]
    if event["type"] == "APOLOGY":
        result = (event.get("detail") or {}).get("result")
        slots["result"] = CONTEXT_APOLOGY_RESULT.get(result, "")
    return pattern.format(**slots)

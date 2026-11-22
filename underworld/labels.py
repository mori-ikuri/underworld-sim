"""Label making for appraisal samples (spec v0.6 sections 9.4 / 10.1).

The label procedure is fixed by the spec:

    sim points (before coefficients)
      -> clamp to +/-3
      -> multiply by the perspective individual's coefficient
      -> round, 0.5 away from zero (never Python's round())
      -> clamp to +/-4
      -> drop zero dimensions
      -> all zero means verdict "no_change"

For events that carry a stage (currently only FRICTION anger), the stage
number itself (light 1 / normal 2 / strong 3) is used as the
before-coefficient points instead of the sim's internal points (2/5/8).
"""

from __future__ import annotations

import math

PRE_CLAMP = 3
POST_CLAMP = 4


def round_half_away_from_zero(value: float) -> int:
    """Round to the nearest integer; halves go away from zero.

    0.5 -> 1, -0.5 -> -1, 2.5 -> 3. Python's built-in round() does
    banker's rounding (0.5 -> 0) and must not be used here (spec 9.4).
    """
    if value >= 0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _clamp(value: float, limit: float) -> float:
    return max(-limit, min(limit, value))


def make_label(points_pre: dict, coefficients: dict,
               *, friction_stage: int | None = None) -> tuple[str, dict, dict]:
    """Build (verdict, deltas, info) from pre-coefficient points.

    points_pre: {dimension: points} as recorded in the log v2 `appraisal`
        record (after habituation / grudge discounts, before coefficients).
    coefficients: the perspective individual's per-dimension coefficients.
    friction_stage: for FRICTION events in strength mode, the stage (1-3);
        its value replaces the recorded anger points (spec 10.1).

    info carries the clamp-hit counts used by the section 9.4 checks:
    {"pre_clamp_hits": int, "post_clamp_hits": int} (dimension counts).
    """
    points = dict(points_pre)
    if friction_stage is not None:
        points["anger"] = float(friction_stage)
    deltas: dict[str, int] = {}
    pre_hits = 0
    post_hits = 0
    for dim, raw in points.items():
        clamped = _clamp(float(raw), PRE_CLAMP)
        if clamped != float(raw):
            pre_hits += 1
        scaled = clamped * float(coefficients[dim])
        rounded = round_half_away_from_zero(scaled)
        final = int(_clamp(rounded, POST_CLAMP))
        if final != rounded:
            post_hits += 1
        if final != 0:
            deltas[dim] = final
    verdict = "change" if deltas else "no_change"
    if not deltas:
        deltas = {}
    return verdict, deltas, {"pre_clamp_hits": pre_hits,
                             "post_clamp_hits": post_hits}

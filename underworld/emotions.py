"""The emotion engine: 8 dimensions, baselines, half-life decay.

Follows the emotion-engine requirements (v0.2, section 7):
  decay:  s <- baseline + (s - baseline) * 0.5 ** (elapsed_hours / half_life)
  apply:  s <- clamp(s + points * point_scale, 0, 1)
  dominant emotion: the dimension with the largest excess over its baseline;
  if that excess is below the calm threshold the individual is "calm".
"""

from __future__ import annotations

# Canonical dimension order (used for deterministic tie-breaking).
DIMENSIONS = (
    "joy",
    "sadness",
    "anger",
    "fear",
    "surprise",
    "affection",
    "trust",
    "fatigue",
)

# Dimensions that describe a feeling *toward the other individual*.
# With only two individuals in the world the target is unambiguous, but the
# log always records the target explicitly (see docs/log-schema.md).
TOWARD_OTHER = ("affection", "anger", "fear", "sadness")


def _clamp(value: float) -> float:
    return min(1.0, max(0.0, value))


class EmotionState:
    """Holds the 8 emotion values for one individual."""

    def __init__(self, emotion_config: dict, values: dict | None = None):
        self._dims = emotion_config["dimensions"]
        self.point_scale = float(emotion_config["point_scale"])
        self.calm_threshold = float(emotion_config["calm_threshold"])
        if values is None:
            values = {dim: float(cfg["baseline"]) for dim, cfg in self._dims.items()}
        self.values = {dim: _clamp(float(values[dim])) for dim in DIMENSIONS}

    # -- basic accessors -------------------------------------------------
    def get(self, dim: str) -> float:
        return self.values[dim]

    def baseline(self, dim: str) -> float:
        return float(self._dims[dim]["baseline"])

    def excess(self, dim: str) -> float:
        return self.values[dim] - self.baseline(dim)

    def set(self, dim: str, value: float) -> None:
        self.values[dim] = _clamp(float(value))

    # -- dynamics --------------------------------------------------------
    def decay(self, hours: float = 1.0, half_life_multipliers: dict | None = None) -> None:
        """Regress every dimension toward its baseline by half-life."""
        multipliers = half_life_multipliers or {}
        for dim in DIMENSIONS:
            cfg = self._dims[dim]
            half_life = float(cfg["half_life_hours"]) * float(multipliers.get(dim, 1.0))
            baseline = float(cfg["baseline"])
            factor = 0.5 ** (hours / half_life)
            self.values[dim] = baseline + (self.values[dim] - baseline) * factor

    def apply(self, dim: str, points: float) -> float:
        """Apply an integer-point delta; returns the actual state change."""
        before = self.values[dim]
        self.values[dim] = _clamp(before + points * self.point_scale)
        return self.values[dim] - before

    # -- interpretation --------------------------------------------------
    def dominant(self) -> tuple[str | None, float]:
        """(dimension, excess) of the strongest above-baseline emotion.

        Returns (None, max_excess) when everything is within the calm band.
        """
        best_dim, best_excess = None, 0.0
        for dim in DIMENSIONS:
            excess = self.excess(dim)
            if excess > best_excess:
                best_dim, best_excess = dim, excess
        if best_excess < self.calm_threshold:
            return None, best_excess
        return best_dim, best_excess

    def dominant_toward_other(self) -> tuple[str | None, float, float]:
        """(dimension, raw value, excess) of the strongest toward-other feeling."""
        best_dim, best_value, best_excess = None, 0.0, 0.0
        for dim in TOWARD_OTHER:
            if self.excess(dim) > best_excess:
                best_dim = dim
                best_value = self.values[dim]
                best_excess = self.excess(dim)
        if best_excess < self.calm_threshold:
            return None, 0.0, best_excess
        return best_dim, best_value, best_excess

    def interrupt_emotion(self) -> tuple[str | None, float]:
        """Strongest approach/avoid drive toward the other, by raw value.

        The interrupt model compares the *raw value* of affection / anger /
        fear against the individual's interrupt threshold (see design memo:
        "affection 0.92 > threshold 0.55 -> other's location").
        """
        best_dim, best_value = None, 0.0
        for dim in ("affection", "anger", "fear"):
            if self.values[dim] > best_value:
                best_dim, best_value = dim, self.values[dim]
        return best_dim, best_value

    # -- serialisation ---------------------------------------------------
    def to_dict(self) -> dict:
        return dict(self.values)

    @classmethod
    def from_dict(cls, emotion_config: dict, values: dict) -> "EmotionState":
        return cls(emotion_config, values)

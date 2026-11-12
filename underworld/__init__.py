"""Underworld: a tiny grid-world simulation of agents with persistent emotions.

Two individuals, each carrying an 8-dimension emotion engine (baseline +
half-life decay) and a body state, live in a 12x12 sandbox. Everything that
happens is recorded as symbolic log events (JSONL). No LLM is used inside
the simulation loop; the daily diary is produced by a pluggable "diary
provider" whose default implementation is a template composer.
"""

__version__ = "1.0.0"

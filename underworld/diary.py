"""Diary providers.

Each individual writes one diary entry per day, just before sleeping,
reflecting on the day's symbolic log events ("kaban" / memory). The diary
writer is a pluggable *provider* so that a future LLM-backed provider can
replace the default without touching the simulation loop.

v1.0 ships only the template provider. An LLM provider is intentionally
left unimplemented: token budget is a deliberate, separate decision, and
the default is OFF (see design memo, "v1.0への方針").
"""

from __future__ import annotations

from collections import Counter

from .emotions import EmotionState

_DIM_JA = {
    "joy": "喜び",
    "sadness": "悲しみ",
    "anger": "怒り",
    "fear": "不安",
    "surprise": "驚き",
    "affection": "好意",
    "trust": "信頼",
    "fatigue": "疲労",
}


class TemplateDiaryProvider:
    """Compose a diary entry from the day's log events with templates."""

    name = "template"

    def compose(
        self,
        *,
        individual_id: str,
        other_id: str,
        day: int,
        events: list[dict],
        emotions: EmotionState,
    ) -> str:
        involved = [
            e for e in events if e.get("actor") == individual_id or e.get("target") == individual_id
        ]
        mine = [e for e in involved if e.get("actor") == individual_id]
        kinds = Counter(
            (e["type"], e.get("kind")) for e in involved if e["type"] not in ("NOTICE", "INTERRUPT")
        )
        parts: list[str] = []

        together = kinds.get(("CONTACT", "together"), 0)
        greet = kinds.get(("CONTACT", "greet"), 0)
        if together:
            parts.append(f"{other_id}と一緒に過ごす時間があった。")
        elif greet:
            parts.append(f"{other_id}と言葉を交わした。")
        if kinds.get(("FRICTION", None), 0):
            parts.append("小さなすれ違いがあった。")
        if kinds.get(("FIGHT", None), 0):
            parts.append("言い合いになった。")
        apology_ok = any(
            e["type"] == "APOLOGY" and e.get("detail", {}).get("result") == "accepted"
            for e in involved
        )
        apology_ng = any(
            e["type"] == "APOLOGY" and e.get("detail", {}).get("result") == "rejected"
            for e in involved
        )
        if apology_ok:
            parts.append("謝罪で仲直りできた場面があった。")
        elif apology_ng:
            parts.append("謝ったけれど、うまく届かなかった。")
        if any(e["type"] == "REACT" and e.get("kind") == "avoided" for e in mine):
            parts.append("近づいたのに離れられて、少し引っかかった。")
        if any(e["type"] == "REACT" and e.get("kind") == "mutual_approach" for e in mine):
            parts.append("互いに近づけたのが嬉しかった。")
        if any(e["type"] == "SURVIVAL" for e in mine):
            parts.append("限界までお腹が空いて、慌てて食べに行った。")
        meals = kinds.get(("EAT", None), 0)
        if meals:
            parts.append(f"ごはんは{meals}回食べた。")

        dominant, _excess = emotions.dominant()
        if dominant is None:
            parts.append("眠る前の気持ちは穏やかだった。")
        else:
            parts.append(
                f"眠る前の気持ちは「{_DIM_JA[dominant]}」が一番強い"
                f"（{emotions.get(dominant):.2f}）。"
            )
        return "".join(parts)


_PROVIDERS = {"template": TemplateDiaryProvider}


def get_provider(name: str):
    """Resolve a diary provider by name (config: ``diary.provider``)."""
    if name == "llm":
        raise NotImplementedError(
            "The LLM diary provider is intentionally not implemented in v1.0. "
            "Implement a provider with a compose(...) method and register it "
            "in underworld/diary.py; keep the default 'template' to run with "
            "zero token cost."
        )
    try:
        return _PROVIDERS[name]()
    except KeyError:
        raise ValueError(f"unknown diary provider: {name!r}") from None

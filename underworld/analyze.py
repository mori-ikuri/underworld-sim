"""Summaries over a run's symbolic log."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from .episodes import detect_episodes


def load_log(path: str | Path) -> list[dict]:
    events = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                events.append(json.loads(line))
    return events


def summarize(events: list[dict], final_state: dict | None = None) -> dict:
    type_counts = Counter(e["type"] for e in events)
    kind_counts = Counter(
        f"{e['type']}:{e['kind']}" for e in events if e.get("kind"))
    per_day: dict[str, Counter] = {}
    for e in events:
        per_day.setdefault(str(e["day"]), Counter())[e["type"]] += 1
    episodes = detect_episodes(events)
    episode_counts = Counter(e["pattern"] for e in episodes)
    days = max((e["day"] for e in events), default=0)
    return {
        "days": days,
        "total_events": len(events),
        "event_counts": dict(sorted(type_counts.items())),
        "kind_counts": dict(sorted(kind_counts.items())),
        "per_day_event_counts": {day: dict(sorted(c.items()))
                                 for day, c in sorted(per_day.items(),
                                                      key=lambda kv: int(kv[0]))},
        "episode_counts": dict(sorted(episode_counts.items())),
        "episodes": episodes,
        "final_state": final_state,
    }


def render_markdown(summary: dict) -> str:
    lines = ["# Run summary", ""]
    lines.append(f"- Days simulated: {summary['days']}")
    lines.append(f"- Total log events: {summary['total_events']}")
    lines.append("")
    lines.append("## Event counts")
    for name, count in summary["event_counts"].items():
        lines.append(f"- {name}: {count}")
    lines.append("")
    lines.append("## Episodes detected")
    if summary["episodes"]:
        for ep in summary["episodes"]:
            lines.append(
                f"- pattern {ep['pattern']} ({ep['label']}): "
                f"{ep['actor']} -> {ep['target']}, "
                f"day {ep['day_start']}..{ep['day_end']} "
                f"(steps {ep['step_start']}..{ep['step_end']})")
    else:
        lines.append("- (none)")
    lines.append("")
    if summary.get("final_state"):
        lines.append("## Final state")
        for ind_id, ind in summary["final_state"]["individuals"].items():
            emo = ", ".join(f"{k}={v:.2f}" for k, v in ind["emotions"].items())
            lines.append(
                f"- {ind_id}: pos={ind['pos']}, hunger={ind['hunger']:.2f}, "
                f"stamina={ind['stamina']:.2f}, sleepiness={ind['sleepiness']:.2f}")
            lines.append(f"  - emotions: {emo}")
        lines.append("")
    return "\n".join(lines)


def render_diaries(events: list[dict]) -> str:
    lines = ["# Diaries", ""]
    for e in events:
        if e["type"] == "DIARY":
            lines.append(f"## Day {e['day']} — individual {e['actor']}")
            lines.append("")
            lines.append(e.get("detail", {}).get("text", ""))
            lines.append("")
    return "\n".join(lines)

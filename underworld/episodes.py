"""Episode extraction: cut multi-step chains out of the symbolic log.

The simulation logs single events; the *stories* are chains. This module
detects the three chain types defined in the design memo:

  pattern A  approach (angry) -> the other flees -> anger amplifies
  pattern B  approach -> warm contact -> later mutual approach dissolves fear
  reconcile  fight -> avoidance -> apology accepted (making up)

Detection is purely a function of the event list, so it can be re-run on
any existing log.jsonl at any time.
"""

from __future__ import annotations

# maximum step gaps that still count as one chain
_A_INTERRUPT_WINDOW = 12
_B_WINDOW = 72
_RECONCILE_WINDOW = 96


def _pair(event: dict) -> tuple[str | None, str | None]:
    return event.get("actor"), event.get("target")


def detect_episodes(events: list[dict]) -> list[dict]:
    episodes: list[dict] = []

    interrupts: dict[str, list[dict]] = {}
    warm_contacts: list[dict] = []
    fights: list[dict] = []

    for event in events:
        etype = event.get("type")
        actor, _target = _pair(event)

        if etype == "INTERRUPT" and event.get("kind") == "approach":
            interrupts.setdefault(actor, []).append(event)

        elif etype == "CONTACT" and event.get("kind") in ("greet", "together"):
            warm_contacts.append(event)

        elif etype == "REACT" and event.get("kind") == "avoided":
            # pattern A: an angry approach that got fled from, anger grows
            if event.get("detail", {}).get("approach_emotion") == "anger":
                start = event["step"]
                for prev in interrupts.get(actor, []):
                    if 0 <= event["step"] - prev["step"] <= _A_INTERRUPT_WINDOW:
                        start = prev["step"]
                episodes.append({
                    "pattern": "A",
                    "label": "approach_fled_anger_amplifies",
                    "actor": actor,
                    "target": event.get("target"),
                    "day_start": _day_of(events, start),
                    "day_end": event["day"],
                    "step_start": start,
                    "step_end": event["step"],
                })

        elif etype == "REACT" and event.get("kind") == "mutual_approach":
            # pattern B: approach -> warm contact -> mutual approach, fear down
            fear_delta = event.get("effects", {}).get(actor, {}).get("fear", 0.0)
            if fear_delta < 0:
                contact = None
                for cand in warm_contacts:
                    if cand["step"] < event["step"] and \
                            event["step"] - cand["step"] <= _B_WINDOW and \
                            actor in (cand.get("actor"), cand.get("target")):
                        contact = cand
                approach = None
                if contact is not None:
                    for prev in interrupts.get(actor, []):
                        if prev["step"] <= contact["step"] and \
                                contact["step"] - prev["step"] <= _B_WINDOW:
                            approach = prev
                if contact is not None and approach is not None:
                    episodes.append({
                        "pattern": "B",
                        "label": "approach_talk_mutual_fear_dissolves",
                        "actor": actor,
                        "target": event.get("target"),
                        "day_start": approach["day"],
                        "day_end": event["day"],
                        "step_start": approach["step"],
                        "step_end": event["step"],
                    })
                    # consume the chain so one contact seeds one episode
                    warm_contacts.remove(contact)
                    interrupts[actor].remove(approach)

        elif etype == "FIGHT":
            fights.append(event)

        elif etype == "APOLOGY" and event.get("detail", {}).get("result") == "accepted":
            for fight in list(fights):
                if 0 <= event["step"] - fight["step"] <= _RECONCILE_WINDOW and \
                        {fight.get("actor"), fight.get("target")} == \
                        {event.get("actor"), event.get("target")}:
                    episodes.append({
                        "pattern": "reconcile",
                        "label": "fight_avoid_apology_makeup",
                        "actor": fight.get("actor"),
                        "target": fight.get("target"),
                        "day_start": fight["day"],
                        "day_end": event["day"],
                        "step_start": fight["step"],
                        "step_end": event["step"],
                    })
                    fights.remove(fight)
                    break

    episodes.sort(key=lambda e: e["step_start"])
    return episodes


def _day_of(events: list[dict], step: int) -> int:
    for event in events:
        if event["step"] == step:
            return event["day"]
    return -1

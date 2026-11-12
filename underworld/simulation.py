"""The Underworld simulation.

A 12x12 grid, two individuals (A and B), one step = one hour.
The inner loop is purely symbolic: every happening is appended to an
in-memory event list (the caller persists it as JSONL) and emotion changes
are resolved through point tables from the config file. No LLM is called
anywhere in this module.

Decision order inside a step (for each awake individual):
  1. survival needs (hunger / stamina at the limit) override everything
  2. anger at/above the interrupt threshold breaks avoidance mode
     (the individual stops enduring and goes at the other)
  3. avoidance mode (after a fight) suppresses the other emotional
     interrupts (affection / fear)
  4. emotional interrupt: a toward-other emotion (affection / anger / fear)
     whose raw value reaches the individual's threshold hijacks the goal
  5. ordinary needs (food / rest) generate a destination
  6. otherwise the individual wanders
"""

from __future__ import annotations

import random

from .diary import get_provider
from .emotions import EmotionState

TOWARD_FOR_INTERRUPT = ("affection", "anger", "fear")


def _sign(value: int) -> int:
    return (value > 0) - (value < 0)


def _chebyshev(a: list[int], b: list[int]) -> int:
    return max(abs(a[0] - b[0]), abs(a[1] - b[1]))


def _rng_state_to_json(rng: random.Random) -> list:
    version, internal, gauss = rng.getstate()
    return [version, list(internal), gauss]


def _rng_state_from_json(rng: random.Random, data: list) -> None:
    version, internal, gauss = data
    rng.setstate((version, tuple(internal), gauss))


class Individual:
    def __init__(self, ind_id: str, cfg: dict, emotions: EmotionState, start: list[int]):
        self.id = ind_id
        self.threshold = float(cfg["interrupt_threshold"])
        self.emotions = emotions
        self.pos = [int(start[0]), int(start[1])]
        # body state: hunger / sleepiness grow toward 1; stamina drains toward 0
        self.hunger = 0.2
        self.stamina = 0.9
        self.sleepiness = 0.1
        self.asleep = False
        self.resting = False
        # relationship / episode state
        self.grudge_until = -1  # step at which the grudge expires
        self.avoid = False
        self.avoid_since = -1
        self.avoid_eroded_total = 0.0
        self.last_diary_day = -1
        self.survival_active = False
        self.prev_reason = "wander"

    # -- helpers ---------------------------------------------------------
    def grudging(self, step: int) -> bool:
        return step < self.grudge_until

    def to_dict(self, step: int) -> dict:
        return {
            "pos": list(self.pos),
            "emotions": self.emotions.to_dict(),
            "hunger": self.hunger,
            "stamina": self.stamina,
            "sleepiness": self.sleepiness,
            "asleep": self.asleep,
            "resting": self.resting,
            "grudge_until": self.grudge_until,
            "avoid": self.avoid,
            "avoid_since": self.avoid_since,
            "avoid_eroded_total": self.avoid_eroded_total,
            "last_diary_day": self.last_diary_day,
            "survival_active": self.survival_active,
            "prev_reason": self.prev_reason,
        }

    def load_dict(self, data: dict) -> None:
        self.pos = [int(data["pos"][0]), int(data["pos"][1])]
        self.hunger = float(data["hunger"])
        self.stamina = float(data["stamina"])
        self.sleepiness = float(data["sleepiness"])
        self.asleep = bool(data["asleep"])
        self.resting = bool(data["resting"])
        self.grudge_until = int(data["grudge_until"])
        self.avoid = bool(data["avoid"])
        self.avoid_since = int(data["avoid_since"])
        self.avoid_eroded_total = float(data["avoid_eroded_total"])
        self.last_diary_day = int(data["last_diary_day"])
        self.survival_active = bool(data["survival_active"])
        self.prev_reason = str(data["prev_reason"])


class Simulation:
    def __init__(self, config: dict, seed: int = 0):
        self.config = config
        self.rng = random.Random(seed)
        self.step_count = 0
        self.events: list[dict] = []
        self.individuals: dict[str, Individual] = {}
        for ind_id in ("A", "B"):
            cfg = config["individuals"][ind_id]
            emotions = EmotionState(config["emotions"])
            self.individuals[ind_id] = Individual(ind_id, cfg, emotions, cfg["start"])
        self.pair_last_contact = -10**9
        self.pair_consecutive_contacts = 0
        self.visible_prev = {"A": False, "B": False}
        self.diary_provider = get_provider(config["diary"]["provider"])

    # ------------------------------------------------------------------
    # small utilities
    # ------------------------------------------------------------------
    def other_id(self, ind_id: str) -> str:
        return "B" if ind_id == "A" else "A"

    def day_index(self) -> int:
        return self.step_count // int(self.config["time"]["steps_per_day"])

    def day_number(self) -> int:
        return self.day_index() + 1

    def clock(self) -> int:
        steps_per_day = int(self.config["time"]["steps_per_day"])
        start = int(self.config["time"]["day_start_clock"])
        return (start + self.step_count % steps_per_day) % 24

    def is_night(self) -> bool:
        clock = self.clock()
        night_start = int(self.config["time"]["night_start_clock"])
        morning = int(self.config["time"]["morning_clock"])
        return clock >= night_start or clock < morning

    def _log(self, type_: str, actor: str | None, target: str | None,
             kind: str | None = None, effects: dict | None = None,
             detail: dict | None = None) -> dict:
        event = {
            "step": self.step_count,
            "day": self.day_number(),
            "clock": self.clock(),
            "type": type_,
            "actor": actor,
            "target": target,
        }
        if kind is not None:
            event["kind"] = kind
        if effects:
            event["effects"] = {
                who: {dim: round(delta, 4) for dim, delta in dims.items() if delta != 0.0}
                for who, dims in effects.items()
            }
            event["effects"] = {k: v for k, v in event["effects"].items() if v}
            if not event["effects"]:
                del event["effects"]
        if detail:
            event["detail"] = detail
        self.events.append(event)
        return event

    def apply_points(self, ind: Individual, dim: str, points: float,
                     *, grudge_discount: bool = True) -> float:
        """Apply a point delta, honouring the grudge discount (減圧1):
        while an individual holds a grudge, positive gains to affection and
        joy are discounted (they cannot simply enjoy kindness right now)."""
        if (
            grudge_discount
            and points > 0
            and dim in ("affection", "joy")
            and ind.grudging(self.step_count)
        ):
            points = points * float(self.config["grudge"]["positive_discount"])
        if points == 0:
            return 0.0
        return ind.emotions.apply(dim, points)

    def apply_table(self, ind: Individual, table: dict, *, scale: float = 1.0,
                    grudge_discount: bool = True) -> dict:
        effects = {}
        for dim, points in table.items():
            delta = self.apply_points(ind, dim, points * scale,
                                       grudge_discount=grudge_discount)
            if delta != 0.0:
                effects[dim] = delta
        return effects

    # ------------------------------------------------------------------
    # geometry
    # ------------------------------------------------------------------
    def _nearest_spot(self, pos: list[int], spots: list) -> list[int]:
        best, best_dist = None, None
        for spot in spots:
            dist = _chebyshev(pos, list(spot))
            if best_dist is None or dist < best_dist:
                best, best_dist = list(spot), dist
        return best

    def _away_target(self, ind: Individual, other: Individual) -> list[int]:
        width = int(self.config["world"]["width"])
        height = int(self.config["world"]["height"])
        corners = [[0, 0], [width - 1, 0], [0, height - 1], [width - 1, height - 1]]
        best, best_dist = None, -1
        for corner in corners:
            dist = _chebyshev(corner, other.pos)
            if dist > best_dist:
                best, best_dist = corner, dist
        return best

    # ------------------------------------------------------------------
    # destination / interrupt decision (exposed for tests)
    # ------------------------------------------------------------------
    def decide_destination(self, ind_id: str) -> tuple[list[int] | None, str, dict]:
        ind = self.individuals[ind_id]
        other = self.individuals[self.other_id(ind_id)]
        world = self.config["world"]
        body = self.config["body"]
        survival = self.config["survival"]

        if ind.resting and ind.stamina < float(body["stamina_rest_full"]):
            return list(ind.pos), "rest_stay", {}
        # 1. survival overrides emotional interrupts (SURVIVAL)
        if ind.hunger >= float(survival["hunger"]):
            return self._nearest_spot(ind.pos, world["food_spots"]), "survival_food", {}
        if ind.stamina <= float(survival["stamina"]):
            return self._nearest_spot(ind.pos, world["rest_spots"]), "survival_rest", {}
        # 2. threshold-crossing anger breaks avoidance: the individual
        #    stops enduring and goes at the other (pattern A's first act).
        #    Avoidance still suppresses affection / fear interrupts.
        emotion, value = ind.emotions.interrupt_emotion()
        if emotion == "anger" and value >= ind.threshold:
            return list(other.pos), "interrupt_approach", {
                "emotion": emotion, "value": round(value, 4)}
        # 3. avoidance mode suppresses the remaining emotional interrupts
        if ind.avoid:
            return self._away_target(ind, other), "avoid", {}
        # 4. emotional interrupt
        if emotion is not None and value >= ind.threshold:
            if emotion == "fear":
                return self._away_target(ind, other), "interrupt_flee", {
                    "emotion": emotion, "value": round(value, 4)}
            return list(other.pos), "interrupt_approach", {
                "emotion": emotion, "value": round(value, 4)}
        # 4. ordinary needs
        if ind.hunger >= float(body["eat_threshold"]):
            return self._nearest_spot(ind.pos, world["food_spots"]), "need_food", {}
        if ind.stamina <= float(body["stamina_rest_threshold"]):
            return self._nearest_spot(ind.pos, world["rest_spots"]), "need_rest", {}
        # 5. wander
        return None, "wander", {}

    # ------------------------------------------------------------------
    # main loop
    # ------------------------------------------------------------------
    def run_steps(self, steps: int) -> None:
        for _ in range(steps):
            self.step()

    def run_days(self, days: int) -> None:
        self.run_steps(days * int(self.config["time"]["steps_per_day"]))

    def step(self) -> None:
        cfg = self.config
        body = cfg["body"]
        step = self.step_count
        inds = [self.individuals["A"], self.individuals["B"]]

        # --- decay + body drift ----------------------------------------
        for ind in inds:
            multipliers = {}
            if ind.grudging(step):
                multipliers["anger"] = float(cfg["grudge"]["decay_multiplier"])
            ind.emotions.decay(1.0, multipliers)
            if ind.asleep:
                ind.hunger = min(1.0, ind.hunger + float(body["hunger_rate_asleep"]))
                ind.sleepiness = max(0.0, ind.sleepiness - float(body["sleepiness_recover"]))
                ind.stamina = min(1.0, ind.stamina + float(body["stamina_sleep_recover"]))
            else:
                ind.hunger = min(1.0, ind.hunger + float(body["hunger_rate_awake"]))
                ind.sleepiness = min(1.0, ind.sleepiness + float(body["sleepiness_rate"]))

        # --- sleep / wake transitions (+ diary just before sleep) ------
        for ind in inds:
            other = self.individuals[self.other_id(ind.id)]
            if not ind.asleep:
                if self.is_night() or ind.sleepiness >= float(body["sleepiness_force"]):
                    if self.clock() >= 21 or self.is_night():
                        if ind.last_diary_day < self.day_index():
                            text = self.diary_provider.compose(
                                individual_id=ind.id,
                                other_id=other.id,
                                day=self.day_number(),
                                events=[e for e in self.events
                                        if e["day"] == self.day_number()],
                                emotions=ind.emotions,
                            )
                            self._log("DIARY", ind.id, None, detail={"text": text})
                            ind.last_diary_day = self.day_index()
                    ind.asleep = True
                    ind.resting = False
                    self._log("SLEEP", ind.id, None)
            else:
                if not self.is_night() and ind.sleepiness <= 0.15:
                    ind.asleep = False
                    effects = self.apply_table(ind, cfg["deltas"]["wake"],
                                               grudge_discount=False)
                    self._log("WAKE", ind.id, None,
                              effects={ind.id: effects} if effects else None)

        # --- grudge / avoidance bookkeeping -----------------------------
        for ind in inds:
            self._update_grudge(ind)
            self._update_avoidance(ind)

        # --- eating / resting in place ----------------------------------
        acted: set[str] = set()
        for ind in inds:
            if ind.asleep:
                acted.add(ind.id)
                continue
            if ind.pos in [list(s) for s in cfg["world"]["food_spots"]] \
                    and ind.hunger >= float(body["eat_min_hunger"]):
                ind.hunger = float(body["hunger_after_eat"])
                ind.stamina = min(1.0, ind.stamina + float(body["eat_stamina_gain"]))
                effects = self.apply_table(ind, cfg["deltas"]["eat"])
                self._log("EAT", ind.id, None,
                          effects={ind.id: effects} if effects else None,
                          detail={"spot": list(ind.pos)})
                acted.add(ind.id)
                continue
            on_rest_spot = ind.pos in [list(s) for s in cfg["world"]["rest_spots"]]
            if on_rest_spot and (ind.resting or ind.stamina <= float(body["stamina_rest_start"])):
                ind.resting = True
                ind.stamina = min(1.0, ind.stamina + float(body["rest_rate"]))
                effects = self.apply_table(ind, cfg["deltas"]["rest"])
                self._log("REST", ind.id, None,
                          effects={ind.id: effects} if effects else None,
                          detail={"spot": list(ind.pos)})
                if ind.stamina >= float(body["stamina_rest_full"]):
                    ind.resting = False
                acted.add(ind.id)

        # --- destination decisions --------------------------------------
        decisions: dict[str, tuple[list[int] | None, str, dict]] = {}
        for ind in inds:
            if ind.id in acted:
                ind.prev_reason = "act"
                continue
            dest, reason, meta = self.decide_destination(ind.id)
            decisions[ind.id] = (dest, reason, meta)
            other = self.individuals[self.other_id(ind.id)]
            if reason.startswith("survival"):
                if not ind.survival_active:
                    ind.survival_active = True
                    self._log("SURVIVAL", ind.id, other.id,
                              kind="food" if reason == "survival_food" else "rest",
                              detail={"hunger": round(ind.hunger, 4),
                                      "stamina": round(ind.stamina, 4)})
            else:
                ind.survival_active = False
            if reason.startswith("interrupt") and ind.prev_reason != reason:
                self._log("INTERRUPT", ind.id, other.id,
                          kind="flee" if reason == "interrupt_flee" else "approach",
                          detail={"emotion": meta.get("emotion"),
                                  "value": meta.get("value"),
                                  "threshold": ind.threshold,
                                  "dest": dest})
            ind.prev_reason = reason

        # --- movement -----------------------------------------------------
        dist_before = _chebyshev(inds[0].pos, inds[1].pos)
        for ind in inds:
            if ind.id not in decisions:
                continue
            dest, reason, _meta = decisions[ind.id]
            other = self.individuals[self.other_id(ind.id)]
            if dest is None:  # wander
                dest = self._wander_dest(ind)
            if dest is not None and list(dest) != ind.pos:
                nxt = [ind.pos[0] + _sign(dest[0] - ind.pos[0]),
                       ind.pos[1] + _sign(dest[1] - ind.pos[1])]
                if nxt != other.pos:  # never stack onto the other's cell
                    ind.pos = nxt
                    ind.stamina = max(0.0, ind.stamina
                                      - float(body["stamina_drain_awake"])
                                      - float(body["stamina_drain_move"]))
                else:
                    ind.stamina = max(0.0, ind.stamina - float(body["stamina_drain_awake"]))
            else:
                ind.stamina = max(0.0, ind.stamina - float(body["stamina_drain_awake"]))
        dist_after = _chebyshev(inds[0].pos, inds[1].pos)

        # --- reactions to the other's movement ---------------------------
        self._react(decisions, dist_before, dist_after)

        # --- noticing -----------------------------------------------------
        self._notice(dist_after)

        # --- contact ------------------------------------------------------
        if dist_after == 1 and not inds[0].asleep and not inds[1].asleep:
            cooldown = int(cfg["contact"]["cooldown_steps"])
            if step - self.pair_last_contact >= cooldown:
                if step - self.pair_last_contact <= cooldown + 2:
                    self.pair_consecutive_contacts += 1
                else:
                    self.pair_consecutive_contacts = 1
                self.pair_last_contact = step
                self._contact(inds[0], inds[1])

        for ind in inds:
            self._update_grudge(ind)

        self.step_count += 1

    # ------------------------------------------------------------------
    # step phases
    # ------------------------------------------------------------------
    def _wander_dest(self, ind: Individual) -> list[int] | None:
        width = int(self.config["world"]["width"])
        height = int(self.config["world"]["height"])
        candidates = [list(ind.pos)]
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                if dx == 0 and dy == 0:
                    continue
                nx, ny = ind.pos[0] + dx, ind.pos[1] + dy
                if 0 <= nx < width and 0 <= ny < height:
                    candidates.append([nx, ny])
        return self.rng.choice(candidates)

    def _update_grudge(self, ind: Individual) -> None:
        cfg = self.config["grudge"]
        step = self.step_count
        anger = ind.emotions.get("anger")
        if anger >= float(cfg["anger_threshold"]):
            if not ind.grudging(step):
                self._log("GRUDGE", ind.id, self.other_id(ind.id), kind="start",
                          detail={"anger": round(anger, 4)})
            ind.grudge_until = max(ind.grudge_until,
                                   step + int(cfg["duration_hours"]))
        elif ind.grudge_until == step:
            ind.grudge_until = -1
            self._log("GRUDGE", ind.id, self.other_id(ind.id), kind="end",
                      detail={"anger": round(anger, 4)})

    def _update_avoidance(self, ind: Individual) -> None:
        if not ind.avoid:
            return
        cfg = self.config["avoidance"]
        other = self.individuals[self.other_id(ind.id)]
        if ind.emotions.get("anger") <= float(cfg["release_anger"]):
            ind.avoid = False
            ind.avoid_since = -1
            ind.avoid_eroded_total = 0.0
            self._log("AVOID", ind.id, other.id, kind="end")
            return
        # 減圧3: a full day or more of avoidance slowly erodes trust and
        # affection -- but never below baseline (no absence penalty).
        duration = self.step_count - ind.avoid_since
        if duration > int(cfg["erosion_after_hours"]):
            rate = float(cfg["erosion_per_hour"])
            effects = {}
            for dim in ("trust", "affection"):
                value = ind.emotions.get(dim)
                baseline = ind.emotions.baseline(dim)
                if value > baseline:
                    new_value = max(baseline, value - rate)
                    ind.emotions.set(dim, new_value)
                    applied = value - new_value
                    if applied > 0:
                        effects[dim] = -applied
                        ind.avoid_eroded_total += applied
            if effects and ind.avoid_eroded_total >= float(cfg["erosion_log_chunk"]):
                ind.avoid_eroded_total = 0.0
                self._log("ERODE", ind.id, other.id, effects={ind.id: effects},
                          detail={"avoid_hours": duration})

    def _react(self, decisions: dict, dist_before: int, dist_after: int) -> None:
        cfg = self.config
        inds = self.individuals
        info = {}
        for ind_id, (_dest, reason, meta) in decisions.items():
            info[ind_id] = (reason, meta.get("emotion"))
        for ind_id in ("A", "B"):
            ind = inds[ind_id]
            other = inds[self.other_id(ind_id)]
            reason, emotion = info.get(ind_id, ("act", None))
            if reason != "interrupt_approach":
                continue
            if dist_after > dist_before:
                # approached, but the other got away: the reaction lands here
                if emotion == "anger":
                    table = cfg["deltas"]["react_avoided_anger"]
                else:
                    table = cfg["deltas"]["react_avoided_affection"]
                effects = self.apply_table(ind, table)
                self._log("REACT", ind_id, other.id, kind="avoided",
                          effects={ind_id: effects} if effects else None,
                          detail={"approach_emotion": emotion,
                                  "dist_before": dist_before,
                                  "dist_after": dist_after})
            elif dist_after < dist_before:
                other_reason, _ = info.get(other.id, ("act", None))
                if other_reason == "interrupt_approach":
                    effects = self.apply_table(ind, cfg["deltas"]["react_mutual"])
                    if effects:
                        self._log("REACT", ind_id, other.id, kind="mutual_approach",
                                  effects={ind_id: effects},
                                  detail={"dist_before": dist_before,
                                          "dist_after": dist_after})

    def _notice(self, dist: int) -> None:
        cfg = self.config
        notice = cfg["notice"]
        radius = int(cfg["world"]["vision_radius"])
        for ind in (self.individuals["A"], self.individuals["B"]):
            other = self.individuals[self.other_id(ind.id)]
            visible_now = dist <= radius and not ind.asleep
            was_visible = self.visible_prev[ind.id]
            if visible_now and not was_visible:
                effects: dict[str, float] = {}
                dim, value, _excess = ind.emotions.dominant_toward_other()
                if dim is not None:
                    points = float(notice["base_points"])
                    if value >= float(notice["high_value_threshold"]):
                        points += float(notice["high_bonus_points"])
                    delta = self.apply_points(ind, dim, points)
                    if delta:
                        effects[dim] = delta
                kind = "spot"
                if dist <= int(cfg["contact"]["eye_contact_range"]):
                    kind = "eye_contact"
                    delta = self.apply_points(
                        ind, "surprise", float(notice["eye_contact_surprise_points"]))
                    if delta:
                        effects["surprise"] = delta
                detail = {"distance": dist, "dominant": dim}
                if ind.grudging(self.step_count):
                    # a grudge reignites a little every time they meet eyes
                    delta = self.apply_points(
                        ind, "anger", float(cfg["grudge"]["reignite_points"]),
                        grudge_discount=False)
                    if delta:
                        effects["anger"] = effects.get("anger", 0.0) + delta
                    detail["grudge_reignite"] = True
                self._log("NOTICE", ind.id, other.id, kind=kind,
                          effects={ind.id: effects} if effects else None,
                          detail=detail)
            elif was_visible and not visible_now and not ind.asleep:
                dim, _value, _excess = ind.emotions.dominant_toward_other()
                effects = {}
                if dim in ("affection", "fear"):
                    for d, pts in (("fear", notice["lost_fear_points"]),
                                   ("sadness", notice["lost_sadness_points"])):
                        delta = self.apply_points(ind, d, float(pts))
                        if delta:
                            effects[d] = delta
                self._log("LOST_SIGHT", ind.id, other.id,
                          effects={ind.id: effects} if effects else None,
                          detail={"dominant": dim})
            self.visible_prev[ind.id] = visible_now

    def _contact(self, a: Individual, b: Individual) -> None:
        cfg = self.config
        deltas = cfg["deltas"]

        # accident first: the seed of anger, independent of goodwill.
        # Each side takes the accident a little differently (anger jitter),
        # which is what later lets one side cool down before the other.
        if self.rng.random() < float(cfg["contact"]["friction_probability"]):
            effects = {}
            jitter_span = int(cfg["contact"].get("friction_jitter", 1))
            for ind in (a, b):
                table = dict(deltas["friction"])
                jitter = self.rng.randint(-jitter_span, jitter_span)
                table["anger"] = max(1, table["anger"] + jitter)
                eff = self.apply_table(ind, table, grudge_discount=False)
                if eff:
                    effects[ind.id] = eff
            self._log("FRICTION", a.id, b.id, effects=effects or None,
                      detail={"consecutive_contacts": self.pair_consecutive_contacts})
            return

        dom_a, _ = a.emotions.dominant()
        dom_b, _ = b.emotions.dominant()
        a_angry = dom_a == "anger" or (
            a.grudging(self.step_count)
            and a.emotions.excess("anger") >= a.emotions.calm_threshold)
        b_angry = dom_b == "anger" or (
            b.grudging(self.step_count)
            and b.emotions.excess("anger") >= b.emotions.calm_threshold)

        if a_angry and b_angry:
            effects = {}
            for ind in (a, b):
                eff = self.apply_table(ind, deltas["fight"], grudge_discount=False)
                if eff:
                    effects[ind.id] = eff
            self._log("FIGHT", a.id, b.id, effects=effects or None)
            for ind in (a, b):
                if not ind.avoid:
                    ind.avoid = True
                    ind.avoid_since = self.step_count
                    ind.avoid_eroded_total = 0.0
                    self._log("AVOID", ind.id, self.other_id(ind.id), kind="start")
            return

        if a_angry != b_angry:
            angry = a if a_angry else b
            calm_side = b if a_angry else a
            warm = float(cfg["apology"]["warm_affection"])
            if calm_side.emotions.get("affection") >= warm \
                    or calm_side.emotions.get("trust") >= warm + 0.05:
                if angry.emotions.get("anger") > float(cfg["apology"]["fail_anger"]):
                    eff_actor = self.apply_table(
                        calm_side, deltas["apology_rejected_actor"], grudge_discount=False)
                    eff_target = self.apply_table(
                        angry, deltas["apology_rejected_target"], grudge_discount=False)
                    effects = {}
                    if eff_actor:
                        effects[calm_side.id] = eff_actor
                    if eff_target:
                        effects[angry.id] = eff_target
                    self._log("APOLOGY", calm_side.id, angry.id,
                              effects=effects or None,
                              detail={"result": "rejected",
                                      "anger": round(angry.emotions.get("anger"), 4)})
                else:
                    eff_target = self.apply_table(
                        angry, deltas["apology_accepted_target"], grudge_discount=False)
                    eff_actor = self.apply_table(
                        calm_side, deltas["apology_accepted_actor"])
                    effects = {}
                    if eff_actor:
                        effects[calm_side.id] = eff_actor
                    if eff_target:
                        effects[angry.id] = eff_target
                    self._log("APOLOGY", calm_side.id, angry.id,
                              effects=effects or None,
                              detail={"result": "accepted",
                                      "anger": round(angry.emotions.get("anger"), 4)})
                return

        if dom_a == "fear" or dom_b == "fear":
            effects = {}
            for ind, dom in ((a, dom_a), (b, dom_b)):
                if dom == "fear":
                    eff = self.apply_table(ind, deltas["keep_distance_self"])
                else:
                    eff = self.apply_table(ind, deltas["keep_distance_other"])
                if eff:
                    effects[ind.id] = eff
            self._log("CONTACT", a.id, b.id, kind="keep_distance",
                      effects=effects or None)
            return

        if dom_a == "affection" and dom_b == "affection":
            hab = cfg["habituation"]
            factor = max(float(hab["floor"]),
                         float(hab["factor"]) ** max(0, self.pair_consecutive_contacts - 1))
            effects = {}
            for ind in (a, b):
                eff = self.apply_table(ind, deltas["together"], scale=factor)
                if eff:
                    effects[ind.id] = eff
            self._log("CONTACT", a.id, b.id, kind="together",
                      effects=effects or None,
                      detail={"consecutive_contacts": self.pair_consecutive_contacts,
                              "habituation_factor": round(factor, 4)})
            return

        # greeting: at least one side warm, or both calm (the bootstrap)
        effects = {}
        for ind, dom in ((a, dom_a), (b, dom_b)):
            if dom in ("affection", "trust"):
                eff = self.apply_table(ind, deltas["greet_warm"])
            elif dom is None:
                partner_dom = dom_b if ind is a else dom_a
                if partner_dom in ("affection", "trust"):
                    eff = self.apply_table(ind, deltas["greet_met"])
                else:
                    eff = self.apply_table(ind, deltas["greet_calm"])
            else:
                eff = {}
            if eff:
                effects[ind.id] = eff
        self._log("CONTACT", a.id, b.id, kind="greet", effects=effects or None)

    # ------------------------------------------------------------------
    # persistence
    # ------------------------------------------------------------------
    def state_dict(self) -> dict:
        return {
            "version": 1,
            "step": self.step_count,
            "rng_state": _rng_state_to_json(self.rng),
            "pair_last_contact": self.pair_last_contact,
            "pair_consecutive_contacts": self.pair_consecutive_contacts,
            "visible_prev": dict(self.visible_prev),
            "individuals": {
                ind_id: {**ind.to_dict(self.step_count),
                         "emotions": ind.emotions.to_dict()}
                for ind_id, ind in self.individuals.items()
            },
        }

    @classmethod
    def from_state(cls, config: dict, state: dict) -> "Simulation":
        sim = cls.__new__(cls)
        sim.config = config
        sim.rng = random.Random()
        _rng_state_from_json(sim.rng, state["rng_state"])
        sim.step_count = int(state["step"])
        sim.events = []
        sim.individuals = {}
        for ind_id in ("A", "B"):
            cfg = config["individuals"][ind_id]
            emotions = EmotionState.from_dict(
                config["emotions"], state["individuals"][ind_id]["emotions"])
            ind = Individual(ind_id, cfg, emotions,
                             state["individuals"][ind_id]["pos"])
            ind.load_dict(state["individuals"][ind_id])
            sim.individuals[ind_id] = ind
        sim.pair_last_contact = int(state["pair_last_contact"])
        sim.pair_consecutive_contacts = int(state["pair_consecutive_contacts"])
        sim.visible_prev = {k: bool(v) for k, v in state["visible_prev"].items()}
        sim.diary_provider = get_provider(config["diary"]["provider"])
        return sim

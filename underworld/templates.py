"""Sentence templates for decoding symbolic events into text.

AUTHORSHIP (needed for audit checkpoint 3, spec section 7): every sentence
pattern in this file -- the event templates, the context forms and the
fixed-form renderers -- was written by Muse Spark (the assistant
"Kinako"). No text from any other AI service or human is included.

Rules these templates follow (spec sections 1, 2, 9.5-4):
  * event_text states only what is visible from outside: no emotion words,
    no numbers, nothing about the perspective individual's inner state.
  * Template selection depends only on the event type x kind (plus the
    split side the run belongs to). Visible facts -- the greeting
    partner's demeanour, the apology result, the friction strength, who
    took distance, eye contact vs. spotting -- are filled in as slots,
    never encoded in template ids.
  * Each type x kind has at least 3 templates, at least one per split
    side (train / validation / test), so test-side phrasings never appear
    in training data (spec sections 2, 4, 9.5-6).

Slots: {p} perspective individual, {o} the other individual,
{demeanour} greeting partner demeanour phrase, {strength} friction
strength phrase, {who} distance-taker clause for keep_distance,
{ap} apologiser, {at} apologised-to, {bumper} the individual who
bumped (friction, spec 11.6), {bumped} the individual who was bumped.
"""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Event templates: combo -> list of templates.
# combo keys: "<type>" or "<type>_<kind>" in lower case.
# ---------------------------------------------------------------------------

TEMPLATES: dict[str, list[dict]] = {
    "notice_spot": [
        {"id": "notice_spot/v1", "side": "train",
         "pattern": "{p}は{o}の姿を見た。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "notice_spot/v2", "side": "val",
         "pattern": "{o}の姿が{p}の目に入った。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "notice_spot/v3", "side": "test",
         "pattern": "{p}は{o}を見かけた。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "notice_eye_contact": [
        {"id": "notice_eye_contact/v1", "side": "train",
         "pattern": "{p}と{o}の目が合った。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "notice_eye_contact/v2", "side": "val",
         "pattern": "{p}は{o}と目が合った。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "notice_eye_contact/v3", "side": "test",
         "pattern": "{p}と{o}は目を合わせた。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "lost_sight": [
        {"id": "lost_sight/v1", "side": "train",
         "pattern": "{p}は{o}を見失った。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "lost_sight/v2", "side": "val",
         "pattern": "{o}の姿が{p}の視界から消えた。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "lost_sight/v3", "side": "test",
         "pattern": "{o}の姿が{p}には見えなくなった。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "react_avoided": [
        {"id": "react_avoided/v1", "side": "train",
         "pattern": "{p}が{o}に近づいたが、{o}は距離を取った。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "react_avoided/v2", "side": "val",
         "pattern": "{p}は{o}に近づいたが、{o}は離れていった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "react_avoided/v3", "side": "test",
         "pattern": "{o}に近づいた{p}だったが、{o}は距離を置いた。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "react_mutual_approach": [
        {"id": "react_mutual_approach/v1", "side": "train",
         "pattern": "{p}と{o}は互いに近づいた。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "react_mutual_approach/v2", "side": "val",
         "pattern": "{p}と{o}は近づき合った。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "react_mutual_approach/v3", "side": "test",
         "pattern": "{o}も{p}も、互いの方へ近づいていった。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "contact_greet": [
        {"id": "contact_greet/v1", "side": "train",
         "pattern": "{p}と{o}は挨拶を交わした。{o}は{demeanour}だった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "contact_greet/v2", "side": "val",
         "pattern": "{o}と挨拶を交わした。{o}は{demeanour}だった。",
         "axis": {"subject_drop": True, "register": "plain"}},
        {"id": "contact_greet/v3", "side": "test",
         "pattern": "{p}は{o}と挨拶を交わした。{o}は{demeanour}だった。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "contact_together": [
        {"id": "contact_together/v1", "side": "train",
         "pattern": "{p}と{o}は一緒に過ごした。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "contact_together/v2", "side": "val",
         "pattern": "{p}は{o}と並んで過ごした。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "contact_together/v3", "side": "test",
         "pattern": "{o}と{p}はしばらく一緒にいた。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "contact_keep_distance": [
        {"id": "contact_keep_distance/v1", "side": "train",
         "pattern": "{who}距離を取った。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "contact_keep_distance/v2", "side": "val",
         "pattern": "{who}距離を置いた。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "contact_keep_distance/v3", "side": "test",
         "pattern": "{who}間を空けた。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    # Friction (spec 11.6, v0.7): observer's viewpoint, both names,
    # who bumped into whom is stated. The same sentence serves both
    # perspectives of an event. No "〜てしまう" / "〜られる" forms and
    # no subject dropping (the bumper must stay visible). The v1-v3
    # ids are retired, never reused: v4 onward is the observer form.
    "friction": [
        {"id": "friction/v4", "side": "train",
         "pattern": "{bumper}が{bumped}に{strength}ぶつかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "friction/v5", "side": "train",
         "pattern": "{bumper}は{bumped}に{strength}ぶつかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "friction/v6", "side": "val",
         "pattern": "{bumped}に{bumper}が{strength}ぶつかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "friction/v7", "side": "val",
         "pattern": "{bumper}が{strength}{bumped}にぶつかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "friction/v8", "side": "test",
         "pattern": "接触の拍子に、{bumper}が{bumped}に{strength}ぶつかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "friction/v9", "side": "test",
         "pattern": "そのとき、{bumper}が{bumped}に{strength}ぶつかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "fight": [
        {"id": "fight/v1", "side": "train",
         "pattern": "{p}と{o}は喧嘩になった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "fight/v2", "side": "val",
         "pattern": "{p}と{o}の間で喧嘩が起きた。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "fight/v3", "side": "test",
         "pattern": "{o}と{p}は喧嘩をした。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
    "apology": [
        {"id": "apology/v1", "side": "train", "result": "accepted",
         "pattern": "{ap}が{at}に謝り、{at}はそれを受け入れた。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "apology/v2", "side": "train", "result": "rejected",
         "pattern": "{ap}が{at}に謝ったが、{at}は受け入れなかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "apology/v3", "side": "val", "result": "accepted",
         "pattern": "{ap}は{at}に謝罪し、{at}は受け入れた。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "apology/v4", "side": "val", "result": "rejected",
         "pattern": "{ap}は{at}に謝罪したが、{at}は受け入れなかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "apology/v5", "side": "test", "result": "accepted",
         "pattern": "謝る{ap}。{at}はその謝罪を受け入れた。",
         "axis": {"subject_drop": False, "register": "plain"}},
        {"id": "apology/v6", "side": "test", "result": "rejected",
         "pattern": "謝る{ap}だったが、{at}は受け入れなかった。",
         "axis": {"subject_drop": False, "register": "plain"}},
    ],
}

# Friction strength phrases: exactly one fixed phrase per stage
# (spec 9.3-1 -- the stage is a visible fact, so stating it is not a leak).
STRENGTH_PHRASES = {
    1: "軽くぶつかった",
    2: "ぶつかった",
    3: "強くぶつかった",
}

# Friction strength as an adverb, for the observer-form templates
# (spec 11.6: "{bumper}が{bumped}に{strength}ぶつかった" shape).
# One fixed word per stage, one-to-one with the stage names.
FRICTION_STRENGTH_ADVERBS = {
    1: "軽く",
    2: "ふつうに",
    3: "強く",
}

# Greeting partner demeanour phrases (spec 9.5-4: must appear in the text).
DEMEANOUR_SMILING = "にこやか"
DEMEANOUR_PLAIN = "ふつうの様子"

# ---------------------------------------------------------------------------
# Context forms: compact factual renderings used only for recent_context
# (spec section 6). One form per combo; slots are the log's actor {a} and
# target {t}, plus {strength} / {result} where the combo carries them.
# Friction uses the same observer form as the event templates (spec
# 11.6): {bumper} / {bumped} names and the {strength} adverb.
# ---------------------------------------------------------------------------

CONTEXT_TEMPLATES: dict[str, str] = {
    "notice_spot": "{a}は{t}の姿を見た。",
    "notice_eye_contact": "{a}と{t}の目が合った。",
    "lost_sight": "{a}は{t}を見失った。",
    "react_avoided": "{a}が{t}に近づいたが、{t}は距離を取った。",
    "react_mutual_approach": "{a}と{t}は互いに近づいた。",
    "contact_greet": "{a}と{t}は挨拶を交わした。",
    "contact_together": "{a}と{t}は一緒に過ごした。",
    "contact_keep_distance": "{a}と{t}は距離を取った。",
    "friction": "{bumper}が{bumped}に{strength}ぶつかった。",
    "fight": "{a}と{t}は喧嘩になった。",
    "apology": "{a}が{t}に謝り、{result}。",
}

CONTEXT_APOLOGY_RESULT = {
    "accepted": "受け入れられた",
    "rejected": "受け入れられなかった",
}

# ---------------------------------------------------------------------------
# Fixed-form renderers (spec sections 1, 2, 6, 9.3-3). These functions are
# the single source of the model-facing sentences for char_profile,
# contact_streak and recent_context, so training and production use the
# exact same wording.
# ---------------------------------------------------------------------------

DIMENSION_NAMES = {
    "joy": "喜び",
    "sadness": "悲しみ",
    "anger": "怒り",
    "fear": "不安",
    "surprise": "驚き",
    "affection": "好意",
    "trust": "信頼",
    "fatigue": "疲労",
}

DIMENSION_ORDER = (
    "joy", "sadness", "anger", "fear",
    "surprise", "affection", "trust", "fatigue",
)

COEFFICIENT_STAGES = {0.5: "低い", 1.0: "ふつう", 1.5: "高い"}


def render_char_profile(coefficients: dict) -> str:
    """Fixed-form sentence for a char_profile (numeric coefficients).

    One-to-one with the 3 coefficient stages; no paraphrasing (spec 1).
    """
    parts = []
    for dim in DIMENSION_ORDER:
        stage = COEFFICIENT_STAGES[float(coefficients[dim])]
        parts.append(f"{DIMENSION_NAMES[dim]}は{stage}")
    return "気質：" + "、".join(parts) + "。"


def render_contact_streak(streak: int) -> str:
    """Fixed-form sentence for contact_streak (6 means "6 or more")."""
    if streak >= 6:
        return "続けて6回以上、顔を合わせている。"
    return f"続けて{streak}回、顔を合わせている。"


def render_recent_context(items: list[str]) -> str:
    """Fixed-form sentence for recent_context (list of context forms)."""
    if not items:
        return "同じ日の直前の出来事はない。"
    return "同じ日の直前の出来事：" + "".join(items)

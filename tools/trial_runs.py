"""Trial data for the section 9.4 checks (spec v0.7).

Generates, under runs/trial/ (git-ignored, never committed):
  * 10 runs x 100 days, seeds 1-10, with a random temperament
    (per-dimension coefficients from 0.5 / 1.0 / 1.5) per run and
    individual
  * 1 control run: seed 42, all coefficients 1.0
All runs use the data-generation config (config/datagen.json: friction
in "strength" mode with method B, fail_anger 0.25 -- spec 11.3), read
through load_config, never rewritten in code. The script then builds
appraisal samples from all runs and prints the aggregation the spec
asks for (per-temperament point spread including sadness, clamp hits,
per-run event counts, apology scenes with the rejected share, bumper /
bumped counts, keep_distance count), plus the apology / fight
breakdowns requested at audit time.

Usage:  python tools/trial_runs.py
Standard library only.
"""

from __future__ import annotations

import hashlib
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from underworld.config import load_config
from underworld.emotions import DIMENSIONS
from underworld.samples import assign_splits, build_samples
from underworld.simulation import Simulation

ROOT = Path(__file__).resolve().parent.parent
OUT = ROOT / "runs" / "trial"
DAYS = 100
STAGES = (0.5, 1.0, 1.5)


def _config_hash(config: dict) -> str:
    return hashlib.sha256(json.dumps(
        config, sort_keys=True, ensure_ascii=False).encode("utf-8")
    ).hexdigest()


DATAGEN_CONFIG = ROOT / "config" / "datagen.json"


def _make_run(run_id: str, seed: int, coefficients: dict) -> dict:
    config = load_config(DATAGEN_CONFIG)
    for ind_id, coefs in coefficients.items():
        config["individuals"][ind_id]["coefficients"] = coefs
    sim = Simulation(config, seed=seed)
    sim.run_days(DAYS)
    run_dir = OUT / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    with open(run_dir / "log.jsonl", "w", encoding="utf-8") as fh:
        for event in sim.events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    meta = {
        "run_id": run_id,
        "seed": seed,
        "coefficients": coefficients,
        "thresholds": {i: config["individuals"][i]["interrupt_threshold"]
                       for i in ("A", "B")},
        "config_hash": _config_hash(config),
        "generator_commit": "trial",
    }
    with open(run_dir / "run.json", "w", encoding="utf-8") as fh:
        json.dump(meta, fh, ensure_ascii=False, indent=1)
    return {"meta": meta, "events": sim.events}


def main() -> int:
    runs = []
    for seed in range(1, 11):
        rng = random.Random(20261122 + seed)
        coefficients = {
            ind: {dim: rng.choice(STAGES) for dim in DIMENSIONS}
            for ind in ("A", "B")}
        runs.append(_make_run(f"trial-seed{seed}", seed, coefficients))
    flat = {ind: {dim: 1.0 for dim in DIMENSIONS} for ind in ("A", "B")}
    runs.append(_make_run("trial-seed42-flat", 42, flat))

    metas = [r["meta"] for r in runs]
    assignment = assign_splits(metas)
    all_samples: dict[str, list[dict]] = {"train": [], "val": [], "test": []}
    index = 1
    details: list[dict] = []
    total_pre_hits = total_post_hits = total_grudge = 0
    per_type_total: Counter = Counter()

    print("== per-run ==")
    print("run_id               angerA angerB same?  events fight "
          "apology(acc/rej) samples grudge_excl")
    apology_groups: dict[str, dict] = {
        "same": {"runs": 0, "apologies": 0, "zero_runs": 0},
        "different": {"runs": 0, "apologies": 0, "zero_runs": 0},
    }
    rejected_total = 0
    apology_total = 0
    bumper_counts: Counter = Counter()
    keep_distance_total = 0
    apology_target_anger: list[float] = []
    fight_over_500 = []
    for run in runs:
        meta, events = run["meta"], run["events"]
        counts = Counter(e["type"] for e in events)
        apology_total += counts["APOLOGY"]
        for e in events:
            if e["type"] == "FRICTION":
                bumper_counts[e["detail"]["bumper"]] += 1
            if e["type"] == "CONTACT" and e.get("kind") == "keep_distance":
                keep_distance_total += 1
        apologies = [e for e in events if e["type"] == "APOLOGY"]
        accepted = sum(1 for e in apologies
                       if e["detail"]["result"] == "accepted")
        rejected = len(apologies) - accepted
        rejected_total += rejected
        for e in apologies:
            record = (e.get("appraisal") or {}).get(e["target"]) or {}
            anger = (record.get("state_before") or {}).get("anger")
            if anger is not None:
                apology_target_anger.append(anger)
        anger_a = meta["coefficients"]["A"]["anger"]
        anger_b = meta["coefficients"]["B"]["anger"]
        group = "same" if anger_a == anger_b else "different"
        apology_groups[group]["runs"] += 1
        apology_groups[group]["apologies"] += len(apologies)
        if not apologies:
            apology_groups[group]["zero_runs"] += 1
        if counts["FIGHT"] > 500:
            fight_over_500.append((meta["run_id"], counts["FIGHT"],
                                   meta["coefficients"]))
        side = assignment[meta["run_id"]]
        samples, stats, index = build_samples(
            events, meta, side, start_index=index, collect_details=True)
        all_samples[side].extend(samples)
        details.extend(stats["label_details"])
        total_pre_hits += stats["pre_clamp_hits"]
        total_post_hits += stats["post_clamp_hits"]
        total_grudge += stats["grudge_excluded"]
        per_type_total.update(stats["per_type"])
        print(f"{meta['run_id']:<20} {anger_a:>5} {anger_b:>6} "
              f"{group:>6} {len(events):>7} {counts['FIGHT']:>5} "
              f"{accepted:>4}/{rejected:<4} {len(samples):>7} "
              f"{stats['grudge_excluded']:>6}  side={side}")

    samples_dir = OUT / "samples"
    samples_dir.mkdir(parents=True, exist_ok=True)
    for side, samples in all_samples.items():
        with open(samples_dir / f"{side}.jsonl", "w", encoding="utf-8") as fh:
            for sample in samples:
                fh.write(json.dumps(sample, ensure_ascii=False) + "\n")

    print()
    print("== apology by anger-coefficient pairing ==")
    for group, g in apology_groups.items():
        mean = g["apologies"] / g["runs"] if g["runs"] else 0.0
        print(f"{group}: runs={g['runs']}, apologies={g['apologies']}, "
              f"mean/run={mean:.2f}, zero-apology runs={g['zero_runs']}")
    print(f"rejected total: {rejected_total}")
    if apology_total:
        print(f"rejected share of all apologies: "
              f"{rejected_total}/{apology_total} "
              f"({rejected_total / apology_total * 100:.1f}%)")
    if apology_target_anger:
        over = sum(1 for a in apology_target_anger if a > 0.25)
        print(f"apology target's anger just before the event: "
              f"n={len(apology_target_anger)}, "
              f"max={max(apology_target_anger):.3f}, "
              f"mean={sum(apology_target_anger) / len(apology_target_anger):.3f}, "
              f"over fail_anger 0.25: {over}")

    print()
    print("== friction bumper / bumped (method B) ==")
    friction_total = sum(bumper_counts.values())
    for ind in ("A", "B"):
        bumped = friction_total - bumper_counts[ind]
        print(f"{ind}: bumper={bumper_counts[ind]}, bumped={bumped}")
    print(f"friction total: {friction_total}")
    print(f"keep_distance events total: {keep_distance_total}")

    print()
    print("== clamp hits (all samples) ==")
    print(f"pre-clamp (+/-3) dimension hits: {total_pre_hits}")
    print(f"post-clamp (+/-4) dimension hits: {total_post_hits}")
    print(f"grudge-excluded samples: {total_grudge}")

    print()
    print("== label spread by dimension x coefficient ==")
    by_key: dict[tuple, list[int]] = defaultdict(list)
    for d in details:
        by_key[(d["dim"], d["coef"])].append(d["delta"])
    for dim in DIMENSIONS:
        for coef in STAGES:
            values = by_key.get((dim, coef), [])
            if not values:
                continue
            nonzero = sum(1 for v in values if v != 0)
            mean = sum(values) / len(values)
            print(f"{dim:<10} x{coef}: n={len(values):>5}, "
                  f"nonzero={nonzero / len(values) * 100:5.1f}%, "
                  f"mean={mean:+.3f}")

    print()
    print("== samples per type ==")
    for key, count in sorted(per_type_total.items()):
        print(f"{key}: {count}")
    print("split sizes:", {k: len(v) for k, v in all_samples.items()})

    print()
    print("== runs with FIGHT > 500 ==")
    if fight_over_500:
        for run_id, fights, coefs in fight_over_500:
            print(f"{run_id}: FIGHT={fights}")
            for ind in ("A", "B"):
                print(f"  {ind}: " + ", ".join(
                    f"{dim}={coefs[ind][dim]}" for dim in DIMENSIONS))
    else:
        print("(none)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

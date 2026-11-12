"""Command line interface.

  python -m underworld run --days 100 --seed 42 --out runs/run1
  python -m underworld analyze --log runs/run1/log.jsonl

`run` writes into the output directory:
  state.json   full simulation state (resumable: run again to continue)
  log.jsonl    symbolic event log, one JSON object per line
  summary.json machine-readable summary incl. detected episodes
  summary.md   human-readable summary
  diaries.md   all diary entries written during the run(s)

Re-running `run` against the same output directory *continues* from the
saved state unless --fresh is given.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from .analyze import load_log, render_diaries, render_markdown, summarize
from .config import load_config
from .simulation import Simulation


def _cmd_run(args: argparse.Namespace) -> int:
    config = load_config(args.config)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "state.json"
    log_path = out / "log.jsonl"

    if state_path.exists() and not args.fresh:
        with open(state_path, encoding="utf-8") as fh:
            state = json.load(fh)
        sim = Simulation.from_state(config, state)
        mode = "a"
        print(f"resuming from step {sim.step_count} ({log_path})")
    else:
        sim = Simulation(config, seed=args.seed)
        mode = "w"
        print(f"starting fresh: seed={args.seed}, days={args.days}")

    before = len(sim.events)
    sim.run_days(args.days)
    new_events = sim.events[before:]

    with open(log_path, mode, encoding="utf-8") as fh:
        for event in new_events:
            fh.write(json.dumps(event, ensure_ascii=False) + "\n")
    with open(state_path, "w", encoding="utf-8") as fh:
        json.dump(sim.state_dict(), fh, ensure_ascii=False, indent=1)

    events = load_log(log_path)
    summary = summarize(events, final_state=sim.state_dict())
    with open(out / "summary.json", "w", encoding="utf-8") as fh:
        json.dump(summary, fh, ensure_ascii=False, indent=1)
    with open(out / "summary.md", "w", encoding="utf-8") as fh:
        fh.write(render_markdown(summary))
    with open(out / "diaries.md", "w", encoding="utf-8") as fh:
        fh.write(render_diaries(events))

    print(f"wrote {len(new_events)} new events "
          f"({summary['total_events']} total over {summary['days']} days)")
    print(f"episodes: {summary['episode_counts']}")
    print(f"output: {out}")
    return 0


def _cmd_analyze(args: argparse.Namespace) -> int:
    events = load_log(args.log)
    summary = summarize(events)
    print(json.dumps(summary, ensure_ascii=False, indent=1))
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="underworld")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="run the simulation")
    run.add_argument("--days", type=int, default=7)
    run.add_argument("--seed", type=int, default=42)
    run.add_argument("--out", default="runs/run")
    run.add_argument("--config", default=None,
                     help="override config JSON (deep-merged over the default)")
    run.add_argument("--fresh", action="store_true",
                     help="ignore any saved state in --out and start over")
    run.set_defaults(func=_cmd_run)

    analyze = sub.add_parser("analyze", help="summarize an existing log.jsonl")
    analyze.add_argument("--log", required=True)
    analyze.set_defaults(func=_cmd_analyze)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())

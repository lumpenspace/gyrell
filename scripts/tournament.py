"""Run a round-robin model tournament and print the standings.

Usage:
    PYTHONPATH=src python3 scripts/tournament.py \
        --models anthropic/claude-haiku-4.5,openai/gpt-5.4-nano \
        --schemes uniform,mixed --guessers 1 --workers 6 \
        --out replays/tournaments/<name>

Slugs without a provider prefix (e.g. "baseline") seat the scripted actor —
useful for offline dry runs. Results land as standard replays plus a
tournament-results.json; point the broadcast server's replay dir at the
output to browse the matches in LUDICA, or feed it to
server.leaderboard.aggregate for Elo.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path  # noqa: F401

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from turngames.adapters.tournament import report_tournament, run_tournament  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="comma-separated model slugs")
    parser.add_argument("--game", default="codewords")
    parser.add_argument("--schemes", default="uniform,mixed")
    parser.add_argument("--guessers", type=int, default=1)
    parser.add_argument("--rounds", type=int, default=1)
    parser.add_argument("--workers", type=int, default=6)
    parser.add_argument("--out", default="replays/tournaments/latest")
    parser.add_argument(
        "--reporter",
        default=None,
        help="model slug that writes color commentary on each match "
        "(reports/<seed>.json + tournament-report.md; never affects results)",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="skip playing; just run the reporter over --out's existing replays",
    )
    parser.add_argument(
        "--endpoints",
        default=None,
        help="JSON file mapping a model label to any OpenAI-compatible "
        'endpoint: {"<label>": {"base_url": ..., "model": ..., '
        '"api_key_env": ..., "timeout": ...}}',
    )
    args = parser.parse_args()

    endpoints = None
    if args.endpoints:
        endpoints = json.loads(Path(args.endpoints).read_text())
        for label, spec in endpoints.items():
            if "api_key_env" in spec:
                spec["api_key"] = os.environ.get(spec["api_key_env"], "local")

    models = [m.strip() for m in args.models.split(",") if m.strip()]
    schemes = tuple(s.strip() for s in args.schemes.split(",") if s.strip())
    for label in endpoints or {}:
        if label not in models and not args.report_only:
            print(f"note: endpoint '{label}' not in --models", file=sys.stderr)
    api_key = os.environ.get("OPENROUTER_API_KEY", "")
    if not api_key and (args.reporter or any("/" in m for m in models)):
        print("OPENROUTER_API_KEY is required for API-backed models", file=sys.stderr)
        return 1

    if args.report_only:
        if not args.reporter:
            print("--report-only needs --reporter", file=sys.stderr)
            return 1
        path = report_tournament(Path(args.out), args.reporter, api_key)
        print(f"digest: {path}")
        return 0

    _, table = run_tournament(
        models,
        api_key=api_key,
        game_id=args.game,
        schemes=schemes,
        guessers=args.guessers,
        rounds=args.rounds,
        workers=args.workers,
        out_dir=Path(args.out),
        endpoints=endpoints,
    )

    width = max(len(m) for m in table) if table else 10
    print(f"\n{'model':<{width}}  W-L-D   win%  as cluegiver")
    for model, b in table.items():
        print(
            f"{model:<{width}}  {b['wins']}-{b['losses']}-{b['draws']}"
            f"   {b['win_rate']:.0%}"
            f"   {b['as_cluegiver_wins']}/{b['as_cluegiver_matches']}"
        )
    print(f"\nreplays + tournament-results.json in {args.out}")
    if args.reporter:
        path = report_tournament(Path(args.out), args.reporter, api_key)
        print(f"reporter digest: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

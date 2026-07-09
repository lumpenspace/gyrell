"""Spectator replay server.

Streams recorded Codewords replays (produced by scripts/record_codewords_replay.py)
over a WebSocket at a watchable pace. This is milestone 2's backend half (see
docs/spectator-app.md): the live match runner will later feed the same message
shapes, so the frontend never needs to know whether it is watching a replay or
a live game.

Run with:
    .venv/bin/uvicorn server.main:app --reload --port 8000
"""

from __future__ import annotations

import asyncio
import contextlib
import gzip
import json
import sys
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from server import games, replay_store  # noqa: E402
from server.channel import LiveChannel  # noqa: E402
from server.leaderboard import aggregate  # noqa: E402
from server.tournament import duel_standings  # noqa: E402

REPLAY_DIR = Path(__file__).resolve().parent.parent / "replays"

# Pacing: deliberations play out word by word (mirroring the game's word
# clock), other events get a beat so reveals are legible.
SECONDS_PER_DELIBERATION_WORD = 0.22
SECONDS_PER_EVENT = 0.9
SECONDS_MIN = 0.05

channel = LiveChannel()


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Restore archived replays before the channel starts: the replay list,
    # leaderboard, and lineup rotation are all derived from REPLAY_DIR.
    await asyncio.to_thread(replay_store.sync_down, REPLAY_DIR)
    task = asyncio.create_task(channel.run())
    yield
    task.cancel()
    with contextlib.suppress(asyncio.CancelledError):
        await task


app = FastAPI(title="codewords-spectator", lifespan=lifespan)

app.add_middleware(
    CORSMiddleware,
    allow_origin_regex=r"http://(localhost|127\.0\.0\.1):\d+",
    allow_methods=["*"],
    allow_headers=["*"],
)


def _replay_name(path: Path) -> str:
    return path.name.removesuffix(".gz").removesuffix(".jsonl")


def list_replays() -> list[str]:
    if not REPLAY_DIR.is_dir():
        return []
    names = {
        _replay_name(path)
        for path in (*REPLAY_DIR.glob("*.jsonl"), *REPLAY_DIR.glob("*.jsonl.gz"))
    }
    return sorted(names)


def _open_replay(name: str):
    """Prefer the gzipped archive; fall back to a plain .jsonl during migration."""
    for suffix, opener in ((".jsonl.gz", gzip.open), (".jsonl", open)):
        path = (REPLAY_DIR / f"{name}{suffix}").resolve()
        if path.parent == REPLAY_DIR.resolve() and path.is_file():
            return opener(path, "rt")
    raise FileNotFoundError(name)


def load_replay(name: str) -> list[dict]:
    with _open_replay(name) as f:
        return [json.loads(line) for line in f if line.strip()]


def _project_snapshot(record: dict, viewer: str) -> dict:
    """Archives store the omniscient snapshot; the spoiler-safe view is that
    minus whatever the game keeps secret — each board cell's key (codewords)
    and the card in play (taboo). Re-derive it here so we only store one
    viewer."""
    if viewer == "audience_omniscient":
        return record
    view = dict(record["view"])
    board = view.get("board")
    if isinstance(board, dict) and "cells" in board:
        cells = tuple(
            {k: v for k, v in cell.items() if k != "key"} for cell in board["cells"]
        )
        view["board"] = {**board, "cells": cells}
    view.pop("current_card", None)
    return {"kind": "snapshot", "viewer": viewer, "view": view}


def record_delay(record: dict) -> float:
    if record["kind"] == "host":
        text = record.get("text") or ""
        return max(SECONDS_PER_EVENT, len(text.split()) * SECONDS_PER_DELIBERATION_WORD)
    if record["kind"] != "event":
        return SECONDS_MIN
    event = record["event"]
    if event["type"] == "actor_responded":
        deliberation = event["payload"].get("public_deliberation") or ""
        words = len(deliberation.split())
        return max(SECONDS_PER_EVENT, words * SECONDS_PER_DELIBERATION_WORD)
    return SECONDS_PER_EVENT


@app.get("/replays")
def get_replays() -> dict:
    return {"replays": list_replays()}


@app.get("/replay/{name}")
def get_replay(name: str) -> dict:
    """The whole archive at once (no pacing) for transcript-style views.
    Same omniscient records the replay socket streams."""
    try:
        return {"records": load_replay(name)}
    except FileNotFoundError:
        return {"records": [], "error": f"unknown replay: {name}"}


@app.get("/games")
def get_games(
    model: str | None = None,
    game: str | None = None,
    kind: str | None = None,
    winner_model: str | None = None,
    team: str | None = None,
    moment: str | None = None,
    word: str | None = None,
    limit: int = 10,
    offset: int = 0,
) -> dict:
    return games.search(
        REPLAY_DIR,
        model=model,
        game=game,
        kind=kind,
        winner_model=winner_model,
        team=team,
        moment=moment,
        word=word,
        limit=limit,
        offset=offset,
    )


@app.get("/leaderboard")
def get_leaderboard() -> dict:
    return aggregate(REPLAY_DIR)


@app.get("/tournament")
def get_tournament() -> dict:
    return duel_standings(REPLAY_DIR)


@app.get("/live/status")
def live_status() -> dict:
    return {
        "status": channel.status,
        "match_number": channel.match_number,
        "lineup": channel.lineup,
        "duel": channel.duel,
        "intermission_seconds": channel.intermission_seconds,
    }


@app.websocket("/ws/live")
async def ws_live(websocket: WebSocket) -> None:
    await websocket.accept()
    viewer = websocket.query_params.get("viewer", "spoiler_safe")
    sub = channel.subscribe(viewer)
    try:
        while True:
            record = await sub.queue.get()
            await websocket.send_json(record)
    except WebSocketDisconnect:
        pass
    finally:
        channel.unsubscribe(sub)


@app.websocket("/ws/replay/{name}")
async def ws_replay(websocket: WebSocket, name: str) -> None:
    await websocket.accept()
    viewer = websocket.query_params.get("viewer", "spoiler_safe")
    try:
        speed = float(websocket.query_params.get("speed", "1"))
    except ValueError:
        speed = 1.0
    speed = min(max(speed, 0.1), 10.0)

    try:
        records = load_replay(name)
    except FileNotFoundError:
        await websocket.send_json({"kind": "error", "message": f"unknown replay: {name}"})
        await websocket.close()
        return

    try:
        for record in records:
            if record["kind"] == "snapshot":
                # Only the omniscient snapshot is authoritative; older archives
                # may still carry a redundant spoiler_safe one — skip it and
                # derive the requested viewer from omniscient instead.
                if record.get("viewer") != "audience_omniscient":
                    continue
                record = _project_snapshot(record, viewer)
            await websocket.send_json(record)
            await asyncio.sleep(record_delay(record) / speed)
        await websocket.send_json({"kind": "stream_end"})
    except WebSocketDisconnect:
        return
    await websocket.close()

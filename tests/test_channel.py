"""Late-joiner catch-up: a mid-game subscriber gets the table talk so far."""

from server.channel import LiveChannel


def snapshot(viewer: str, turn: int) -> dict:
    return {"kind": "snapshot", "viewer": viewer, "view": {"turn_number": turn}}


def drain(sub) -> list[dict]:
    records = []
    while not sub.queue.empty():
        records.append(sub.queue.get_nowait())
    return records


def play_two_turns(channel: LiveChannel) -> None:
    channel._broadcast({"kind": "meta", "game_id": "codewords", "live": True})
    for viewer in ("spoiler_safe", "audience_omniscient"):
        channel._broadcast(snapshot(viewer, 1))
    channel._broadcast({"kind": "host", "text": "welcome!"})
    channel._broadcast({"kind": "event", "event": {"seq": 1, "type": "clue_given"}})
    # Mid-turn snapshots (no turn change) must not bloat the backlog.
    for viewer in ("spoiler_safe", "audience_omniscient"):
        channel._broadcast(snapshot(viewer, 1))
    channel._broadcast({"kind": "event", "event": {"seq": 2, "type": "guess_confirmed"}})
    for viewer in ("spoiler_safe", "audience_omniscient"):
        channel._broadcast(snapshot(viewer, 2))
    channel._broadcast({"kind": "channel", "status": "live", "match_number": 1})


def test_late_joiner_gets_feed_backlog():
    channel = LiveChannel()
    play_two_turns(channel)

    sub = channel.subscribe("spoiler_safe")
    records = drain(sub)

    kinds = [r["kind"] for r in records]
    assert kinds == ["meta", "backlog", "snapshot", "channel"]

    backlog = records[1]["records"]
    # In broadcast order: turn-1 snapshot, host line, clue, guess, turn-2
    # snapshot — mid-turn snapshots skipped, other viewer's snapshots skipped.
    assert [r["kind"] for r in backlog] == ["snapshot", "host", "event", "event", "snapshot"]
    assert all(r.get("viewer", "spoiler_safe") == "spoiler_safe" for r in backlog)
    assert [r["view"]["turn_number"] for r in backlog if r["kind"] == "snapshot"] == [1, 2]
    # The trailing latest-snapshot still brings the view fully current.
    assert records[2]["view"]["turn_number"] == 2


def test_from_start_subscriber_has_no_backlog():
    channel = LiveChannel()
    sub = channel.subscribe("spoiler_safe")
    assert drain(sub) == []


def test_new_match_resets_backlog():
    channel = LiveChannel()
    play_two_turns(channel)
    channel._broadcast({"kind": "meta", "game_id": "taboo", "live": True})

    sub = channel.subscribe("audience_omniscient")
    records = drain(sub)
    assert [r["kind"] for r in records] == ["meta", "channel"]
    assert records[0]["game_id"] == "taboo"

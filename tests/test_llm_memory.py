import json
import unittest
from unittest.mock import patch

from turngames.actors import BaselineCodewordsActor, OpenRouterActor
from turngames.actors.llm import NOTES_MAX_CHARS, build_messages, clip_note
from turngames.core import ActorInput, StateMachine
from turngames.games.codewords import CodewordsConfig, CodewordsSpec

WORDS = tuple(f"word{i}" for i in range(25))
KEY = tuple(
    ["red"] * 9 + ["blue"] * 8 + ["neutral"] * 7 + ["assassin"]
)


def cluegiver_input(transcript=()) -> ActorInput:
    spec = CodewordsSpec()
    machine = StateMachine.new(
        spec, "memory-test", CodewordsConfig(words=WORDS, key=KEY)
    )
    seat = machine.state.seat("red_cluegiver")
    return ActorInput(
        seat=seat,
        role=spec.role_specs()["cluegiver"],
        observation=machine.observe("red_cluegiver"),
        transcript=tuple(transcript),
    )


class MemoryPromptTest(unittest.TestCase):
    def test_prompt_includes_transcript_and_notes(self) -> None:
        actor_input = cluegiver_input(["HOST: welcome", "Ruby: I like word3"])
        user = build_messages(actor_input, notes=("targets: word1, word2",))[1]["content"]
        self.assertIn("Table talk so far", user)
        self.assertIn("Ruby: I like word3", user)
        self.assertIn("Your private notes from earlier turns", user)
        self.assertIn("- targets: word1, word2", user)

    def test_prompt_omits_empty_memory_sections(self) -> None:
        user = build_messages(cluegiver_input())[1]["content"]
        self.assertNotIn("Table talk so far", user)
        self.assertNotIn("private notes", user)

    def test_clip_note(self) -> None:
        self.assertIsNone(clip_note(None))
        self.assertIsNone(clip_note("   "))
        self.assertIsNone(clip_note(42))
        self.assertEqual(len(clip_note("x" * 1000)), NOTES_MAX_CHARS)


class NotesAccumulationTest(unittest.TestCase):
    def test_actor_feeds_notes_back_on_next_call(self) -> None:
        actor = OpenRouterActor(
            model="test/model",
            api_key="k",
            fallback=BaselineCodewordsActor(seed="fb"),
        )
        response = json.dumps(
            {
                "public_deliberation": "Thinking.",
                "action": {"type": "say"},
                "notes": "plan: music for word1+word2",
            }
        )
        seen_prompts = []

        def fake_chat(messages):
            seen_prompts.append(messages[1]["content"])
            return response, "hidden thinking"

        with patch.object(actor, "_chat", side_effect=fake_chat):
            first = actor.act(cluegiver_input())
            second = actor.act(cluegiver_input())

        self.assertEqual(first.notes, "plan: music for word1+word2")
        self.assertEqual(first.reasoning, "hidden thinking")
        self.assertNotIn("plan: music", seen_prompts[0])
        self.assertIn("- plan: music for word1+word2", seen_prompts[1])


if __name__ == "__main__":
    unittest.main()

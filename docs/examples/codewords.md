# Example: codewords

Source:
[`src/turngames/games/codewords/`](https://github.com/lumpenspace/gyrell/tree/main/src/turngames/games/codewords)

[Codenames](https://en.wikipedia.org/wiki/Codenames_(board_game))-style
hidden information: cluegivers see the key, guessers don't, one word is the
assassin, and public deliberation is taxed per word by a team clock.

Kernel features it demonstrates: asymmetric `observe` projections
(cluegiver/guesser/spectator), clue legality via `validate` with survivable
`handle_invalid`, the word-clock timer, and multi-seat play under one
policy. This is the game behind
[`lumpenspace/gyrell`](https://app.primeintellect.ai/dashboard/environments/lumpenspace/gyrell).

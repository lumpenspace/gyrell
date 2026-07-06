#!/usr/bin/env bash
# Evaluate an arbitrary model on the Codewords environment against live
# OpenRouter adversaries in varied combinations.
#
# The evaluated model always plays the red team. Suites:
#   crossplay        red team = evaluated model (cluegiver + guesser),
#                    blue team drawn per-board from the OpenRouter combo pool
#   mixed-cluegiver  evaluated model plays only the red cluegiver; its guesser
#                    teammate AND the blue team come from the OpenRouter pool
#   mixed-guesser    evaluated model plays only the red guesser; its cluegiver
#                    AND the blue team come from the OpenRouter pool
#   all              run all three suites
#
# Examples:
#   # OpenRouter-served model
#   scripts/eval_codewords.sh -m openai/gpt-4.1-mini
#   # local model behind any OpenAI-compatible server (ollama, LM Studio, vLLM, mlx)
#   scripts/eval_codewords.sh -m qwen3-4b -b http://localhost:11434/v1 -k LOCAL_API_KEY
#   # bigger sweep
#   scripts/eval_codewords.sh -m anthropic/claude-haiku-4.5 -n 40 -r 2 -s all
#
# Requires: prime CLI (logged in), OPENROUTER_API_KEY exported (adversaries).
# Results are saved under ./outputs/evals/ and uploaded to the PI dashboard;
# slice win rates per adversary matchup with:
#   python3 scripts/codewords_eval_report.py

set -euo pipefail

ENV_ID="${CODEWORDS_ENV_ID:-lumpenspace/codewords}"
MODEL=""
PROVIDER="openrouter"
BASE_URL=""
KEY_VAR=""
NUM_EXAMPLES=20
ROLLOUTS=1
SUITE="crossplay"
MAX_TURNS=40
MAX_TOKENS=400
TEMPERATURE=0.7

usage() { grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 1; }

while getopts "m:p:b:k:n:r:s:t:h" opt; do
  case $opt in
    m) MODEL="$OPTARG" ;;
    p) PROVIDER="$OPTARG" ;;
    b) BASE_URL="$OPTARG" ;;
    k) KEY_VAR="$OPTARG" ;;
    n) NUM_EXAMPLES="$OPTARG" ;;
    r) ROLLOUTS="$OPTARG" ;;
    s) SUITE="$OPTARG" ;;
    t) MAX_TURNS="$OPTARG" ;;
    h|*) usage ;;
  esac
done
[ -n "$MODEL" ] || usage

# Pool of live adversary matchups: uniform teams from three labs plus two
# cross-lab role-splits. One entry is chosen per board (deterministic in the
# seed) and recorded in game_result.seat_models.
POOL='[
  {"model": "openai/gpt-4.1-mini"},
  {"model": "anthropic/claude-haiku-4.5"},
  {"model": "google/gemini-2.5-flash-lite"},
  {"cluegiver": {"model": "anthropic/claude-haiku-4.5"}, "guesser": {"model": "openai/gpt-4.1-mini"}},
  {"cluegiver": {"model": "openai/gpt-4.1-mini"}, "guesser": {"model": "google/gemini-2.5-flash-lite"}}
]'
POOL=$(echo "$POOL" | tr -d '\n ')

client_flags=(-m "$MODEL")
if [ -n "$BASE_URL" ]; then
  client_flags+=(--api-base-url "$BASE_URL")
  [ -n "$KEY_VAR" ] && client_flags+=(--api-key-var "$KEY_VAR")
else
  client_flags+=(-p "$PROVIDER")
fi

run_suite() {
  local name="$1" args="$2"
  echo "=== suite: $name (model: $MODEL, $NUM_EXAMPLES examples x $ROLLOUTS) ==="
  prime eval run "$ENV_ID" \
    "${client_flags[@]}" \
    -n "$NUM_EXAMPLES" -r "$ROLLOUTS" \
    --max-tokens "$MAX_TOKENS" --temperature "$TEMPERATURE" \
    -a "$args" \
    --state-columns game_result --save-results \
    --disable-tui --abbreviated-summary
}

base_args() { # $1 = role, $2 = extra json fields (may be empty)
  local extra=""
  [ -n "$2" ] && extra=", $2"
  echo "{\"role\": \"$1\", \"max_turns\": $MAX_TURNS, \"num_train_examples\": 0, \"num_eval_examples\": $NUM_EXAMPLES, \"opponent\": $POOL$extra}"
}

case $SUITE in
  crossplay)        run_suite crossplay        "$(base_args team "")" ;;
  mixed-cluegiver)  run_suite mixed-cluegiver  "$(base_args cluegiver "\"partner\": $POOL")" ;;
  mixed-guesser)    run_suite mixed-guesser    "$(base_args guesser "\"partner\": $POOL")" ;;
  all)
    run_suite crossplay       "$(base_args team "")"
    run_suite mixed-cluegiver "$(base_args cluegiver "\"partner\": $POOL")"
    run_suite mixed-guesser   "$(base_args guesser "\"partner\": $POOL")"
    ;;
  *) echo "unknown suite: $SUITE"; exit 1 ;;
esac

echo
echo "Per-matchup breakdown: python3 scripts/codewords_eval_report.py"

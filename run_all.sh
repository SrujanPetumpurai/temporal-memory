#!/usr/bin/env bash
# One command to run the memory system and the action system.
#
#   bash run_all.sh                      # train sets -> outputs/, then score them
#   bash run_all.sh QUESTIONS [COMMANDS] # any other sets (e.g. hidden test): no scoring
#
# Optional env: PYTHON (default: first of python3, python, py), MEM_OUT, ACT_OUT, JUDGE (default none).
set -euo pipefail
cd "$(dirname "$0")"

# Mac/Linux call it python3; Windows (Git Bash/WSL-less) usually only has python or py.
PY="${PYTHON:-}"
if [ -z "$PY" ]; then
  for c in python3 python py; do
    if command -v "$c" >/dev/null 2>&1; then PY="$c"; break; fi
  done
fi
[ -n "$PY" ] || { echo "No Python found. Install Python 3.10+ or set PYTHON=..." >&2; exit 1; }
"$PY" -c 'import sys; sys.exit(sys.version_info < (3, 10))' \
  || { echo "Need Python 3.10+ (set PYTHON=/path/to/python3.x)" >&2; exit 1; }

if [ -z "${API_KEY:-}" ] && ! grep -qs '^API_KEY=.\+' .env; then
  echo "WARNING: no API_KEY in the environment or .env (see .env.example)." >&2
  echo "         Memory will use the weaker extractive fallback; actions will use rules only." >&2
fi

mkdir -p outputs
MEM_IN="${1:-evals/memory_train.jsonl}"
ACT_IN="${2:-evals/actions_train.jsonl}"
MEM_OUT="${MEM_OUT:-outputs/memory_answers.jsonl}"
ACT_OUT="${ACT_OUT:-outputs/actions_out.jsonl}"

echo "== memory: $MEM_IN -> $MEM_OUT"
"$PY" -m candor memory --questions "$MEM_IN" --out "$MEM_OUT"

if [ -f "$ACT_IN" ]; then
  echo "== actions: $ACT_IN -> $ACT_OUT"
  "$PY" -m candor actions --commands "$ACT_IN" --out "$ACT_OUT"
fi

if [ $# -eq 0 ]; then
  echo "== scoring (train sets)"
  cd eval_harness
  "$PY" score_retrieval.py --gold ../evals/memory_train.jsonl  --answers "../$MEM_OUT"
  "$PY" score_memory.py    --gold ../evals/memory_train.jsonl  --answers "../$MEM_OUT" --judge "${JUDGE:-none}"
  "$PY" score_actions.py   --gold ../evals/actions_train.jsonl --predictions "../$ACT_OUT" --out ../outputs/results_actions.json
fi
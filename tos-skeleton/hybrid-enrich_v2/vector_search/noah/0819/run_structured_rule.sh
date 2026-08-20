#!/usr/bin/env bash
set -e
QIDS="../../../filesearch/out/qids_s30.json"
python3 agent_runner.py --run s30v3_structured_core --arm c5_structured_core --qids "$QIDS" --reps 2 --workers 6 --model sonnet
python3 agent_runner.py --run s30v3_structured_full --arm c5_structured_full --qids "$QIDS" --reps 2 --workers 6 --model sonnet

#!/usr/bin/env bash
# Score a forecaster on the synthetic finals and the practice pseudo-answers.
# usage: tools/evaluate.sh <main.py> <sim dir> <practice answers dir> <kit input dir>
set -euo pipefail
bot=$1 sim=$2 pans=$3 kin=$4
work=$(mktemp -d)
mkdir -p "$work/sim" "$work/prac"
for f in "$sim"/*.in; do
  d=$(mktemp -d); cp "$f" "$d/"
  MAKEFUN_INPUT_DIR=$d MAKEFUN_OUTPUT_DIR=$work/sim python3 -I "$bot"
  rm -rf "$d"
done
MAKEFUN_INPUT_DIR=$kin MAKEFUN_OUTPUT_DIR=$work/prac python3 -I "$bot"
echo "practice: $(python3 "$(dirname "$0")/score.py" "$pans" "$work/prac" | tail -1)"
echo "sim:      $(python3 "$(dirname "$0")/score.py" "$sim" "$work/sim" | tail -1)"
echo "$work"

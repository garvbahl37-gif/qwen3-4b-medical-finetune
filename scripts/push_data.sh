#!/usr/bin/env bash
# Upload run 2's prepared data as a private Kaggle Dataset.
#
#   bash scripts/push_data.sh [data-dir]    # default data/v2
#
# Build it first: python -m training.prepare_data_v2 --out data/v2
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${1:-data/v2}"
FILES=(train.jsonl eval_medqa.jsonl eval_medmcqa.jsonl eval_pubmedqa.jsonl
       eval_mmlu_medical.jsonl data_report.json)
for f in "${FILES[@]}"; do
  [ -s "$DATA/$f" ] || { echo "missing $DATA/$f -- run training.prepare_data_v2 first"; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
for f in "${FILES[@]}"; do cp "$DATA/$f" "$STAGE/"; done
bash scripts/kaggle_dataset.sh medical-ft-data "Medical Fine-tune Data" "$STAGE"

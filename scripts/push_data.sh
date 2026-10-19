#!/usr/bin/env bash
# Upload run 2's prepared data as a private Kaggle Dataset.
#
#   bash scripts/push_data.sh [data-dir] [dataset-slug]   # default data/v2 medical-ft-data
#   bash scripts/push_data.sh data/v3 medical-ft-data-v3   # E2's pool, dev split, benchmarks
#
# Build it first: python -m training.prepare_data_v2 --out data/v2, or
# python -m training.prepare_pool_v3 --out data/v3.
set -euo pipefail
cd "$(dirname "$0")/.."

DATA="${1:-data/v2}"
SLUG="${2:-medical-ft-data}"
if [ -f "$DATA/pilot_report.json" ]; then
  FILES=(train.jsonl e2_report.json pilot_report.json)
elif [ -f "$DATA/eval_weak.jsonl" ]; then
  FILES=(eval_weak.jsonl weak_report.json)
elif [ -f "$DATA/e2_report.json" ]; then
  FILES=(train.jsonl e2_report.json)
elif [ -f "$DATA/eval_pool.jsonl" ]; then
  FILES=(eval_pool.jsonl eval_dev_medqa.jsonl eval_dev_medmcqa.jsonl eval_medqa.jsonl
         eval_medmcqa.jsonl eval_pubmedqa.jsonl eval_mmlu_medical.jsonl data_report.json)
else
  FILES=(train.jsonl eval_medqa.jsonl eval_medmcqa.jsonl eval_pubmedqa.jsonl
         eval_mmlu_medical.jsonl data_report.json)
fi
for f in "${FILES[@]}"; do
  [ -s "$DATA/$f" ] || { echo "missing $DATA/$f -- build the data first"; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
for f in "${FILES[@]}"; do cp "$DATA/$f" "$STAGE/"; done
TITLE="Medical Fine-tune Data"
[ "$SLUG" = "medical-ft-data" ] || TITLE="Medical Fine-tune Data ${SLUG#medical-ft-data-}"
bash scripts/kaggle_dataset.sh "$SLUG" "$TITLE" "$STAGE"

#!/usr/bin/env bash
# Upload the trained LoRA adapter as a private Kaggle Dataset.
#
#   bash scripts/push_adapter.sh [adapter-dir]    # default training/outputs/run1
#
# Only the two files PEFT needs go up. The tokenizer saved beside them is the
# base model's own, which evaluation loads from the base.
set -euo pipefail
cd "$(dirname "$0")/.."

ADAPTER="${1:-training/outputs/run1}"
for f in adapter_config.json adapter_model.safetensors; do
  [ -s "$ADAPTER/$f" ] || { echo "missing $ADAPTER/$f"; exit 1; }
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
cp "$ADAPTER/adapter_config.json" "$ADAPTER/adapter_model.safetensors" "$STAGE/"
bash scripts/kaggle_dataset.sh medical-ft-adapter "Medical Fine-tune Adapter" "$STAGE"

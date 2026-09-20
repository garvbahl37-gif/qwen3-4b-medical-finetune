#!/usr/bin/env bash
# Upload the training code as a private Kaggle Dataset, then push the notebook.
#
#   bash scripts/push_kaggle.sh
#
# Credentials come from ~/.kaggle/kaggle.json, already present for gb1105.

set -euo pipefail
cd "$(dirname "$0")/.."

KAGGLE="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
SLUG="medical-ft-code"
USER="$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")"

command -v "$KAGGLE" >/dev/null || { echo "kaggle CLI not found at $KAGGLE"; exit 1; }

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
# Flat, no subdirectories: --dir-mode's "zip" would upload training/ as
# training.zip, and whether Kaggle extracts that back into a directory is not
# something worth discovering during a six-hour GPU run. The notebook
# reassembles the package itself.
cp training/*.py "$STAGE/"
cp training/requirements.txt "$STAGE/"

cat > "$STAGE/dataset-metadata.json" <<JSON
{
  "title": "Medical Fine-tune Code",
  "id": "$USER/$SLUG",
  "licenses": [{"name": "CC0-1.0"}]
}
JSON

if "$KAGGLE" datasets status "$USER/$SLUG" >/dev/null 2>&1; then
  echo "==> updating dataset $USER/$SLUG"
  "$KAGGLE" datasets version -p "$STAGE" -m "code update $(date -u +%FT%TZ)" --dir-mode skip
else
  echo "==> creating dataset $USER/$SLUG"
  "$KAGGLE" datasets create -p "$STAGE" --dir-mode skip
fi

echo "==> waiting for the dataset to finish processing"
# `datasets create` is asynchronous. Pushing the kernel before the dataset is
# ready means Kaggle starts the run with nothing attached, and the notebook
# dies at the code-fetch cell after burning a queue slot.
for attempt in $(seq 1 60); do
  status="$("$KAGGLE" datasets status "$USER/$SLUG" 2>&1 || true)"
  case "$status" in
    *ready*)  echo "    dataset ready"; break ;;
    *error*)  echo "    dataset processing FAILED: $status"; exit 1 ;;
  esac
  if [ "$attempt" -eq 60 ]; then
    echo "    dataset still not ready after 5 minutes: $status"
    exit 1
  fi
  sleep 5
done

echo "==> pushing the notebook"
"$KAGGLE" kernels push -p training

echo
echo "Open it, set Accelerator to GPU T4 x2 and Internet On, then Save & Run All:"
echo "  https://www.kaggle.com/code/$USER/qwen3-4b-medical-fine-tune-training"

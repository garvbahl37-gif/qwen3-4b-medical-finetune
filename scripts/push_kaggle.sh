#!/usr/bin/env bash
# Upload the code as a private Kaggle Dataset, then push a notebook.
#
#   bash scripts/push_kaggle.sh              # the training notebook
#   bash scripts/push_kaggle.sh evaluation   # the evaluation notebook
#
# The evaluation notebook also needs the adapter: run scripts/push_adapter.sh
# first. Credentials come from ~/.kaggle/kaggle.json.
set -euo pipefail
cd "$(dirname "$0")/.."

KERNEL_DIR="${1:-training}"
KAGGLE="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
[ -f "$KERNEL_DIR/kernel-metadata.json" ] || {
  echo "no kernel-metadata.json in $KERNEL_DIR"; exit 1; }

# Rebuild both notebooks so each one's baked-in EXPECTED_FINGERPRINT matches
# the training/*.py bytes this run is about to upload, not whatever was on
# disk the last time someone ran build_notebook.py by hand.
"${PYTHON:-.venv/bin/python}" scripts/build_notebook.py

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT
# Flat, no subdirectories: --dir-mode zip would upload training/ as a zip, and
# whether Kaggle extracts it is not worth discovering during a GPU run. The
# notebook reassembles the package itself.
cp training/*.py training/requirements.txt "$STAGE/"
bash scripts/kaggle_dataset.sh medical-ft-code "Medical Fine-tune Code" "$STAGE"

echo "==> pushing the notebook in $KERNEL_DIR"
"$KAGGLE" kernels push -p "$KERNEL_DIR"

ID="$(python3 -c "import json;print(json.load(open('$KERNEL_DIR/kernel-metadata.json'))['id'])")"
echo
echo "  https://www.kaggle.com/code/$ID"

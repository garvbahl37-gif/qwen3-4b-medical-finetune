#!/usr/bin/env bash
# Create or version a private Kaggle Dataset from a directory, then block until
# Kaggle reports it ready.
#
#   bash scripts/kaggle_dataset.sh <slug> "<title>" <dir>
#
# `datasets create` and `datasets version` are asynchronous. A kernel pushed
# before its dataset is ready starts with nothing attached; that cost a run.
set -euo pipefail

SLUG="$1"; TITLE="$2"; SRC="$3"
KAGGLE="${KAGGLE_BIN:-$HOME/.local/bin/kaggle}"
USER="$(python3 -c "import json;print(json.load(open('$HOME/.kaggle/kaggle.json'))['username'])")"

command -v "$KAGGLE" >/dev/null || { echo "kaggle CLI not found at $KAGGLE"; exit 1; }
[ -d "$SRC" ] || { echo "no such directory: $SRC"; exit 1; }

cat > "$SRC/dataset-metadata.json" <<JSON
{"title": "$TITLE", "id": "$USER/$SLUG", "licenses": [{"name": "CC0-1.0"}]}
JSON

if "$KAGGLE" datasets status "$USER/$SLUG" >/dev/null 2>&1; then
  echo "==> updating dataset $USER/$SLUG"
  "$KAGGLE" datasets version -p "$SRC" -m "update $(date -u +%FT%TZ)" --dir-mode skip
else
  echo "==> creating dataset $USER/$SLUG"
  "$KAGGLE" datasets create -p "$SRC" --dir-mode skip
fi

echo "==> waiting for $USER/$SLUG to finish processing"
status=""
for _ in $(seq 1 120); do
  status="$("$KAGGLE" datasets status "$USER/$SLUG" 2>&1 || true)"
  case "$status" in
    *ready*) echo "    ready"; exit 0 ;;
    *error*) echo "    processing FAILED: $status"; exit 1 ;;
  esac
  sleep 5
done
echo "    still not ready after 10 minutes: $status"
exit 1

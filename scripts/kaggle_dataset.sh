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

# What this upload should leave on Kaggle, as name,size. After `datasets version`
# the status can still read "ready" from the PREVIOUS version, so readiness alone
# would let a kernel start on stale code; the listing has to match too.
expected="$(cd "$SRC" && for f in *; do
  [ "$f" = dataset-metadata.json ] && continue
  printf '%s,%s\n' "$f" "$(wc -c < "$f" | tr -d ' ')"
done | sort)"

if "$KAGGLE" -W datasets status "$USER/$SLUG" >/dev/null 2>&1; then
  echo "==> updating dataset $USER/$SLUG"
  "$KAGGLE" -W datasets version -p "$SRC" -m "update $(date -u +%FT%TZ)" --dir-mode skip
else
  echo "==> creating dataset $USER/$SLUG"
  "$KAGGLE" -W datasets create -p "$SRC" --dir-mode skip
fi

echo "==> waiting for $USER/$SLUG to finish processing"
status=""
for _ in $(seq 1 120); do
  status="$("$KAGGLE" -W datasets status "$USER/$SLUG" 2>&1 || true)"
  case "$status" in
    *ready*)
      # Keep only rows after the header, regardless of any preamble (a stale
      # CLI's version warning, printed to stdout, would otherwise be read as
      # the first "file").
      listed="$("$KAGGLE" -W datasets files "$USER/$SLUG" --csv 2>/dev/null \
                | sed -n '/^name,size,creationDate/,$p' | tail -n +2 \
                | cut -d, -f1,2 | sort || true)"
      if [ "$listed" = "$expected" ]; then
        echo "    ready, serving the files just uploaded"; exit 0
      fi
      echo "    reads ready but still lists the previous files; waiting" ;;
    *error*|*failed*) echo "    processing FAILED: $status"; exit 1 ;;
  esac
  sleep 5
done
echo "    still not ready after 10 minutes: $status"
exit 1

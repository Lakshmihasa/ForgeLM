#!/usr/bin/env bash
# Vendor the official Spider + test-suite evaluator into eval/official/_vendor/.
# These are third-party files (keep their upstream licenses; see the official README).
set -euo pipefail

DEST="src/text2sql/eval/official/_vendor"
TMP="$(mktemp -d)"
mkdir -p "$DEST"

echo "Cloning upstream evaluators into $TMP ..."
git clone --depth 1 https://github.com/taoyds/test-suite-sql-eval "$TMP/tsse"
git clone --depth 1 https://github.com/taoyds/spider "$TMP/spider"

echo "Copying files into $DEST ..."
# test-suite matcher (single-instance exec match)
cp "$TMP/tsse/exec_eval.py"     "$DEST/exec_eval.py"
# hardness + SQL parsing (from Spider; test-suite also ships compatible copies)
cp "$TMP/spider/evaluation.py"  "$DEST/evaluation.py"
cp "$TMP/spider/process_sql.py" "$DEST/process_sql.py"
touch "$DEST/__init__.py"

echo "Done. Verify:  python -c 'from text2sql.eval.official import is_available; print(is_available())'"
echo "NOTE: keep upstream license headers and add attribution to NOTICE."
rm -rf "$TMP"
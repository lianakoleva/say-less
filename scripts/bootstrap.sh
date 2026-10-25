#!/usr/bin/env bash
# One-command SayLess bootstrap.
#
# Usage:
#   ./scripts/bootstrap.sh            # interactive
#   ./scripts/bootstrap.sh --yes      # non-interactive setup
set -euo pipefail

cd "$(dirname "$0")/.."

echo "Installing SayLess Python dependencies..."
python -m pip install -e ".[stt,llm,tools]"

echo "Initializing SayLess..."
sayless init

echo "Verifying the LiteRT-LM model and starting the server..."
sayless setup "$@"

echo
echo "Done. Run 'sayless run' or 'sayless chat'."

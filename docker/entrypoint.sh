#!/usr/bin/env bash
set -e

# Allow running any command
# If no args, show help
if [ $# -eq 0 ]; then
  python -m cbers_colorize.cli --help
  exit 0
fi

exec "$@"
#!/usr/bin/env bash
# Vendor the canonical turngames engine into this environment package so the
# built wheel is self-contained (turngames is not published to PyPI).
# Run after any change under src/turngames, then rebuild/re-push the env.
set -euo pipefail
cd "$(dirname "$0")"
rsync -a --delete --exclude '__pycache__' ../../src/turngames/ turngames/
echo "Synced ../../src/turngames -> $(pwd)/turngames"

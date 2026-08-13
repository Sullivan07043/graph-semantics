#!/bin/bash
# Build causalget into ./site (no venv pollution). Needs gcc and Python headers.
set -eu
cd "$(dirname "$0")"
PY=${PY:-/data2/shuhao/venv/bin/python}
rm -rf build site
$PY -m pip install --target site --no-deps . >/dev/null
rm -rf build causal_get.egg-info
echo "built: $(ls site/causalget/c_backend*.so)"

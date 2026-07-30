#!/bin/bash
# Cache TLVD's released Multitasking graph + descriptions in the DATA.md layout.
set -euo pipefail
script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_dir="$(cd "$script_dir/.." && pwd)"
default_data_root="$(cd "$repo_dir/.." && pwd)/data"
data_root="${GRAPHSEM_DATA:-$default_data_root}"
dest="$data_root/TLVD"
mkdir -p "$dest"
cd "$dest"
for f in multitasking_alpha0.05_rtscale1_N-1.dot multitasking_description.json; do
  curl -fSL "https://raw.githubusercontent.com/HYJ9999/TLVD/main/mac_collab/data/$f" -o "$f"
  echo "cached $f ($(wc -c < "$f") bytes)"
done

#!/bin/bash
# Cache TLVD's released Multitasking graph + description files (from their public repo) into data_cache/.
set -e
cd "$(dirname "$0")/data_cache"
for f in multitasking_alpha0.05_rtscale1_N-1.dot multitasking_description.json; do
  curl -sL "https://raw.githubusercontent.com/HYJ9999/TLVD/main/mac_collab/data/$f" -o "$f"
  echo "cached $f ($(wc -c < "$f") bytes)"
done

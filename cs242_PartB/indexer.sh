#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./indexer.sh <input-dir> <output-dir>
#
# Example:
#   ./indexer.sh \
#     /home/cs242/PartB/cs242_PartB \
#     /home/cs242/PartB/cs242_PartB/indexes/faiss

INPUT_DIR="${1:?input directory (containing newdocs.jsonl) required}"
OUTPUT_DIR="${2:?output directory for FAISS index required}"

SEARCH_DENSE="/home/cs242/PartB/cs242_PartB/Code/search_dense.py"

echo "[0/2] Installing dependencies..."
pip3 install torch transformers faiss-cpu numpy --quiet

echo ""
echo "Input:  $INPUT_DIR/newdocs.jsonl"
echo "Output: $OUTPUT_DIR"
echo ""
echo "NOTE: BERT index was pre-built using CS242_PartB_BERT_Indexing.ipynb on a T4 GPU."
echo "      Pre-built index is available at: $OUTPUT_DIR"
echo ""

echo "[1/2] Smoke test — BERT search"
python3 "$SEARCH_DENSE" search \
  --index_dir /home/cs242/PartB/cs242_PartB \
  --query "attention mechanism transformers" \
  -k 5

echo ""
echo "DONE"
echo ""
echo "To search with BERT:"
echo "  python3 $SEARCH_DENSE search --index_dir /home/cs242/PartB/cs242_PartB --query \"your query\" -k 10"

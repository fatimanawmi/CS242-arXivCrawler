#!/usr/bin/env bash
set -euo pipefail

# Usage:
#   ./indexbuilder.sh \
#     /path/to/json_folder \
#     /path/to/output_dir \
#     /path/to/index_dir \
#     /path/to/docstore.sqlite \
#     [max_docs]
#
# Example:
#   ./indexbuilder.sh \
#     /home/cs242/PartA/Sparse_retrieval/data_all/papers_json \
#     /home/cs242/PartA/Sparse_retrieval/data_all/processed \
#     /home/cs242/PartA/Sparse_retrieval/indexes/lucene_newdocs \
#     /home/cs242/PartA/Sparse_retrieval/data_all/docstore/newdocs.sqlite \
#     0

INP="${1:?input folder (.json files)}"
OUTDIR="${2:?processed output dir}"
INDEXDIR="${3:?lucene index dir}"
DOCSTORE="${4:?docstore sqlite path}"
MAXDOCS="${5:-0}"   # 0 means no limit

JSONL="${OUTDIR%/}/newdocs.jsonl"

echo "[1/3] Processing JSON files -> JSONL + docstore"
mkdir -p "$OUTDIR" "$(dirname "$DOCSTORE")"

if [[ "$MAXDOCS" -gt 0 ]]; then
  python3 data_process.py --input "$INP" --out "$OUTDIR" --docstore_db "$DOCSTORE" --max_docs "$MAXDOCS"
else
  python3 data_process.py --input "$INP" --out "$OUTDIR" --docstore_db "$DOCSTORE"
fi

echo "[2/3] Building Lucene index"
python3 build_lucene_index.py --input "$JSONL" --index_dir "$INDEXDIR"

echo "[3/3] Smoke test search"
python3 search_and_display.py search --index_dir "$INDEXDIR" --query "algorithm" -k 5 --docstore_db "$DOCSTORE" || true

echo "DONE"
echo "JSONL:     $JSONL"
echo "Index:     $INDEXDIR"
echo "Docstore:  $DOCSTORE"

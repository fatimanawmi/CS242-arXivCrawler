# Dense Retrieval

## Files
- `Code/CS242_PartB_BERT_Indexing.ipynb` - BERT indexing notebook
- `Code/search_dense.py` - Dense search implementation
- `indexer.sh` - Script to build the FAISS index
- `results/` - Runtime benchmarks and plots

## Requirements
pip install faiss-cpu

pip install sentence-transformers

pip install torch

## How to Run
bash indexer.sh

python Code/search_dense.py

## Results
- `results/bert_runtime.csv` - BERT runtime benchmarks
- `results/bert_vs_lucene_runtime.png` - Comparison plot

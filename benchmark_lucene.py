import csv
import os

from build_lucene_index import build_index

INPUT = "/home/cs242/PartA/Sparse_retrieval/data_all/processed/newdocs.jsonl"
OUTDIR_BASE = "/home/cs242/PartA/Sparse_retrieval/bench_indexes"
RESULTS_CSV = "/home/cs242/PartA/Sparse_retrieval/results/lucene_runtime.csv"

os.makedirs(OUTDIR_BASE, exist_ok=True)
os.makedirs(os.path.dirname(RESULTS_CSV), exist_ok=True)

Ns = [1000, 2000, 5000, 9000]

with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["N_docs", "seconds"])

    for N in Ns:
        idx_dir = os.path.join(OUTDIR_BASE, f"lucene_{N}")
        _, secs = build_index(
            input_jsonl=INPUT,
            index_dir=idx_dir,
            max_docs=N,
            commit_every=2000,   
            recreate=True
        )
        w.writerow([N, secs])
        f.flush()
        print(f"N={N}: {secs:.2f} sec")

print("Wrote:", RESULTS_CSV)

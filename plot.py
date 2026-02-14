import csv
import matplotlib.pyplot as plt

csv_path = "/home/cs242/PartA/Sparse_retrieval/results/lucene_runtime.csv"
xs, ys = [], []
with open(csv_path, "r", encoding="utf-8") as f:
    r = csv.DictReader(f)
    for row in r:
        xs.append(int(row["N_docs"]))
        ys.append(float(row["seconds"]))

plt.figure()
plt.plot(xs, ys, marker="o")
plt.xlabel("Number of documents")
plt.ylabel("Indexing runtime (seconds)")
plt.title("Lucene indexing runtime vs. number of documents")
plt.grid(True)
plt.savefig("results/lucene_runtime.png", dpi=200)
print("Saved: results/lucene_runtime.png")

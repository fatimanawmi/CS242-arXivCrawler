#!/usr/bin/env python3
"""
search_dense.py  —  Part B1/B2: Dense (BERT + FAISS) Search

Usage
-----
  python3 search_dense.py search \
      --index_dir /home/cs242/PartB/Dense_retrieval \
      --query "attention mechanism transformers" \
      -k 10 \
      --year_min 2024 --year_max 2025 \
      --docstore_db /home/cs242/PartA/Sparse_retrieval/data_all/docstore/newdocs.sqlite
"""

import argparse
import json
import os
import sqlite3
import time
from textwrap import shorten

import numpy as np
import torch
import faiss
from transformers import AutoTokenizer, AutoModel


# ── Query embedding ───────────────────────────────────────────────────────────
@torch.no_grad()
def encode_query(query: str, tokenizer, model, device) -> np.ndarray:
    t0 = time.time()
    encoded = tokenizer(
        [query], padding=True, truncation=True,
        max_length=512, return_tensors="pt"
    ).to(device)
    out = model(**encoded)
    cls = out.last_hidden_state[:, 0, :]                   # [CLS] token
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)   # L2-normalize
    print(f"Query embedding time: {time.time()-t0:.4f}s")
    return cls.cpu().numpy().astype(np.float32)


# ── Load FAISS index + passage metadata ──────────────────────────────────────
def load_dense_index(index_dir: str):
    t_total   = time.time()
    faiss_dir = os.path.join(index_dir, "indexes", "faiss")

    meta_path  = os.path.join(faiss_dir, "index_meta.json")
    faiss_path = os.path.join(faiss_dir, "faiss.index")
    ids_path   = os.path.join(faiss_dir, "passage_ids.json")
    pmeta_path = os.path.join(faiss_dir, "passage_meta.json")

    with open(meta_path) as f:
        meta = json.load(f)

    t0    = time.time()
    index = faiss.read_index(faiss_path)
    print(f"FAISS index load time : {time.time()-t0:.4f}s  ({index.ntotal:,} vectors, dim={index.d})")

    t0 = time.time()
    with open(ids_path) as f:
        passage_ids = json.load(f)
    print(f"passage_ids load time : {time.time()-t0:.4f}s  ({len(passage_ids):,} entries)")

    t0 = time.time()
    with open(pmeta_path, encoding="utf-8") as f:
        passage_meta = json.load(f)
    print(f"passage_meta load time: {time.time()-t0:.4f}s  ({len(passage_meta):,} passages)")

    t0        = time.time()
    device    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(meta["model"])
    model_obj = AutoModel.from_pretrained(meta["model"]).to(device)
    model_obj.eval()
    print(f"BERT model load time  : {time.time()-t0:.4f}s")

    print(f"Total load time       : {time.time()-t_total:.4f}s")
    print(f"Model: {meta['model']}  device: {device}")
    return index, passage_ids, passage_meta, tokenizer, model_obj, device


# ── Docstore (same SQLite from Part A) ───────────────────────────────────────
def fetch_from_docstore(db_path: str, doc_ids: list[str]) -> dict[str, dict]:
    if not db_path or not os.path.exists(db_path):
        return {}
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    out = {}
    for doc_id in doc_ids:
        cur.execute("SELECT json FROM docs WHERE doc_id = ?", (doc_id,))
        row = cur.fetchone()
        if row:
            out[doc_id] = json.loads(row[0])
    con.close()
    return out


# ── Dense search core ─────────────────────────────────────────────────────────
def dense_search(
    index, passage_ids, passage_meta, tokenizer, model_obj, device,
    query: str,
    k: int,
    year_min: int | None = None,
    year_max: int | None = None,
    docstore_db: str | None = None,
) -> list[dict]:
    q_vec = encode_query(query, tokenizer, model_obj, device)

    fetch_k = k * 10
    t0 = time.time()
    scores, indices = index.search(q_vec, fetch_k)
    print(f"FAISS search time     : {time.time()-t0:.4f}s")

    needs_year_filter = (year_min is not None or year_max is not None)

    seen: dict[str, dict] = {}
    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(passage_ids):
            continue
        pid    = passage_ids[idx]
        m      = passage_meta.get(pid, {})
        doc_id = m.get("doc_id", pid)

        title = (m.get("title") or "").strip().lower()
        if not title or len(title) < 10:
            continue

        if doc_id not in seen:
            seen[doc_id] = {"score": float(score), "passage": m}

    ranked = sorted(seen.values(), key=lambda x: x["score"], reverse=True)

    if needs_year_filter:
        filtered = []
        for r in ranked:
            year = r["passage"].get("year")
            try:
                year = int(year)
            except (TypeError, ValueError):
                year = None
            if year_min is not None and (year is None or year < year_min):
                continue
            if year_max is not None and (year is None or year > year_max):
                continue
            filtered.append(r)
        ranked = filtered

    return ranked[:k]


# ── Display ───────────────────────────────────────────────────────────────────
def print_results(ranked: list[dict], full_docs: dict):
    if not ranked:
        print("No results.")
        return

    for rank, r in enumerate(ranked, start=1):
        p      = r["passage"]
        doc_id = p.get("doc_id", "")
        score  = r["score"]

        obj     = full_docs.get(doc_id, {})
        ds_meta = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}

        title      = p.get("title")      or obj.get("title")      or p.get("text", "")[:60]
        authors    = p.get("authors")    or obj.get("authors",    "")
        conference = p.get("conference") or ds_meta.get("conference", "")
        year       = p.get("year")       or ds_meta.get("year",       "")
        url        = p.get("url")        or ds_meta.get("url",        "")
        abstract   = (p.get("abstract") or obj.get("abstract") or p.get("text", "")).replace("\n", " ").strip()

        snippet = shorten(abstract, width=260, placeholder="...")

        print("=" * 100)
        print(f"[{rank}] score={score:.4f}  year={year}  conf={conference}")
        print(f"doc_id={doc_id}")
        print(title)
        if authors:
            print(f"Authors: {authors}")
        if url:
            print("URL:", url)
        print("Snippet:", snippet)

    print("=" * 100)
    print(f"Returned {len(ranked)} results.")


# ── Command ───────────────────────────────────────────────────────────────────
def cmd_search(args):
    index, passage_ids, passage_meta, tokenizer, model_obj, device = load_dense_index(args.index_dir)
    ranked  = dense_search(
        index, passage_ids, passage_meta, tokenizer, model_obj, device,
        query=args.query,
        k=args.k,
        year_min=args.year_min,
        year_max=args.year_max,
        docstore_db=args.docstore_db,
    )
    doc_ids = [r["passage"].get("doc_id", "") for r in ranked]
    full    = fetch_from_docstore(args.docstore_db, doc_ids) if args.docstore_db else {}
    print_results(ranked, full)


# ── CLI ───────────────────────────────────────────────────────────────────────
def main():
    ap  = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Search BERT FAISS index")
    s.add_argument("--index_dir",   required=True)
    s.add_argument("--query",       required=True)
    s.add_argument("-k",            type=int, default=10)
    s.add_argument("--year_min",    type=int, default=None)
    s.add_argument("--year_max",    type=int, default=None)
    s.add_argument("--docstore_db", default=None)

    args = ap.parse_args()
    if args.cmd == "search":
        cmd_search(args)


if __name__ == "__main__":
    main()

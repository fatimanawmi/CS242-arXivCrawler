import json
import os
import sqlite3
from textwrap import shorten

import faiss
import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer


_DENSE_CACHE = {
    "loaded": False,
    "index_dir": None,
    "index": None,
    "passage_ids": None,
    "passage_meta": None,
    "tokenizer": None,
    "model": None,
    "device": None,
    "meta": None,
}


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


@torch.no_grad()
def encode_query(query: str, tokenizer, model, device) -> np.ndarray:
    encoded = tokenizer(
        [query],
        padding=True,
        truncation=True,
        max_length=512,
        return_tensors="pt",
    ).to(device)

    out = model(**encoded)
    cls = out.last_hidden_state[:, 0, :]  # [CLS]
    cls = torch.nn.functional.normalize(cls, p=2, dim=1)
    return cls.cpu().numpy().astype(np.float32)


def load_dense_index(index_dir: str):
    global _DENSE_CACHE

    if _DENSE_CACHE["loaded"] and _DENSE_CACHE["index_dir"] == index_dir:
        return (
            _DENSE_CACHE["index"],
            _DENSE_CACHE["passage_ids"],
            _DENSE_CACHE["passage_meta"],
            _DENSE_CACHE["tokenizer"],
            _DENSE_CACHE["model"],
            _DENSE_CACHE["device"],
            _DENSE_CACHE["meta"],
        )

    faiss_dir = os.path.join(index_dir, "indexes", "faiss")

    meta_path = os.path.join(faiss_dir, "index_meta.json")
    faiss_path = os.path.join(faiss_dir, "faiss.index")
    ids_path = os.path.join(faiss_dir, "passage_ids.json")
    pmeta_path = os.path.join(faiss_dir, "passage_meta.json")

    with open(meta_path, "r", encoding="utf-8") as f:
        meta = json.load(f)

    index = faiss.read_index(faiss_path)

    with open(ids_path, "r", encoding="utf-8") as f:
        passage_ids = json.load(f)

    with open(pmeta_path, "r", encoding="utf-8") as f:
        passage_meta = json.load(f)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(meta["model"])
    model = AutoModel.from_pretrained(meta["model"]).to(device)
    model.eval()

    _DENSE_CACHE = {
        "loaded": True,
        "index_dir": index_dir,
        "index": index,
        "passage_ids": passage_ids,
        "passage_meta": passage_meta,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "meta": meta,
    }

    return index, passage_ids, passage_meta, tokenizer, model, device, meta


def dense_search(
    index_dir: str,
    query: str,
    k: int,
    year_min: int | None = None,
    year_max: int | None = None,
    docstore_db: str | None = None,
) -> list[dict]:
    (
        index,
        passage_ids,
        passage_meta,
        tokenizer,
        model,
        device,
        meta,
    ) = load_dense_index(index_dir)

    q_vec = encode_query(query, tokenizer, model, device)

    fetch_k = max(k * 10, 20)
    scores, indices = index.search(q_vec, fetch_k)

    needs_year_filter = year_min is not None or year_max is not None

    seen: dict[str, dict] = {}

    for score, idx in zip(scores[0], indices[0]):
        if idx < 0 or idx >= len(passage_ids):
            continue

        pid = passage_ids[idx]
        p = passage_meta.get(pid, {})
        doc_id = p.get("doc_id", pid)

        title = (p.get("title") or "").strip()
        if not title or len(title) < 3:
            continue

        if doc_id not in seen:
            seen[doc_id] = {
                "score": float(score),
                "passage": p,
            }

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

    ranked = ranked[:k]

    doc_ids = [r["passage"].get("doc_id", "") for r in ranked]
    full_docs = fetch_from_docstore(docstore_db, doc_ids) if docstore_db else {}

    results = []
    for rank, r in enumerate(ranked, start=1):
        p = r["passage"]
        doc_id = p.get("doc_id", "")
        obj = full_docs.get(doc_id, {})
        ds_meta = obj.get("metadata", {}) if isinstance(obj.get("metadata"), dict) else {}

        title = p.get("title") or obj.get("title") or p.get("text", "")[:60]
        authors = p.get("authors") or obj.get("authors", "")
        conference = p.get("conference") or ds_meta.get("conference", "") or obj.get("conference", "")
        year = p.get("year") or ds_meta.get("year", "") or obj.get("year", "")
        url = (
            p.get("url")
            or obj.get("url", "")
            or ds_meta.get("url", "")
            or obj.get("paper_url", "")
            or ds_meta.get("paper_url", "")
            or obj.get("arxiv_url", "")
            or ds_meta.get("arxiv_url", "")
        )
        if not url and doc_id and "/" not in doc_id and " " not in doc_id:
            url = f"https://arxiv.org/abs/{doc_id}"
        abstract = (
            p.get("abstract")
            or obj.get("abstract")
            or p.get("text", "")
            or ""
        ).replace("\n", " ").strip()

        snippet = shorten(abstract, width=260, placeholder="...")
        categories = obj.get("categories", [])
        if isinstance(categories, str):
            categories = [categories]

        print(
            "DENSE DEBUG:",
            "doc_id=", doc_id,
            "| p.url=", repr(p.get("url")),
            "| obj.url=", repr(obj.get("url") if isinstance(obj, dict) else None),
            "| metadata.url=", repr(ds_meta.get("url") if isinstance(ds_meta, dict) else None),
        )

        print("DENSE DEBUG: docstore_found=", doc_id in full_docs, "| doc_id=", doc_id)

        results.append({
            "rank": rank,
            "score": float(r["score"]),
            "doc_id": doc_id,
            "title": title,
            "authors": authors,
            "conference": conference,
            "year": year,
            "url": url,
            "snippet": snippet,
            "categories": categories,
            "model_name": meta.get("model", ""),
        })

    return results
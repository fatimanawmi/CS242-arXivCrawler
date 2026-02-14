import argparse
import json
import os
import re
import sqlite3
from pathlib import Path

def flatten_text(x):
    """
    Recursively collect text from dict/list/scalars.
    Includes keys + values so headings/checklist keys become searchable too.
    """
    out = []

    def rec(v):
        if v is None:
            return
        if isinstance(v, str):
            s = v.strip()
            if s:
                out.append(s)
        elif isinstance(v, (int, float, bool)):
            out.append(str(v))
        elif isinstance(v, list):
            for it in v:
                rec(it)
        elif isinstance(v, dict):
            for k, val in v.items():
                if isinstance(k, str) and k.strip():
                    out.append(k.strip())
                rec(val)
        else:
            out.append(str(v))

    rec(x)

    # light dedup while preserving order
    seen = set()
    uniq = []
    for s in out:
        s2 = re.sub(r"\s+", " ", s).strip()
        if not s2 or s2 in seen:
            continue
        seen.add(s2)
        uniq.append(s2)

    return "\n".join(uniq)

def normalize(obj: dict, fallback_id: str):
    meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}

    doc_id = str(meta.get("id") or obj.get("id") or meta.get("sanitized_title") or fallback_id)

    title = (meta.get("title") or obj.get("title") or "").strip()
    authors = (meta.get("authors") or obj.get("authors") or "").strip()

    # prefer metadata.abstract (usually clean); fall back to top-level abstract
    abstract = (meta.get("abstract") or obj.get("abstract") or "").strip()

    conference = (meta.get("conference") or "").strip()
    url = (meta.get("url") or obj.get("url") or "").strip()

    year = meta.get("year") or obj.get("year")
    try:
        year = int(year) if year is not None else None
    except Exception:
        year = None

    # body from sections if present
    body = ""
    sections = obj.get("sections")
    if isinstance(sections, dict):
        parts = []
        for sec_title, sec_text in sections.items():
            if sec_title:
                parts.append(str(sec_title).strip())
            if sec_text:
                parts.append(str(sec_text).strip())
        body = "\n\n".join([p for p in parts if p]).strip()

    categories = obj.get("categories") or meta.get("categories") or []
    if isinstance(categories, str):
        categories = categories.split()
    if not isinstance(categories, list):
        categories = []

    all_text = flatten_text(obj)

    return {
        "doc_id": doc_id,
        "paper_id": doc_id,
        "sanitized_title": meta.get("sanitized_title") or "",
        "title": title,
        "authors": authors,
        "abstract": abstract,
        "body": body,
        "all_text": all_text,          # <<< uses all new info for search
        "conference": conference,
        "year": year,
        "url": url,
        "categories": categories,
    }

def iter_json_files(input_path: Path):
    if input_path.is_dir():
        yield from sorted(input_path.rglob("*.json"))
    else:
        yield input_path

def open_docstore(db_path: str):
    if not db_path:
        return None, None
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, json TEXT NOT NULL)")
    con.commit()
    return con, cur

def main(input_path: str, out_path: str, max_docs: int | None, docstore_db: str | None):
    inp = Path(input_path)
    out = Path(out_path)

    # Allow --out to be a directory OR a file path
    if out.suffix.lower() != ".jsonl":
        out.mkdir(parents=True, exist_ok=True)
        out_jsonl = out / "newdocs.jsonl"
    else:
        out.parent.mkdir(parents=True, exist_ok=True)
        out_jsonl = out

    # fresh output
    if out_jsonl.exists():
        out_jsonl.unlink()

    con, cur = open_docstore(docstore_db) if docstore_db else (None, None)

    n = 0
    commit_every = 200

    with out_jsonl.open("w", encoding="utf-8") as fout:
        for fp in iter_json_files(inp):
            with fp.open("r", encoding="utf-8") as f:
                obj = json.load(f)

            doc = normalize(obj, fallback_id=fp.stem)

            # write JSONL for Lucene indexing
            fout.write(json.dumps(doc, ensure_ascii=False) + "\n")

            # store raw JSON for exact retrieval/display
            if cur is not None:
                cur.execute(
                    "INSERT OR REPLACE INTO docs(doc_id, json) VALUES(?, ?)",
                    (doc["doc_id"], json.dumps(obj, ensure_ascii=False)),
                )

            n += 1
            if n % commit_every == 0:
                if con is not None:
                    con.commit()
                print(f"processed {n:,} files...")

            if max_docs is not None and n >= max_docs:
                break

    if con is not None:
        con.commit()
        con.close()

    print(f"\nDONE: wrote {n:,} docs to {out_jsonl}")
    if docstore_db:
        print(f"DONE: docstore at {docstore_db}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Directory of .json files OR a single .json file")
    ap.add_argument("--out", required=True, help="Output directory OR output .jsonl path")
    ap.add_argument("--max_docs", type=int, default=None)
    ap.add_argument("--docstore_db", default=None, help="SQLite docstore path (optional)")
    args = ap.parse_args()

    main(args.input, args.out, args.max_docs, args.docstore_db)

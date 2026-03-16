#!/usr/bin/env python3
"""
search_and_display.py

Commands:
  1) Build a SQLite docstore for retrieval/display:
     python3 search_and_display.py build_docstore --json_path DATA.jsonl --db data/docstore/docs.sqlite

  2) Search a Lucene index and display results (optionally enriching with docstore):
     python3 search_and_display.py search --index_dir indexes/lucene_newdocs --query "pareto set" -k 10 --docstore_db data/docstore/docs.sqlite

Notes:
- Avoids MultiFieldQueryParser/JArray issues by building per-field QueryParser queries and combining them.
- Queries multiple fields including all_text
- Year filter uses IntPoint("year", year).
"""

import argparse
import json
import os
import sqlite3
from textwrap import shorten

import lucene
from java.nio.file import Paths

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader, Term
from org.apache.lucene.search import (
    IndexSearcher,
    BooleanQuery,
    BooleanClause,
    TermQuery,
    BoostQuery,
)
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.document import IntPoint


_JVM_STARTED = False


def init_jvm():
    global _JVM_STARTED
    if not _JVM_STARTED:
        lucene.initVM(vmargs=["-Djava.awt.headless=true"])
        _JVM_STARTED = True


# ---------------------------
# JSON reader (jsonl / array / single object)
# ---------------------------
def iter_docs_auto(path: str):
    """
    Yields documents from either:
      - JSONL: one JSON object per line
      - JSON array: [ {...}, {...}, ... ]
      - single JSON object: { ... }

    Streaming for JSON arrays requires ijson.
    """
    with open(path, "r", encoding="utf-8") as f:
        # Peek first non-whitespace char
        first = ""
        while True:
            c = f.read(1)
            if not c:
                return
            if not c.isspace():
                first = c
                break
        f.seek(0)

        if first == "[":
            # Streaming parse JSON array
            try:
                import ijson
            except ImportError as e:
                raise RuntimeError(
                    "file is a JSON array. Install ijson for streaming:\n"
                    "  pip install ijson\n"
                    "OR convert to JSONL with:\n"
                    "  jq -c '.[]' file.json > file.jsonl"
                ) from e

            with open(path, "rb") as fb:
                for obj in ijson.items(fb, "item"):
                    yield obj

        elif first == "{":
            # Could be JSONL or a single JSON object; try JSONL first
            ok = True
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError:
                    ok = False
                    break
            if not ok:
                f.seek(0)
                yield json.load(f)
        else:
            raise RuntimeError(f"Unknown JSON format. First char = {first!r}")


# ---------------------------
# Docstore (SQLite)
# ---------------------------
def build_sqlite_docstore(json_path: str, db_path: str, commit_every: int = 5000):
    """
    Builds a SQLite docstore: doc_id -> full JSON (as text).
    Input can be JSONL, JSON array, or a single object.
    """
    os.makedirs(os.path.dirname(db_path) or ".", exist_ok=True)
    con = sqlite3.connect(db_path)
    cur = con.cursor()
    cur.execute("PRAGMA journal_mode=WAL;")
    cur.execute("CREATE TABLE IF NOT EXISTS docs (doc_id TEXT PRIMARY KEY, json TEXT NOT NULL)")
    con.commit()

    n = 0
    for obj in iter_docs_auto(json_path):
        doc_id = obj.get("doc_id")
        if not doc_id:
            continue
        cur.execute(
            "INSERT OR REPLACE INTO docs(doc_id, json) VALUES(?, ?)",
            (doc_id, json.dumps(obj, ensure_ascii=False)),
        )
        n += 1
        if n % commit_every == 0:
            con.commit()
            print(f"Docstore: inserted {n:,} records...")

    con.commit()
    con.close()
    print(f"Docstore DONE: {n:,} records in {db_path}")


def fetch_from_docstore(db_path: str, doc_ids: list[str]) -> dict[str, dict]:
    if not db_path:
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


# ---------------------------
# Lucene search + display
# ---------------------------
def open_searcher(index_dir: str) -> IndexSearcher:
    directory = FSDirectory.open(Paths.get(os.path.abspath(index_dir)))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())  # BM25 scoring
    return searcher


def build_query(query_str: str, analyzer, year_min: int | None, year_max: int | None):
    """
    Multi-field query without MultiFieldQueryParser (avoids JArray issues).
    Uses per-field QueryParser + BoostQuery and combines with BooleanQuery.
    """
    boosts = {
        "title": 3.0,
        "abstract": 2.0,
        "authors": 1.0,
        "conference": 1.0,
        "body": 1.0,
        "all_text": 0.6,   # catch-all: uses all new info
    }

    should_builder = BooleanQuery.Builder()
    for field, w in boosts.items():
        p = QueryParser(field, analyzer)
        p.setDefaultOperator(QueryParser.Operator.AND)
        qf = p.parse(query_str)
        should_builder.add(BoostQuery(qf, float(w)), BooleanClause.Occur.SHOULD)

    should_builder.setMinimumNumberShouldMatch(1)
    text_q = should_builder.build()

    bq = BooleanQuery.Builder()
    bq.add(text_q, BooleanClause.Occur.MUST)



    if year_min is not None or year_max is not None:
        lo = year_min if year_min is not None else -10**9
        hi = year_max if year_max is not None else 10**9
        bq.add(IntPoint.newRangeQuery("year", lo, hi), BooleanClause.Occur.FILTER)

    return bq.build()


def doc_to_display_dict(doc):
    # Stored fields 
    doc_id = doc.get("doc_id") or ""
    paper_id = doc.get("paper_id") or ""
    sanitized_title = doc.get("sanitized_title") or ""

    title = doc.get("title") or ""
    authors = doc.get("authors") or ""
    conference = doc.get("conference") or ""
    url = doc.get("url") or ""
    abstract = doc.get("abstract") or ""

    categories = list(doc.getValues("categories")) if doc.getValues("categories") is not None else []
    year = doc.get("year")  # may be None (StoredField Int is retrievable via doc.get in PyLucene often as string)

    snippet = abstract.replace("\n", " ").strip()
    snippet = shorten(snippet, width=260, placeholder="...")

    return {
        "doc_id": doc_id,
        "paper_id": paper_id,
        "sanitized_title": sanitized_title,
        "title": title,
        "authors": authors,
        "conference": conference,
        "year": year,
        "categories": categories,
        "url": url,
        "snippet": snippet,
    }


def search_and_print(index_dir: str, query_str: str, k: int, year_min: int | None, year_max: int | None,
                     docstore_db: str | None):
    init_jvm()

    analyzer = EnglishAnalyzer()
    searcher = open_searcher(index_dir)
    q = build_query(query_str, analyzer, year_min, year_max)

    hits = searcher.search(q, k).scoreDocs
    if not hits:
        print("No results.")
        return

    results = []
    doc_ids = []
    for rank, sd in enumerate(hits, start=1):
        doc = searcher.doc(sd.doc)
        row = doc_to_display_dict(doc)
        row["rank"] = rank
        row["score"] = float(sd.score)
        results.append(row)
        if row["doc_id"]:
            doc_ids.append(row["doc_id"])

    full = fetch_from_docstore(docstore_db, doc_ids) if docstore_db else {}

    for r in results:
        print("=" * 100)
        print(f"[{r['rank']}] score={r['score']:.4f}  year={r['year']}  conf={r['conference']}")
        if r["paper_id"] and r["paper_id"] != r["doc_id"]:
            print(f"doc_id={r['doc_id']}  paper_id={r['paper_id']}")
        else:
            print(f"doc_id={r['doc_id']}")

        print(r["title"])
        if r["authors"]:
            print(f"Authors: {r['authors']}")
        if r["categories"]:
            print("Categories:", " ".join(r["categories"]))
        if r["url"]:
            print("URL:", r["url"])
        print("Snippet:", r["snippet"])

        if docstore_db and r["doc_id"] in full:
            obj = full[r["doc_id"]]
            # show useful extra info (if present)
            meta = obj.get("metadata") if isinstance(obj.get("metadata"), dict) else {}
            yr = meta.get("year") if meta else obj.get("year")
            conf = meta.get("conference") if meta else obj.get("conference")
            if conf or yr:
                print(f"Docstore meta: conference={conf} year={yr}")
            # show key list as proof retrieval works
            print("Docstore keys:", ", ".join(sorted(obj.keys())))

    print("=" * 100)
    print(f"Returned {len(results)} results.")


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("search", help="Search Lucene and display results")
    s.add_argument("--index_dir", required=True)
    s.add_argument("--query", required=True)
    s.add_argument("-k", type=int, default=10)
    s.add_argument("--year_min", type=int, default=None)
    s.add_argument("--year_max", type=int, default=None)
    s.add_argument("--docstore_db", default=None, help="Optional SQLite docstore to fetch full JSON")

    d = sub.add_parser("build_docstore", help="Build SQLite docstore (doc_id -> full JSON)")
    d.add_argument("--json_path", required=True)
    d.add_argument("--db", required=True)
    d.add_argument("--commit_every", type=int, default=5000)

    args = ap.parse_args()

    if args.cmd == "build_docstore":
        build_sqlite_docstore(args.json_path, args.db, commit_every=args.commit_every)
    else:
        search_and_print(
            index_dir=args.index_dir,
            query_str=args.query,
            k=args.k,
            year_min=args.year_min,
            year_max=args.year_max,
            docstore_db=args.docstore_db,
        )


if __name__ == "__main__":
    main()

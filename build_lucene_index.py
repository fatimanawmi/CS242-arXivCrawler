#!/usr/bin/env python3
"""
build_lucene_index.py

Builds a Lucene index from a JSONL file where each line is a JSON dict with keys:

['doc_id', 'paper_id', 'sanitized_title', 'title', 'authors', 'abstract',
 'body', 'all_text', 'conference', 'year', 'url', 'categories']

Notes:
- Uses PerFieldAnalyzerWrapper with a Java HashMap (works with PyLucene).
- Indexes large text fields (body, all_text) but does NOT store them to save space.
- Stores display fields (title, abstract, authors, conference, url, ids).
- Adds IntPoint("year", year) for range queries + StoredField("year") for display.
"""

import argparse
import json
import os
import shutil
import time

import lucene
from java.nio.file import Paths
from java.util import HashMap

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import IndexWriter, IndexWriterConfig
from org.apache.lucene.document import Document, StringField, TextField, StoredField, IntPoint, Field

from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.analysis.standard import StandardAnalyzer
from org.apache.lucene.analysis.core import KeywordAnalyzer
from org.apache.lucene.analysis.miscellaneous import PerFieldAnalyzerWrapper


_JVM_STARTED = False


def init_jvm_once():
    global _JVM_STARTED
    if not _JVM_STARTED:
        lucene.initVM(vmargs=["-Djava.awt.headless=true"])
        _JVM_STARTED = True


def make_analyzer():
    """
    Default: EnglishAnalyzer (tokenize + lowercase + stemming + stopwords)
    Per-field overrides:
      - exact IDs/urls/categories: KeywordAnalyzer
      - authors/conference: StandardAnalyzer (no stemming; better for names/venues)
    """
    default = EnglishAnalyzer()

    field_map = HashMap()
    # exact / keyword-ish
    field_map.put("doc_id", KeywordAnalyzer())
    field_map.put("paper_id", KeywordAnalyzer())
    field_map.put("sanitized_title", KeywordAnalyzer())
    field_map.put("url", KeywordAnalyzer())
    field_map.put("categories", KeywordAnalyzer())
    field_map.put("year_str", KeywordAnalyzer()) 

    
    field_map.put("authors", StandardAnalyzer())
    field_map.put("conference", StandardAnalyzer())

    return PerFieldAnalyzerWrapper(default, field_map)


def safe_int(x):
    try:
        if x is None:
            return None
        return int(x)
    except Exception:
        return None




def build_index(input_jsonl, index_dir, max_docs=None, commit_every=5000, recreate=True):
    init_jvm_once()

    index_dir = os.path.abspath(index_dir)
    if recreate and os.path.exists(index_dir):
        shutil.rmtree(index_dir)
    os.makedirs(index_dir, exist_ok=True)

    directory = FSDirectory.open(Paths.get(index_dir))
    analyzer = make_analyzer()

    config = IndexWriterConfig(analyzer)
    config.setOpenMode(IndexWriterConfig.OpenMode.CREATE)

    writer = IndexWriter(directory, config)

    t0 = time.time()
    n = 0

    with open(input_jsonl, "r", encoding="utf-8") as f:
        for line in f:
            if max_docs is not None and n >= max_docs:
                break

            line = line.strip()
            if not line:
                continue

            try:
                obj = json.loads(line)
            except json.JSONDecodeError:
                continue

            # ---- read fields (your schema) ----
            doc_id = (obj.get("doc_id") or "").strip()
            paper_id = (obj.get("paper_id") or "").strip()
            sanitized_title = (obj.get("sanitized_title") or "").strip()

            title = (obj.get("title") or "").strip()
            abstract = (obj.get("abstract") or "").strip()
            body = (obj.get("body") or "").strip()
            all_text = (obj.get("all_text") or "").strip()

            authors = (obj.get("authors") or "").strip()
            conference = (obj.get("conference") or "").strip()
            url = (obj.get("url") or "").strip()

            year = safe_int(obj.get("year"))

            d = Document()

            # ---- ID / exact-match fields (stored) ----
            if doc_id:
                d.add(StringField("doc_id", doc_id, Field.Store.YES))
            if paper_id:
                d.add(StringField("paper_id", paper_id, Field.Store.YES))
            if sanitized_title:
                d.add(StringField("sanitized_title", sanitized_title, Field.Store.YES))
            if url:
                d.add(StringField("url", url, Field.Store.YES))

            # ---- Display + searchable text fields ----
            # Store title/abstract for display in results
            if title:
                d.add(TextField("title", title, Field.Store.YES))
            if abstract:
                d.add(TextField("abstract", abstract, Field.Store.YES))

            # Authors + conference stored for display
            if authors:
                d.add(TextField("authors", authors, Field.Store.YES))
            if conference:
                d.add(TextField("conference", conference, Field.Store.YES))

            # ---- Large fields: index only (Store.NO) ----
            if body:
                d.add(TextField("body", body, Field.Store.NO))
            if all_text:
                d.add(TextField("all_text", all_text, Field.Store.NO))


            # ---- Year: range filter + display ----
            if year is not None:
                d.add(IntPoint("year", year))
                d.add(StoredField("year", year))
                # optional helper; can be useful for exact matching/debug
                d.add(StringField("year_str", str(year), Field.Store.NO))

            writer.addDocument(d)
            n += 1

            if commit_every and commit_every > 0 and (n % commit_every == 0):
                writer.commit()
                print(f"Indexed {n:,} docs...")

    writer.commit()
    writer.close()

    t1 = time.time()
    print(f"\nDONE: Indexed {n:,} docs into {index_dir}")
    print(f"Time: {t1 - t0:.2f} seconds")
    return n, (t1 - t0)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="Path to input JSONL (one JSON object per line)")
    ap.add_argument("--index_dir", required=True, help="Output Lucene index directory")
    ap.add_argument("--max_docs", type=int, default=None, help="Index only first N docs (testing)")
    ap.add_argument("--commit_every", type=int, default=5000)
    ap.add_argument("--no_recreate", action="store_true", help="Don't delete existing index_dir")
    args = ap.parse_args()

    build_index(
        input_jsonl=args.input,
        index_dir=args.index_dir,
        max_docs=args.max_docs,
        commit_every=args.commit_every,
        recreate=not args.no_recreate,
    )

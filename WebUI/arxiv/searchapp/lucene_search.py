import os
import sqlite3
import json
from textwrap import shorten

import lucene
from java.nio.file import Paths

from org.apache.lucene.store import FSDirectory
from org.apache.lucene.index import DirectoryReader
from org.apache.lucene.search import (
    IndexSearcher,
    BooleanQuery,
    BooleanClause,
    BoostQuery,
)
from org.apache.lucene.search.similarities import BM25Similarity
from org.apache.lucene.analysis.en import EnglishAnalyzer
from org.apache.lucene.queryparser.classic import QueryParser
from org.apache.lucene.document import IntPoint


JVM_STARTED = False


def init_jvm():
    global JVM_STARTED

    if not JVM_STARTED:
        lucene.initVM(vmargs=["-Djava.awt.headless=true"])
        JVM_STARTED = True

    # Important for Django/threaded environments:
    lucene.getVMEnv().attachCurrentThread()


def fetch_from_docstore(db_path, doc_ids):
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


def open_searcher(index_dir):
    init_jvm()
    directory = FSDirectory.open(Paths.get(os.path.abspath(index_dir)))
    reader = DirectoryReader.open(directory)
    searcher = IndexSearcher(reader)
    searcher.setSimilarity(BM25Similarity())
    return searcher


def build_query(query_str, analyzer, year_min=None, year_max=None):
    init_jvm()

    boosts = {
        "title": 3.0,
        "abstract": 2.0,
        "authors": 1.0,
        "conference": 1.0,
        "body": 1.0,
        "all_text": 0.6,
    }

    should_builder = BooleanQuery.Builder()

    for field, weight in boosts.items():
        parser = QueryParser(field, analyzer)
        parser.setDefaultOperator(QueryParser.Operator.AND)
        field_query = parser.parse(query_str)
        should_builder.add(
            BoostQuery(field_query, float(weight)),
            BooleanClause.Occur.SHOULD
        )

    should_builder.setMinimumNumberShouldMatch(1)
    text_query = should_builder.build()

    final_builder = BooleanQuery.Builder()
    final_builder.add(text_query, BooleanClause.Occur.MUST)

    if year_min is not None or year_max is not None:
        lo = year_min if year_min is not None else -10**9
        hi = year_max if year_max is not None else 10**9
        final_builder.add(
            IntPoint.newRangeQuery("year", lo, hi),
            BooleanClause.Occur.FILTER
        )

    return final_builder.build()


def doc_to_display_dict(doc):
    doc_id = doc.get("doc_id") or ""
    paper_id = doc.get("paper_id") or ""
    sanitized_title = doc.get("sanitized_title") or ""

    title = doc.get("title") or ""
    authors = doc.get("authors") or ""
    conference = doc.get("conference") or ""
    url = doc.get("url") or ""
    abstract = doc.get("abstract") or ""

    categories = list(doc.getValues("categories")) if doc.getValues("categories") is not None else []
    year = doc.get("year")

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


def search_documents(index_dir, query_str, k=10, year_min=None, year_max=None, docstore_db=None):
    init_jvm()

    analyzer = EnglishAnalyzer()
    searcher = open_searcher(index_dir)
    query = build_query(query_str, analyzer, year_min, year_max)

    hits = searcher.search(query, k).scoreDocs
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

    full_docs = fetch_from_docstore(docstore_db, doc_ids) if docstore_db else {}

    for row in results:
        extra = full_docs.get(row["doc_id"])
        row["docstore"] = extra if extra else None

    return results
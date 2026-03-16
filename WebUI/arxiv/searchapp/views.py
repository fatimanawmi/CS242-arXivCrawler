from django.shortcuts import render

from .lucene_search import search_documents
from .dense_search import dense_search

SPARSE_INDEX_DIR = "/home/cs242/PartA/Sparse_retrieval/indexes/lucene_newdocs"
DOCSTORE_DB = "/home/cs242/PartA/Sparse_retrieval/data_all/docstore/newdocs.sqlite"
# DENSE_INDEX_DIR = "/home/cs242/PartB/Dense_retrieval"
DENSE_INDEX_DIR = "/home/cs242/PartB/cs242_PartB"


def search_view(request):
    query = request.GET.get("q", "").strip()
    k_raw = request.GET.get("k", "10").strip()
    year_min_raw = request.GET.get("year_min", "").strip()
    year_max_raw = request.GET.get("year_max", "").strip()

    try:
        k = int(k_raw)
    except ValueError:
        k = 10
    k = max(1, min(k, 50))

    try:
        year_min = int(year_min_raw) if year_min_raw else None
    except ValueError:
        year_min = None

    try:
        year_max = int(year_max_raw) if year_max_raw else None
    except ValueError:
        year_max = None

    sparse_results = []
    dense_results = []
    sparse_error = None
    dense_error = None

    if query:
        try:
            sparse_results = search_documents(
                index_dir=SPARSE_INDEX_DIR,
                query_str=query,
                k=k,
                year_min=year_min,
                year_max=year_max,
                docstore_db=DOCSTORE_DB,
            )
        except Exception as e:
            sparse_error = str(e)

        try:
            dense_results = dense_search(
                index_dir=DENSE_INDEX_DIR,
                query=query,
                k=k,
                year_min=year_min,
                year_max=year_max,
                docstore_db=DOCSTORE_DB,
            )
        except Exception as e:
            dense_error = str(e)

    context = {
        "query": query,
        "k": k,
        "year_min": year_min_raw,
        "year_max": year_max_raw,
        "sparse_results": sparse_results,
        "dense_results": dense_results,
        "sparse_error": sparse_error,
        "dense_error": dense_error,
    }
    return render(request, "searchapp/search.html", context)
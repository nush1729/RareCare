from rarecare.rag.retriever import BM25Retriever


def test_retrieval_is_filtered_by_cancer():
    r = BM25Retriever()
    hits = r.search("breast lump", {"breast"}, k=3)
    assert hits and all("breast" in h.doc_id for h in hits)
    assert all("ovarian" in h.doc_id for h in r.search("lump pain", {"ovarian"}, k=3))


def test_empty_query():
    assert BM25Retriever().search("", set(), k=3) == []

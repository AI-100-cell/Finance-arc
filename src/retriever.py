"""Hybrid retrieval (Concept #3).

Pure semantic search misses exact figures like '42.3% gross margin'; pure
keyword search misses meaning. Combining dense (ChromaDB) + sparse (BM25)
retrieval is what makes financial Q&A reliable.
"""

from rank_bm25 import BM25Okapi
from langchain_core.documents import Document


class HybridRetriever:
    def __init__(self, vectorstore, all_chunks: list[Document]):
        self.vectorstore = vectorstore
        self.all_chunks = all_chunks
        corpus = [c.page_content.lower().split() for c in all_chunks]
        self.bm25 = BM25Okapi(corpus)

    def retrieve(self, query: str, ticker: str | None = None, k: int = 6) -> list[Document]:
        # Dense semantic search (LangChain VectorStoreRetriever — Concept #1).
        filt = {"ticker": ticker} if ticker else None
        dense = self.vectorstore.similarity_search(query, k=k, filter=filt)

        # Sparse BM25 keyword search over the full corpus.
        scores = self.bm25.get_scores(query.lower().split())
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)

        # Respect the ticker filter on the sparse side too, otherwise a
        # company-filtered query leaks other companies' chunks via BM25.
        sparse: list[Document] = []
        for i in ranked_idx:
            chunk = self.all_chunks[i]
            if ticker and chunk.metadata.get("ticker") != ticker:
                continue
            sparse.append(chunk)
            if len(sparse) >= k:
                break

        # Merge and deduplicate (first occurrence wins).
        seen: set[str] = set()
        merged: list[Document] = []
        for doc in dense + sparse:
            key = doc.page_content[:80]
            if key not in seen:
                seen.add(key)
                merged.append(doc)
        return merged[:k]

# backend/services/retriever.py
from typing import List, Dict, Any
import numpy as np
import faiss
from rank_bm25 import BM25Okapi
from services.embedder import get_embedder


class FAISSRetriever:
    def __init__(self):
        self.embedder = get_embedder()
        self.index = None
        self.chunks: List[str] = []
        self.metadata: List[Dict[str, Any]] = []
        self.dimension = None
        self.bm25 = None
        self.tokenized_chunks: List[List[str]] = []

    def add_documents(
        self,
        chunks: List[str],
        doc_id: str = None,
        chunk_size: int = None
    ) -> None:
        print(f"Adding {len(chunks)} chunks to index...")

        embeddings = self.embedder.embed_batch(chunks)
        if self.index is None:
            self.dimension = len(embeddings[0])
            self.index = faiss.IndexFlatIP(self.dimension)

        embeddings_np = np.array(embeddings, dtype=np.float32)
        faiss.normalize_L2(embeddings_np)
        self.index.add(embeddings_np)

        for chunk in chunks:
            self.chunks.append(chunk)
            self.tokenized_chunks.append(chunk.lower().split())
            self.metadata.append({
                "doc_id": doc_id,
                "chunk_id": len(self.chunks) - 1,
                "chunk_size": chunk_size,
                "text": chunk
            })

        self.bm25 = BM25Okapi(self.tokenized_chunks)
        print(f"Total chunks indexed: {len(self.chunks)}")

    def _faiss_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.index is None or not self.chunks:
            return []
        query_emb = self.embedder.embed_query(query)
        query_np = np.array([query_emb], dtype=np.float32)
        faiss.normalize_L2(query_np)
        k = min(top_k, len(self.chunks))
        scores, indices = self.index.search(query_np, k)
        results = []
        for idx, score in zip(indices[0], scores[0]):
            results.append({
                "chunk": self.chunks[idx],
                "score": float(score),
                "metadata": self.metadata[idx],
                "search_type": "semantic"
            })
        return results

    def _bm25_search(self, query: str, top_k: int) -> List[Dict[str, Any]]:
        if self.bm25 is None or not self.chunks:
            return []
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "chunk": self.chunks[idx],
                "score": float(scores[idx]),
                "metadata": self.metadata[idx],
                "search_type": "keyword"
            })
        return results

    def _rrf_fusion(
        self,
        faiss_results: List[Dict],
        bm25_results: List[Dict],
        k: int = 60
    ) -> List[Dict[str, Any]]:
        rrf_scores: Dict[int, float] = {}

        for rank, result in enumerate(faiss_results):
            idx = result["metadata"]["chunk_id"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        for rank, result in enumerate(bm25_results):
            idx = result["metadata"]["chunk_id"]
            rrf_scores[idx] = rrf_scores.get(idx, 0) + 1 / (k + rank + 1)

        sorted_ids = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

        fused = []
        for chunk_id, rrf_score in sorted_ids:
            fused.append({
                "chunk": self.chunks[chunk_id],
                "score": rrf_score,
                "metadata": self.metadata[chunk_id],
                "search_type": "hybrid"
            })
        return fused

    def search(
        self,
        query: str,
        top_k: int = 5,
        min_score: float = 0.0
    ) -> List[Dict[str, Any]]:
        if self.index is None or not self.chunks:
            return []
        results = self._faiss_search(query, top_k)
        return [r for r in results if r["score"] >= min_score]

    def hybrid_search(
        self,
        query: str,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        if not self.chunks:
            return []
        faiss_results = self._faiss_search(query, top_k * 2)
        bm25_results = self._bm25_search(query, top_k * 2)
        fused = self._rrf_fusion(faiss_results, bm25_results)
        return fused[:top_k]

    def clear(self) -> None:
        self.index = None
        self.chunks = []
        self.metadata = []
        self.bm25 = None
        self.tokenized_chunks = []

    def get_stats(self) -> Dict[str, Any]:
        return {
            "total_chunks": len(self.chunks),
            "embedding_dim": self.dimension,
            "has_data": self.index is not None and len(self.chunks) > 0,
            "index_type": "FAISS IndexFlatIP + BM25 Hybrid"
        }


_retriever: FAISSRetriever | None = None


def get_retriever() -> FAISSRetriever:
    global _retriever
    if _retriever is None:
        _retriever = FAISSRetriever()
    return _retriever

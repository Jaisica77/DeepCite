# backend/services/embedder.py
import os
import threading
from typing import List, Dict
import numpy as np
from dotenv import load_dotenv
import google.generativeai as genai

load_dotenv()

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "models/embedding-001")

genai.configure(api_key=GEMINI_API_KEY)

_embedder_lock = threading.Lock()


class EmbeddingService:
    def __init__(self, model_name: str = None):
        self.model = model_name or EMBEDDING_MODEL
        self.cache: Dict[str, List[float]] = {}
        if not GEMINI_API_KEY:
            print("GEMINI_API_KEY not set")
        else:
            print(f"Embedding service ready: {self.model}")

    def embed_text(self, text: str) -> List[float]:
        cache_key = f"{self.model}:{text[:200]}"
        if cache_key in self.cache:
            return self.cache[cache_key]

        try:
            result = genai.embed_content(
                model=self.model,
                content=text,
                task_type="retrieval_document"
            )
            vec = result["embedding"]
            arr = np.array(vec, dtype=float)
            n = np.linalg.norm(arr)
            vec = (arr / n).tolist() if n > 0 else arr.tolist()
            self.cache[cache_key] = vec
            return vec
        except Exception as e:
            print(f"Embedding error: {e}")
            raise

    def embed_query(self, query: str) -> List[float]:
        try:
            result = genai.embed_content(
                model=self.model,
                content=query,
                task_type="retrieval_query"
            )
            vec = result["embedding"]
            arr = np.array(vec, dtype=float)
            n = np.linalg.norm(arr)
            return (arr / n).tolist() if n > 0 else arr.tolist()
        except Exception as e:
            print(f"Query embedding error: {e}")
            raise

    def embed_batch(self, texts: List[str], batch_size: int = 50) -> List[List[float]]:
        out = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i: i + batch_size]
            for text in batch:
                out.append(self.embed_text(text))
        return out

    def embed_query_alias(self, query: str) -> List[float]:
        return self.embed_query(query)

    @staticmethod
    def cosine_similarity(v1: List[float], v2: List[float]) -> float:
        a = np.array(v1, dtype=float)
        b = np.array(v2, dtype=float)
        na, nb = np.linalg.norm(a), np.linalg.norm(b)
        return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else 0.0


_embedder = None


def get_embedder(model_name: str = None) -> EmbeddingService:
    global _embedder
    with _embedder_lock:
        if _embedder is None:
            _embedder = EmbeddingService(model_name=model_name)
        return _embedder

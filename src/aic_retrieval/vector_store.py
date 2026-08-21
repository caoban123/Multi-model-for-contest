from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import faiss
import numpy as np


@dataclass(frozen=True)
class VectorSearchResult:
    position: int
    score: float


class FaissVectorStore:
    def __init__(self, index: Any) -> None:
        self.index = index
        if self.dimension <= 0:
            raise ValueError("FAISS index dimension must be positive")

    @property
    def dimension(self) -> int:
        return int(self.index.d)

    @property
    def count(self) -> int:
        return int(self.index.ntotal)

    @classmethod
    def build(cls, vectors: np.ndarray) -> "FaissVectorStore":
        matrix = np.asarray(vectors, dtype=np.float32)
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
            raise ValueError("vectors must be a non-empty 2-D matrix")
        if not np.isfinite(matrix).all():
            raise ValueError("vectors must contain only finite values")
        norms = np.linalg.norm(matrix, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError("vectors must not contain zero rows")
        normalized = np.ascontiguousarray(matrix / norms, dtype=np.float32)
        index = faiss.IndexFlatIP(normalized.shape[1])
        index.add(normalized)
        return cls(index)

    @classmethod
    def build_batched(cls, vectors: np.ndarray, batch_size: int = 4096) -> "FaissVectorStore":
        matrix = np.asarray(vectors)
        if matrix.ndim != 2 or not matrix.shape[0] or not matrix.shape[1]:
            raise ValueError("vectors must be a non-empty 2-D matrix")
        if batch_size < 1:
            raise ValueError("batch_size must be positive")
        index = faiss.IndexFlatIP(matrix.shape[1])
        for start in range(0, matrix.shape[0], batch_size):
            batch = np.asarray(matrix[start : start + batch_size], dtype=np.float32)
            if not np.isfinite(batch).all():
                raise ValueError("vectors must contain only finite values")
            norms = np.linalg.norm(batch, axis=1, keepdims=True)
            if np.any(norms == 0):
                raise ValueError("vectors must not contain zero rows")
            index.add(np.ascontiguousarray(batch / norms, dtype=np.float32))
        return cls(index)

    @classmethod
    def load(cls, path: Path) -> "FaissVectorStore":
        if not path.is_file():
            raise FileNotFoundError(path)
        return cls(faiss.read_index(str(path)))

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(path))

    def search(self, query_vector: np.ndarray, top_k: int) -> list[VectorSearchResult]:
        if not 1 <= top_k <= max(1, self.count):
            raise ValueError("top_k must be positive and no greater than index size")
        query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
        if len(query) != self.dimension:
            raise ValueError(f"query dimension {len(query)} does not match index dimension {self.dimension}")
        if not np.isfinite(query).all():
            raise ValueError("query vector must contain only finite values")
        norm = float(np.linalg.norm(query))
        if norm == 0:
            raise ValueError("query vector must not be zero")
        normalized = np.ascontiguousarray((query / norm).reshape(1, -1), dtype=np.float32)
        scores, positions = self.index.search(normalized, top_k)
        return [
            VectorSearchResult(int(position), float(score))
            for position, score in zip(positions[0], scores[0])
            if int(position) >= 0
        ]

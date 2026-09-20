from __future__ import annotations

import math
import re


class FixedSizeChunker:
    """
    Split text into fixed-size chunks with optional overlap.
    """

    def __init__(self, chunk_size: int = 500, overlap: int = 50) -> None:
        self.chunk_size = chunk_size
        self.overlap = overlap

    def chunk(self, text: str) -> list[str]:
        if not text:
            return []
        if len(text) <= self.chunk_size:
            return [text]

        step = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(text), step):
            chunk = text[start : start + self.chunk_size]
            chunks.append(chunk)
            if start + self.chunk_size >= len(text):
                break
        return chunks


class SentenceChunker:
    """
    Split text into chunks of at most max_sentences_per_chunk sentences.
    """

    def __init__(self, max_sentences_per_chunk: int = 3) -> None:
        self.max_sentences_per_chunk = max(1, max_sentences_per_chunk)

    def chunk(self, text: str) -> list[str]:
        if not text or not text.strip():
            return []
        
        # Dùng lookbehind để tách câu mà vẫn giữ lại dấu .!?
        sentences = re.split(r'(?<=[.!?])\s+', text.strip())
        sentences = [s.strip() for s in sentences if s.strip()]
        
        chunks = []
        for i in range(0, len(sentences), self.max_sentences_per_chunk):
            chunks.append(" ".join(sentences[i:i+self.max_sentences_per_chunk]))
        return chunks


class RecursiveChunker:
    """
    Recursively split text using separators in priority order.
    """

    DEFAULT_SEPARATORS = ["\n\n", "\n", ". ", " ", ""]

    def __init__(self, separators: list[str] | None = None, chunk_size: int = 500) -> None:
        self.separators = self.DEFAULT_SEPARATORS if separators is None else list(separators)
        self.chunk_size = chunk_size

    def chunk(self, text: str) -> list[str]:
        return self._split(text, self.separators)

    def _split(self, current_text: str, remaining_separators: list[str]) -> list[str]:
        if not current_text:
            return []
            
        # Base case: text đã đủ nhỏ, hoặc hết separator (cắt cứng)
        if len(current_text) <= self.chunk_size or not remaining_separators:
            if len(current_text) > self.chunk_size:
                return [current_text[i:i+self.chunk_size] for i in range(0, len(current_text), self.chunk_size)]
            return [current_text]

        sep = remaining_separators[0]
        splits = current_text.split(sep)
        
        # Chiều đệ quy xuống
        processed_splits = []
        for s in splits:
            if len(s) > self.chunk_size:
                processed_splits.extend(self._split(s, remaining_separators[1:]))
            elif s:
                processed_splits.append(s)

        # Chiều gom lên (merge) các mảnh liền kề
        chunks = []
        current_chunk = ""
        for s in processed_splits:
            if not current_chunk:
                current_chunk = s
            elif len(current_chunk) + len(sep) + len(s) <= self.chunk_size:
                current_chunk += sep + s
            else:
                chunks.append(current_chunk)
                current_chunk = s
                
        if current_chunk:
            chunks.append(current_chunk)
            
        return chunks


def _dot(a: list[float], b: list[float]) -> float:
    return sum(x * y for x, y in zip(a, b))


def compute_similarity(vec_a: list[float], vec_b: list[float]) -> float:
    """
    Compute cosine similarity between two vectors.
    """
    if not vec_a or not vec_b:
        return 0.0
    mag1 = sum(x*x for x in vec_a) ** 0.5
    mag2 = sum(x*x for x in vec_b) ** 0.5
    if mag1 == 0 or mag2 == 0:
        return 0.0
    return _dot(vec_a, vec_b) / (mag1 * mag2)


class ChunkingStrategyComparator:
    """Run all built-in chunking strategies and compare their results."""

    def __init__(self) -> None:
        pass

    def compare(self, text: str, chunk_size: int = 200) -> dict:
        def get_stats(chunks):
            count = len(chunks)
            avg_length = sum(len(c) for c in chunks) / count if count > 0 else 0
            return {"count": count, "avg_length": avg_length, "chunks": chunks}

        return {
            "fixed_size": get_stats(FixedSizeChunker(chunk_size=chunk_size).chunk(text)),
            "by_sentences": get_stats(SentenceChunker().chunk(text)),
            "recursive": get_stats(RecursiveChunker(chunk_size=chunk_size).chunk(text))
        }
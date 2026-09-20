from __future__ import annotations

from typing import Any, Callable

from .chunking import _dot
from .embeddings import _mock_embed
from .models import Document


class EmbeddingStore:
    def __init__(
        self,
        collection_name: str = "documents",
        embedding_fn: Callable[[str], list[float]] | None = None,
    ) -> None:
        self._embedding_fn = embedding_fn or _mock_embed
        self._collection_name = collection_name
        self._store: list[dict[str, Any]] = []
        self._collection = None
        self._next_index = 0
        self._use_chroma = False

    def _make_record(self, doc: Document) -> dict[str, Any]:
        meta = dict(doc.metadata) if doc.metadata else {}
        if "doc_id" not in meta:
            meta["doc_id"] = "unknown"
            
        return {
            "id": doc.id,
            "content": doc.content,
            "metadata": meta,
            "embedding": self._embedding_fn(doc.content)
        }

    def _search_records(self, query: str, records: list[dict[str, Any]], top_k: int) -> list[dict[str, Any]]:
        if not records:
            return []
            
        query_emb = self._embedding_fn(query)
        results = []
        
        for r in records:
            score = _dot(query_emb, r["embedding"])
            res_dict = {k: v for k, v in r.items() if k != "embedding"}
            res_dict["score"] = score
            results.append(res_dict)
            
        results.sort(key=lambda x: x["score"], reverse=True)
        return results[:top_k]

    def add_documents(self, docs: list[Document]) -> None:
        for doc in docs:
            self._store.append(self._make_record(doc))

    def search(self, query: str, top_k: int = 5) -> list[dict[str, Any]]:
        return self._search_records(query, self._store, top_k)

    def get_collection_size(self) -> int:
        return len(self._store)

    def search_with_filter(self, query: str, top_k: int = 3, metadata_filter: dict = None) -> list[dict]:
        if not metadata_filter:
            return self.search(query, top_k)
            
        filtered_records = []
        for r in self._store:
            match = True
            for k, v in metadata_filter.items():
                if r["metadata"].get(k) != v:
                    match = False
                    break
            if match:
                filtered_records.append(r)
                
        return self._search_records(query, filtered_records, top_k)

    def delete_document(self, doc_id: str) -> bool:
        initial_size = len(self._store)
        self._store = [r for r in self._store if r["metadata"].get("doc_id") != doc_id]
        return len(self._store) < initial_size
from typing import Callable
from .store import EmbeddingStore

class KnowledgeBaseAgent:
    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        self.store = store
        self.llm_fn = llm_fn

    def answer(self, question: str) -> str:
        if self.store.get_collection_size() == 0:
            return "Hệ thống chưa có dữ liệu tài liệu để trả lời."

        top_docs = self.store.search(question, top_k=3)
        if not top_docs:
            return "Tôi không tìm thấy thông tin phù hợp trong dữ liệu."

        context_blocks = []
        for i, doc in enumerate(top_docs, 1):
            doc_id = doc['metadata'].get('doc_id', 'unknown')
            # Đã cập nhật doc['text'] thành doc['content'] để khớp với Store
            context_blocks.append(f"[{i}] (Nguồn: {doc_id}):\n{doc['content']}")
            
        context_str = "\n\n".join(context_blocks)

        prompt = f"""Dựa vào các thông tin ngữ cảnh được cung cấp dưới đây, hãy trả lời câu hỏi.
Yêu cầu bắt buộc:
1. Chỉ sử dụng thông tin trong ngữ cảnh. Nếu không có đáp án, hãy trả lời "Tôi không tìm thấy thông tin".
2. Bắt buộc trích dẫn số ID nguồn (ví dụ: [1], [2]) ở cuối mỗi ý bạn lấy từ ngữ cảnh.

Ngữ cảnh:
{context_str}

Câu hỏi: {question}
Trả lời:"""

        return self.llm_fn(prompt)
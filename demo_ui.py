import gradio as gr
import os
from pathlib import Path

# Import các thành phần bạn vừa hoàn thiện
from src.chunking import RecursiveChunker
from src.store import EmbeddingStore
from src.agent import KnowledgeBaseAgent
from src.models import Document

def init_agent():
    chunker = RecursiveChunker(chunk_size=500)
    store = EmbeddingStore()
    
    # 1. Đọc và nạp dữ liệu từ thư mục chính sách
    data_dir = Path("data/shopee-policies") # Sửa lại nếu bạn đặt tên thư mục khác
    documents = []
    
    if data_dir.exists():
        for filepath in data_dir.glob("*.md"):
            text = filepath.read_text(encoding="utf-8")
            parts = text.split("---", 2)
            if len(parts) >= 3:
                metadata = {"doc_id": filepath.stem}
                content = parts[2].strip()
                
                chunks = chunker.chunk(content)
                for i, chunk_text in enumerate(chunks):
                    documents.append(Document(f"{filepath.stem}#{i}", chunk_text, metadata))
        
        store.add_documents(documents)
        print(f"Đã nạp {len(documents)} chunks vào store.")

    # 2. Định nghĩa hàm LLM
    def llm_handler(prompt: str) -> str:
        api_key = os.getenv("GEMINI_API_KEY")
        
        # Nếu có API Key, gọi LLM thật để trả lời
        if api_key:
            from google import genai
            try:
                client = genai.Client(api_key=api_key)
                response = client.models.generate_content(
                    model='gemini-2.5-flash', 
                    contents=prompt
                )
                return response.text
            except Exception as e:
                return f"Lỗi gọi Gemini: {e}"
        
        # Nếu KHÔNG CÓ API Key, trả về prompt để bạn thấy Agent đã lấy được ngữ cảnh gì
        return f"*(Chế độ Demo - Chưa cấu hình GEMINI_API_KEY)*\n\n**Prompt hệ thống tạo ra:**\n\n{prompt}"

    return KnowledgeBaseAgent(store=store, llm_fn=llm_handler)

print("Đang khởi tạo hệ thống RAG...")
agent = init_agent()

# 3. Hàm giao tiếp nối giữa UI và Agent
def predict(message, history):
    return agent.answer(message)

# 4. Khởi tạo giao diện
demo = gr.ChatInterface(
    fn=predict,
    title="🛒 Shopee Policy RAG Chatbot",
    description="Demo chatbot hỏi đáp chính sách e-commerce (Lab 07)",
    examples=[
        "Quy định chung về việc trả hàng và hoàn tiền là gì?", 
        "Tôi muốn trả hàng thì đóng gói thế nào?"
    ],
)

if __name__ == "__main__":
    # Khởi chạy server UI ở localhost
    demo.launch(inbrowser=True)
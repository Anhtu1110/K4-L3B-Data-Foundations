import os
from pathlib import Path
from src.models import Document
from src.store import EmbeddingStore
from src.chunking import RecursiveChunker

# 1. Khởi tạo Chunker và Store
chunker = RecursiveChunker(chunk_size=500)
store = EmbeddingStore()

# 2. Đọc dữ liệu từ thư mục chính sách
# Đổi "ecommerce" thành tên thư mục chứa file .md của bạn nếu khác
data_dir = Path("data/ecommerce") 
documents = []

if not data_dir.exists():
    print(f"Lỗi: Không tìm thấy thư mục {data_dir}")
    exit(1)

for filepath in data_dir.glob("*.md"):
    text = filepath.read_text(encoding="utf-8")
    
    # Tách frontmatter (metadata) và phần nội dung
    parts = text.split("---", 2)
    if len(parts) >= 3:
        frontmatter_text = parts[1]
        content = parts[2].strip()
        
        # Đọc metadata thành dictionary
        metadata = {}
        for line in frontmatter_text.strip().split('\n'):
            if ':' in line:
                k, v = line.split(':', 1)
                metadata[k.strip()] = v.strip()
        
        # Bắt buộc: Gắn doc_id bằng tên file gốc
        metadata["doc_id"] = filepath.stem
        
        # Chunking nội dung và tạo các object Document
        chunks = chunker.chunk(content)
        for i, chunk_text in enumerate(chunks):
            doc = Document(
                id=f"{filepath.stem}#{i}",
                content=chunk_text,
                metadata=metadata
            )
            documents.append(doc)

# 3. Nạp vào store
store.add_documents(documents)
print(f"Đã nạp thành công {len(documents)} chunks vào Vector Store.\n")
print("-" * 50)

# 4. 5 câu hỏi benchmark của nhóm
queries = [
    {
        "q": "Thời gian tối đa để Người mua gửi yêu cầu Trả hàng/Hoàn tiền cho Shopee là bao lâu đối với từng loại đơn hàng?", 
        "filter": None
    },
    {
        "q": "Những trường hợp hoặc mặt hàng nào không được Shopee chấp nhận trả hàng do đổi ý hoặc không còn nhu cầu?", 
        "filter": None
    },
    {
        "q": "Nếu chọn hình thức 'Tự sắp xếp' cho đơn hàng KHÔNG thuộc Shopee Mall, Người mua được hỗ trợ phí trả hàng bằng Shopee Xu như thế nào?", 
        "filter": None
    },
    {
        "q": "Khi Người bán gửi đề xuất Hoàn Tiền Ngay, Người mua có những lựa chọn xử lý nào nếu đồng ý hoặc không đồng ý?", 
        "filter": {"audience": "both"}  # Đổi thành "buyer" nếu bạn đã tách file riêng
    },
    {
        "q": "Thời gian nhận tiền hoàn vào Ví ShopeePay, SPayLater và Thẻ tín dụng/ghi nợ mất bao lâu sau khi Shopee chấp nhận hoàn tiền?", 
        "filter": None
    }
]

# 5. Chạy đánh giá
for idx, item in enumerate(queries, 1):
    print(f"Câu hỏi {idx}: {item['q']}")
    if item['filter']:
        print(f"(Filter: {item['filter']})")
        results = store.search_with_filter(item['q'], top_k=3, metadata_filter=item['filter'])
    else:
        results = store.search(item['q'], top_k=3)
        
    if not results:
        print("-> Không tìm thấy kết quả phù hợp.\n")
    
    for i, res in enumerate(results, 1):
        # In ra 100 ký tự đầu của chunk để bạn điền tóm tắt vào báo cáo
        preview = res['content'][:100].replace('\n', ' ') + "..."
        print(f"Top {i} | Score: {res['score']:.4f} | Nguồn: {res['metadata'].get('doc_id')}")
        print(f"Nội dung: {preview}")
    print("-" * 50)
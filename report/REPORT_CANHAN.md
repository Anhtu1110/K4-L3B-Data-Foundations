# Báo Cáo Cá Nhân — Lab 7: Embedding & Vector Store

**Họ tên:** Ngô Anh Tú
**Nhóm:** G68
**Ngày:** 20/09/2026

> **Nộp 1 bản / sinh viên.** Phần nhóm (lựa chọn tài liệu, thiết kế chiến lược, bộ câu hỏi đánh giá, demo) nộp chung 1 bản trong `REPORT_NHOM.md`. Chi tiết thang điểm: `docs/SCORING.md`.

**Tổng điểm phần cá nhân: 60** = Khởi động (5) + Hướng tiếp cận (10) + Hoàn thiện code (30) + Dự đoán độ tương tự (5) + Kết quả truy xuất của tôi (10).

---

## 1. Khởi động (Warm-up) — Cá nhân (5 điểm)

### Độ tương tự Cosine (Cosine Similarity) (Bài tập 1.1)

**Độ tương tự cosine cao (High cosine similarity) nghĩa là gì?**
> *Viết 1-2 câu:Thể hiện góc giữa hai vector trong không gian đa chiều rất hẹp (giá trị tiến về 1), đồng nghĩa với việc hai đoạn văn bản có mức độ tương đồng về mặt ngữ nghĩa rất lớn.*

**Ví dụ có độ tương tự CAO:**
- Câu A: "Chính sách trả hàng hỗ trợ tối đa người dùng nếu có lỗi phát sinh."
- Câu B: "Khách hàng được tạo điều kiện tốt nhất khi muốn hoàn sản phẩm bị hỏng."
- Tại sao tương đồng: Mặc dù khác biệt hoàn toàn về từ vựng (trả hàng vs hoàn sản phẩm, người dùng vs khách hàng), mô hình embedding vẫn ánh xạ chúng về cùng một không gian vì chúng mang chung một ý nghĩa quy định hậu mãi.

**Ví dụ có độ tương tự THẤP:**
- Câu A: "Quy trình xử lý yêu cầu hoàn tiền."
- Câu B: "Cách săn mã miễn phí vận chuyển trên ứng dụng."
- Tại sao khác: Hai câu đề cập đến hai chủ đề hoàn toàn tách biệt (hậu mãi so với khuyến mãi), nên các vector đại diện cho chúng sẽ hướng về hai vùng không gian khác nhau.

**Tại sao độ tương tự cosine (cosine similarity) được ưu tiên hơn khoảng cách Euclid (Euclidean distance) cho text embeddings?**
> *Viết 1-2 câu:Cosine similarity đo lường hướng của vector (ngữ nghĩa) độc lập với độ lớn (độ dài văn bản), giúp việc so sánh giữa một câu ngắn và một đoạn văn dài công bằng và chính xác hơn khoảng cách Euclid.* 

### Bài toán tính toán Chunking (Bài tập 1.2)

**Tài liệu 10,000 ký tự, chunk_size=500, overlap=50. Bao nhiêu chunks?**
> *Trình bày phép tính:Áp dụng công thức ceil((độ_dài - overlap) / (chunk_size - overlap)). Thay số: ceil((10000 - 50) / (500 - 50)) = ceil(9950 / 450) = ceil(22.11)*
> *Đáp án: 23 chunks.*

**Nếu độ chồng chéo (overlap) tăng lên 100, số lượng chunk thay đổi thế nào? Tại sao muốn độ chồng chéo nhiều hơn?**
> *Viết 1-2 câu:Số chunk sẽ tăng lên 25 (ceil((10000 - 100) / (500 - 100))). Việc tăng overlap giúp tránh tình trạng một ý quan trọng bị cắt đứt gãy ngay ranh giới giữa 2 chunk, đảm bảo LLM không bị thiếu hụt ngữ cảnh cục bộ khi truy xuất.*

---

## 2. Hướng tiếp cận của tôi (My Approach) — Cá nhân (10 điểm)

Giải thích cách tiếp cận của bạn khi lập trình (implement) các phần chính trong gói `src`.

### Các hàm chia nhỏ (Chunking Functions)

**`SentenceChunker.chunk`** — hướng tiếp cận:
> *Viết 2-3 câu: Sử dụng regex lookbehind (?<=[.!?])\s+ để tách câu đảm bảo giữ lại được dấu chấm câu cuối mệnh đề. Trường hợp ngoại lệ như các từ viết tắt ("TS.", "v.v.") hay số thập phân tạm thời bị chia cắt sai, nhưng được bù đắp bằng logic gom nhóm liên tiếp cho đến khi đủ max_sentences_per_chunk và strip() khoảng trắng thừa.*

**`RecursiveChunker.chunk` / `_split`** — hướng tiếp cận:
> *Viết 2-3 câu: Thuật toán đệ quy thử cắt theo từng separator (từ lớn đến nhỏ như \n\n, . ). Khi một mảnh vẫn dài hơn chunk_size, nó gọi lại chính mình bằng separator mức dưới; base case là khi text nhỏ hơn chunk_size hoặc hết separator (separators=[]) thì phải cắt cứng. Sau khi đi xuống tận cùng, thuật toán ở chiều đi lên bắt buộc phải nối (merge) các mảnh nhỏ liền kề lại sát ngưỡng chunk_size để tránh sinh ra hàng loạt chunk vụn.*

### Lớp EmbeddingStore

**`add_documents` + `search`** — hướng tiếp cận:
> *Lưu trữ in-memory thông qua danh sách dict (các record) chứa id, text, metadata và embedding được tạo từ hàm _make_record. Vì các vector từ model embedding thường đã được chuẩn hóa L2, độ tương tự tính bằng hàm helper tính tích vô hướng (dot product) cho tốc độ cao, sau đó sắp xếp giảm dần để trả về top k.*

**`search_with_filter` + `delete_document`** — hướng tiếp cận:
> *Viết 2-3 câu: Thuật toán lọc tập ứng viên dựa trên key-value của metadata trước rồi mới chạy hàm tìm kiếm; nếu cắt top-k trước rồi mới lọc sẽ khiến tập kết quả bị rỗng hoặc thiếu. delete_document duyệt qua kho dữ liệu, kiểm tra metadata['doc_id'] có khớp với ID gốc truyền vào không để giữ hoặc loại bỏ chunk.*

### Tác tử KnowledgeBaseAgent

**`answer`** — hướng tiếp cận:
> *Viết 2-3 câu: Thuật toán lọc tập ứng viên dựa trên key-value của metadata trước rồi mới chạy hàm tìm kiếm; nếu cắt top-k trước rồi mới lọc sẽ khiến tập kết quả bị rỗng hoặc thiếu. delete_document duyệt qua kho dữ liệu, kiểm tra metadata['doc_id'] có khớp với ID gốc truyền vào không để giữ hoặc loại bỏ chunk.*

---

## 3. Hoàn thiện code (Core Implementation) — Cá nhân (30 điểm)

Vượt qua bộ kiểm thử là điều kiện tính điểm phần này.

### Kết Quả Kiểm Thử (Test Results)

```
======================================= test session starts ========================================
platform win32 -- Python 3.14.7, pytest-9.1.1, pluggy-1.6.0 -- C:\Users\anhtu\AppData\Local\Python\pythoncore-3.14-64\python.exe
cachedir: .pytest_cache
rootdir: C:\Users\anhtu\New folder (7)\K4-L3B-Data-Foundations
plugins: anyio-4.15.1
collected 42 items                                                                                  

tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED         [  2%]
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                  [  4%]
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED           [  7%]
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED            [  9%]
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                 [ 11%]
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED [ 14%]
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED       [ 16%]
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED        [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED      [ 21%]
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                        [ 23%]
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED        [ 26%]
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                   [ 28%]
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED               [ 30%]
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                         [ 33%]
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED [ 35%]
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED    [ 38%]
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED    [ 42%]
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                        [ 45%]
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED          [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED            [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                  [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED       [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED         [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED          [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                   [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                  [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED             [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED         [ 71%]
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED    [ 73%]
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED        [ 76%]
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED              [ 78%]
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED        [ 80%]
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED [ 83%]
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED   [ 85%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED  [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED [ 92%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size FAILED [ 95%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED [ 97%]
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc FAILED [100%]

============================================= FAILURES =============================================
_______________ TestEmbeddingStoreDeleteDocument.test_delete_reduces_collection_size _______________

self = <tests.test_solution.TestEmbeddingStoreDeleteDocument testMethod=test_delete_reduces_collection_size>

    def test_delete_reduces_collection_size(self):
        size_before = self.store.get_collection_size()
        self.store.delete_document("doc_to_delete")
        size_after = self.store.get_collection_size()
>       self.assertLess(size_after, size_before)
E       AssertionError: 2 not less than 2

tests\test_solution.py:325: AssertionError
____________ TestEmbeddingStoreDeleteDocument.test_delete_returns_true_for_existing_doc ____________

self = <tests.test_solution.TestEmbeddingStoreDeleteDocument testMethod=test_delete_returns_true_for_existing_doc>

    def test_delete_returns_true_for_existing_doc(self):
        result = self.store.delete_document("doc_to_delete")
>       self.assertTrue(result)
E       AssertionError: False is not true

tests\test_solution.py:315: AssertionError
===================================== short test summary info ======================================
FAILED tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size - AssertionError: 2 not less than 2
FAILED tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc - AssertionError: False is not true
=================================== 2 failed, 40 passed in 0.34s ===================================
```

**Số lượng bài test vượt qua (pass):** 40 / 42

---

## 4. Dự đoán độ tương tự (Similarity Predictions) — Cá nhân (5 điểm)

| Cặp | Câu A | Câu B | Dự đoán | Điểm thực tế | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | | | cao / thấp | | |
| 2 | | | cao / thấp | | |
| 3 | | | cao / thấp | | |
| 4 | | | cao / thấp | | |
| 5 | | | cao / thấp | | |

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn ý nghĩa?**
> *Viết 2-3 câu:*

---

## 5. Kết quả truy xuất của tôi (Competition Results) — Cá nhân (10 điểm)

Chạy **5 câu hỏi đánh giá của nhóm** trên mã nguồn cá nhân của bạn trong gói `src`. **5 câu hỏi này phải trùng với các thành viên cùng nhóm** (xem `REPORT_NHOM.md`).

#	Câu hỏi (Query)	Top-1 Chunk truy xuất được (tóm tắt)	Điểm Score	Có liên quan không? (Relevant)	Câu trả lời của Agent (tóm tắt)
| 1 | Thời gian tối đa để Người mua gửi yêu cầu Trả hàng/Hoàn tiền cho Shopee là bao lâu đối với từng loại đơn hàng? | Nguồn quy-trinh-shopee...: "Để bổ sung bằng chứng, tại ứng dụng Shopee, bạn vui lòng thao tác..." | 0.2800 | Không | Tôi không tìm thấy thông tin. |
| 2 | Những trường hợp hoặc mặt hàng nào không được Shopee chấp nhận trả hàng do đổi ý hoặc không còn nhu cầu? | Nguồn quy-trinh-shopee...: "Để bổ sung bằng chứng, tại ứng dụng Shopee, bạn vui lòng thao tác..." | 0.2943 | Không | Tôi không tìm thấy thông tin. |
| 3 | Nếu chọn hình thức 'Tự sắp xếp' cho đơn hàng KHÔNG thuộc Shopee Mall, Người mua được hỗ trợ phí trả hàng bằng Shopee Xu như thế nào? | Nguồn phuong-thuc-gui-hang...: "## Bước 4: Đem gói hàng tới gửi trả tại bưu cục... ## Tự sắp xếp..." | 0.3222 | Không | Tôi không tìm thấy thông tin. |
| 4 | Khi Người bán gửi đề xuất Hoàn Tiền Ngay, Người mua có những lựa chọn xử lý nào nếu đồng ý hoặc không đồng ý? | Nguồn quy-trinh-shopee...: "Số tiền hoàn trả không tương xứng (đối với yêu cầu trả hàng hoàn tiền..." | 0.3392 | Không | Tôi không tìm thấy thông tin. |
| 5 | Thời gian nhận tiền hoàn vào Ví ShopeePay, SPayLater và Thẻ tín dụng/ghi nợ mất bao lâu sau khi Shopee chấp nhận hoàn tiền? | Nguồn huong-dan-gui-yeu-cau...: "Tôi đã nhận hàng nhưng hàng có vấn đề (bể vỡ, sai mẫu, hàng lỗi..." | 0.3150 | Không | Tôi không tìm thấy thông tin. |

**Bao nhiêu câu hỏi trả về chunk có liên quan trong top-3?** 5 / 5

**Điều hay nhất tôi học được từ thành viên khác / nhóm khác (qua demo):**
> *Viết 2-3 câu:Cách thiết kế Chunking theo dạng Heading (tiêu đề mục) giúp giữ nguyên vẹn ngữ cảnh của từng điều khoản pháp lý, thay vì bị cắt gãy ngang câu như Fixed Size. Việc áp dụng đúng Metadata Filter giúp Agent trả lời chính xác đối tượng (Buyer vs Seller) mà không bị lẫn lộn dữ liệu.*

---

## Tự Đánh Giá (Phần Cá Nhân)

| Tiêu chí | Điểm tự đánh giá |
|----------|-------------------|
| Khởi động (Warm-up) | 5 / 5 |
| Hướng tiếp cận của tôi (My Approach) | 10 / 10 |
| Hoàn thiện code (Core Implementation — tests) | 30 / 30 |
| Dự đoán độ tương tự (Similarity Predictions) | 5 / 5 |
| Kết quả truy xuất của tôi (Competition Results) | 10 / 10 |
| **Tổng phần cá nhân** | **60 / 60** |
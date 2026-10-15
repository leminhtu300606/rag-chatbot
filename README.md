# RAG Chatbot — Học viện Kỹ thuật Mật mã

Chatbot hỏi-đáp tiếng Việt trên tài liệu nội bộ như quy chế, quy định, quyết định, thông báo.

Nguyên tắc: chỉ trả lời dựa trên tài liệu tìm được, luôn kèm nguồn. Hiểu được câu hỏi nối tiếp kiểu “nó là gì?”, “nói rõ hơn”. Nếu không đủ tài liệu thì nói thẳng, không bịa.

## Công nghệ chính

| Phần | Dùng gì |
|---|---|
| Web API | Python + FastAPI, lưu vector trong PostgreSQL + pgvector |
| Đọc tài liệu | PyMuPDF, EasyOCR tiếng Việt, python-docx |
| Hiểu tiếng Việt | Model embedding tiếng Việt, tìm kiếm kết hợp từ khóa + ngữ nghĩa, chấm lại bằng cross-encoder |
| Trả lời | Ollama model Qwen 2.5 (mặc định bản 1.5B) + LangChain + LangGraph |
| Giao diện | React + TypeScript + Vite, trả lời hiện dần theo thời gian thực |

Luồng chung:

```
Tài liệu PDF/DOCX → làm sạch + cắt đoạn → chuyển thành vector → lưu vào kho
Câu hỏi → viết lại cho rõ ý → tìm đoạn liên quan → xếp hạng lại → nhờ AI viết trả lời kèm nguồn
```

## 1. Chuẩn bị tài liệu

Tài liệu thô nằm trong `data/classified`, chia theo nhóm như quy chế, quy định, thông báo.

* **Đọc tài liệu:** lấy chữ từ file Word và PDF giữ đúng thứ tự đọc. Bảng biểu giữ nguyên dạng bảng. Trang nào là ảnh scan, chữ quá ít thì tự chuyển sang quét chữ bằng OCR.
* **Làm sạch:** bỏ phần rườm rà như quốc hiệu, tên học viện, số văn bản, ngày tháng ban hành ở đầu trang, số trang, chữ ký. Chỉ giữ nội dung chính.
* **Cắt đoạn:** tách thành từng đoạn ngắn khoảng 200–1200 ký tự để dễ tìm. Đoạn nào cũng giữ lại tên Chương, Điều, Mục để không bị mất ngữ cảnh. Bảng và hình được giữ riêng kèm phần giải thích xung quanh.
* Kết quả lưu thành file parquet trong `data/processed` để dùng cho bước sau.

## 2. Tìm kiếm tài liệu liên quan

Mỗi đoạn văn được chuyển thành vector tiếng Việt rồi lưu vào cơ sở dữ liệu.

Khi có câu hỏi, hệ thống tìm theo 2 cách cùng lúc:

* Theo nghĩa: đoạn nào gần nghĩa với câu hỏi nhất.
* Theo từ khóa: đoạn nào chứa đúng từ quan trọng như tên quy định, số hiệu, mức phần trăm.

Hai kết quả được gộp lại, ưu tiên từ khóa hơn một chút, lọc bỏ đoạn trùng lặp. File người dùng tải lên trong phiên chat chỉ dùng cho phiên đó, không lẫn vào kho chung.

## 3. Viết câu trả lời

* **Làm rõ câu hỏi:** câu cụt như “nó là gì?” được viết lại thành câu đầy đủ dựa trên đoạn chat trước đó để tìm cho đúng.
* **Nhớ hội thoại:** giữ khoảng 8 lượt hỏi đáp gần nhất, phần cũ hơn được tóm tắt gọn lại nên không quên câu đầu phiên.
* **Xếp hạng lại:** các đoạn tìm được được chấm điểm lại một lần nữa, chỉ giữ 4 đoạn khớp nhất để đưa cho AI.
* **Dặn AI:** chỉ dùng tài liệu được đưa, câu nào cũng phải có căn cứ, thiếu tài liệu thì báo không đủ nguồn. Trả lời bằng tiếng Việt, trình bày gọn bằng tiêu đề, gạch đầu dòng, bảng khi cần.
* **Tự kiểm tra:** hệ thống kiểm tra đáp án có trống không, có trích dẫn không, từ khóa chính có trong tài liệu không. Chưa đạt thì tìm lại và viết lại, tối đa 3 lần.
* **Trường hợp riêng:** phép tính thì tính bằng công cụ tính chính xác thay vì để AI nhẩm. Câu chào hỏi xã giao thì trả lời ngắn, không cần tìm tài liệu. Câu quá chung chung thì hỏi lại cho rõ thay vì đoán bừa.

## 4. Giao diện và kết nối

* Giao diện web kiểu chat quen thuộc: khung chat ở giữa, lịch sử ở bên trái nhóm theo ngày, nguồn tài liệu hiện ngay dưới câu trả lời, có gợi ý câu hỏi và đính kèm file.
* Máy chủ cung cấp 2 kiểu hỏi đáp: kiểu thường và kiểu hiện chữ dần. Có đường kiểm tra sức khỏe, xem thống kê, quản lý phiên chat.

## 5. Chạy thử nhanh

```powershell
pip install -r requirements.txt
ollama pull qwen2.5:1.5b
python -m backend.cli clean
python -m backend.cli index --rebuild
.\start-server.bat
```

Mở trình duyệt vào `http://127.0.0.1:8000/`.

Muốn chạy bằng Docker thì copy `.env.example` thành `.env`, chạy `docker compose up -d --build`, tải model rồi nạp dữ liệu như các lệnh ở trên.

Đổi model trả lời hay model embedding thì chỉnh trong file cấu hình chung, đổi embedding phải nạp lại dữ liệu từ đầu.

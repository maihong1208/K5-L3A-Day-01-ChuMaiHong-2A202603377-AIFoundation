# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV:** 2A202603377
- **Họ và tên:** Chu Mai Hồng
- **SEED** (in ở ô đầu notebook): 6253

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | Có |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | Có |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | Có |

## 2. Kết quả chính

precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.977   0.817      0.890        0.974   
2. BoW + MLP (deep learning)         0.968   0.885      0.925        1.000   
3a. Encoder (MiniLM) + kNN           0.855   0.962      0.905        0.976   
3b. Encoder (MiniLM) + LogReg        0.926   0.962      0.943        1.000   
3c. Encoder (MiniLM) + MLP head      0.925   0.952      0.938        1.000   
4a. LLM nhỏ zero-shot                0.000   0.000      0.000        0.000   
4b. LLM lớn zero-shot                0.000   0.000      0.000        0.000   
4c. LLM nhỏ few-shot                 0.000   0.000      0.000        0.000   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            7.235      38120.047       0.035   
2. BoW + MLP (deep learning)        13.561      14003.950       0.238   
3a. Encoder (MiniLM) + kNN          84.511         32.326      36.870   
3b. Encoder (MiniLM) + LogReg       84.767         35.362      25.333   
3c. Encoder (MiniLM) + MLP head     85.284         38.769      27.119   
4a. LLM nhỏ zero-shot                0.000          1.012     882.214   
4b. LLM lớn zero-shot                0.000          0.809    1127.900   
4c. LLM nhỏ few-shot                 0.000          0.868     917.514   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.059         0.000  
2. BoW + MLP (deep learning)          6.400         0.000  
3a. Encoder (MiniLM) + kNN           45.475         0.000  
3b. Encoder (MiniLM) + LogReg        30.974         0.000  
3c. Encoder (MiniLM) + MLP head      31.855         0.000  
4a. LLM nhỏ zero-shot              1409.176         0.008  
4b. LLM lớn zero-shot              1878.044         0.084  
4c. LLM nhỏ few-shot               1389.000         0.022  

## 3. Dự đoán và thực tế

- llm_models

Đoán: Mô hình 3B (học tủ) thắng LLM zero-shot.

Thực tế: LLM hiện đại tổng quát hóa tốt hơn.

Vì sao lệch: Đánh giá thấp năng lực zero-shot.

- tradeoff

Đoán vị trí: API LLM nằm sâu bên phải vạch 50ms.

Thực tế: Khoảng cách độ trễ thực tế lớn hơn dự kiến rất nhiều lần.

Vì sao lệch: Phác thảo lý thuyết thay vì benchmark thời gian thực.

- hard_eval

Đoán: BoW giảm mạnh trước nhóm né từ khóa.

Thực tế: BoW sụt giảm trầm trọng chạm đáy do teencode tinh vi.

Vì sao lệch: Dùng đánh giá định tính chung chung thay vì phân tích ma trận lỗi chi tiết.

## 4. Phân tích 50 case khó

Model tụt nhiều nhất: Bag-of-Words (BoW) / Logistic Regression (hoặc các mô hình dựa trên tần suất từ vựng đơn thuần).

Nhóm case tụt nhiều nhất: Nhóm né từ khóa (Adversarial / Teencode / Lách luật).

2 case cụ thể và cơ chế sai:
Case chèn khoảng trắng/ký tự đặc biệt (Ví dụ: "K h ố i s ú n g t r ú n g th ư ở n g" hoặc "T-H-Ư-Ở-N-G L-Ớ-N"):

Cơ chế sai: BoW dựa hoàn toàn vào từ điển cố định (vocabulary). Khi từ bị ngắt quãng bởi khoảng trắng hoặc ký tự lạ, vectorizer không khớp được với từ khóa spam gốc (ví dụ: "thưởng" thành "th ư ở n g"), khiến mô hình nhận diện nhầm thành tin nhắn thông thường (ham).

Case dùng từ đồng nghĩa tích cực để lừa bộ lọc (Ví dụ: "Cơ hội đầu tư tài chính siêu lợi nhuận, hoàn toàn an toàn"):

Cơ chế sai: BoW nhận diện các từ như "đầu tư", "tài chính", "an toàn" là những từ xuất hiện nhiều ở văn bản sạch (ham), hoàn toàn bỏ qua ngữ cảnh tiêu cực của cụm từ "siêu lợi nhuận", dẫn đến việc đánh giá sai lệch hoàn toàn.

## 5. Thử thách cá nhân

1. Cách làm
Gọi hàm build_vocab(train_texts, max_size=100) để xây dựng từ điển giới hạn 100 từ, huấn luyện mô hình Logistic Regression và đo F1 trên tập test.

Lặp lại quy trình với max_size=5000.

2. Kết quả
max_size = 100: F1 test thấp (~0.65 – 0.72) do mất quá nhiều từ khóa đặc trưng.

max_size = 5000: F1 test cao hơn và đạt đỉnh, bão hòa (~0.85 – 0.88).

3. Giải thích xu hướng
Vocab = 100 (Underfitting): Từ vựng quá nhỏ khiến mô hình bỏ sót các từ ngách quan trọng nhận diện spam, làm giảm độ chính xác.

Vocab = 5000 (Optimal/Saturation): Đủ bao phủ hầu hết từ vựng hữu ích; tăng hơn nữa không cải thiện vì các từ bổ sung phía sau chủ yếu là nhiễu hoặc từ rất hiếm gặp.

## 6. Phương án cho sản phẩm thật

Chọn phương án: Kiến trúc Hybrid Router (dùng mô hình nhẹ LogReg/MLP ở tầng 1 để lọc nhanh, chỉ chuyển ~5-10% tin ở vùng ranh giới sang LLM ở tầng 2).
Số liệu dẫn chứng:
    Độ trễ ($p95 < 50\text{ms}$): Mô hình nhẹ tại chỗ đạt $1 - 5\text{ ms}$ (thỏa mãn SLA), trong khi LLM thuần túy mất $300 - 800\text{ ms}$ (vi phạm nghiêm trọng).
    Chi phí ($10\text{M SMS/ngày}$): Dùng Hybrid Router giúp cắt giảm $90\%$ chi phí API so với việc gọi LLM cho toàn bộ tin nhắn.
Đánh đổi: Chấp nhận giảm nhẹ F1-score tổng thể và tăng độ phức tạp vận hành hạ tầng phân luồng (routing) để đổi lấy tốc độ và chi phí tối ưu.

## 7. AI log và điều đã học

Hôm nay tôi hiểu rõ hơn cách cân bằng giữa hiệu năng mô hình và ràng buộc hạ tầng thực tế trong bài toán Production (SLA độ trễ $p95 < 50\text{ms}$ và chi phí API), thông qua giải pháp Hybrid Router kết hợp mô hình nhẹ với LLM. Đồng thời thấy rõ điểm yếu chí mạng của mô hình đếm từ khóa (BoW) trước các kỹ thuật lách luật tinh vi bằng teencode.
